"""Dữ liệu của chính bệnh nhân đang đăng nhập, và cấu hình đang chạy cho UI."""

from typing import Annotated, Any

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException

from patient_assistant.application.ports.clock import Clock
from patient_assistant.application.ports.repositories import ConversationLogRepository
from patient_assistant.application.services.auth_service import AuthService
from patient_assistant.application.tools.toolbox import Toolbox
from patient_assistant.config import Settings
from patient_assistant.container import Container
from patient_assistant.domain.account import ChatMessage, ConversationSummary
from patient_assistant.interface.api.dependencies import AuthenticatedUser, PatientUser

router = APIRouter(prefix="/v1", tags=["me"])

ToolboxDependency = Annotated[Toolbox, Depends(Provide[Container.toolbox])]
ClockDependency = Annotated[Clock, Depends(Provide[Container.clock])]
LogDependency = Annotated[ConversationLogRepository, Depends(Provide[Container.log])]


@router.get("/meta")
@inject
async def meta(
    user: AuthenticatedUser,
    toolbox: ToolboxDependency,
    clock: ClockDependency,
    settings: Annotated[Settings, Depends(Provide[Container.settings])],
    auth: Annotated[AuthService, Depends(Provide[Container.auth])],
    patient_labels: Annotated[dict[str, str], Depends(Provide[Container.patient_labels])],
) -> dict[str, Any]:
    """Cấu hình đang chạy và dữ liệu giả lập — để UI hiển thị gợi ý."""
    departments, doctors = await toolbox.catalog()
    return {
        "user": user.model_dump(mode="json"),
        "llm_model": settings.llm_model_id,
        "stt_enabled": settings.stt_enabled,
        "stt_model": settings.stt_model if settings.stt_enabled else None,
        "stt_language": settings.stt_language,
        "now": clock.now().isoformat(),
        "patients": [
            {
                "id": patient.id,
                "label": patient_labels.get(patient.id, patient.display_name),
                "display_name": patient.display_name,
                "date_of_birth": patient.date_of_birth.isoformat() if patient.date_of_birth else None,
                "phone": patient.phone,
            }
            for patient in (await auth.list_patients() if user.can_view_conversations else [])
        ],
        "departments": [department.model_dump() for department in departments],
        "doctors": [doctor.model_dump() for doctor in doctors],
        "limits": {
            "max_corrections": settings.max_corrections,
            "max_clarify_rounds": settings.max_clarify_rounds,
            "max_questions": settings.max_questions,
            "max_slot_search_rounds": settings.max_slot_search_rounds,
        },
    }


@router.get("/me/appointments")
@inject
async def my_appointments(toolbox: ToolboxDependency, clock: ClockDependency, user: PatientUser) -> Any:
    assert user.patient_id
    result = await toolbox.list_my_appointments(user.patient_id, clock.now())
    if not result.ok:
        raise HTTPException(503, "Không đọc được lịch hẹn")
    return result.data


@router.get("/me/reports")
@inject
async def my_reports(toolbox: ToolboxDependency, user: PatientUser) -> Any:
    assert user.patient_id
    result = await toolbox.get_report_history(user.patient_id, limit=20)
    if not result.ok:
        raise HTTPException(503, "Không đọc được báo cáo")
    return result.data


@router.get("/me/conversations")
@inject
async def my_conversations(log: LogDependency, user: PatientUser) -> list[ConversationSummary]:
    return await log.list_conversations(user.patient_id, 50)


@router.get("/me/conversations/{conversation_id}")
@inject
async def my_conversation(conversation_id: str, log: LogDependency, user: PatientUser) -> list[ChatMessage]:
    if await log.owner(conversation_id) != user.patient_id:
        raise HTTPException(404, "Không có hội thoại này")
    messages: list[ChatMessage] = await log.list_messages(conversation_id)
    return messages
