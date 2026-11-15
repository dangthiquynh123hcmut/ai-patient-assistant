"""Nhân viên y tế xem hội thoại; quản trị viên quản lý tài khoản."""

from typing import Annotated

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException

from patient_assistant.application.ports.repositories import ConversationLogRepository, UsernameTakenError
from patient_assistant.application.services.auth_service import AuthError, AuthService
from patient_assistant.container import Container
from patient_assistant.domain.account import ChatMessage, ConversationSummary, NewUser, User
from patient_assistant.interface.api.dependencies import AdminUser, StaffUser
from patient_assistant.interface.api.schemas import UserUpdateIn

router = APIRouter(prefix="/v1", tags=["staff"])

LogDependency = Annotated[ConversationLogRepository, Depends(Provide[Container.log])]
AuthDependency = Annotated[AuthService, Depends(Provide[Container.auth])]


# ------------------------------------------------------------------ nhân viên y tế


@router.get("/staff/conversations")
@inject
async def staff_conversations(
    log: LogDependency, _: StaffUser, patient_id: str | None = None, limit: int = 100
) -> list[ConversationSummary]:
    return await log.list_conversations(patient_id, min(max(limit, 1), 500))


@router.get("/staff/conversations/{conversation_id}")
@inject
async def staff_conversation(conversation_id: str, log: LogDependency, _: StaffUser) -> list[ChatMessage]:
    if await log.owner(conversation_id) is None:
        raise HTTPException(404, "Không có hội thoại này")
    messages: list[ChatMessage] = await log.list_messages(conversation_id)
    return messages


# ------------------------------------------------------------------ quản trị tài khoản


@router.get("/admin/users")
@inject
async def list_users(auth: AuthDependency, _: AdminUser) -> list[User]:
    return await auth.list_users()


@router.post("/admin/users", status_code=201)
@inject
async def create_user(body: NewUser, auth: AuthDependency, _: AdminUser) -> User:
    try:
        return await auth.create_user(body)
    except UsernameTakenError as exception:
        raise HTTPException(409, "Username đã tồn tại") from exception
    except AuthError as exception:
        raise HTTPException(400, str(exception)) from exception


@router.patch("/admin/users/{user_id}")
@inject
async def update_user(user_id: str, body: UserUpdateIn, auth: AuthDependency, actor: AdminUser) -> User:
    try:
        return await auth.update_user(actor, user_id, active=body.active, password=body.password)
    except AuthError as exception:
        raise HTTPException(400, str(exception)) from exception
