"""Bộ tool của agent.

- Tool đọc: có thể được expose cho LLM qua `read_only_specs` (patient_id đã bind sẵn, LLM không điền).
- Tool ghi (`save_patient_report`, `create_appointment`): KHÔNG BAO GIỜ expose cho LLM; chỉ node code gọi
  sau khi validation đạt và người dùng xác nhận.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from patient_assistant.application.ports.llm import ToolSpecification
from patient_assistant.application.ports.repositories import (
    CatalogRepository,
    KnowledgeRepository,
    PatientRepository,
    PlanRepository,
)
from patient_assistant.application.services.booking_service import BookingService
from patient_assistant.application.services.report_service import ReportService
from patient_assistant.application.tools.result import ToolErrorCode, ToolResult
from patient_assistant.domain.booking.models import Department, Doctor
from patient_assistant.domain.booking.policy import BookingPolicy
from patient_assistant.domain.report.models import ReportExtraction
from patient_assistant.domain.text.vi_dates import DateWindow

WRITE_TOOLS = frozenset({"save_patient_report", "create_appointment", "cancel_appointment", "reschedule_appointment"})


async def _guarded(tool: str, call: Callable[[], Awaitable[ToolResult]]) -> ToolResult:
    """Đo thời gian và biến exception bất ngờ thành lỗi chuẩn hóa — không bao giờ thành công giả."""
    started = time.perf_counter()
    try:
        result = await call()
    except Exception as exception:  # noqa: BLE001 - ranh giới hạ tầng
        result = ToolResult.failure(
            tool, ToolErrorCode.UNAVAILABLE, f"Lỗi hệ thống: {type(exception).__name__}", retryable=True
        )
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Search query for the approved guidance documents (in the user's language)")


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlotsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    department_code: str
    date_from: date = Field(description="Concrete date YYYY-MM-DD, already resolved by the system")
    date_to: date


class Toolbox:
    def __init__(
        self,
        reports: ReportService,
        booking: BookingService,
        catalog: CatalogRepository,
        plans: PlanRepository,
        knowledge: KnowledgeRepository,
        patients: PatientRepository,
        slot_limit: int = 3,
    ) -> None:
        self._reports = reports
        self._booking = booking
        self._catalog = catalog
        self._plans = plans
        self._knowledge = knowledge
        self._patients = patients
        self._slot_limit = slot_limit

    @property
    def booking_policy(self) -> BookingPolicy:
        return self._booking.policy

    # ------------------------------------------------------------------ đọc

    async def catalog(self) -> tuple[list[Department], list[Doctor]]:
        return await self._catalog.list_departments(), await self._catalog.list_doctors()

    async def get_patient_plan(self, patient_id: str) -> ToolResult:
        async def call() -> ToolResult:
            plan = await self._plans.get_active_plan(patient_id)
            if plan is None:
                return ToolResult.failure("get_patient_plan", ToolErrorCode.NOT_FOUND, "Chưa có kế hoạch theo dõi.")
            return ToolResult.success("get_patient_plan", plan.model_dump(mode="json"), [plan.plan_id])

        return await _guarded("get_patient_plan", call)

    async def get_patient_summary(self, patient_id: str, now: datetime) -> ToolResult:
        """Tổng hợp thông tin của CHÍNH bệnh nhân đang đăng nhập: hồ sơ, kế hoạch, báo cáo gần đây, lịch hẹn.

        Chỉ đọc; ghép nhiều nguồn backend bằng code. patient_id luôn lấy từ phiên đăng nhập.
        """

        async def call() -> ToolResult:
            profile = await self._patients.get_profile(patient_id)
            if profile is None:
                return ToolResult.failure("get_patient_summary", ToolErrorCode.NOT_FOUND, "Không có hồ sơ bệnh nhân.")
            plan = await self._plans.get_active_plan(patient_id)
            departments = {department.code: department.name for department in await self._catalog.list_departments()}
            reports = (await self._reports.history(patient_id, 4)).data or []
            upcoming = (await self._booking.upcoming(patient_id, now)).data or []
            data = {
                "profile": profile.model_dump(),
                "plan": plan.model_dump() | {"department_name": departments.get(plan.department_code)}
                if plan
                else None,
                "recent_reports": reports,
                "upcoming_appointments": upcoming,
            }
            ids = [
                profile.id,
                *([plan.plan_id] if plan else []),
                *[report["id"] for report in reports],
                *[upcoming_appointment["id"] for upcoming_appointment in upcoming],
            ]
            return ToolResult.success("get_patient_summary", data, ids)

        return await _guarded("get_patient_summary", call)

    async def get_report_history(self, patient_id: str, limit: int = 5) -> ToolResult:
        return await _guarded("get_report_history", lambda: self._reports.history(patient_id, limit))

    async def get_report(self, patient_id: str, report_id: str) -> ToolResult:
        return await _guarded("get_report", lambda: self._reports.get(patient_id, report_id))

    async def search_approved_knowledge(self, query: str, limit: int = 3) -> ToolResult:
        async def call() -> ToolResult:
            documents = await self._knowledge.search(query, limit)
            return ToolResult.success(
                "search_approved_knowledge",
                [document.model_dump() for document in documents],
                [document.doc_id for document in documents],
            )

        return await _guarded("search_approved_knowledge", call)

    async def get_available_slots(
        self, patient_id: str, department_code: str, window: DateWindow, doctor_id: str | None, now: datetime
    ) -> ToolResult:
        return await _guarded(
            "get_available_slots",
            lambda: self._booking.search(patient_id, department_code, window, doctor_id, self._slot_limit, now),
        )

    async def get_alternative_slots(
        self, patient_id: str, department_code: str, window: DateWindow, now: datetime
    ) -> ToolResult:
        return await _guarded(
            "get_available_slots",
            lambda: self._booking.alternatives(patient_id, department_code, window, self._slot_limit, now),
        )

    async def find_my_appointments(
        self,
        patient_id: str,
        now: datetime,
        when: str = "upcoming",
        department_code: str | None = None,
        limit: int = 5,
    ) -> ToolResult:
        """Lịch hẹn của bệnh nhân đang đăng nhập, chỉ gồm các trường được phép đưa cho LLM."""

        async def call() -> ToolResult:
            result = await self._booking.find(patient_id, now, when, department_code, limit)
            if not result.ok:
                return result
            names = {department.code: department.name for department in await self._catalog.list_departments()}
            result.data = [
                {
                    "id": appointment["id"],
                    "slot_id": appointment["slot"]["slot_id"],
                    "department_code": appointment["slot"]["department_code"],
                    "department_name": names.get(appointment["slot"]["department_code"]),
                    "doctor_id": appointment["slot"]["doctor_id"],
                    "doctor_name": appointment["slot"]["doctor_name"],
                    "start_at": appointment["slot"]["start_at"],
                    "end_at": appointment["slot"]["end_at"],
                    "reason": appointment.get("reason"),
                }
                for appointment in result.data
            ]
            return result

        return await _guarded("find_my_appointments", call)

    async def list_my_appointments(self, patient_id: str, now: datetime) -> ToolResult:
        return await _guarded("list_my_appointments", lambda: self._booking.upcoming(patient_id, now))

    async def read_back_appointment(self, patient_id: str, appointment_id: str) -> ToolResult:
        async def call() -> ToolResult:
            appointment = await self._booking.get_appointment(patient_id, appointment_id)
            if appointment is None:
                return ToolResult.failure("get_appointment", ToolErrorCode.NOT_FOUND, "Không đọc lại được lịch hẹn.")
            return ToolResult.success("get_appointment", appointment.model_dump(mode="json"), [appointment.id])

        return await _guarded("get_appointment", call)

    # ------------------------------------------------------------------ ghi (chỉ code gọi)

    async def save_patient_report(
        self, patient_id: str, draft: ReportExtraction, idempotency_key: str, now: datetime
    ) -> ToolResult:
        return await _guarded(
            "save_patient_report", lambda: self._reports.save(patient_id, draft, idempotency_key, now)
        )

    async def create_appointment(
        self, patient_id: str, slot_id: str, reason: str | None, idempotency_key: str, now: datetime
    ) -> ToolResult:
        return await _guarded(
            "create_appointment", lambda: self._booking.book(patient_id, slot_id, reason, idempotency_key, now)
        )

    async def cancel_appointment(self, patient_id: str, appointment_id: str, now: datetime) -> ToolResult:
        return await _guarded("cancel_appointment", lambda: self._booking.cancel(patient_id, appointment_id, now))

    async def reschedule_appointment(
        self,
        patient_id: str,
        appointment_id: str,
        slot_id: str,
        reason: str | None,
        idempotency_key: str,
        now: datetime,
    ) -> ToolResult:
        return await _guarded(
            "reschedule_appointment",
            lambda: self._booking.reschedule(patient_id, appointment_id, slot_id, reason, idempotency_key, now),
        )

    # ------------------------------------------------------------------ tool cho LLM

    def read_only_specifications(self, patient_id: str, now: datetime) -> list[ToolSpecification]:
        async def search(input_data: BaseModel) -> dict[str, Any]:
            assert isinstance(input_data, SearchKnowledgeInput)
            return (await self.search_approved_knowledge(input_data.query)).model_dump(mode="json")

        async def plan(_: BaseModel) -> dict[str, Any]:
            return (await self.get_patient_plan(patient_id)).model_dump(mode="json")

        async def slots(input_data: BaseModel) -> dict[str, Any]:
            assert isinstance(input_data, SlotsInput)
            window = DateWindow(date_from=input_data.date_from, date_to=input_data.date_to)
            return (
                await self.get_available_slots(patient_id, input_data.department_code, window, None, now)
            ).model_dump(mode="json")

        async def appointments(_: BaseModel) -> dict[str, Any]:
            return (await self.list_my_appointments(patient_id, now)).model_dump(mode="json")

        async def summary(_: BaseModel) -> dict[str, Any]:
            result = (await self.get_patient_summary(patient_id, now)).model_dump(mode="json")
            # LLM không cần ngày sinh / số điện thoại để trả lời -> không gửi dữ liệu cá nhân này cho LLM.
            for key in ("date_of_birth", "phone"):
                ((result.get("data") or {}).get("profile") or {}).pop(key, None)
            return result

        return [
            ToolSpecification(
                "search_approved_knowledge",
                "Search passages in the approved guidance documents. Returns doc_id for citing the source.",
                SearchKnowledgeInput,
                search,
            ),
            ToolSpecification(
                "get_patient_plan", "Get the logged-in patient's current follow-up plan.", EmptyInput, plan
            ),
            ToolSpecification(
                "get_available_slots",
                "View free appointment slots by department and concrete date range (reference only; does not book).",
                SlotsInput,
                slots,
            ),
            ToolSpecification(
                "get_patient_summary",
                "Summary of the logged-in patient's record, follow-up plan, recent reports and appointments.",
                EmptyInput,
                summary,
            ),
            ToolSpecification(
                "list_my_appointments",
                "View the logged-in patient's upcoming appointments.",
                EmptyInput,
                appointments,
            ),
        ]
