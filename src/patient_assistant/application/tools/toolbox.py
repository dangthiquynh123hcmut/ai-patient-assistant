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

from patient_assistant.application.ports.llm import ToolSpec
from patient_assistant.application.ports.repositories import (
    CatalogRepository,
    KnowledgeRepository,
    PlanRepository,
)
from patient_assistant.application.services.booking_service import BookingService
from patient_assistant.application.services.report_service import ReportService
from patient_assistant.application.tools.result import ToolErrorCode, ToolResult
from patient_assistant.domain.booking.models import Department, Doctor
from patient_assistant.domain.booking.policy import BookingPolicy
from patient_assistant.domain.report.models import ReportExtraction
from patient_assistant.domain.text.vi_dates import DateWindow

WRITE_TOOLS = frozenset({"save_patient_report", "create_appointment"})


async def _guarded(tool: str, call: Callable[[], Awaitable[ToolResult]]) -> ToolResult:
    """Đo thời gian và biến exception bất ngờ thành lỗi chuẩn hóa — không bao giờ thành công giả."""
    started = time.perf_counter()
    try:
        result = await call()
    except Exception as exc:  # noqa: BLE001 - ranh giới hạ tầng
        result = ToolResult.failure(
            tool, ToolErrorCode.UNAVAILABLE, f"Lỗi hệ thống: {type(exc).__name__}", retryable=True
        )
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Câu truy vấn tìm trong tài liệu hướng dẫn đã duyệt")


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlotsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    department_code: str
    date_from: date = Field(description="Ngày cụ thể YYYY-MM-DD đã được hệ thống quy đổi")
    date_to: date


class Toolbox:
    def __init__(
        self,
        reports: ReportService,
        booking: BookingService,
        catalog: CatalogRepository,
        plans: PlanRepository,
        knowledge: KnowledgeRepository,
        slot_limit: int = 3,
    ) -> None:
        self._reports = reports
        self._booking = booking
        self._catalog = catalog
        self._plans = plans
        self._knowledge = knowledge
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

    async def get_report_history(self, patient_id: str, limit: int = 5) -> ToolResult:
        return await _guarded("get_report_history", lambda: self._reports.history(patient_id, limit))

    async def get_report(self, patient_id: str, report_id: str) -> ToolResult:
        return await _guarded("get_report", lambda: self._reports.get(patient_id, report_id))

    async def search_approved_knowledge(self, query: str, limit: int = 3) -> ToolResult:
        async def call() -> ToolResult:
            docs = await self._knowledge.search(query, limit)
            return ToolResult.success(
                "search_approved_knowledge", [d.model_dump() for d in docs], [d.doc_id for d in docs]
            )

        return await _guarded("search_approved_knowledge", call)

    async def get_available_slots(
        self, department_code: str, window: DateWindow, doctor_id: str | None, now: datetime
    ) -> ToolResult:
        return await _guarded(
            "get_available_slots",
            lambda: self._booking.search(department_code, window, doctor_id, self._slot_limit, now),
        )

    async def get_alternative_slots(self, department_code: str, window: DateWindow, now: datetime) -> ToolResult:
        return await _guarded(
            "get_available_slots", lambda: self._booking.alternatives(department_code, window, self._slot_limit, now)
        )

    async def list_my_appointments(self, patient_id: str, now: datetime) -> ToolResult:
        return await _guarded("list_my_appointments", lambda: self._booking.upcoming(patient_id, now))

    async def read_back_appointment(self, patient_id: str, appointment_id: str) -> ToolResult:
        async def call() -> ToolResult:
            appt = await self._booking.get_appointment(patient_id, appointment_id)
            if appt is None:
                return ToolResult.failure("get_appointment", ToolErrorCode.NOT_FOUND, "Không đọc lại được lịch hẹn.")
            return ToolResult.success("get_appointment", appt.model_dump(mode="json"), [appt.id])

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

    # ------------------------------------------------------------------ tool cho LLM

    def read_only_specs(self, patient_id: str, now: datetime) -> list[ToolSpec]:
        async def search(inp: BaseModel) -> dict[str, Any]:
            assert isinstance(inp, SearchKnowledgeInput)
            return (await self.search_approved_knowledge(inp.query)).model_dump(mode="json")

        async def plan(_: BaseModel) -> dict[str, Any]:
            return (await self.get_patient_plan(patient_id)).model_dump(mode="json")

        async def slots(inp: BaseModel) -> dict[str, Any]:
            assert isinstance(inp, SlotsInput)
            window = DateWindow(date_from=inp.date_from, date_to=inp.date_to)
            return (await self.get_available_slots(inp.department_code, window, None, now)).model_dump(mode="json")

        async def appointments(_: BaseModel) -> dict[str, Any]:
            return (await self.list_my_appointments(patient_id, now)).model_dump(mode="json")

        return [
            ToolSpec(
                "search_approved_knowledge",
                "Tìm đoạn trong tài liệu hướng dẫn đã được duyệt. Trả về doc_id để dẫn nguồn.",
                SearchKnowledgeInput,
                search,
            ),
            ToolSpec(
                "get_patient_plan", "Lấy kế hoạch theo dõi hiện tại của bệnh nhân đang đăng nhập.", EmptyInput, plan
            ),
            ToolSpec(
                "get_available_slots",
                "Xem slot khám còn trống theo khoa và khoảng ngày cụ thể (chỉ để tham khảo, không đặt lịch).",
                SlotsInput,
                slots,
            ),
            ToolSpec(
                "list_my_appointments",
                "Xem các lịch hẹn sắp tới của bệnh nhân đang đăng nhập.",
                EmptyInput,
                appointments,
            ),
        ]
