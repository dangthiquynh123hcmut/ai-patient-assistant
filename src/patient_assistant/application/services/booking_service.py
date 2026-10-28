"""Nghiệp vụ đặt lịch phía backend: tìm slot, nới điều kiện, đặt lịch có kiểm tra lại và chống trùng."""

from __future__ import annotations

from datetime import datetime, time, timedelta

from patient_assistant.application.ports.repositories import (
    AppointmentRepository,
    SlotRepository,
    SlotTakenError,
)
from patient_assistant.application.tools.result import ToolErrorCode, ToolResult
from patient_assistant.domain.booking.models import Appointment, AppointmentStatus, Slot
from patient_assistant.domain.booking.policy import BookingPolicy, slot_issues
from patient_assistant.domain.text.vi_dates import DateWindow

ALTERNATIVE_SPAN_DAYS = 7


class BookingService:
    def __init__(self, slots: SlotRepository, appointments: AppointmentRepository, policy: BookingPolicy) -> None:
        self._slots = slots
        self._appointments = appointments
        self.policy = policy

    def _bounds(self, window: DateWindow, now: datetime) -> tuple[datetime, datetime]:
        tz = now.tzinfo
        start = datetime.combine(window.date_from, window.time_from or time(0, 0), tz)
        end = datetime.combine(window.date_to, window.time_to or time(23, 59), tz)
        return max(start, now + self.policy.min_lead_time), end

    async def _find(
        self,
        department_code: str,
        start: datetime,
        end: datetime,
        doctor_id: str | None,
        limit: int,
        window: DateWindow | None,
        now: datetime,
    ) -> list[Slot]:
        found = await self._slots.find_available(department_code, start, end, doctor_id, limit * 4)
        result = []
        for slot in found:
            local = slot.start_at.astimezone(now.tzinfo)
            if not self.policy.is_working_day(local.date()):
                continue
            # Khung giờ áp dụng cho từng ngày trong khoảng, không chỉ ngày đầu/cuối.
            if (
                window
                and window.time_from
                and window.time_to
                and not (window.time_from <= local.time() <= window.time_to)
            ):
                continue
            result.append(slot)
        return result[:limit]

    async def search(
        self, department_code: str, window: DateWindow, doctor_id: str | None, limit: int, now: datetime
    ) -> ToolResult:
        start, end = self._bounds(window, now)
        slots = await self._find(department_code, start, end, doctor_id, limit, window, now)
        return ToolResult.success(
            "get_available_slots", [s.model_dump(mode="json") for s in slots], [s.slot_id for s in slots]
        )

    async def alternatives(self, department_code: str, window: DateWindow, limit: int, now: datetime) -> ToolResult:
        """Nới theo luật cố định: ±7 ngày quanh khoảng đã chọn, bỏ ràng buộc giờ và bác sĩ."""
        relaxed = DateWindow(
            date_from=max(window.date_from - timedelta(days=ALTERNATIVE_SPAN_DAYS), now.date()),
            date_to=window.date_to + timedelta(days=ALTERNATIVE_SPAN_DAYS),
        )
        start, end = self._bounds(relaxed, now)
        horizon = datetime.combine(now.date() + timedelta(days=self.policy.max_horizon_days), time(23, 59), now.tzinfo)
        slots = await self._find(department_code, start, min(end, horizon), None, limit, None, now)
        return ToolResult.success(
            "get_available_slots", [s.model_dump(mode="json") for s in slots], [s.slot_id for s in slots]
        )

    async def book(
        self, patient_id: str, slot_id: str, reason: str | None, idempotency_key: str, now: datetime
    ) -> ToolResult:
        tool = "create_appointment"
        existing = await self._appointments.get_by_idempotency_key(idempotency_key)
        if existing:
            if existing.patient_id != patient_id:
                return ToolResult.failure(tool, ToolErrorCode.FORBIDDEN, "Khóa idempotency không hợp lệ.")
            return ToolResult.success(tool, existing.model_dump(mode="json"), [existing.id])
        slot = await self._slots.get(slot_id)
        if slot is None:
            return ToolResult.failure(tool, ToolErrorCode.NOT_FOUND, "Slot không tồn tại.")
        mine = await self._appointments.list_for_patient(patient_id)
        issues = slot_issues(slot, now, self.policy, mine)
        if issues:
            return ToolResult.failure(
                tool, ToolErrorCode.INVALID_ARGS, "Không thể đặt slot này.", details=[i.message for i in issues]
            )
        try:
            appt = await self._appointments.book(patient_id, slot, reason, idempotency_key)
        except SlotTakenError:
            return ToolResult.failure(tool, ToolErrorCode.CONFLICT, "Slot vừa được người khác đặt.", retryable=False)
        return ToolResult.success(tool, appt.model_dump(mode="json"), [appt.id])

    async def get_appointment(self, patient_id: str, appointment_id: str) -> Appointment | None:
        appt = await self._appointments.get(appointment_id)
        return appt if appt and appt.patient_id == patient_id else None

    async def upcoming(self, patient_id: str, now: datetime) -> ToolResult:
        appts = [
            a
            for a in await self._appointments.list_for_patient(patient_id)
            if a.status == AppointmentStatus.BOOKED and a.slot.start_at >= now
        ]
        appts.sort(key=lambda a: a.slot.start_at)
        return ToolResult.success(
            "list_my_appointments", [a.model_dump(mode="json") for a in appts], [a.id for a in appts]
        )
