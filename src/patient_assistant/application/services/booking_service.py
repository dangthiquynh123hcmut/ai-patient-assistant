"""Nghiệp vụ đặt lịch phía backend: tìm slot, nới điều kiện, đặt lịch có kiểm tra lại và chống trùng."""

from __future__ import annotations

from datetime import datetime, time, timedelta

from patient_assistant.application.ports.repositories import (
    AppointmentRepository,
    SlotRepository,
    SlotTakenError,
)
from patient_assistant.application.tools.result import ToolErrorCode, ToolResult
from patient_assistant.domain.booking.models import Appointment, AppointmentStatus
from patient_assistant.domain.booking.policy import BookingPolicy, slot_issues
from patient_assistant.domain.booking.ranking import SlotFit, fit_to_schedule
from patient_assistant.domain.text.vi_dates import DateWindow

ALTERNATIVE_SPAN_DAYS = 7
# Số slot trống lấy từ DB trước khi lọc/xếp hạng — đủ rộng để thấy cả những ngày bệnh nhân đã có lịch.
CANDIDATE_POOL = 200
# Lỗi khi ghi mà đổi sang slot khác là xử lý được: tìm lại, giữ nguyên yêu cầu thay vì kết thúc luồng.
RECOVERABLE_ISSUES = frozenset({"overlapping_appointment", "too_soon"})


class BookingService:
    def __init__(self, slots: SlotRepository, appointments: AppointmentRepository, policy: BookingPolicy) -> None:
        self._slots = slots
        self._appointments = appointments
        self.policy = policy

    def _bounds(self, window: DateWindow, now: datetime) -> tuple[datetime, datetime]:
        timezone = now.tzinfo
        start = datetime.combine(window.date_from, window.time_from or time(0, 0), timezone)
        end = datetime.combine(window.date_to, window.time_to or time(23, 59), timezone)
        return max(start, now + self.policy.min_lead_time), end

    async def _find(
        self,
        patient_id: str,
        department_code: str,
        start: datetime,
        end: datetime,
        doctor_id: str | None,
        limit: int,
        window: DateWindow | None,
        now: datetime,
    ) -> list[SlotFit]:
        found = await self._slots.find_available(department_code, start, end, doctor_id, CANDIDATE_POOL)
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
        # Bỏ slot trùng/sát lịch của chính bệnh nhân, đưa slot cùng ngày với lịch đã có lên đầu.
        mine = await self._appointments.list_for_patient(patient_id)
        return fit_to_schedule(result, mine, now, self.policy.travel_buffer)[:limit]

    @staticmethod
    def _slots_result(fits: list[SlotFit]) -> ToolResult:
        data = []
        source_ids = []
        for fit in fits:
            item = fit.slot.model_dump(mode="json")
            source_ids.append(fit.slot.slot_id)
            if fit.same_day:
                item["same_day_appointment"] = {
                    "id": fit.same_day.id,
                    "department_code": fit.same_day.slot.department_code,
                    "start_at": fit.same_day.slot.start_at.isoformat(),
                }
                source_ids.append(fit.same_day.id)
            data.append(item)
        return ToolResult.success("get_available_slots", data, list(dict.fromkeys(source_ids)))

    async def search(
        self,
        patient_id: str,
        department_code: str,
        window: DateWindow,
        doctor_id: str | None,
        limit: int,
        now: datetime,
    ) -> ToolResult:
        start, end = self._bounds(window, now)
        return self._slots_result(
            await self._find(patient_id, department_code, start, end, doctor_id, limit, window, now)
        )

    async def alternatives(
        self, patient_id: str, department_code: str, window: DateWindow, limit: int, now: datetime
    ) -> ToolResult:
        """Nới theo luật cố định: ±7 ngày quanh khoảng đã chọn, bỏ ràng buộc giờ và bác sĩ."""
        relaxed = DateWindow(
            date_from=max(window.date_from - timedelta(days=ALTERNATIVE_SPAN_DAYS), now.date()),
            date_to=window.date_to + timedelta(days=ALTERNATIVE_SPAN_DAYS),
        )
        start, end = self._bounds(relaxed, now)
        horizon = datetime.combine(now.date() + timedelta(days=self.policy.max_horizon_days), time(23, 59), now.tzinfo)
        return self._slots_result(
            await self._find(patient_id, department_code, start, min(end, horizon), None, limit, None, now)
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
        if issues and all(issue.code in RECOVERABLE_ISSUES for issue in issues):
            return ToolResult.failure(tool, ToolErrorCode.CONFLICT, " ".join(issue.message for issue in issues))
        if issues:
            return ToolResult.failure(
                tool,
                ToolErrorCode.INVALID_ARGUMENTS,
                "Không thể đặt slot này.",
                details=[issue.message for issue in issues],
            )
        try:
            appointment = await self._appointments.book(patient_id, slot, reason, idempotency_key)
        except SlotTakenError:
            return ToolResult.failure(
                tool, ToolErrorCode.CONFLICT, "Slot bạn chọn vừa được người khác đặt.", retryable=False
            )
        return ToolResult.success(tool, appointment.model_dump(mode="json"), [appointment.id])

    async def _own_upcoming(
        self, tool: str, patient_id: str, appointment_id: str, now: datetime
    ) -> Appointment | ToolResult:
        appointment = await self.get_appointment(patient_id, appointment_id)
        if appointment is None:
            return ToolResult.failure(tool, ToolErrorCode.NOT_FOUND, "Không có lịch hẹn này.")
        if appointment.status != AppointmentStatus.BOOKED:
            return ToolResult.failure(tool, ToolErrorCode.INVALID_ARGUMENTS, "Lịch hẹn này đã bị hủy trước đó.")
        if appointment.slot.start_at < now:
            return ToolResult.failure(tool, ToolErrorCode.INVALID_ARGUMENTS, "Lịch hẹn này đã qua, không đổi/hủy được.")
        return appointment

    async def cancel(self, patient_id: str, appointment_id: str, now: datetime) -> ToolResult:
        tool = "cancel_appointment"
        existing = await self.get_appointment(patient_id, appointment_id)
        if existing and existing.status == AppointmentStatus.CANCELLED:  # gọi lại sau lỗi mạng -> như đã hủy
            return ToolResult.success(tool, existing.model_dump(mode="json"), [existing.id])
        appointment = await self._own_upcoming(tool, patient_id, appointment_id, now)
        if isinstance(appointment, ToolResult):
            return appointment
        cancelled = await self._appointments.cancel(appointment.id)
        if cancelled is None:
            return ToolResult.failure(tool, ToolErrorCode.NOT_FOUND, "Không có lịch hẹn này.")
        return ToolResult.success(tool, cancelled.model_dump(mode="json"), [cancelled.id])

    async def reschedule(
        self,
        patient_id: str,
        appointment_id: str,
        slot_id: str,
        reason: str | None,
        idempotency_key: str,
        now: datetime,
    ) -> ToolResult:
        """Dời lịch: đặt slot mới và hủy lịch cũ trong một transaction. Lịch cũ không tính là trùng với lịch mới."""
        tool = "reschedule_appointment"
        existing = await self._appointments.get_by_idempotency_key(idempotency_key)
        if existing:
            if existing.patient_id != patient_id:
                return ToolResult.failure(tool, ToolErrorCode.FORBIDDEN, "Khóa idempotency không hợp lệ.")
            return ToolResult.success(tool, existing.model_dump(mode="json"), [existing.id])
        appointment = await self._own_upcoming(tool, patient_id, appointment_id, now)
        if isinstance(appointment, ToolResult):
            return appointment
        slot = await self._slots.get(slot_id)
        if slot is None:
            return ToolResult.failure(tool, ToolErrorCode.NOT_FOUND, "Slot không tồn tại.")
        others = [
            other for other in await self._appointments.list_for_patient(patient_id) if other.id != appointment.id
        ]
        issues = slot_issues(slot, now, self.policy, others)
        if issues and all(issue.code in RECOVERABLE_ISSUES for issue in issues):
            return ToolResult.failure(tool, ToolErrorCode.CONFLICT, " ".join(issue.message for issue in issues))
        if issues:
            return ToolResult.failure(
                tool,
                ToolErrorCode.INVALID_ARGUMENTS,
                "Không thể dời sang slot này.",
                details=[issue.message for issue in issues],
            )
        try:
            new = await self._appointments.reschedule(
                appointment.id, patient_id, slot, reason if reason is not None else appointment.reason, idempotency_key
            )
        except SlotTakenError:
            return ToolResult.failure(tool, ToolErrorCode.CONFLICT, "Slot bạn chọn vừa được người khác đặt.")
        return ToolResult.success(tool, new.model_dump(mode="json"), [new.id, appointment.id])

    async def get_appointment(self, patient_id: str, appointment_id: str) -> Appointment | None:
        appointment = await self._appointments.get(appointment_id)
        return appointment if appointment and appointment.patient_id == patient_id else None

    async def find(
        self,
        patient_id: str,
        now: datetime,
        when: str = "upcoming",
        department_code: str | None = None,
        limit: int = 5,
    ) -> ToolResult:
        """Lịch hẹn của chính bệnh nhân, kể cả lịch đã qua — để agent hiểu "lịch trước", "lịch hiện tại".

        `when`: upcoming (sắp tới, gần nhất trước) | past (đã qua, gần nhất trước) | all (mới nhất trước).
        Chỉ lịch đang `booked`; lịch đã hủy không dùng làm mốc.
        """
        appointments = [
            booked
            for booked in await self._appointments.list_for_patient(patient_id)
            if booked.status == AppointmentStatus.BOOKED
            and (department_code is None or booked.slot.department_code == department_code)
            and (when == "all" or (booked.slot.start_at >= now) == (when == "upcoming"))
        ]
        appointments.sort(key=lambda booked: booked.slot.start_at, reverse=when != "upcoming")
        appointments = appointments[:limit]
        return ToolResult.success(
            "find_my_appointments",
            [booked.model_dump(mode="json") for booked in appointments],
            [booked.id for booked in appointments],
        )

    async def upcoming(self, patient_id: str, now: datetime) -> ToolResult:
        appointments = [
            booked
            for booked in await self._appointments.list_for_patient(patient_id)
            if booked.status == AppointmentStatus.BOOKED and booked.slot.start_at >= now
        ]
        appointments.sort(key=lambda booked: booked.slot.start_at)
        return ToolResult.success(
            "list_my_appointments",
            [booked.model_dump(mode="json") for booked in appointments],
            [booked.id for booked in appointments],
        )
