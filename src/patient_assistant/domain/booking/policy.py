"""Quy tắc nghiệp vụ đặt lịch. Dùng chung cho validator của agent và `BookingService` ở backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from patient_assistant.domain.booking.models import Appointment, AppointmentStatus, Slot
from patient_assistant.domain.common import Layer, Resolution, ValidationIssue
from patient_assistant.domain.text.vi_dates import WEEKDAY_NAMES, DateWindow, DayParts


def _no_holidays(_: date) -> bool:
    return False


@dataclass(frozen=True)
class BookingPolicy:
    day_parts: DayParts = field(default_factory=DayParts)
    working_weekdays: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5})  # Thứ Hai – Thứ Bảy
    min_lead_time: timedelta = timedelta(hours=2)
    max_horizon_days: int = 30
    max_active_appointments: int = 3
    is_holiday: Callable[[date], bool] = _no_holidays

    def is_working_day(self, day: date) -> bool:
        return day.weekday() in self.working_weekdays and not self.is_holiday(day)


def _issue(code: str, message: str, resolution: Resolution = Resolution.ASK_USER) -> ValidationIssue:
    return ValidationIssue(
        field="date_expression", layer=Layer.BUSINESS, code=code, message=message, resolution=resolution
    )


def window_issues(window: DateWindow, now: datetime, policy: BookingPolicy) -> list[ValidationIssue]:
    today = now.date()
    if window.date_to < today:
        return [_issue("date_in_past", "Khoảng ngày đã qua.")]
    if window.date_from > today + timedelta(days=policy.max_horizon_days):
        return [_issue("too_far", f"Chỉ nhận đặt lịch trong vòng {policy.max_horizon_days} ngày tới.")]
    days = [window.date_from + timedelta(days=i) for i in range((window.date_to - window.date_from).days + 1)]
    if not any(policy.is_working_day(d) for d in days):
        if len(days) == 1:
            d = days[0]
            why = "ngày nghỉ lễ" if policy.is_holiday(d) else "ngày không làm việc"
            return [_issue("non_working_day", f"{WEEKDAY_NAMES[d.weekday()]} {d:%d/%m/%Y} là {why}.")]
        return [_issue("non_working_day", "Khoảng ngày này không có ngày làm việc.")]
    return []


def slot_issues(
    slot: Slot,
    now: datetime,
    policy: BookingPolicy,
    patient_appointments: list[Appointment],
) -> list[ValidationIssue]:
    """Kiểm tra lại một slot ngay trước khi ghi (backend, FR-09.8)."""
    issues: list[ValidationIssue] = []
    local_day = slot.start_at.astimezone(now.tzinfo).date()
    if slot.start_at < now + policy.min_lead_time:
        issues.append(_issue("too_soon", "Slot quá gần thời điểm hiện tại.", Resolution.BLOCK))
    if local_day > now.date() + timedelta(days=policy.max_horizon_days):
        issues.append(_issue("too_far", "Slot nằm ngoài khoảng được phép đặt.", Resolution.BLOCK))
    if not policy.is_working_day(local_day):
        issues.append(_issue("non_working_day", "Slot rơi vào ngày không làm việc.", Resolution.BLOCK))
    active = [a for a in patient_appointments if a.status == AppointmentStatus.BOOKED and a.slot.start_at >= now]
    if len(active) >= policy.max_active_appointments:
        issues.append(
            _issue(
                "too_many_active",
                f"Đã có {len(active)} lịch hẹn sắp tới (tối đa {policy.max_active_appointments}).",
                Resolution.BLOCK,
            )
        )
    for appt in active:
        if appt.slot.start_at < slot.end_at and slot.start_at < appt.slot.end_at:
            issues.append(_issue("overlapping_appointment", "Trùng giờ với một lịch hẹn đã có.", Resolution.BLOCK))
            break
    return issues
