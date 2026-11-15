"""Lọc và xếp slot theo lịch hẹn sẵn có của chính bệnh nhân — luật cố định bằng code, không dùng LLM.

- Bỏ slot trùng giờ hoặc cách một lịch đã có ít hơn `travel_buffer` (không kịp di chuyển giữa các khoa).
- Slot cùng ngày với một lịch đã có được đưa lên đầu: bệnh nhân chỉ phải đến bệnh viện một lần.
  Trong nhóm đó, slot sát lịch cũ hơn đứng trước; ngoài ra giữ nguyên thứ tự thời gian.

Hàm chỉ sắp xếp lại / bỏ bớt slot backend đã trả, không bao giờ thêm slot mới (FR-09.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo

from patient_assistant.domain.booking.models import Appointment, AppointmentStatus, Slot


@dataclass(frozen=True)
class SlotFit:
    slot: Slot
    same_day: Appointment | None = None  # lịch đã có cùng ngày, gần slot nhất


def gap_between(first: Slot, second: Slot) -> timedelta:
    """Khoảng trống giữa hai slot; âm nghĩa là trùng giờ."""
    return max(second.start_at - first.end_at, first.start_at - second.end_at)


def active_appointments(appointments: list[Appointment], now: datetime) -> list[Appointment]:
    return [
        appointment
        for appointment in appointments
        if appointment.status == AppointmentStatus.BOOKED and appointment.slot.end_at > now
    ]


def fit_to_schedule(
    slots: list[Slot], appointments: list[Appointment], now: datetime, travel_buffer: timedelta
) -> list[SlotFit]:
    timezone: tzinfo | None = now.tzinfo
    active = active_appointments(appointments, now)
    fits: list[tuple[timedelta | None, int, SlotFit]] = []
    for i, slot in enumerate(slots):
        gaps = [(gap_between(slot, appointment.slot), appointment) for appointment in active]
        if any(gap < travel_buffer for gap, _ in gaps):
            continue
        day = slot.start_at.astimezone(timezone).date()
        same_day = [
            (gap, appointment)
            for gap, appointment in gaps
            if appointment.slot.start_at.astimezone(timezone).date() == day
        ]
        nearest = min(same_day, key=lambda gap_pair: gap_pair[0], default=None)
        fits.append((nearest[0] if nearest else None, i, SlotFit(slot, nearest[1] if nearest else None)))
    # Cùng ngày trước (gap nhỏ trước), còn lại theo thứ tự backend trả (theo giờ).
    fits.sort(key=lambda candidate: (candidate[0] is None, candidate[0] or timedelta(0), candidate[1]))
    return [fit for _, _, fit in fits]
