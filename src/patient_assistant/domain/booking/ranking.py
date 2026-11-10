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


def gap_between(a: Slot, b: Slot) -> timedelta:
    """Khoảng trống giữa hai slot; âm nghĩa là trùng giờ."""
    return max(b.start_at - a.end_at, a.start_at - b.end_at)


def active_appointments(appointments: list[Appointment], now: datetime) -> list[Appointment]:
    return [a for a in appointments if a.status == AppointmentStatus.BOOKED and a.slot.end_at > now]


def fit_to_schedule(
    slots: list[Slot], appointments: list[Appointment], now: datetime, travel_buffer: timedelta
) -> list[SlotFit]:
    tz: tzinfo | None = now.tzinfo
    active = active_appointments(appointments, now)
    fits: list[tuple[timedelta | None, int, SlotFit]] = []
    for i, slot in enumerate(slots):
        gaps = [(gap_between(slot, a.slot), a) for a in active]
        if any(gap < travel_buffer for gap, _ in gaps):
            continue
        day = slot.start_at.astimezone(tz).date()
        same_day = [(gap, a) for gap, a in gaps if a.slot.start_at.astimezone(tz).date() == day]
        nearest = min(same_day, key=lambda x: x[0], default=None)
        fits.append((nearest[0] if nearest else None, i, SlotFit(slot, nearest[1] if nearest else None)))
    # Cùng ngày trước (gap nhỏ trước), còn lại theo thứ tự backend trả (theo giờ).
    fits.sort(key=lambda x: (x[0] is None, x[0] or timedelta(0), x[1]))
    return [f for _, _, f in fits]
