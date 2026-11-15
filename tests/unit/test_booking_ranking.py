from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from patient_assistant.domain.booking.models import Appointment, AppointmentStatus, Slot
from patient_assistant.domain.booking.ranking import fit_to_schedule

TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
NOW = datetime(2026, 10, 28, 10, 0, tzinfo=TIMEZONE)
BUFFER = timedelta(minutes=30)


def _slot(day: int, hh: int, mm: int = 0, department: str = "PHCN") -> Slot:
    start = datetime(2026, 11, day, hh, mm, tzinfo=TIMEZONE)
    return Slot(
        slot_id=f"{department}-{day}-{hh:02d}{mm:02d}",
        department_code=department,
        doctor_id="doc",
        doctor_name="BS. A",
        start_at=start,
        end_at=start + timedelta(minutes=30),
    )


def _appointment(slot: Slot, status: AppointmentStatus = AppointmentStatus.BOOKED) -> Appointment:
    return Appointment(
        id=f"appt-{slot.slot_id}",
        patient_id="p1",
        slot=slot,
        status=status,
        reason=None,
        idempotency_key=f"k-{slot.slot_id}",
        created_at=NOW,
    )


def _ids(fits: list) -> list[str]:
    return [fit.slot.slot_id for fit in fits]


def test_overlapping_and_too_close_slots_are_dropped_exact_buffer_is_kept() -> None:
    existing = _appointment(_slot(5, 14, 0, "TM"))  # 14:00–14:30
    slots = [_slot(5, 13, 30), _slot(5, 14, 0), _slot(5, 14, 30), _slot(5, 15, 0), _slot(5, 13, 0)]
    fits = fit_to_schedule(slots, [existing], NOW, BUFFER)
    # 13:30 và 14:30 sát ngay lịch cũ (khoảng trống 0), 14:00 trùng giờ. 13:00 và 15:00 cách đúng 30 phút.
    assert set(_ids(fits)) == {"PHCN-5-1300", "PHCN-5-1500"}
    assert all(fit.same_day == existing for fit in fits)


def test_same_day_slots_come_first_nearest_gap_first_others_keep_time_order() -> None:
    existing = _appointment(_slot(5, 14, 0, "TM"))
    slots = [_slot(3, 8), _slot(3, 9), _slot(5, 8), _slot(5, 15, 30), _slot(5, 15)]
    fits = fit_to_schedule(slots, [existing], NOW, BUFFER)
    assert _ids(fits) == ["PHCN-5-1500", "PHCN-5-1530", "PHCN-5-0800", "PHCN-3-0800", "PHCN-3-0900"]
    assert [fit.same_day is not None for fit in fits] == [True, True, True, False, False]


def test_cancelled_and_past_appointments_are_ignored() -> None:
    cancelled = _appointment(_slot(5, 9, 0, "TM"), AppointmentStatus.CANCELLED)
    past_slot = Slot(
        slot_id="past",
        department_code="TM",
        doctor_id="doc",
        doctor_name="BS. A",
        start_at=NOW - timedelta(days=1),
        end_at=NOW - timedelta(days=1) + timedelta(minutes=30),
    )
    slots = [_slot(5, 9), _slot(5, 10)]
    fits = fit_to_schedule(slots, [cancelled, _appointment(past_slot)], NOW, BUFFER)
    assert _ids(fits) == ["PHCN-5-0900", "PHCN-5-1000"]
    assert all(fit.same_day is None for fit in fits)


def test_nothing_is_added_only_reordered_or_dropped() -> None:
    existing = [_appointment(_slot(4, 9, 0, "TM")), _appointment(_slot(6, 15, 0, "NOI"))]
    slots = [_slot(day, hour) for day in (3, 4, 5, 6) for hour in (8, 9, 10, 14, 15)]
    fits = fit_to_schedule(slots, existing, NOW, BUFFER)
    assert set(_ids(fits)) <= {slot.slot_id for slot in slots}
    assert len(_ids(fits)) == len(set(_ids(fits)))
