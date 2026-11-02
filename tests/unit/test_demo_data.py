from __future__ import annotations

from datetime import timedelta

from patient_assistant.infrastructure.persistence import seed
from tests.conftest import NOW, TZ


def test_demo_json_loads_with_valid_references() -> None:
    assert {d.code for d in seed.DEPARTMENTS} == {"PHCN", "CXK", "NOI"}
    assert {d.department_code for d in seed.DOCTORS} <= {d.code for d in seed.DEPARTMENTS}
    assert seed.DEMO_PATIENT_ID in {p.id for p in seed.PATIENTS}
    assert {p.patient_id for p in seed.PLANS} <= {p.id for p in seed.PATIENTS}


def test_slots_follow_schedule() -> None:
    slots = seed.generate_slots(NOW.date(), TZ)
    assert slots and all(s.start_at.weekday() in seed.SCHEDULE.working_weekdays for s in slots)
    assert all(s.end_at - s.start_at == timedelta(minutes=seed.SCHEDULE.slot_minutes) for s in slots)
    assert len({s.slot_id for s in slots}) == len(slots)


def test_history_is_relative_to_today_and_points_to_real_slots() -> None:
    reports = seed.history_reports(NOW.date(), TZ)
    assert [r.reporting_period for r in reports if r.patient_id == seed.DEMO_PATIENT_ID] == [
        "2026-W41",
        "2026-W42",
        "2026-W43",
    ]
    slots = seed.generate_slots(NOW.date(), TZ)
    appointments = seed.history_appointments(NOW.date(), slots)
    assert all(a.slot in slots and a.slot.start_at.date() > NOW.date() for a in appointments)
