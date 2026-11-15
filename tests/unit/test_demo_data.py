from __future__ import annotations

from datetime import timedelta

from patient_assistant.infrastructure.persistence import seed
from tests.conftest import NOW, TIMEZONE


def test_demo_json_loads_with_valid_references() -> None:
    assert {department.code for department in seed.DEPARTMENTS} == {
        "PHCN",
        "CXK",
        "NOI",
        "DL",
        "TM",
        "TK",
        "TMH",
        "MAT",
    }
    assert {doctor.department_code for doctor in seed.DOCTORS} <= {department.code for department in seed.DEPARTMENTS}
    assert seed.DEMO_PATIENT_ID in {patient.id for patient in seed.PATIENTS}
    assert {plan.patient_id for plan in seed.PLANS} <= {patient.id for patient in seed.PATIENTS}


def test_slots_follow_schedule() -> None:
    slots = seed.generate_slots(NOW.date(), TIMEZONE)
    assert slots and all(slot.start_at.weekday() in seed.SCHEDULE.working_weekdays for slot in slots)
    assert all(slot.end_at - slot.start_at == timedelta(minutes=seed.SCHEDULE.slot_minutes) for slot in slots)
    assert len({slot.slot_id for slot in slots}) == len(slots)


def test_history_is_relative_to_today_and_points_to_real_slots() -> None:
    reports = seed.history_reports(NOW.date(), TIMEZONE)
    assert [report.reporting_period for report in reports if report.patient_id == seed.DEMO_PATIENT_ID] == [
        "2026-W41",
        "2026-W42",
        "2026-W43",
    ]
    slots = seed.generate_slots(NOW.date(), TIMEZONE)
    appointments = seed.history_appointments(NOW.date(), slots)
    assert all(
        appointment.slot in slots and appointment.slot.start_at.date() > NOW.date() for appointment in appointments
    )
