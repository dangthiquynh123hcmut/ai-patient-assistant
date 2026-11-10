"""Dữ liệu demo đọc từ các file JSON trong `demo_data/` — sửa JSON là đổi dữ liệu, không cần sửa code.

Mọi thời điểm (slot, lịch sử báo cáo, lịch hẹn) được tính tương đối theo "hôm nay" lúc khởi động để dữ liệu
không bị cũ. File được validate bằng Pydantic khi nạp: sai định dạng thì báo lỗi rõ ràng ngay khi khởi động.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from functools import cache
from importlib import resources
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from patient_assistant.application.ports.repositories import KnowledgeDoc
from patient_assistant.domain.account import NewUser, Role
from patient_assistant.domain.booking.models import (
    Appointment,
    AppointmentStatus,
    Department,
    Doctor,
    PatientPlan,
    Slot,
)
from patient_assistant.domain.common import Evidence
from patient_assistant.domain.patient import PatientProfile
from patient_assistant.domain.report.models import Report, SymptomChange
from patient_assistant.domain.report.rules import current_period

DEMO_PATIENT_ID = "patient-demo-001"


class DemoPatient(BaseModel):
    id: str
    display_name: str
    label: str
    date_of_birth: date | None = None
    phone: str | None = None


class Schedule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    days_ahead: int = Field(ge=1, le=120)
    working_weekdays: list[int]
    slot_minutes: int = Field(ge=5, le=240)
    start_times: list[time]


class HistoryReport(BaseModel):
    patient_id: str
    weeks_ago: int = Field(ge=1)
    pain_level: int
    exercise_days: int
    symptom_change: SymptomChange | None = None
    triggering_activity: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class HistoryAppointment(BaseModel):
    patient_id: str
    doctor_id: str
    days_from_today: int = Field(ge=1)
    time: time
    reason: str | None = None


class History(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reports: list[HistoryReport] = Field(default_factory=list)
    appointments: list[HistoryAppointment] = Field(default_factory=list)


def _read(name: str) -> Any:
    path = resources.files("patient_assistant.infrastructure.persistence").joinpath(f"demo_data/{name}")
    return json.loads(path.read_text("utf-8"))


def _load_list(name: str, model: Any) -> list[Any]:
    return TypeAdapter(list[model]).validate_python(_read(name))


@cache
def _data() -> dict[str, Any]:
    data: dict[str, Any] = {
        "departments": _load_list("departments.json", Department),
        "doctors": _load_list("doctors.json", Doctor),
        "patients": _load_list("patients.json", DemoPatient),
        "plans": _load_list("plans.json", PatientPlan),
        "knowledge": _load_list("knowledge.json", KnowledgeDoc),
        "schedule": Schedule.model_validate(_read("schedule.json")),
        "history": History.model_validate(_read("history.json")),
    }
    _check_references(data)
    return data


def _check_references(data: dict[str, Any]) -> None:
    """Bắt lỗi tham chiếu trong JSON (ví dụ bác sĩ thuộc khoa không tồn tại) ngay lúc nạp."""
    departments = {d.code for d in data["departments"]}
    doctors = {d.id for d in data["doctors"]}
    patients = {p.id for p in data["patients"]}
    problems = [
        f"doctors.json: {d.id} -> khoa '{d.department_code}' không có"
        for d in data["doctors"]
        if d.department_code not in departments
    ]
    problems += [
        f"plans.json: {p.plan_id} -> khoa '{p.department_code}' không có"
        for p in data["plans"]
        if p.department_code not in departments
    ]
    problems += [
        f"plans.json: {p.plan_id} -> bệnh nhân '{p.patient_id}' không có"
        for p in data["plans"]
        if p.patient_id not in patients
    ]
    problems += [
        f"history.json: lịch hẹn -> bác sĩ '{a.doctor_id}' không có"
        for a in data["history"].appointments
        if a.doctor_id not in doctors
    ]
    if problems:
        raise ValueError("Dữ liệu demo không hợp lệ:\n" + "\n".join(problems))


class DemoAccount(BaseModel):
    username: str
    display_name: str
    role: Role
    patient_id: str | None = None


def demo_accounts(password: str) -> list[NewUser]:
    """Tài khoản demo trong `demo_data/users.json`, cùng một mật khẩu (cấu hình qua PA_SEED_PASSWORD)."""
    patients = {p.id for p in PATIENTS}
    accounts = _load_list("users.json", DemoAccount)
    missing = [a.username for a in accounts if a.patient_id and a.patient_id not in patients]
    if missing:
        raise ValueError(f"users.json: bệnh nhân không có cho {missing}")
    return [NewUser(**a.model_dump(), password=password) for a in accounts]


def profiles() -> list[PatientProfile]:
    return [PatientProfile(**p.model_dump(exclude={"label"})) for p in PATIENTS]


DEPARTMENTS: list[Department] = _data()["departments"]
DOCTORS: list[Doctor] = _data()["doctors"]
PATIENTS: list[DemoPatient] = _data()["patients"]
PLANS: list[PatientPlan] = _data()["plans"]
KNOWLEDGE: list[KnowledgeDoc] = _data()["knowledge"]
SCHEDULE: Schedule = _data()["schedule"]
HISTORY: History = _data()["history"]


def _slot_id(doctor_id: str, day: date, t: time) -> str:
    return f"{doctor_id}-{day:%Y%m%d}-{t:%H%M}"


def generate_slots(today: date, tz: tzinfo, days: int | None = None) -> list[Slot]:
    """Slot của mọi bác sĩ theo `schedule.json`, từ hôm nay tới `days_ahead` ngày."""
    duration = timedelta(minutes=SCHEDULE.slot_minutes)
    slots: list[Slot] = []
    for offset in range(days or SCHEDULE.days_ahead):
        day = today + timedelta(days=offset)
        if day.weekday() not in SCHEDULE.working_weekdays:
            continue
        for doctor in DOCTORS:
            for t in SCHEDULE.start_times:
                start = datetime.combine(day, t, tz)
                slots.append(
                    Slot(
                        slot_id=_slot_id(doctor.id, day, t),
                        department_code=doctor.department_code,
                        doctor_id=doctor.id,
                        doctor_name=doctor.name,
                        start_at=start,
                        end_at=start + duration,
                    )
                )
    return slots


def history_reports(today: date, tz: tzinfo) -> list[Report]:
    out = []
    for i, r in enumerate(HISTORY.reports, start=1):
        day = today - timedelta(weeks=r.weeks_ago)
        created = datetime.combine(day, time(20, 0), tz).astimezone(UTC)
        out.append(
            Report(
                id=f"rep-demo-{i:03d}",
                patient_id=r.patient_id,
                reporting_period=current_period(created),
                pain_level=r.pain_level,
                exercise_days=r.exercise_days,
                symptom_change=r.symptom_change,
                triggering_activity=r.triggering_activity,
                evidence=r.evidence,
                idempotency_key=f"demo-history-report-{i}",
                created_at=created,
            )
        )
    return out


def history_appointments(today: date, slots: list[Slot]) -> list[Appointment]:
    """Lịch hẹn demo gắn vào slot thật; ngày nghỉ thì dời sang ngày làm việc kế tiếp."""
    by_id = {s.slot_id: s for s in slots}
    out = []
    for i, a in enumerate(HISTORY.appointments, start=1):
        day = today + timedelta(days=a.days_from_today)
        while day.weekday() not in SCHEDULE.working_weekdays:
            day += timedelta(days=1)
        slot = by_id.get(_slot_id(a.doctor_id, day, a.time))
        if slot is None:
            raise ValueError(f"history.json: lịch hẹn {i} không khớp slot nào ({a.doctor_id}, {day}, {a.time})")
        out.append(
            Appointment(
                id=f"appt-demo-{i:03d}",
                patient_id=a.patient_id,
                slot=slot,
                status=AppointmentStatus.BOOKED,
                reason=a.reason,
                idempotency_key=f"demo-history-appointment-{i}",
                created_at=datetime.combine(today, time(8, 0), slot.start_at.tzinfo).astimezone(UTC),
            )
        )
    return out
