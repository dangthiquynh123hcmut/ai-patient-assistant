"""Repository in-memory: dùng cho test và demo không cần Postgres. Hành vi khớp với bản SQL."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from patient_assistant.application.ports.repositories import KnowledgeDoc, NewReport, SlotTakenError
from patient_assistant.domain.booking.models import (
    Appointment,
    AppointmentStatus,
    Department,
    Doctor,
    PatientPlan,
    Slot,
)
from patient_assistant.domain.report.models import Report
from patient_assistant.domain.text.normalize import tokens


class InMemoryReportRepository:
    def __init__(self) -> None:
        self._rows: dict[str, Report] = {}

    async def add(self, report: NewReport) -> Report:
        if any(r.idempotency_key == report.idempotency_key for r in self._rows.values()):
            raise ValueError("duplicate idempotency_key")
        saved = Report(id=str(uuid.uuid4()), created_at=datetime.now(UTC), **report.model_dump())
        self._rows[saved.id] = saved
        return saved

    async def get(self, report_id: str) -> Report | None:
        return self._rows.get(report_id)

    async def get_by_idempotency_key(self, key: str) -> Report | None:
        return next((r for r in self._rows.values() if r.idempotency_key == key), None)

    def all(self) -> list[Report]:
        """Toàn bộ bản ghi — chỉ dùng cho kiểm tra/evaluation, không thuộc port."""
        return list(self._rows.values())

    async def list_by_patient(self, patient_id: str, limit: int) -> list[Report]:
        rows = sorted(
            (r for r in self._rows.values() if r.patient_id == patient_id), key=lambda r: r.created_at, reverse=True
        )
        return rows[:limit]


class InMemoryCatalogRepository:
    def __init__(self, departments: list[Department], doctors: list[Doctor]) -> None:
        self._departments, self._doctors = departments, doctors

    async def list_departments(self) -> list[Department]:
        return list(self._departments)

    async def list_doctors(self) -> list[Doctor]:
        return list(self._doctors)


class InMemoryPlanRepository:
    def __init__(self, plans: list[PatientPlan]) -> None:
        self._plans = {p.patient_id: p for p in plans}

    async def get_active_plan(self, patient_id: str) -> PatientPlan | None:
        return self._plans.get(patient_id)


class InMemoryBookingStore:
    """Slot + lịch hẹn chung một store để thao tác `book` nguyên tử (giống unique index ở DB)."""

    def __init__(self, slots: list[Slot]) -> None:
        self._slots = {s.slot_id: s for s in slots}
        self._appointments: dict[str, Appointment] = {}
        self._lock = asyncio.Lock()

    def all_appointments(self) -> list[Appointment]:
        """Toàn bộ lịch hẹn — chỉ dùng cho kiểm tra/evaluation (ví dụ phát hiện đặt trùng)."""
        return list(self._appointments.values())

    def slot_ids(self) -> set[str]:
        return set(self._slots)

    def _taken(self) -> set[str]:
        return {a.slot.slot_id for a in self._appointments.values() if a.status == AppointmentStatus.BOOKED}

    # SlotRepository
    async def find_available(
        self, department_code: str, start: datetime, end: datetime, doctor_id: str | None, limit: int
    ) -> list[Slot]:
        taken = self._taken()
        rows = [
            s
            for s in self._slots.values()
            if s.department_code == department_code
            and start <= s.start_at <= end
            and (doctor_id is None or s.doctor_id == doctor_id)
            and s.slot_id not in taken
        ]
        return sorted(rows, key=lambda s: s.start_at)[:limit]

    async def get(self, slot_id: str) -> Slot | None:
        return self._slots.get(slot_id)

    # AppointmentRepository
    async def book(self, patient_id: str, slot: Slot, reason: str | None, idempotency_key: str) -> Appointment:
        async with self._lock:
            existing = await self.get_by_idempotency_key(idempotency_key)
            if existing:
                return existing
            if slot.slot_id in self._taken():
                raise SlotTakenError(slot.slot_id)
            appt = Appointment(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                slot=slot,
                status=AppointmentStatus.BOOKED,
                reason=reason,
                idempotency_key=idempotency_key,
                created_at=datetime.now(UTC),
            )
            self._appointments[appt.id] = appt
            return appt

    async def get_appointment(self, appointment_id: str) -> Appointment | None:
        return self._appointments.get(appointment_id)

    async def get_by_idempotency_key(self, key: str) -> Appointment | None:
        return next((a for a in self._appointments.values() if a.idempotency_key == key), None)

    async def list_for_patient(self, patient_id: str) -> list[Appointment]:
        return [a for a in self._appointments.values() if a.patient_id == patient_id]


class InMemoryAppointmentRepository:
    """Adapter mỏng để `InMemoryBookingStore` khớp đúng port AppointmentRepository (tên `get`)."""

    def __init__(self, store: InMemoryBookingStore) -> None:
        self._store = store

    async def book(self, patient_id: str, slot: Slot, reason: str | None, idempotency_key: str) -> Appointment:
        return await self._store.book(patient_id, slot, reason, idempotency_key)

    async def get(self, appointment_id: str) -> Appointment | None:
        return await self._store.get_appointment(appointment_id)

    async def get_by_idempotency_key(self, key: str) -> Appointment | None:
        return await self._store.get_by_idempotency_key(key)

    async def list_for_patient(self, patient_id: str) -> list[Appointment]:
        return await self._store.list_for_patient(patient_id)


class InMemoryKnowledgeRepository:
    def __init__(self, docs: list[KnowledgeDoc]) -> None:
        self._docs = docs

    async def search(self, query: str, limit: int) -> list[KnowledgeDoc]:
        """Chấm điểm theo số token trùng — đủ cho tập tài liệu giả lập nhỏ."""
        q = set(tokens(query))
        scored = [(len(q & set(tokens(f"{d.title} {d.content}"))), d) for d in self._docs]
        return [d for score, d in sorted(scored, key=lambda x: -x[0]) if score >= 2][:limit]


class InMemoryAuditLog:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def record(self, kind: str, thread_id: str, payload: dict[str, Any]) -> None:
        self.events.append({"kind": kind, "thread_id": thread_id, **payload})
