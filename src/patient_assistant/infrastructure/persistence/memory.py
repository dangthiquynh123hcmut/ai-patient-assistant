"""Repository in-memory: dùng cho test và demo không cần Postgres. Hành vi khớp với bản SQL."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from patient_assistant.application.ports.repositories import (
    KnowledgeDoc,
    NewReport,
    SlotTakenError,
    UsernameTakenError,
)
from patient_assistant.domain.account import ChatMessage, ConversationSummary, User
from patient_assistant.domain.booking.models import (
    Appointment,
    AppointmentStatus,
    Department,
    Doctor,
    PatientPlan,
    Slot,
)
from patient_assistant.domain.patient import PatientProfile
from patient_assistant.domain.report.models import Report
from patient_assistant.domain.text.normalize import tokens


class InMemoryReportRepository:
    def __init__(self, initial: list[Report] | None = None) -> None:
        self._rows: dict[str, Report] = {r.id: r for r in initial or []}

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


class InMemoryPatientRepository:
    def __init__(self, patients: list[PatientProfile]) -> None:
        self._patients = {p.id: p for p in patients}

    async def get_profile(self, patient_id: str) -> PatientProfile | None:
        return self._patients.get(patient_id)


class InMemoryPlanRepository:
    def __init__(self, plans: list[PatientPlan]) -> None:
        self._plans = {p.patient_id: p for p in plans}

    async def get_active_plan(self, patient_id: str) -> PatientPlan | None:
        return self._plans.get(patient_id)


class InMemoryBookingStore:
    """Slot + lịch hẹn chung một store để thao tác `book` nguyên tử (giống unique index ở DB)."""

    def __init__(self, slots: list[Slot], appointments: list[Appointment] | None = None) -> None:
        self._slots = {s.slot_id: s for s in slots}
        self._appointments: dict[str, Appointment] = {a.id: a for a in appointments or []}
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


class InMemoryUserRepository:
    def __init__(self, initial: list[tuple[User, str]] | None = None) -> None:
        self._rows: dict[str, tuple[User, str]] = {u.id: (u, h) for u, h in initial or []}

    async def add(self, user: User, password_hash: str) -> User:
        if any(u.username == user.username for u, _ in self._rows.values()):
            raise UsernameTakenError(user.username)
        self._rows[user.id] = (user, password_hash)
        return user

    async def get(self, user_id: str) -> User | None:
        row = self._rows.get(user_id)
        return row[0] if row else None

    async def get_credentials(self, username: str) -> tuple[User, str] | None:
        return next((row for row in self._rows.values() if row[0].username == username), None)

    async def list(self) -> list[User]:
        return sorted((u for u, _ in self._rows.values()), key=lambda u: u.created_at)

    async def update(
        self, user_id: str, *, active: bool | None = None, password_hash: str | None = None
    ) -> User | None:
        row = self._rows.get(user_id)
        if row is None:
            return None
        user = row[0].model_copy(update={"active": active}) if active is not None else row[0]
        self._rows[user_id] = (user, password_hash or row[1])
        return user

    async def count(self) -> int:
        return len(self._rows)

    async def set_login_state(self, user_id: str, failed_logins: int, locked_until: datetime | None) -> None:
        user, password_hash = self._rows[user_id]
        update = {"failed_logins": failed_logins, "locked_until": locked_until}
        self._rows[user_id] = (user.model_copy(update=update), password_hash)


class InMemorySessionRepository:
    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, datetime]] = {}

    async def add(self, token_hash: str, user_id: str, expires_at: datetime) -> None:
        self._rows[token_hash] = (user_id, expires_at)

    async def get(self, token_hash: str) -> tuple[str, datetime] | None:
        return self._rows.get(token_hash)

    async def delete(self, token_hash: str) -> None:
        self._rows.pop(token_hash, None)

    async def delete_for_user(self, user_id: str) -> None:
        self._rows = {k: v for k, v in self._rows.items() if v[0] != user_id}


class InMemoryConversationLog:
    def __init__(self) -> None:
        self._conversations: dict[str, ConversationSummary] = {}
        self._messages: dict[str, list[ChatMessage]] = {}

    async def open(self, thread_id: str, user_id: str, patient_id: str, title: str, now: datetime) -> str:
        found = self._conversations.get(thread_id)
        if found is None:
            found = ConversationSummary(
                thread_id=thread_id,
                patient_id=patient_id,
                user_id=user_id,
                title=title,
                message_count=0,
                last_kind=None,
                created_at=now,
                updated_at=now,
            )
            self._conversations[thread_id] = found
        return found.patient_id

    async def owner(self, thread_id: str) -> str | None:
        found = self._conversations.get(thread_id)
        return found.patient_id if found else None

    async def add_message(self, message: ChatMessage) -> None:
        self._messages.setdefault(message.thread_id, []).append(message)
        conv = self._conversations[message.thread_id]
        self._conversations[message.thread_id] = conv.model_copy(
            update={
                "message_count": conv.message_count + 1,
                "updated_at": message.created_at,
                "last_kind": message.action if message.role == "assistant" else conv.last_kind,
            }
        )

    async def list_conversations(self, patient_id: str | None, limit: int) -> list[ConversationSummary]:
        rows = [c for c in self._conversations.values() if patient_id is None or c.patient_id == patient_id]
        return sorted(rows, key=lambda c: c.updated_at, reverse=True)[:limit]

    async def list_messages(self, thread_id: str) -> list[ChatMessage]:
        return list(self._messages.get(thread_id, []))
