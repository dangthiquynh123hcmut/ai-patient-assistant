"""Repository SQLAlchemy (async). Mỗi phương thức mở một session ngắn — không giữ transaction qua lượt chat."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, tzinfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_assistant.application.ports.repositories import KnowledgeDoc, NewReport, SlotTakenError
from patient_assistant.domain.booking.models import (
    Appointment,
    AppointmentStatus,
    Department,
    Doctor,
    PatientPlan,
    Slot,
)
from patient_assistant.domain.common import Evidence
from patient_assistant.domain.report.models import Report, SymptomChange
from patient_assistant.domain.text.normalize import tokens
from patient_assistant.infrastructure.persistence.sql.models import (
    AppointmentRow,
    DepartmentRow,
    DoctorRow,
    KnowledgeDocRow,
    PatientPlanRow,
    ReportRow,
    SlotRow,
)

Sessions = async_sessionmaker[AsyncSession]


def _report(row: ReportRow) -> Report:
    return Report(
        id=row.id,
        patient_id=row.patient_id,
        reporting_period=row.reporting_period,
        pain_level=row.pain_level,
        exercise_days=row.exercise_days,
        symptom_change=SymptomChange(row.symptom_change) if row.symptom_change else None,
        triggering_activity=row.triggering_activity,
        evidence=[Evidence.model_validate(e) for e in row.evidence],
        idempotency_key=row.idempotency_key,
        created_at=row.created_at,
    )


class SqlReportRepository:
    def __init__(self, sessions: Sessions) -> None:
        self._sessions = sessions

    async def add(self, report: NewReport) -> Report:
        row = ReportRow(
            id=str(uuid.uuid4()),
            **report.model_dump(mode="json", exclude={"evidence"}),
            evidence=[e.model_dump() for e in report.evidence],
        )
        async with self._sessions.begin() as session:
            session.add(row)
        return _report(row)

    async def get(self, report_id: str) -> Report | None:
        async with self._sessions() as session:
            row = await session.get(ReportRow, report_id)
            return _report(row) if row else None

    async def get_by_idempotency_key(self, key: str) -> Report | None:
        async with self._sessions() as session:
            row = await session.scalar(select(ReportRow).where(ReportRow.idempotency_key == key))
            return _report(row) if row else None

    async def list_by_patient(self, patient_id: str, limit: int) -> list[Report]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(ReportRow)
                .where(ReportRow.patient_id == patient_id)
                .order_by(ReportRow.created_at.desc())
                .limit(limit)
            )
            return [_report(r) for r in rows]


class SqlCatalogRepository:
    def __init__(self, sessions: Sessions) -> None:
        self._sessions = sessions

    async def list_departments(self) -> list[Department]:
        async with self._sessions() as session:
            rows = await session.scalars(select(DepartmentRow).order_by(DepartmentRow.code))
            return [Department(code=r.code, name=r.name, aliases=tuple(r.aliases)) for r in rows]

    async def list_doctors(self) -> list[Doctor]:
        async with self._sessions() as session:
            rows = await session.scalars(select(DoctorRow).order_by(DoctorRow.id))
            return [Doctor(id=r.id, name=r.name, department_code=r.department_code) for r in rows]


class SqlPlanRepository:
    def __init__(self, sessions: Sessions) -> None:
        self._sessions = sessions

    async def get_active_plan(self, patient_id: str) -> PatientPlan | None:
        async with self._sessions() as session:
            row = await session.scalar(select(PatientPlanRow).where(PatientPlanRow.patient_id == patient_id))
            return (
                PatientPlan(
                    plan_id=row.plan_id,
                    patient_id=row.patient_id,
                    department_code=row.department_code,
                    summary=row.summary,
                )
                if row
                else None
            )


async def _slot(session: AsyncSession, row: SlotRow, tz: tzinfo) -> Slot:
    """DB lưu UTC; trả về theo múi giờ hiển thị để nhãn slot đúng giờ địa phương."""
    doctor = await session.get(DoctorRow, row.doctor_id)
    return Slot(
        slot_id=row.slot_id,
        department_code=row.department_code,
        doctor_id=row.doctor_id,
        doctor_name=doctor.name if doctor else row.doctor_id,
        start_at=row.start_at.astimezone(tz),
        end_at=row.end_at.astimezone(tz),
    )


class SqlSlotRepository:
    def __init__(self, sessions: Sessions, tz: tzinfo = UTC) -> None:
        self._sessions = sessions
        self._tz = tz

    async def find_available(
        self, department_code: str, start: datetime, end: datetime, doctor_id: str | None, limit: int
    ) -> list[Slot]:
        booked = select(AppointmentRow.slot_id).where(AppointmentRow.status == AppointmentStatus.BOOKED.value)
        stmt = (
            select(SlotRow)
            .where(
                SlotRow.department_code == department_code,
                SlotRow.start_at >= start,
                SlotRow.start_at <= end,
                SlotRow.slot_id.not_in(booked),
            )
            .order_by(SlotRow.start_at)
            .limit(limit)
        )
        if doctor_id:
            stmt = stmt.where(SlotRow.doctor_id == doctor_id)
        async with self._sessions() as session:
            return [await _slot(session, r, self._tz) for r in await session.scalars(stmt)]

    async def get(self, slot_id: str) -> Slot | None:
        async with self._sessions() as session:
            row = await session.get(SlotRow, slot_id)
            return await _slot(session, row, self._tz) if row else None


class SqlAppointmentRepository:
    def __init__(self, sessions: Sessions, tz: tzinfo = UTC) -> None:
        self._sessions = sessions
        self._tz = tz

    async def _to_domain(self, session: AsyncSession, row: AppointmentRow) -> Appointment:
        slot_row = await session.get(SlotRow, row.slot_id)
        assert slot_row is not None
        return Appointment(
            id=row.id,
            patient_id=row.patient_id,
            slot=await _slot(session, slot_row, self._tz),
            status=AppointmentStatus(row.status),
            reason=row.reason,
            idempotency_key=row.idempotency_key,
            created_at=row.created_at,
        )

    async def book(self, patient_id: str, slot: Slot, reason: str | None, idempotency_key: str) -> Appointment:
        row = AppointmentRow(
            id=str(uuid.uuid4()),
            patient_id=patient_id,
            slot_id=slot.slot_id,
            status=AppointmentStatus.BOOKED.value,
            reason=reason,
            idempotency_key=idempotency_key,
        )
        try:
            async with self._sessions.begin() as session:
                session.add(row)
        except IntegrityError as exc:
            # Retry cùng khóa (lỗi mạng) -> trả bản đã có; còn lại là slot đã bị chiếm.
            existing = await self.get_by_idempotency_key(idempotency_key)
            if existing:
                return existing
            raise SlotTakenError(slot.slot_id) from exc
        async with self._sessions() as session:
            return await self._to_domain(session, row)

    async def get(self, appointment_id: str) -> Appointment | None:
        async with self._sessions() as session:
            row = await session.get(AppointmentRow, appointment_id)
            return await self._to_domain(session, row) if row else None

    async def get_by_idempotency_key(self, key: str) -> Appointment | None:
        async with self._sessions() as session:
            row = await session.scalar(select(AppointmentRow).where(AppointmentRow.idempotency_key == key))
            return await self._to_domain(session, row) if row else None

    async def list_for_patient(self, patient_id: str) -> list[Appointment]:
        async with self._sessions() as session:
            rows = await session.scalars(select(AppointmentRow).where(AppointmentRow.patient_id == patient_id))
            return [await self._to_domain(session, r) for r in rows]


class SqlKnowledgeRepository:
    """MVP: chấm điểm token trong Python (tập tài liệu nhỏ). Có thể thay bằng Postgres FTS / pgvector."""

    def __init__(self, sessions: Sessions) -> None:
        self._sessions = sessions

    async def search(self, query: str, limit: int) -> list[KnowledgeDoc]:
        async with self._sessions() as session:
            docs = [
                KnowledgeDoc(doc_id=r.doc_id, title=r.title, content=r.content)
                for r in await session.scalars(select(KnowledgeDocRow))
            ]
        q = set(tokens(query))
        scored = [(len(q & set(tokens(f"{d.title} {d.content}"))), d) for d in docs]
        return [d for score, d in sorted(scored, key=lambda x: -x[0]) if score >= 2][:limit]
