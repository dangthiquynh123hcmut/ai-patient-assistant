"""Tạo engine/schema và nạp dữ liệu giả lập. Production nên dùng Alembic thay cho `create_all`."""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from patient_assistant.infrastructure.persistence import seed
from patient_assistant.infrastructure.persistence.sql.models import (
    AppointmentRow,
    Base,
    DepartmentRow,
    DoctorRow,
    KnowledgeDocRow,
    PatientPlanRow,
    PatientRow,
    ReportRow,
    SlotRow,
)


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def ensure_slots(sessions: async_sessionmaker[AsyncSession], today: date, tz: ZoneInfo) -> int:
    """Bổ sung slot từ hôm nay tới `days_ahead` ngày (theo `schedule.json`) còn thiếu. Trả số slot đã thêm.

    Chạy lại mỗi lần deploy/hằng ngày để lịch không cạn dần; slot đã có (kể cả đã đặt) giữ nguyên.
    """
    slots = seed.generate_slots(today, tz)
    async with sessions.begin() as session:
        existing = set(await session.scalars(select(SlotRow.slot_id).where(SlotRow.start_at >= slots[0].start_at)))
        missing = [s for s in slots if s.slot_id not in existing]
        session.add_all(
            SlotRow(
                slot_id=s.slot_id,
                department_code=s.department_code,
                doctor_id=s.doctor_id,
                start_at=s.start_at,
                end_at=s.end_at,
            )
            for s in missing
        )
    return len(missing)


async def seed_demo_data(
    sessions: async_sessionmaker[AsyncSession], today: date, tz: ZoneInfo, with_history: bool = False
) -> None:
    """Nạp `demo_data/*.json` vào DB nếu DB còn trống. Slot sinh tương đối theo hôm nay."""
    async with sessions.begin() as session:
        if await session.scalar(select(DepartmentRow).limit(1)):
            return  # đã seed
        session.add_all(DepartmentRow(code=d.code, name=d.name, aliases=list(d.aliases)) for d in seed.DEPARTMENTS)
        await session.flush()
        session.add_all(DoctorRow(id=d.id, name=d.name, department_code=d.department_code) for d in seed.DOCTORS)
        session.add_all(PatientRow(id=p.id, display_name=p.display_name) for p in seed.PATIENTS)
        await session.flush()
        session.add_all(PatientPlanRow(**p.model_dump()) for p in seed.PLANS)
        session.add_all(KnowledgeDocRow(**k.model_dump()) for k in seed.KNOWLEDGE)
        await session.flush()
        slots = seed.generate_slots(today, tz)
        session.add_all(
            SlotRow(
                slot_id=s.slot_id,
                department_code=s.department_code,
                doctor_id=s.doctor_id,
                start_at=s.start_at,
                end_at=s.end_at,
            )
            for s in slots
        )
        if not with_history:
            return
        await session.flush()
        session.add_all(
            ReportRow(
                **r.model_dump(mode="json", exclude={"evidence", "created_at"}),
                evidence=[e.model_dump() for e in r.evidence],
                created_at=r.created_at,
            )
            for r in seed.history_reports(today, tz)
        )
        session.add_all(
            AppointmentRow(
                id=a.id,
                patient_id=a.patient_id,
                slot_id=a.slot.slot_id,
                status=a.status.value,
                reason=a.reason,
                idempotency_key=a.idempotency_key,
                created_at=a.created_at,
            )
            for a in seed.history_appointments(today, slots)
        )
