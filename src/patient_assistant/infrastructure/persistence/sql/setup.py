"""Tạo engine/schema và nạp dữ liệu giả lập. Production nên dùng Alembic thay cho `create_all`."""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from patient_assistant.infrastructure.persistence import seed
from patient_assistant.infrastructure.persistence.sql.models import (
    Base,
    DepartmentRow,
    DoctorRow,
    KnowledgeDocRow,
    PatientPlanRow,
    SlotRow,
)


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def seed_demo_data(sessions: async_sessionmaker[AsyncSession], today: date, tz: ZoneInfo) -> None:
    async with sessions.begin() as session:
        if await session.scalar(select(DepartmentRow).limit(1)):
            return  # đã seed
        session.add_all(DepartmentRow(code=d.code, name=d.name, aliases=list(d.aliases)) for d in seed.DEPARTMENTS)
        await session.flush()
        session.add_all(DoctorRow(id=d.id, name=d.name, department_code=d.department_code) for d in seed.DOCTORS)
        session.add_all(PatientPlanRow(**p.model_dump()) for p in seed.PLANS)
        session.add_all(KnowledgeDocRow(**k.model_dump()) for k in seed.KNOWLEDGE)
        await session.flush()
        session.add_all(
            SlotRow(
                slot_id=s.slot_id,
                department_code=s.department_code,
                doctor_id=s.doctor_id,
                start_at=s.start_at,
                end_at=s.end_at,
            )
            for s in seed.generate_slots(today, tz)
        )
