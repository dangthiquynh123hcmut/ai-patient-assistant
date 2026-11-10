"""Engine, migration (Alembic) và nạp dữ liệu giả lập.

Bảng chỉ được tạo/đổi qua migration Alembic (`sql/migrations/`), không dùng `metadata.create_all`, nên
database của dev, test, eval và production luôn đúng cùng một lịch sử schema.
"""

from __future__ import annotations

import asyncio
import re
import threading
from datetime import date
from typing import Any
from zoneinfo import ZoneInfo

from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from patient_assistant.infrastructure.persistence import seed
from patient_assistant.infrastructure.persistence.sql.models import (
    AppointmentRow,
    DepartmentRow,
    DoctorRow,
    KnowledgeDocRow,
    PatientPlanRow,
    PatientRow,
    ReportRow,
    SlotRow,
)

_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def make_engine(url: str, schema: str | None = None) -> AsyncEngine:
    """Engine Postgres (asyncpg). `schema`: mọi bảng được tạo/đọc trong schema đó (qua search_path)."""
    connect_args = {"server_settings": {"search_path": _checked(schema)}} if schema else {}
    return create_async_engine(url, pool_pre_ping=True, connect_args=connect_args)


def _checked(schema: str) -> str:
    if not _SCHEMA_NAME.match(schema):
        raise ValueError(f"Tên schema không hợp lệ: {schema!r}")
    return schema


async def create_db_schema(url: str, schema: str) -> None:
    await _execute(url, f'CREATE SCHEMA IF NOT EXISTS "{_checked(schema)}"')


async def drop_db_schema(url: str, schema: str) -> None:
    await _execute(url, f'DROP SCHEMA IF EXISTS "{_checked(schema)}" CASCADE')


async def _execute(url: str, statement: str) -> None:
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(statement))
    finally:
        await engine.dispose()


MIGRATIONS = "patient_assistant.infrastructure.persistence.sql:migrations"
# Alembic giữ trạng thái chạy migration ở biến toàn cục của module -> chỉ cho một migration chạy mỗi lúc
# trong một tiến trình (eval chạy nhiều case song song, mỗi case một schema).
_MIGRATION_LOCK = threading.Lock()


def alembic_config(url: str, schema: str | None = None) -> Config:
    """Cấu hình Alembic dựng bằng code (không cần alembic.ini): migration nằm trong package nên dùng được ở mọi nơi."""
    config = Config()
    config.set_main_option("script_location", MIGRATIONS)
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # configparser coi % là cú pháp nội suy
    if schema:
        config.set_main_option("pa_schema", _checked(schema))
    return config


async def migrate(url: str, schema: str | None = None) -> None:
    """`alembic upgrade head` — tạo bảng mới hoặc nâng schema lên bản mới nhất (thay cho `create_all`)."""

    def upgrade() -> None:
        with _MIGRATION_LOCK:
            command.upgrade(alembic_config(url, schema), "head")

    # env.py tự chạy event loop riêng (asyncio.run) nên phải chạy ở thread khác, ngoài loop hiện tại.
    await asyncio.to_thread(upgrade)


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


async def _fill_patient_details(sessions: async_sessionmaker[AsyncSession]) -> None:
    """Hồ sơ demo đã có từ trước khi có ngày sinh/số điện thoại: điền các ô còn trống, không ghi đè ô đã có."""
    demo = {p.id: p for p in seed.profiles()}
    async with sessions.begin() as session:
        for row in await session.scalars(select(PatientRow).where(PatientRow.id.in_(demo))):
            row.date_of_birth = row.date_of_birth or demo[row.id].date_of_birth
            row.phone = row.phone or demo[row.id].phone


async def seed_demo_data(
    sessions: async_sessionmaker[AsyncSession], today: date, tz: ZoneInfo, with_history: bool = False
) -> int:
    """Nạp `demo_data/*.json` vào DB, trả số slot khám đã thêm. Slot sinh tương đối theo hôm nay.

    DB trống: nạp toàn bộ (kèm lịch sử nếu `with_history`). DB đã có dữ liệu: chỉ BỔ SUNG danh mục còn thiếu theo
    mã (khoa, bác sĩ, bệnh nhân, kế hoạch, tài liệu mới thêm vào JSON) cùng slot còn thiếu; dữ liệu đã có và lịch
    sử giữ nguyên. Vì vậy thêm khoa/bác sĩ chỉ cần sửa JSON rồi chạy lại seed (hoặc khởi động lại server dev).
    """
    async with sessions.begin() as session:
        fresh = await session.scalar(select(DepartmentRow).limit(1)) is None
        catalog: list[tuple[Any, dict[str, Any]]] = [
            (
                DepartmentRow.code,
                {d.code: DepartmentRow(code=d.code, name=d.name, aliases=list(d.aliases)) for d in seed.DEPARTMENTS},
            ),
            (
                DoctorRow.id,
                {d.id: DoctorRow(id=d.id, name=d.name, department_code=d.department_code) for d in seed.DOCTORS},
            ),
            (PatientRow.id, {p.id: PatientRow(**p.model_dump()) for p in seed.profiles()}),
            (PatientPlanRow.plan_id, {p.plan_id: PatientPlanRow(**p.model_dump()) for p in seed.PLANS}),
            (KnowledgeDocRow.doc_id, {k.doc_id: KnowledgeDocRow(**k.model_dump()) for k in seed.KNOWLEDGE}),
        ]
        for key, rows in catalog:  # theo thứ tự khóa ngoại: khoa -> bác sĩ, bệnh nhân -> kế hoạch
            existing = set(await session.scalars(select(key)))
            session.add_all(row for k, row in rows.items() if k not in existing)
            await session.flush()
    if not fresh:
        await _fill_patient_details(sessions)
        return await ensure_slots(sessions, today, tz)  # slot cho bác sĩ mới (và các ngày tới còn thiếu)
    async with sessions.begin() as session:
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
            return len(slots)
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
    return len(slots)
