"""Môi trường Alembic (async). Dùng chung metadata với app nên `--autogenerate` thấy đúng các bảng."""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from patient_assistant.config import Settings
from patient_assistant.infrastructure.persistence.sql.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    # Test truyền URL (và schema riêng) qua config; còn lại đọc PA_DATABASE_URL giống app.
    url = config.get_main_option("sqlalchemy.url") or Settings().database_url
    if not url:
        raise SystemExit("PA_DATABASE_URL chưa đặt")
    return url


def _configure(**kwargs: object) -> None:
    context.configure(target_metadata=target_metadata, compare_type=True, **kwargs)  # type: ignore[arg-type]


def run_offline() -> None:
    """`alembic upgrade head --sql`: in SQL ra thay vì chạy, để DBA duyệt."""
    _configure(url=_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    schema = config.get_main_option("pa_schema")
    connect_args = {"server_settings": {"search_path": schema}} if schema else {}
    engine = create_async_engine(_url(), poolclass=NullPool, connect_args=connect_args)
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    asyncio.run(run_online())
