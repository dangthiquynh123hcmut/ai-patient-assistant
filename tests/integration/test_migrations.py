"""Migration Alembic phải tạo đúng schema mà models khai báo — nếu lệch, `--autogenerate` sẽ thấy khác biệt."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

alembic = pytest.importorskip("alembic")

from alembic import command  # noqa: E402
from alembic.autogenerate import compare_metadata  # noqa: E402
from alembic.config import Config  # noqa: E402
from alembic.migration import MigrationContext  # noqa: E402
from sqlalchemy import create_engine, inspect  # noqa: E402

from patient_assistant.infrastructure.persistence.sql.models import Base  # noqa: E402
from patient_assistant.infrastructure.persistence.sql.setup import create_db_schema, drop_db_schema  # noqa: E402
from tests.conftest import TEST_DATABASE_URL  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _config(url: str, schema: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    config.set_main_option("pa_schema", schema)
    return config


def test_migrations_match_models_and_downgrade() -> None:
    """Chạy migration trong một schema Postgres trống, riêng cho test này."""
    schema = f"m_{uuid.uuid4().hex[:12]}"
    asyncio.run(create_db_schema(TEST_DATABASE_URL, schema))
    sync = create_engine(
        TEST_DATABASE_URL.replace("+asyncpg", "+psycopg", 1), connect_args={"options": f"-c search_path={schema}"}
    )
    try:
        config = _config(TEST_DATABASE_URL, schema)
        command.upgrade(config, "head")
        with sync.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
        assert diff == [], diff
        tables = set(inspect(sync).get_table_names(schema=schema))
        assert tables >= {"users", "sessions", "conversations", "chat_messages"}

        command.downgrade(config, "base")
        assert set(inspect(sync).get_table_names(schema=schema)) <= {"alembic_version"}
    finally:
        sync.dispose()
        asyncio.run(drop_db_schema(TEST_DATABASE_URL, schema))
