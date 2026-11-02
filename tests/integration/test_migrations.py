"""Migration Alembic phải tạo đúng schema mà models khai báo — nếu lệch, `--autogenerate` sẽ thấy khác biệt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

alembic = pytest.importorskip("alembic")

from alembic import command  # noqa: E402
from alembic.autogenerate import compare_metadata  # noqa: E402
from alembic.config import Config  # noqa: E402
from alembic.migration import MigrationContext  # noqa: E402
from sqlalchemy import create_engine, inspect  # noqa: E402

from patient_assistant.infrastructure.persistence.sql.models import Base  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def test_migrations_match_models_and_downgrade(tmp_path: Any) -> None:
    db = tmp_path / "migrated.db"
    config = _config(f"sqlite+aiosqlite:///{db}")
    command.upgrade(config, "head")

    sync = create_engine(f"sqlite:///{db}")
    with sync.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
    assert diff == [], diff
    assert set(inspect(sync).get_table_names()) >= {"users", "sessions", "conversations", "chat_messages"}

    command.downgrade(config, "base")
    assert set(inspect(sync).get_table_names()) <= {"alembic_version"}
    sync.dispose()
