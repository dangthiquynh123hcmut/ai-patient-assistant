"""Thêm ngày sinh và số điện thoại vào hồ sơ bệnh nhân (cả hai không bắt buộc; hồ sơ cũ để trống).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-09
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("patients", sa.Column("date_of_birth", sa.Date(), nullable=True))
    op.add_column("patients", sa.Column("phone", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("patients", "phone")
    op.drop_column("patients", "date_of_birth")
