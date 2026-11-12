"""Chuyển sang JWT: bảng `sessions` lưu chuỗi refresh token (khóa theo `sid`) thay cho token phiên ngẫu nhiên.

Phiên cũ (token ngẫu nhiên) không chuyển đổi được sang JWT nên bị xóa: mọi người dùng đăng nhập lại một lần.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-12
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _user_fk() -> sa.Column[str]:
    return sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)


def upgrade() -> None:
    op.drop_table("sessions")  # kéo theo các index của bảng
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        _user_fk(),
        sa.Column("refresh_hash", sa.String(64), nullable=False),
        sa.Column("previous_refresh_hash", sa.String(64), nullable=True),
        sa.Column("rotated_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])


def downgrade() -> None:
    op.drop_table("sessions")
    op.create_table(
        "sessions",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        _user_fk(),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])
