"""Schema ban đầu: danh mục, slot, lịch hẹn, báo cáo, tài liệu, tài khoản, phiên đăng nhập, lịch sử hội thoại.

Revision ID: 0001
Revises:
Create Date: 2026-10-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "departments",
        sa.Column("code", sa.String(16), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("aliases", sa.JSON(), nullable=False),
    )
    op.create_table(
        "doctors",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("department_code", sa.String(16), sa.ForeignKey("departments.code"), nullable=False),
    )
    op.create_table(
        "patients",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("display_name", sa.String(128), nullable=False),
    )
    op.create_table(
        "patient_plans",
        sa.Column("plan_id", sa.String(64), primary_key=True),
        sa.Column("patient_id", sa.String(64), nullable=False),
        sa.Column("department_code", sa.String(16), sa.ForeignKey("departments.code"), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
    )
    op.create_index("ix_patient_plans_patient_id", "patient_plans", ["patient_id"])
    op.create_table(
        "slots",
        sa.Column("slot_id", sa.String(96), primary_key=True),
        sa.Column("department_code", sa.String(16), sa.ForeignKey("departments.code"), nullable=False),
        sa.Column("doctor_id", sa.String(64), sa.ForeignKey("doctors.id"), nullable=False),
        sa.Column("start_at", sa.DateTime(), nullable=False),
        sa.Column("end_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_slots_department_code", "slots", ["department_code"])
    op.create_index("ix_slots_start_at", "slots", ["start_at"])
    op.create_table(
        "appointments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(64), nullable=False),
        sa.Column("slot_id", sa.String(96), sa.ForeignKey("slots.slot_id"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_appointments_patient_id", "appointments", ["patient_id"])
    # Một slot chỉ có một lịch hẹn hiệu lực — chống đặt trùng ngay ở DB (FR-09.9).
    op.create_index(
        "uq_appointments_active_slot",
        "appointments",
        ["slot_id"],
        unique=True,
        postgresql_where=sa.text("status = 'booked'"),
    )
    op.create_table(
        "reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_id", sa.String(64), nullable=False),
        sa.Column("reporting_period", sa.String(16), nullable=False),
        sa.Column("pain_level", sa.Integer(), nullable=False),
        sa.Column("exercise_days", sa.Integer(), nullable=False),
        sa.Column("symptom_change", sa.String(32), nullable=True),
        sa.Column("triggering_activity", sa.Text(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_reports_patient_id", "reports", ["patient_id"])
    op.create_table(
        "knowledge_docs",
        sa.Column("doc_id", sa.String(32), primary_key=True),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(32), nullable=False, unique=True),
        sa.Column("display_name", sa.String(128), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("patient_id", sa.String(64), sa.ForeignKey("patients.id"), nullable=True),
        sa.Column("password_hash", sa.String(256), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("failed_logins", sa.Integer(), nullable=False),
        sa.Column("locked_until", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "sessions",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_table(
        "conversations",
        sa.Column("thread_id", sa.String(64), primary_key=True),
        sa.Column("patient_id", sa.String(64), sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(128), nullable=False),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("last_kind", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_conversations_patient_id", "conversations", ["patient_id"])
    op.create_index("ix_conversations_updated_at", "conversations", ["updated_at"])
    op.create_table(
        "chat_messages",
        sa.Column("seq", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("id", sa.String(36), nullable=False, unique=True),
        sa.Column("thread_id", sa.String(64), sa.ForeignKey("conversations.thread_id"), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chat_messages_thread_id", "chat_messages", ["thread_id"])
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])


def downgrade() -> None:
    for table in (
        "chat_messages",
        "conversations",
        "sessions",
        "users",
        "knowledge_docs",
        "reports",
        "appointments",
        "slots",
        "patient_plans",
        "patients",
        "doctors",
        "departments",
    ):
        op.drop_table(table)
