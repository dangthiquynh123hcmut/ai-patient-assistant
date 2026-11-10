"""Đổi tên cột `thread_id` thành `conversation_id` (bảng conversations, chat_messages), kèm index và khóa ngoại.

Chỉ đổi tên, dữ liệu giữ nguyên. Checkpoint của LangGraph không bị ảnh hưởng: nó lưu theo giá trị mã hội thoại,
còn khóa cấu hình `thread_id` là của LangGraph.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-08
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("conversations", "thread_id", new_column_name="conversation_id")
    op.alter_column("chat_messages", "thread_id", new_column_name="conversation_id")
    op.execute("ALTER INDEX ix_chat_messages_thread_id RENAME TO ix_chat_messages_conversation_id")
    op.execute(
        "ALTER TABLE chat_messages RENAME CONSTRAINT chat_messages_thread_id_fkey TO chat_messages_conversation_id_fkey"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE chat_messages RENAME CONSTRAINT chat_messages_conversation_id_fkey TO chat_messages_thread_id_fkey"
    )
    op.execute("ALTER INDEX ix_chat_messages_conversation_id RENAME TO ix_chat_messages_thread_id")
    op.alter_column("chat_messages", "conversation_id", new_column_name="thread_id")
    op.alter_column("conversations", "conversation_id", new_column_name="thread_id")
