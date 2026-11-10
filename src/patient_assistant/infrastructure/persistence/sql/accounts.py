"""Repository SQLAlchemy cho tài khoản, phiên đăng nhập và lịch sử hội thoại."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_assistant.application.ports.repositories import UsernameTakenError
from patient_assistant.domain.account import ChatMessage, ConversationSummary, Role, User
from patient_assistant.infrastructure.persistence.sql.models import (
    ChatMessageRow,
    ConversationRow,
    SessionRow,
    UserRow,
)

Sessions = async_sessionmaker[AsyncSession]


def _user(row: UserRow) -> User:
    return User(
        id=row.id,
        username=row.username,
        display_name=row.display_name,
        role=Role(row.role),
        patient_id=row.patient_id,
        active=row.active,
        created_at=row.created_at,
        failed_logins=row.failed_logins or 0,
        locked_until=row.locked_until,
    )


class SqlUserRepository:
    def __init__(self, sessions: Sessions) -> None:
        self._sessions = sessions

    async def add(self, user: User, password_hash: str) -> User:
        try:
            async with self._sessions.begin() as session:
                session.add(
                    UserRow(
                        id=user.id,
                        username=user.username,
                        display_name=user.display_name,
                        role=user.role.value,
                        patient_id=user.patient_id,
                        password_hash=password_hash,
                        active=user.active,
                        created_at=user.created_at,
                    )
                )
        except IntegrityError as exc:
            raise UsernameTakenError(user.username) from exc
        return user

    async def get(self, user_id: str) -> User | None:
        async with self._sessions() as session:
            row = await session.get(UserRow, user_id)
            return _user(row) if row else None

    async def get_credentials(self, username: str) -> tuple[User, str] | None:
        async with self._sessions() as session:
            row = await session.scalar(select(UserRow).where(UserRow.username == username))
            return (_user(row), row.password_hash) if row else None

    async def list(self) -> list[User]:
        async with self._sessions() as session:
            rows = await session.scalars(select(UserRow).order_by(UserRow.created_at))
            return [_user(r) for r in rows]

    async def update(
        self, user_id: str, *, active: bool | None = None, password_hash: str | None = None
    ) -> User | None:
        async with self._sessions.begin() as session:
            row = await session.get(UserRow, user_id)
            if row is None:
                return None
            if active is not None:
                row.active = active
            if password_hash is not None:
                row.password_hash = password_hash
            return _user(row)

    async def count(self) -> int:
        async with self._sessions() as session:
            return int(await session.scalar(select(func.count()).select_from(UserRow)) or 0)

    async def set_login_state(self, user_id: str, failed_logins: int, locked_until: datetime | None) -> None:
        async with self._sessions.begin() as session:
            await session.execute(
                update(UserRow)
                .where(UserRow.id == user_id)
                .values(failed_logins=failed_logins, locked_until=locked_until)
            )


class SqlSessionRepository:
    def __init__(self, sessions: Sessions) -> None:
        self._sessions = sessions

    async def add(self, token_hash: str, user_id: str, expires_at: datetime) -> None:
        async with self._sessions.begin() as session:
            session.add(SessionRow(token_hash=token_hash, user_id=user_id, expires_at=expires_at))

    async def get(self, token_hash: str) -> tuple[str, datetime] | None:
        async with self._sessions() as session:
            row = await session.get(SessionRow, token_hash)
            return (row.user_id, row.expires_at) if row else None

    async def delete(self, token_hash: str) -> None:
        async with self._sessions.begin() as session:
            await session.execute(delete(SessionRow).where(SessionRow.token_hash == token_hash))

    async def delete_for_user(self, user_id: str) -> None:
        async with self._sessions.begin() as session:
            await session.execute(delete(SessionRow).where(SessionRow.user_id == user_id))


def _summary(row: ConversationRow) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=row.conversation_id,
        patient_id=row.patient_id,
        user_id=row.user_id,
        title=row.title,
        message_count=row.message_count,
        last_kind=row.last_kind,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlConversationLog:
    def __init__(self, sessions: Sessions) -> None:
        self._sessions = sessions

    async def open(self, conversation_id: str, user_id: str, patient_id: str, title: str, now: datetime) -> str:
        try:
            async with self._sessions.begin() as session:
                row = await session.get(ConversationRow, conversation_id)
                if row is not None:
                    return row.patient_id
                session.add(
                    ConversationRow(
                        conversation_id=conversation_id,
                        patient_id=patient_id,
                        user_id=user_id,
                        title=title,
                        created_at=now,
                        updated_at=now,
                    )
                )
                return patient_id
        except IntegrityError:  # hai request cùng tạo một hội thoại -> đọc lại chủ sở hữu thật
            owner = await self.owner(conversation_id)
            assert owner is not None
            return owner

    async def owner(self, conversation_id: str) -> str | None:
        async with self._sessions() as session:
            row = await session.get(ConversationRow, conversation_id)
            return row.patient_id if row else None

    async def add_message(self, message: ChatMessage) -> None:
        async with self._sessions.begin() as session:
            session.add(ChatMessageRow(**message.model_dump()))
            conv = await session.get(ConversationRow, message.conversation_id)
            assert conv is not None
            conv.message_count += 1
            conv.updated_at = message.created_at
            if message.role == "assistant":
                conv.last_kind = message.action

    async def list_conversations(self, patient_id: str | None, limit: int) -> list[ConversationSummary]:
        query = select(ConversationRow).order_by(ConversationRow.updated_at.desc()).limit(limit)
        if patient_id is not None:
            query = query.where(ConversationRow.patient_id == patient_id)
        async with self._sessions() as session:
            return [_summary(r) for r in await session.scalars(query)]

    async def list_messages(self, conversation_id: str) -> list[ChatMessage]:
        query = (
            select(ChatMessageRow).where(ChatMessageRow.conversation_id == conversation_id).order_by(ChatMessageRow.seq)
        )
        async with self._sessions() as session:
            return [
                ChatMessage(
                    id=r.id,
                    conversation_id=r.conversation_id,
                    role=r.role,
                    action=r.action,
                    text=r.text,
                    intent=r.intent,
                    created_at=r.created_at,
                )
                for r in await session.scalars(query)
            ]
