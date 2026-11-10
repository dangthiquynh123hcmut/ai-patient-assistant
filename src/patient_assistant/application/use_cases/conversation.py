"""Hội thoại của người dùng đã đăng nhập: kiểm tra quyền sở hữu, chạy agent, lưu mọi câu hỏi và câu trả lời."""

from __future__ import annotations

import uuid
from typing import Literal

from patient_assistant.application.agent.state import AgentState, InterruptPayload, UserDecision
from patient_assistant.application.ports.clock import Clock
from patient_assistant.application.ports.repositories import ConversationLogRepository
from patient_assistant.application.use_cases.chat import ChatUseCase, TurnResult
from patient_assistant.domain.account import ChatMessage, User
from patient_assistant.domain.common import SttFlag

_TITLE_CHARS = 80


class NotAllowed(Exception):
    """Người dùng không có quyền với hội thoại này (không phải bệnh nhân, hoặc hội thoại của người khác)."""


def _decision_text(decision: UserDecision) -> str:
    if decision.text:
        return decision.text
    if decision.action == "choose":
        return f"Chọn lịch số {decision.option_no}"
    if decision.action == "edit":
        return "Sửa: " + ", ".join(f"{k}={v}" for k, v in decision.fields.items())
    return {"confirm": "Xác nhận", "cancel": "Hủy"}.get(decision.action, decision.action)


class ConversationService:
    def __init__(self, chat: ChatUseCase, log: ConversationLogRepository, clock: Clock) -> None:
        self._chat = chat
        self._log = log
        self._clock = clock

    def _patient(self, user: User) -> str:
        if not user.can_chat or user.patient_id is None:
            raise NotAllowed("Chỉ tài khoản bệnh nhân mới chat với trợ lý.")
        return user.patient_id

    async def _check_owner(self, user: User, conversation_id: str) -> str:
        patient_id = self._patient(user)
        if await self._log.owner(conversation_id) != patient_id:
            raise NotAllowed("Không có hội thoại này.")
        return patient_id

    async def send(
        self,
        user: User,
        conversation_id: str,
        text: str,
        input_source: Literal["text", "voice"] = "text",
        stt_flags: list[SttFlag] | None = None,
    ) -> TurnResult:
        patient_id = self._patient(user)
        now = self._clock.now()
        if await self._log.open(conversation_id, user.id, patient_id, text[:_TITLE_CHARS], now) != patient_id:
            raise NotAllowed("Không có hội thoại này.")
        await self._add(conversation_id, "user", "voice" if input_source == "voice" else "message", text)
        result = await self._chat.send_message(conversation_id, patient_id, text, input_source, stt_flags)
        await self._add_result(conversation_id, result)
        return result

    async def resume(self, user: User, conversation_id: str, decision: UserDecision) -> TurnResult:
        patient_id = await self._check_owner(user, conversation_id)
        result = await self._chat.resume(conversation_id, patient_id, decision)  # NoPendingQuestion -> không ghi gì
        await self._add(conversation_id, "user", decision.action, _decision_text(decision))
        await self._add_result(conversation_id, result)
        return result

    async def state(self, user: User, conversation_id: str) -> AgentState | None:
        await self._check_owner(user, conversation_id)
        return await self._chat.inspect_state(conversation_id)

    async def pending(self, user: User, conversation_id: str) -> InterruptPayload | None:
        await self._check_owner(user, conversation_id)
        return await self._chat.pending_question(conversation_id)

    async def _add_result(self, conversation_id: str, result: TurnResult) -> None:
        intent = next((s.details.get("intent") for s in result.steps if s.node == "classify"), None)
        if result.interrupt is not None:
            await self._add(conversation_id, "assistant", result.interrupt.kind, result.interrupt.message, intent)
        elif result.response is not None:
            await self._add(conversation_id, "assistant", result.response.kind.value, result.response.text, intent)

    async def _add(self, conversation_id: str, role: str, action: str, text: str, intent: str | None = None) -> None:
        await self._log.add_message(
            ChatMessage(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role=role,
                action=action,
                text=text,
                intent=intent,
                created_at=self._clock.now(),
            )
        )
