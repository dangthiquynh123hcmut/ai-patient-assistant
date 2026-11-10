"""Câu người dùng gõ khi agent đang chờ trả lời: là câu trả lời, câu hỏi chen ngang, hay đổi sang việc khác.

Không phân loại thì mọi câu đều bị nhét vào flow đang chạy: câu hỏi "dữ liệu lưu ở đâu" bị coi là câu trả lời
cho "bạn tập mấy ngày", và một yêu cầu mới bị nuốt cho tới khi hết lượt hỏi lại.
"""

from __future__ import annotations

import re
from enum import StrEnum

from patient_assistant.application.agent.memory import render_memory
from patient_assistant.application.agent.presenters import (
    ASIDE_FALLBACK,
    INTENT_LABELS,
    assistant_context_text,
    doctors_text,
)
from patient_assistant.application.agent.state import AgentState
from patient_assistant.application.ports.llm import LLMError, LLMPort
from patient_assistant.application.tools.toolbox import Toolbox
from patient_assistant.domain.booking.validators import department_in_text
from patient_assistant.domain.conversation import (
    TASK_INTENTS,
    ConfirmIntent,
    Intent,
    asks_about_assistant_context,
    classify_by_rules,
    interpret_confirmation,
    is_continuation,
    output_is_safe,
)

_DIGIT = re.compile(r"\d")
# Câu ngắn ("thứ năm", "khoa nội", "buổi sáng") gần như luôn là câu trả lời — không tốn một lần gọi LLM.
_MIN_WORDS_FOR_LLM = 5


class ReplyKind(StrEnum):
    ANSWER = "answer"  # trả lời câu hỏi đang chờ -> resume flow
    ASIDE = "aside"  # hỏi chen ngang -> trả lời riêng rồi hỏi lại đúng câu cũ
    SWITCH = "switch"  # yêu cầu việc khác -> bắt đầu lượt mới, việc đang làm được giữ để nối lại


class SideTalk:
    def __init__(self, llm: LLMPort, tools: Toolbox) -> None:
        self._llm = llm
        self._tools = tools

    async def triage(self, text: str, current: Intent | None) -> ReplyKind:
        # Đang hỏi rõ yêu cầu (chưa có việc nào): câu nào cũng đi vào phân loại lại, không cần tách.
        if not text.strip() or current in (None, Intent.UNKNOWN):
            return ReplyKind.ANSWER
        if interpret_confirmation(text) != ConfirmIntent.UNCLEAR:
            return ReplyKind.ANSWER
        if asks_about_assistant_context(text):
            return ReplyKind.ASIDE
        guess = classify_by_rules(text)
        if guess.primary in (Intent.SMALL_TALK, Intent.GET_DOCTOR_INFO):
            return ReplyKind.ASIDE
        # "lấy thông tin cũ tôi đã trả lời phía trên" là câu trả lời cho việc đang làm, không phải việc khác.
        if guess.primary is None and is_continuation(text):
            return ReplyKind.ANSWER
        if guess.primary in TASK_INTENTS and guess.primary != current:
            return ReplyKind.SWITCH
        if guess.primary is not None or _DIGIT.search(text) or len(text.split()) < _MIN_WORDS_FOR_LLM:
            return ReplyKind.ANSWER
        try:
            intents = await self._llm.classify_intent(text)
        except LLMError:
            return ReplyKind.ANSWER
        first = intents[0] if intents else None
        if first == Intent.SMALL_TALK:
            return ReplyKind.ASIDE
        if first in TASK_INTENTS and first != current:
            return ReplyKind.SWITCH
        return ReplyKind.ANSWER

    async def answer(self, text: str, state: AgentState, question: str) -> str:
        task = INTENT_LABELS.get(state.intent) if state.intent else None
        if classify_by_rules(text).primary == Intent.GET_DOCTOR_INFO:
            departments, doctors = await self._tools.catalog()
            only = department_in_text(text, departments)
            return doctors_text(departments, doctors, only)
        if asks_about_assistant_context(text):
            summary = await self._tools.get_patient_summary(state.patient_id, state.now)
            name = (summary.data or {}).get("profile", {}).get("display_name") if summary.ok else None
            return assistant_context_text(name, state.patient_id, task)
        context = f"Người dùng đang làm dở việc: {task or 'không rõ'}. Câu hỏi đang chờ họ trả lời: “{question}”"
        try:
            reply = (await self._llm.converse(text, context, render_memory(state.memory))).strip()
        except LLMError:
            reply = ""
        return reply if reply and output_is_safe(reply) else ASIDE_FALLBACK
