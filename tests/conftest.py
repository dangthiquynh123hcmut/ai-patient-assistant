from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from patient_assistant.application.ports.llm import (
    BookingPrompt,
    GroundedAnswer,
    LLMError,
    ReportPrompt,
    ToolSpec,
)
from patient_assistant.application.use_cases.chat import TurnResult
from patient_assistant.config import Settings
from patient_assistant.container import Container, assemble, in_memory_repositories, memory_checkpointer
from patient_assistant.domain.booking.models import SlotOption
from patient_assistant.domain.conversation import Intent
from patient_assistant.infrastructure.clock import FixedClock
from patient_assistant.infrastructure.llm.offline import OfflineRuleLLM
from patient_assistant.infrastructure.persistence.memory import InMemoryAuditLog
from patient_assistant.infrastructure.persistence.seed import DEMO_PATIENT_ID
from patient_assistant.infrastructure.stt.faster_whisper import UnavailableSTT

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
NOW = datetime(2026, 10, 28, 10, 0, tzinfo=TZ)  # Thứ Tư
PATIENT = DEMO_PATIENT_ID


class ScriptedLLM(OfflineRuleLLM):
    """LLM giả trả lần lượt các đầu ra định sẵn; hết kịch bản thì dùng luật offline."""

    def __init__(self, reports: list[Any] | None = None, bookings: list[Any] | None = None) -> None:
        self.reports = list(reports or [])
        self.bookings = list(bookings or [])
        self.report_prompts: list[ReportPrompt] = []
        self.booking_prompts: list[BookingPrompt] = []

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        self.report_prompts.append(prompt)
        if self.reports:
            out = self.reports.pop(0)
            if isinstance(out, Exception):
                raise out
            return out
        return await super().extract_report(prompt)

    async def extract_booking(self, prompt: BookingPrompt) -> dict[str, Any]:
        self.booking_prompts.append(prompt)
        if self.bookings:
            out = self.bookings.pop(0)
            if isinstance(out, Exception):
                raise out
            return out
        return await super().extract_booking(prompt)


class FailingLLM(OfflineRuleLLM):
    async def classify_intent(self, text: str) -> list[Intent]:
        raise LLMError("unavailable", "down", retryable=True)

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        raise LLMError("refusal", "declined")

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        raise LLMError("unavailable", "down")

    async def answer_question(self, question: str, tools: list[ToolSpec], max_tool_calls: int) -> GroundedAnswer:
        raise LLMError("unavailable", "down")


@pytest.fixture
def make_app() -> Callable[..., Container]:
    def _make(llm: Any = None, repos: dict[str, Any] | None = None, **settings: Any) -> Container:
        clock = FixedClock(NOW)
        container = assemble(
            Settings(**settings),
            repos or in_memory_repositories(clock),
            llm or OfflineRuleLLM(),
            UnavailableSTT(),
            clock,
            memory_checkpointer(),
            audit=InMemoryAuditLog(),
        )
        return container

    return _make


def waiting(result: TurnResult, kind: str) -> TurnResult:
    assert result.status == "waiting_user", result
    assert result.interrupt is not None and result.interrupt.kind == kind, result.interrupt
    return result


def done(result: TurnResult, kind: str) -> TurnResult:
    assert result.status == "done", result
    assert result.response is not None and result.response.kind.value == kind, result.response
    return result
