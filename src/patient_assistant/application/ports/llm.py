"""Port LLM. Agent chỉ biết interface này; Bedrock hay offline là chi tiết của infrastructure."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel

from patient_assistant.domain.booking.models import Department, SlotOption
from patient_assistant.domain.common import ValidationIssue
from patient_assistant.domain.conversation import Intent


class LLMError(Exception):
    """Lỗi LLM được chuẩn hóa. `code`: refusal | max_tokens | unavailable | invalid_output."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class CorrectionFeedback:
    """Lần sửa: đầu ra trước + lỗi cụ thể từ validator (FR-06.1)."""

    previous: dict[str, Any]
    issues: list[ValidationIssue]


@dataclass(frozen=True)
class ReportPrompt:
    sources: list[str]
    now: datetime
    feedback: CorrectionFeedback | None = None


@dataclass(frozen=True)
class BookingPrompt:
    sources: list[str]
    now: datetime
    departments: list[Department]
    feedback: CorrectionFeedback | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Tool đọc dữ liệu mà LLM được phép gọi. Định danh bệnh nhân đã được bind sẵn trong `handler`."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[dict[str, Any]]]


@dataclass
class GroundedAnswer:
    text: str
    source_ids: list[str] = field(default_factory=list)
    tool_calls: int = 0


class LLMPort(Protocol):
    async def classify_intent(self, text: str) -> list[Intent]:
        """Danh sách intent theo thứ tự ưu tiên; rỗng nếu không xác định được."""
        ...

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        """JSON thô theo schema ReportExtraction. Việc validate là của domain."""
        ...

    async def extract_booking(self, prompt: BookingPrompt) -> dict[str, Any]: ...

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        """Dự phòng khi luật không ghép được câu chọn slot. Trả option_no hoặc None."""
        ...

    async def answer_question(self, question: str, tools: list[ToolSpec], max_tool_calls: int) -> GroundedAnswer: ...
