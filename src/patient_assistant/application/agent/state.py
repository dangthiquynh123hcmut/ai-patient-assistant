"""State của LangGraph. Định danh (thread_id, patient_id, now) do API/use case đặt, không bao giờ do LLM."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, get_args

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from patient_assistant.application.tools.result import ToolCallRecord
from patient_assistant.domain.booking.models import BookingRequest, SlotOption
from patient_assistant.domain.common import Origin, SttFlag, ValidationIssue
from patient_assistant.domain.conversation import Intent, SafetyKind
from patient_assistant.domain.report.models import ReportExtraction
from patient_assistant.domain.text.vi_dates import DateWindow


class OutcomeKind(StrEnum):
    REPORT_SAVED = "report_saved"
    APPOINTMENT_BOOKED = "appointment_booked"
    HISTORY = "history"
    APPOINTMENTS = "appointments"
    ANSWER = "answer"
    CANCELLED = "cancelled"
    HANDOFF = "handoff"
    SAFETY = "safety"
    ERROR = "error"


class Outcome(BaseModel):
    kind: OutcomeKind
    code: str | None = None  # lý do cụ thể: llm_refusal, tool_unavailable, no_slots, too_many_corrections...
    data: Any = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class UserDecision(BaseModel):
    """Giá trị resume của mọi interrupt. UI gửi action tường minh; giọng nói gửi `reply` + text."""

    action: Literal["reply", "confirm", "edit", "cancel", "choose", "other_time"]
    text: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    option_no: int | None = None


class FieldExplanation(BaseModel):
    """Giải thích từng trường (FR-05). Do code dựng từ kết quả validator, không do LLM viết."""

    field: str
    label: str
    value: Any
    evidence: str | None
    origin: Origin
    validation: Literal["passed", "failed", "missing", "uncertain"]
    explanation: str


class InterruptPayload(BaseModel):
    kind: Literal["clarify", "confirm_report", "choose_slot", "confirm_booking"]
    message: str
    explanations: list[FieldExplanation] = Field(default_factory=list)
    options: list[SlotOption] = Field(default_factory=list)
    allowed_actions: list[str]


class AgentResponse(BaseModel):
    kind: OutcomeKind
    text: str
    explanations: list[FieldExplanation] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    data: Any = None


class AttemptRecord(BaseModel):
    """Metadata mỗi lần LLM tạo/sửa đầu ra (FR-06.6)."""

    flow: Literal["report", "booking"]
    attempt_no: int
    output: dict[str, Any] | None
    issue_codes: list[str]
    llm_fixable_codes: list[str] = Field(default_factory=list)  # lỗi do LLM (phân biệt với lỗi cần hỏi người dùng)
    passed: bool


class AgentState(BaseModel):
    # -- định danh lượt
    thread_id: str
    patient_id: str
    now: datetime

    # -- đầu vào
    input_source: Literal["text", "voice"] = "text"
    raw_input: str
    stt_flags: list[SttFlag] = Field(default_factory=list)

    # -- phân loại
    intent: Intent | None = None
    pending_intents: list[Intent] = Field(default_factory=list)
    intent_source: Literal["rule", "llm"] | None = None
    safety: SafetyKind = SafetyKind.OK

    # -- dùng chung cho các flow
    user_replies: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    correction_count: int = 0
    clarify_rounds: int = 0
    last_output: dict[str, Any] | None = None
    decision: UserDecision | None = None
    attempts: list[AttemptRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    # -- report_flow
    report_draft: ReportExtraction | None = None
    edited_fields: list[str] = Field(default_factory=list)

    # -- booking_flow
    booking_request: BookingRequest | None = None
    department_from_plan: bool = False
    resolved_window: DateWindow | None = None
    date_issues: list[ValidationIssue] = Field(default_factory=list)
    doctor_id: str | None = None
    offered_slots: list[SlotOption] = Field(default_factory=list)
    relaxed_search: bool = False
    selected_slot_id: str | None = None
    search_rounds: int = 0
    conflict_retries: int = 0

    # -- ghi dữ liệu: kết quả tool ghi, chờ node verify đọc lại
    write_result: dict[str, Any] | None = None
    notice: str | None = None  # thông báo kèm lần hỏi tiếp theo, ví dụ "slot vừa bị đặt mất"

    # -- kết quả
    outcome: Outcome | None = None
    response: AgentResponse | None = None

    def sources(self) -> list[str]:
        """Nguồn để đối chiếu evidence: lời người dùng ban đầu + các câu trả lời bổ sung."""
        return [self.raw_input, *self.user_replies]


AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def checkpoint_types() -> set[tuple[str, str]]:
    """Mọi kiểu (pydantic model, enum) có thể nằm trong checkpoint của AgentState.

    Dùng làm allowlist cho serializer của checkpointer: chỉ giải mã đúng các kiểu này, thay vì cho phép
    mọi kiểu (mặc định cũ của LangGraph, sẽ bị chặn ở bản sau). Tự suy ra từ type hint nên không bị lệch
    khi thêm trường mới.
    """
    found: set[tuple[str, str]] = set()

    def visit(tp: Any) -> None:
        for arg in get_args(tp):
            visit(arg)
        if isinstance(tp, type) and tp.__module__.startswith("patient_assistant."):
            key = (tp.__module__, tp.__qualname__)
            if key in found:
                return
            found.add(key)
            if issubclass(tp, BaseModel):
                for f in tp.model_fields.values():
                    visit(f.annotation)

    visit(AgentState)
    return found
