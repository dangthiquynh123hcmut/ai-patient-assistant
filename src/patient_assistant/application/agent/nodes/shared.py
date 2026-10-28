"""Node và helper dùng chung giữa các flow: hỏi lại, chuyển người xử lý, ghi attempt/tool call."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langgraph.types import interrupt

from patient_assistant.application.agent.deps import AgentDeps
from patient_assistant.application.agent.presenters import clarify_question
from patient_assistant.application.agent.state import (
    AgentState,
    AttemptRecord,
    InterruptPayload,
    Outcome,
    OutcomeKind,
    UserDecision,
)
from patient_assistant.application.tools.result import ToolCallRecord, ToolErrorCode, ToolResult
from patient_assistant.domain.common import Resolution, ValidationIssue
from patient_assistant.domain.conversation import ConfirmIntent, interpret_confirmation


def ask_user(payload: InterruptPayload) -> UserDecision:
    """Dừng graph chờ người dùng. Khi resume, LangGraph chạy lại node từ đầu và hàm này trả giá trị resume —
    vì vậy mọi code trước lời gọi này trong node phải không có side effect ghi."""
    return UserDecision.model_validate(interrupt(payload.model_dump(mode="json")))


def is_cancel(decision: UserDecision) -> bool:
    if decision.action == "cancel":
        return True
    return (
        decision.action == "reply"
        and bool(decision.text)
        and interpret_confirmation(decision.text or "") == ConfirmIntent.CANCEL
    )


def idempotency_key(*parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def tool_record(state: AgentState, result: ToolResult, called_by: str = "code") -> list[ToolCallRecord]:
    return [
        *state.tool_calls,
        ToolCallRecord(
            tool=result.tool,
            called_by=called_by,
            status=result.status,
            error_code=result.error.code.value if result.error else None,
            latency_ms=result.latency_ms,
        ),
    ]


def tool_error_outcome(result: ToolResult, code: str) -> Outcome:
    assert result.error is not None
    if result.error.code == ToolErrorCode.UNAVAILABLE:
        return Outcome(kind=OutcomeKind.ERROR, code="tool_unavailable")
    return Outcome(kind=OutcomeKind.ERROR, code=code, data=result.error.details or [result.error.message])


async def record_attempt(
    deps: AgentDeps, state: AgentState, flow: str, output: dict[str, Any] | None, issues: list[ValidationIssue]
) -> list[AttemptRecord]:
    attempt = AttemptRecord(
        flow=flow,
        attempt_no=state.correction_count,
        output=output,
        issue_codes=[i.code for i in issues],
        llm_fixable_codes=[i.code for i in issues if i.resolution == Resolution.LLM_FIXABLE],
        passed=not issues,
    )
    await deps.audit.record("attempt", state.thread_id, attempt.model_dump(mode="json"))
    return [*state.attempts, attempt]


class SharedNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    async def clarify(self, state: AgentState) -> dict[str, Any]:
        if state.clarify_rounds >= self.deps.limits.max_clarify_rounds:
            return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds", issues=state.issues)}
        needs_departments = any(i.field == "department_code" for i in state.issues)
        departments = (await self.deps.tools.catalog())[0] if needs_departments else None
        question = clarify_question(state.issues, departments)
        decision = ask_user(
            InterruptPayload(
                kind="clarify",
                message=f"{state.notice} {question}" if state.notice else question,
                allowed_actions=["reply", "cancel"],
            )
        )
        if is_cancel(decision):
            return {"decision": decision, "outcome": Outcome(kind=OutcomeKind.CANCELLED)}
        asked_stt = any(i.code == "stt_uncertain" for i in state.issues)
        return {
            "decision": decision,
            "user_replies": [*state.user_replies, decision.text or ""],
            "clarify_rounds": state.clarify_rounds + 1,
            "correction_count": 0,
            "issues": [],
            "notice": None,
            # Người dùng đã nói/gõ lại đoạn bị nghi nhận dạng sai -> bỏ cờ cũ.
            "stt_flags": [] if asked_stt else state.stt_flags,
        }

    async def handoff(self, state: AgentState) -> dict[str, Any]:
        blocked = any(i.resolution == Resolution.BLOCK for i in state.issues)
        code = "blocked" if blocked else "too_many_corrections"
        await self.deps.audit.record(
            "handoff", state.thread_id, {"code": code, "issues": [i.code for i in state.issues]}
        )
        return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code=code, issues=state.issues)}
