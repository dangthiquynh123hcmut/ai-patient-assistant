"""Node của report_flow: trích xuất -> validate -> (sửa | hỏi lại) -> xác nhận -> lưu -> đọc lại."""

from __future__ import annotations

from typing import Any

from patient_assistant.application.agent.deps import AgentDeps
from patient_assistant.application.agent.nodes.shared import (
    ask_user,
    idempotency_key,
    is_cancel,
    record_attempt,
    tool_error_outcome,
    tool_record,
)
from patient_assistant.application.agent.presenters import REPORT_CONFIRM_MESSAGE, report_explanations
from patient_assistant.application.agent.routing import confirmation_of
from patient_assistant.application.agent.state import AgentState, InterruptPayload, Outcome, OutcomeKind
from patient_assistant.application.ports.llm import CorrectionFeedback, LLMError, ReportPrompt
from patient_assistant.domain.common import Resolution
from patient_assistant.domain.conversation import ConfirmIntent
from patient_assistant.domain.report.models import REPORT_FIELDS, Report
from patient_assistant.domain.report.validators import ReportValidationContext, parse_report, validate_report


class ReportNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    async def _generate(self, state: AgentState, feedback: CorrectionFeedback | None) -> dict[str, Any]:
        try:
            raw = await self.deps.llm.extract_report(ReportPrompt(state.sources(), state.now, feedback))
        except LLMError as exc:
            return {"outcome": Outcome(kind=OutcomeKind.ERROR, code=f"llm_{exc.code}")}
        draft, schema_issues = parse_report(raw)
        return {"last_output": raw, "report_draft": draft, "issues": schema_issues, "edited_fields": []}

    async def extract(self, state: AgentState) -> dict[str, Any]:
        return await self._generate(state, None)

    async def correct(self, state: AgentState) -> dict[str, Any]:
        fixable = [i for i in state.issues if i.resolution == Resolution.LLM_FIXABLE]
        updates = await self._generate(state, CorrectionFeedback(state.last_output or {}, fixable))
        updates["correction_count"] = state.correction_count + 1
        return updates

    async def validate(self, state: AgentState) -> dict[str, Any]:
        if state.report_draft is None:
            issues = state.issues  # lớp schema đã thất bại
        else:
            issues = validate_report(
                state.report_draft,
                ReportValidationContext(
                    sources=state.sources(),
                    now=state.now,
                    stt_flags=state.stt_flags,
                    edited_fields=frozenset(state.edited_fields),
                ),
            )
        output = (
            state.last_output
            if not state.edited_fields
            else (state.report_draft.model_dump(mode="json") if state.report_draft else None)
        )
        return {"issues": issues, "attempts": await record_attempt(self.deps, state, "report", output, issues)}

    async def present_draft(self, state: AgentState) -> dict[str, Any]:
        assert state.report_draft is not None
        if state.clarify_rounds >= self.deps.limits.max_clarify_rounds:
            return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds")}
        decision = ask_user(
            InterruptPayload(
                kind="confirm_report",
                message=f"{state.notice} {REPORT_CONFIRM_MESSAGE}" if state.notice else REPORT_CONFIRM_MESSAGE,
                explanations=report_explanations(state.report_draft, [], state.edited_fields, state.now),
                allowed_actions=["confirm", "edit", "cancel", "reply"],
            )
        )
        updates: dict[str, Any] = {"decision": decision, "notice": None}
        if is_cancel(decision):
            updates["outcome"] = Outcome(kind=OutcomeKind.CANCELLED)
        elif confirmation_of(state.model_copy(update={"decision": decision})) == ConfirmIntent.UNCLEAR:
            updates["clarify_rounds"] = state.clarify_rounds + 1
            updates["notice"] = "Mình chưa rõ bạn đã xác nhận hay chưa."
        return updates

    async def apply_edits(self, state: AgentState) -> dict[str, Any]:
        assert state.report_draft is not None and state.decision is not None
        fields = {k: v for k, v in state.decision.fields.items() if k in REPORT_FIELDS}
        merged = {**state.report_draft.model_dump(mode="json"), **fields}
        draft, schema_issues = parse_report(merged)
        # Giá trị sai kiểu do người dùng gõ -> hỏi lại người dùng, không đưa cho LLM sửa.
        issues = [i.model_copy(update={"resolution": Resolution.ASK_USER}) for i in schema_issues]
        return {
            "report_draft": draft or state.report_draft,
            "edited_fields": sorted(set(state.edited_fields) | set(fields)),
            "issues": issues,
        }

    async def save(self, state: AgentState) -> dict[str, Any]:
        draft = state.report_draft
        # Chốt chặn bằng code: không bao giờ ghi khi chưa validate đạt hoặc chưa xác nhận (FR-04.5, FR-07.3).
        if draft is None or state.issues or confirmation_of(state) != ConfirmIntent.CONFIRM:
            return {"outcome": Outcome(kind=OutcomeKind.ERROR, code="guard_violation")}
        key = idempotency_key("report", state.thread_id, state.patient_id, draft.model_dump(mode="json"))
        result = await self.deps.tools.save_patient_report(state.patient_id, draft, key, state.now)
        updates: dict[str, Any] = {
            "tool_calls": tool_record(state, result),
            "write_result": result.model_dump(mode="json"),
        }
        if not result.ok:
            updates["outcome"] = tool_error_outcome(result, "save_failed")
        return updates

    async def verify_save(self, state: AgentState) -> dict[str, Any]:
        """Lớp 5: chỉ báo thành công khi backend trả ok VÀ đọc lại được đúng bản ghi (FR-07.6)."""
        written = state.write_result or {}
        data = written.get("data") or {}
        report_id = data.get("id")
        if written.get("status") != "ok" or not report_id:
            return {"outcome": Outcome(kind=OutcomeKind.ERROR, code="verify_failed")}
        back = await self.deps.tools.get_report(state.patient_id, report_id)
        tool_calls = tool_record(state, back)
        if not back.ok:
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ERROR, code="verify_failed")}
        saved = Report.model_validate(back.data)
        assert state.report_draft is not None
        if (saved.pain_level, saved.exercise_days) != (state.report_draft.pain_level, state.report_draft.exercise_days):
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ERROR, code="verify_failed")}
        return {
            "tool_calls": tool_calls,
            "outcome": Outcome(kind=OutcomeKind.REPORT_SAVED, data=back.data, source_ids=back.source_ids),
        }
