"""Node của graph chính: nhận input, chặn an toàn, phân loại, truy vấn đọc, trả lời."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from patient_assistant.application.agent.deps import AgentDeps
from patient_assistant.application.agent.nodes.shared import ask_user, is_cancel, tool_record
from patient_assistant.application.agent.presenters import (
    CHAT_FALLBACK,
    INTENT_CHOICES,
    INTENT_LABELS,
    final_text,
    intent_choices,
    intent_question,
)
from patient_assistant.application.agent.state import (
    AgentResponse,
    AgentState,
    InterruptPayload,
    Outcome,
    OutcomeKind,
)
from patient_assistant.application.ports.llm import LLMError, ToolSpec
from patient_assistant.application.tools.result import ToolCallRecord
from patient_assistant.domain.conversation import (
    Intent,
    SafetyKind,
    asks_about_other_patient,
    check_input_safety,
    classify_by_rules,
    is_continuation,
    output_is_safe,
    rank_intents,
)

NO_SOURCE_ANSWER = (
    "Mình không tìm thấy thông tin này trong tài liệu hướng dẫn đã được duyệt, nên không trả lời "
    "để tránh đưa thông tin chưa kiểm chứng. Bạn có thể hỏi trực tiếp nhân viên y tế."
)


def _recorded(spec: ToolSpec, sink: list[ToolCallRecord]) -> ToolSpec:
    """Bọc handler để ghi lại mọi lần LLM gọi tool (FR-08.3)."""

    async def handler(inp: BaseModel) -> dict[str, Any]:
        result = await spec.handler(inp)
        error = result.get("error") or {}
        sink.append(
            ToolCallRecord(
                tool=spec.name,
                called_by="llm",
                status=result.get("status", "error"),
                error_code=error.get("code"),
                latency_ms=result.get("latency_ms", 0),
                source_ids=result.get("source_ids", []),
            )
        )
        return result

    return ToolSpec(spec.name, spec.description, spec.input_model, handler)


class CommonNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    async def ingest(self, state: AgentState) -> dict[str, Any]:
        await self.deps.audit.record(
            "turn_started",
            state.thread_id,
            {"source": state.input_source, "chars": len(state.raw_input), "stt_flags": len(state.stt_flags)},
        )
        return {"raw_input": " ".join(state.raw_input.split())}

    async def guard_input(self, state: AgentState) -> dict[str, Any]:
        return {"safety": check_input_safety(state.raw_input).kind}

    async def safety_response(self, state: AgentState) -> dict[str, Any]:
        await self.deps.audit.record("safety_block", state.thread_id, {"kind": state.safety.value})
        return {"outcome": Outcome(kind=OutcomeKind.SAFETY, code=state.safety.value)}

    async def classify(self, state: AgentState) -> dict[str, Any]:
        text = state.user_replies[-1] if state.user_replies else state.raw_input
        verdict = check_input_safety(text)
        if verdict.kind != SafetyKind.OK:
            return {
                "safety": verdict.kind,
                "intent": Intent.UNKNOWN,
                "outcome": Outcome(kind=OutcomeKind.SAFETY, code=verdict.kind.value),
            }
        guess = classify_by_rules(text)
        carry = state.carryover
        if carry and guess.primary in (None, carry.intent, Intent.SMALL_TALK) and is_continuation(text):
            return {
                "intent": carry.intent,
                "intent_source": "context",
                "context_sources": carry.sources,
                "carryover": None,
                "notice": f"Mình tiếp tục việc {INTENT_LABELS[carry.intent]} lúc nãy (chưa có gì được lưu).",
            }
        if guess.primary:
            return {"intent": guess.primary, "pending_intents": guess.secondary, "intent_source": "rule"}
        try:
            intents = [i for i in await self.deps.llm.classify_intent(text) if i != Intent.UNKNOWN]
        except LLMError:
            intents = []
        if len(intents) > 1:
            intents = [i for i in intents if i != Intent.SMALL_TALK]
        if not intents:
            return {"intent": Intent.UNKNOWN, "intent_source": "llm"}
        return {"intent": intents[0], "pending_intents": intents[1:], "intent_source": "llm"}

    async def clarify_intent(self, state: AgentState) -> dict[str, Any]:
        if state.clarify_rounds >= self.deps.limits.max_clarify_rounds:
            return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds")}
        # Đoán intent gần nhất bằng từ khóa để hỏi lại có trọng tâm, kèm nút chọn nhanh — thay vì menu chung.
        said = state.user_replies[-1] if state.user_replies else state.raw_input
        ranked = [i for i, _ in rank_intents(said) if i in INTENT_CHOICES]
        decision = ask_user(
            InterruptPayload(
                kind="clarify",
                message=intent_question(said, ranked),
                choices=intent_choices(ranked),
                allowed_actions=["reply", "cancel"],
            )
        )
        if is_cancel(decision):
            return {"decision": decision, "outcome": Outcome(kind=OutcomeKind.CANCELLED)}
        return {
            "decision": decision,
            "user_replies": [*state.user_replies, decision.text or ""],
            "clarify_rounds": state.clarify_rounds + 1,
        }

    async def fetch_history(self, state: AgentState) -> dict[str, Any]:
        result = await self.deps.tools.get_report_history(state.patient_id)
        outcome = (
            Outcome(kind=OutcomeKind.HISTORY, data=result.data, source_ids=result.source_ids)
            if result.ok
            else Outcome(kind=OutcomeKind.ERROR, code="tool_unavailable")
        )
        return {"tool_calls": tool_record(state, result), "outcome": outcome}

    async def fetch_patient_info(self, state: AgentState) -> dict[str, Any]:
        """Chỉ trả thông tin của bệnh nhân đang đăng nhập — kể cả khi người dùng hỏi về người khác."""
        result = await self.deps.tools.get_patient_summary(state.patient_id, state.now)
        if not result.ok:
            return {
                "tool_calls": tool_record(state, result),
                "outcome": Outcome(kind=OutcomeKind.ERROR, code="tool_unavailable"),
            }
        other = any(asks_about_other_patient(text) for text in state.sources())
        return {
            "tool_calls": tool_record(state, result),
            "outcome": Outcome(
                kind=OutcomeKind.PATIENT_INFO,
                code="own_data_only" if other else None,
                data=result.data,
                source_ids=result.source_ids,
            ),
        }

    async def fetch_appointments(self, state: AgentState) -> dict[str, Any]:
        result = await self.deps.tools.list_my_appointments(state.patient_id, state.now)
        outcome = (
            Outcome(kind=OutcomeKind.APPOINTMENTS, data=result.data, source_ids=result.source_ids)
            if result.ok
            else Outcome(kind=OutcomeKind.ERROR, code="tool_unavailable")
        )
        return {"tool_calls": tool_record(state, result), "outcome": outcome}

    async def rag_answer(self, state: AgentState) -> dict[str, Any]:
        calls: list[ToolCallRecord] = []
        specs = [_recorded(spec, calls) for spec in self.deps.tools.read_only_specs(state.patient_id, state.now)]
        question = " ".join(state.sources())
        try:
            answer = await self.deps.llm.answer_question(question, specs, self.deps.limits.max_tool_calls)
        except LLMError as exc:
            return {
                "tool_calls": [*state.tool_calls, *calls],
                "outcome": Outcome(kind=OutcomeKind.ERROR, code=f"llm_{exc.code}"),
            }
        tool_calls = [*state.tool_calls, *calls]
        if not output_is_safe(answer.text):
            return {
                "tool_calls": tool_calls,
                "safety": SafetyKind.MEDICAL_ADVICE,
                "outcome": Outcome(kind=OutcomeKind.SAFETY),
            }
        # Chỉ trả lời khi có nguồn đã duyệt mà chính tool trả về — không tin doc_id do LLM tự viết.
        returned = {s for c in calls for s in c.source_ids}
        sources = [s for s in answer.source_ids if s in returned]
        if not sources:
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ANSWER, data=NO_SOURCE_ANSWER)}
        return {
            "tool_calls": tool_calls,
            "outcome": Outcome(kind=OutcomeKind.ANSWER, data=answer.text, source_ids=sources),
        }

    async def converse(self, state: AgentState) -> dict[str, Any]:
        text = state.user_replies[-1] if state.user_replies else state.raw_input
        try:
            reply = (await self.deps.llm.converse(text)).strip()
        except LLMError:
            reply = ""
        # Câu trò chuyện vẫn qua cùng bộ lọc đầu ra; lỗi hay rỗng thì dùng câu chào cố định.
        if not reply or not output_is_safe(reply):
            reply = CHAT_FALLBACK
        return {"outcome": Outcome(kind=OutcomeKind.CHAT, data=reply)}

    async def respond(self, state: AgentState) -> dict[str, Any]:
        outcome = state.outcome or Outcome(kind=OutcomeKind.ERROR, code="no_outcome")
        text = final_text(
            outcome,
            state.safety,
            state.pending_intents if outcome.kind not in (OutcomeKind.SAFETY, OutcomeKind.ERROR) else [],
        )
        if state.carryover and state.carryover.just_left and outcome.kind != OutcomeKind.SAFETY:
            text += (
                f"\n\nViệc {INTENT_LABELS[state.carryover.intent]} lúc nãy chưa xong và chưa được lưu — "
                "nhắn “làm tiếp” khi bạn muốn quay lại."
            )
        response = AgentResponse(
            kind=outcome.kind,
            text=text,
            source_ids=outcome.source_ids,
            data=outcome.data,
        )
        await self.deps.audit.record(
            "turn_finished",
            state.thread_id,
            {
                "outcome": outcome.kind.value,
                "code": outcome.code,
                "intent": state.intent.value if state.intent else None,
                "corrections": state.correction_count,
                "clarify_rounds": state.clarify_rounds,
                "tool_calls": [t.model_dump() for t in state.tool_calls],
            },
        )
        return {"response": response}
