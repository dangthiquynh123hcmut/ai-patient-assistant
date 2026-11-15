"""Node của graph chính: nhận input, chặn an toàn, phân loại, truy vấn đọc, trả lời."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from patient_assistant.application.agent.dependencies import AgentDependencies
from patient_assistant.application.agent.memory import note_for_turn, render_memory
from patient_assistant.application.agent.nodes.shared import ask_user, is_cancel, next_ask, tool_record
from patient_assistant.application.agent.presenters import (
    CHAT_FALLBACK,
    INTENT_CHOICES,
    INTENT_LABELS,
    doctors_text,
    final_text,
    intent_choices,
    intent_question,
)
from patient_assistant.application.agent.state import (
    AgentResponse,
    AgentState,
    Carryover,
    InterruptPayload,
    Outcome,
    OutcomeKind,
)
from patient_assistant.application.ports.llm import LLMError, ToolSpecification
from patient_assistant.application.tools.result import ToolCallRecord
from patient_assistant.domain.booking.validators import department_in_text
from patient_assistant.domain.conversation import (
    Intent,
    SafetyKind,
    asks_about_other_patient,
    check_input_safety,
    classify_by_rules,
    is_continuation,
    looks_like_question,
    output_is_safe,
    rank_intents,
)

# Việc khác hẳn đặt lịch: khi agent đang hỏi dở mà người dùng nói tới các việc này thì rời agent.
_LEAVES_BOOKING_AGENT = frozenset(
    {Intent.CREATE_REPORT, Intent.GET_REPORT_HISTORY, Intent.GET_PATIENT_INFO, Intent.ANSWER_FAQ}
)

# Việc agent đặt lịch tự làm được bằng tool của nó (xem lịch hẹn, hỏi bác sĩ/khoa).
_BOOKING_AGENT_COVERS = frozenset({Intent.LIST_APPOINTMENTS, Intent.GET_DOCTOR_INFO})


def _prefer_booking_agent(result: dict[str, Any]) -> dict[str, Any]:
    """ "Đặt lịch giống lịch hiện tại…" nhắc tới lịch hẹn nên LLM có thể xếp `list_appointments` trước, đặt lịch sau.
    Ở chế độ agent, việc xem lịch là một phần của đặt lịch: vào thẳng agent thay vì trả danh sách rồi bảo nhắn tiếp."""
    pending = result.get("pending_intents") or []
    if result.get("intent") in _BOOKING_AGENT_COVERS and Intent.BOOK_APPOINTMENT in pending:
        return result | {
            "intent": Intent.BOOK_APPOINTMENT,
            "pending_intents": [i for i in pending if i not in (Intent.BOOK_APPOINTMENT, *_BOOKING_AGENT_COVERS)],
        }
    return result


NO_SOURCE_ANSWER = (
    "Mình không tìm thấy thông tin này trong tài liệu hướng dẫn đã được duyệt, nên không trả lời "
    "để tránh đưa thông tin chưa kiểm chứng. Bạn có thể hỏi trực tiếp nhân viên y tế."
)


def _recorded(specification: ToolSpecification, sink: list[ToolCallRecord]) -> ToolSpecification:
    """Bọc handler để ghi lại mọi lần LLM gọi tool (FR-08.3)."""

    async def handler(input_data: BaseModel) -> dict[str, Any]:
        result = await specification.handler(input_data)
        error = result.get("error") or {}
        sink.append(
            ToolCallRecord(
                tool=specification.name,
                called_by="llm",
                status=result.get("status", "error"),
                error_code=error.get("code"),
                latency_ms=result.get("latency_ms", 0),
                source_ids=result.get("source_ids", []),
            )
        )
        return result

    return ToolSpecification(specification.name, specification.description, specification.input_model, handler)


def _resume_carryover(carry: Carryover, pending: Sequence[Intent]) -> dict[str, Any]:
    return {
        "intent": carry.intent,
        "pending_intents": [i for i in pending if i != carry.intent],
        "intent_source": "context",
        "context_sources": carry.sources,
        "carryover": None,
        "notice": f"Mình tiếp tục việc {INTENT_LABELS[carry.intent]} lúc nãy, dùng lại thông tin bạn đã nói "
        "(chưa có gì được lưu).",
    }


class CommonNodes:
    def __init__(self, dependencies: AgentDependencies) -> None:
        self.dependencies = dependencies

    async def ingest(self, state: AgentState) -> dict[str, Any]:
        await self.dependencies.audit.record(
            "turn_started",
            state.conversation_id,
            {"source": state.input_source, "chars": len(state.raw_input), "stt_flags": len(state.stt_flags)},
        )
        return {"raw_input": " ".join(state.raw_input.split())}

    async def guard_input(self, state: AgentState) -> dict[str, Any]:
        return {"safety": check_input_safety(state.raw_input).kind}

    async def safety_response(self, state: AgentState) -> dict[str, Any]:
        await self.dependencies.audit.record("safety_block", state.conversation_id, {"kind": state.safety.value})
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
        if state.agent_active and self.dependencies.agent_mode == "booking":
            # Agent đặt lịch đang hỏi dở: câu tiếp theo là câu trả lời cho agent ("9 giờ", "bác sĩ khác", "cảm ơn",
            # "lịch hiện tại của tôi ngày nào?"), trừ khi người dùng rõ ràng chuyển sang việc khác.
            if guess.primary not in _LEAVES_BOOKING_AGENT:
                return {"intent": Intent.BOOK_APPOINTMENT, "intent_source": "context", "agent_input": text}
            return {**(await self._classify(state, text, guess)), "agent_active": False}
        result = await self._classify(state, text, guess)
        if self.dependencies.agent_mode == "booking":
            result = _prefer_booking_agent(result)
        return result

    async def _classify(self, state: AgentState, text: str, guess: Any) -> dict[str, Any]:
        carry = state.carryover
        # Có việc dang dở: nói "làm tiếp"/"lấy thông tin cũ", hoặc yêu cầu lại đúng việc đó ("tôi muốn đặt lịch
        # khám") -> nối lại và dùng lại những gì người dùng đã nói, không hỏi lại từ đầu.
        if carry and (
            guess.primary == carry.intent or (guess.primary in (None, Intent.SMALL_TALK) and is_continuation(text))
        ):
            return _resume_carryover(carry, pending=guess.secondary)
        if guess.primary:
            return {"intent": guess.primary, "pending_intents": guess.secondary, "intent_source": "rule"}
        try:
            memory = render_memory(state.memory)
            intents = [i for i in await self.dependencies.llm.classify_intent(text, memory) if i != Intent.UNKNOWN]
        except LLMError:
            intents = []
        if len(intents) > 1:
            intents = [i for i in intents if i != Intent.SMALL_TALK]
        # Một câu hỏi không thuộc việc nào ("hệ thống có những ... nào", câu ngoài phạm vi) -> trò chuyện
        # trả lời tự nhiên; không ép vào việc đang dở, cũng không hỏi lại "bạn muốn làm việc nào".
        if not intents and looks_like_question(text):
            return {"intent": Intent.SMALL_TALK, "intent_source": "llm"}
        # Câu chung chung ("tôi muốn xử lí ngay bây giờ") khi đang có việc dang dở -> hiểu là làm tiếp việc đó.
        if carry and (not intents or intents[0] == carry.intent):
            return _resume_carryover(carry, pending=intents[1:])
        if not intents:
            return {"intent": Intent.UNKNOWN, "intent_source": "llm"}
        return {"intent": intents[0], "pending_intents": intents[1:], "intent_source": "llm"}

    async def clarify_intent(self, state: AgentState) -> dict[str, Any]:
        counters = next_ask(state, "intent", self.dependencies)
        if counters is None:
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
        return {**counters, "decision": decision, "user_replies": [*state.user_replies, decision.text or ""]}

    async def fetch_history(self, state: AgentState) -> dict[str, Any]:
        result = await self.dependencies.tools.get_report_history(state.patient_id)
        outcome = (
            Outcome(kind=OutcomeKind.HISTORY, data=result.data, source_ids=result.source_ids)
            if result.ok
            else Outcome(kind=OutcomeKind.ERROR, code="tool_unavailable")
        )
        return {"tool_calls": tool_record(state, result), "outcome": outcome}

    async def fetch_patient_info(self, state: AgentState) -> dict[str, Any]:
        """Chỉ trả thông tin của bệnh nhân đang đăng nhập — kể cả khi người dùng hỏi về người khác."""
        result = await self.dependencies.tools.get_patient_summary(state.patient_id, state.now)
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

    async def fetch_doctors(self, state: AgentState) -> dict[str, Any]:
        """Danh mục khoa/bác sĩ (dữ liệu chung của phòng khám, không phải hồ sơ bệnh nhân)."""
        departments, doctors = await self.dependencies.tools.catalog()
        said = " ".join(state.sources())
        only = department_in_text(said, departments)
        return {"outcome": Outcome(kind=OutcomeKind.ANSWER, data=doctors_text(departments, doctors, only))}

    async def fetch_appointments(self, state: AgentState) -> dict[str, Any]:
        result = await self.dependencies.tools.list_my_appointments(state.patient_id, state.now)
        outcome = (
            Outcome(kind=OutcomeKind.APPOINTMENTS, data=result.data, source_ids=result.source_ids)
            if result.ok
            else Outcome(kind=OutcomeKind.ERROR, code="tool_unavailable")
        )
        return {"tool_calls": tool_record(state, result), "outcome": outcome}

    async def rag_answer(self, state: AgentState) -> dict[str, Any]:
        calls: list[ToolCallRecord] = []
        specifications = [
            _recorded(specification, calls)
            for specification in self.dependencies.tools.read_only_specifications(state.patient_id, state.now)
        ]
        question = " ".join(state.sources())
        try:
            answer = await self.dependencies.llm.answer_question(
                question, specifications, self.dependencies.limits.max_tool_calls, render_memory(state.memory)
            )
        except LLMError as exception:
            return {
                "tool_calls": [*state.tool_calls, *calls],
                "outcome": Outcome(kind=OutcomeKind.ERROR, code=f"llm_{exception.code}"),
            }
        tool_calls = [*state.tool_calls, *calls]
        if not output_is_safe(answer.text):
            return {
                "tool_calls": tool_calls,
                "safety": SafetyKind.MEDICAL_ADVICE,
                "outcome": Outcome(kind=OutcomeKind.SAFETY),
            }
        # Chỉ trả lời khi có nguồn đã duyệt mà chính tool trả về — không tin doc_id do LLM tự viết.
        returned = {source_id for call in calls for source_id in call.source_ids}
        sources = [source_id for source_id in answer.source_ids if source_id in returned]
        if not sources:
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ANSWER, data=NO_SOURCE_ANSWER)}
        return {
            "tool_calls": tool_calls,
            "outcome": Outcome(kind=OutcomeKind.ANSWER, data=answer.text, source_ids=sources),
        }

    async def converse(self, state: AgentState) -> dict[str, Any]:
        text = state.user_replies[-1] if state.user_replies else state.raw_input
        try:
            reply = (await self.dependencies.llm.converse(text, None, render_memory(state.memory))).strip()
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
        if state.agent_done:  # agent đã ghi nhiều việc trong lượt này: liệt kê đủ, theo thứ tự
            # Việc cuối bị hủy/lỗi: chỉ việc đó chưa ghi — không nói "chưa có dữ liệu nào được lưu" sau khi đã lưu.
            text = text.replace("Chưa có dữ liệu nào được lưu", "Việc này chưa được ghi; các việc ở trên vẫn đã lưu")
            text = "\n\n".join([*(done_text for done_text in state.agent_done if done_text != text), text])
        if state.agent_followup:
            text += f"\n\n{state.agent_followup}"
        if state.carryover and state.carryover.just_left and outcome.kind != OutcomeKind.SAFETY:
            text += (
                f"\n\nViệc {INTENT_LABELS[state.carryover.intent]} lúc nãy chưa xong và chưa được lưu — "
                "nhắn “làm tiếp” khi bạn muốn quay lại."
            )
        # Ghi chú cho các lượt sau. Danh mục chỉ để nhận ra tên khoa/bác sĩ; lỗi đọc danh mục không làm hỏng lượt.
        try:
            departments, doctors = await self.dependencies.tools.catalog()
        except Exception:  # noqa: BLE001
            departments, doctors = [], []
        turn_note = note_for_turn(
            " ".join([state.raw_input, *state.user_replies]),
            state.intent,
            outcome.kind.value,
            outcome.data,
            outcome.source_ids,
            departments,
            doctors,
        )
        response = AgentResponse(
            kind=outcome.kind,
            text=text,
            source_ids=outcome.source_ids,
            data=outcome.data,
        )
        await self.dependencies.audit.record(
            "turn_finished",
            state.conversation_id,
            {
                "outcome": outcome.kind.value,
                "code": outcome.code,
                "intent": state.intent.value if state.intent else None,
                "corrections": state.correction_count,
                "clarify_rounds": state.clarify_rounds,
                "tool_calls": [tool_call.model_dump() for tool_call in state.tool_calls],
            },
        )
        return {"response": response, "turn_note": turn_note}
