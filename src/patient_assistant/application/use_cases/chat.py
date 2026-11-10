"""Use case hội thoại: bọc graph, ẩn chi tiết LangGraph (config, Command, interrupt) khỏi lớp API."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel, Field

from patient_assistant.application.agent.memory import TurnNote, append_note, note_for_turn
from patient_assistant.application.agent.side_talk import ReplyKind, SideTalk
from patient_assistant.application.agent.state import (
    AgentResponse,
    AgentState,
    Carryover,
    InterruptPayload,
    OutcomeKind,
    UserDecision,
)
from patient_assistant.application.agent.trace import NODES, StepTrace, describe_interrupt, describe_step
from patient_assistant.application.ports.clock import Clock
from patient_assistant.domain.common import SttFlag
from patient_assistant.domain.conversation import FLOW_INTENTS, is_continuation


class TurnResult(BaseModel):
    conversation_id: str
    status: Literal["waiting_user", "done"]
    interrupt: InterruptPayload | None = None
    response: AgentResponse | None = None
    steps: list[StepTrace] = Field(default_factory=list)  # các node đã chạy trong lượt này, kèm diễn giải


SUBGRAPH_NODES = ("report_flow", "booking_flow")


class NoPendingQuestion(Exception):
    """Gọi resume khi hội thoại không có câu hỏi nào đang chờ."""


class ChatUseCase:
    def __init__(self, graph: Any, clock: Clock, recursion_limit: int, side_talk: SideTalk | None = None) -> None:
        self._graph = graph
        self._clock = clock
        self._recursion_limit = recursion_limit
        self._side_talk = side_talk

    def _config(self, conversation_id: str) -> dict[str, Any]:
        # "thread_id" là khóa do LangGraph quy định cho checkpointer; giá trị là mã hội thoại của mình.
        return {"configurable": {"thread_id": conversation_id}, "recursion_limit": self._recursion_limit}

    async def send_message(
        self,
        conversation_id: str,
        patient_id: str,
        text: str,
        input_source: Literal["text", "voice"] = "text",
        stt_flags: list[SttFlag] | None = None,
    ) -> TurnResult:
        """Bắt đầu một lượt mới. State của lượt được đặt lại (mỗi lượt một intent chính); chỉ giữ việc dang dở
        của lượt trước để người dùng có thể nói "lưu đi", "làm tiếp" mà không phải kể lại từ đầu."""
        previous = await self.inspect_state(conversation_id)
        fresh = AgentState(
            conversation_id=conversation_id,
            patient_id=patient_id,
            now=self._clock.now(),
            input_source=input_source,
            raw_input=text,
            stt_flags=stt_flags or [],
            carryover=_carryover(previous, patient_id),
            memory=_memory(previous, patient_id),
        )
        steps = await self._run(fresh.model_dump(), conversation_id)
        return await self._result(conversation_id, steps)

    async def resume(self, conversation_id: str, patient_id: str, decision: UserDecision) -> TurnResult:
        config = self._config(conversation_id)
        snapshot = await self._graph.aget_state(config)
        if not snapshot or not _pending(snapshot) or snapshot.values.get("patient_id") != patient_id:
            raise NoPendingQuestion(conversation_id)
        if self._side_talk and decision.action == "reply" and decision.text:
            state = await self.inspect_state(conversation_id)
            assert state is not None
            carry = state.carryover
            if carry and carry.intent == state.intent and is_continuation(decision.text):
                # "lấy thông tin cũ tôi đã trả lời phía trên" giữa chừng -> bắt đầu lại đúng việc này, kèm lời cũ.
                restarted = await self.send_message(conversation_id, patient_id, decision.text)
                note = _step("reuse_context", "Người dùng nhắc lại thông tin đã nói → nối lời cũ vào việc đang làm.")
                steps = [note, *restarted.steps]
                return restarted.model_copy(
                    update={"steps": [s.model_copy(update={"index": i}) for i, s in enumerate(steps)]}
                )
            kind = await self._side_talk.triage(decision.text, state.intent)
            if kind == ReplyKind.SWITCH:
                switched = await self.send_message(conversation_id, patient_id, decision.text)
                note = _step(
                    "switch_task",
                    "Câu trả lời là một yêu cầu khác → bắt đầu việc mới; việc đang làm được "
                    "giữ lại, nói “làm tiếp” hoặc “lưu đi” để quay lại.",
                )
                steps = [note, *switched.steps]
                return switched.model_copy(
                    update={"steps": [s.model_copy(update={"index": i}) for i, s in enumerate(steps)]}
                )
            if kind == ReplyKind.ASIDE:
                pending = InterruptPayload.model_validate(_pending(snapshot)[0])
                reply = await self._side_talk.answer(decision.text, state, pending.message)
                note = _step(
                    "side_reply",
                    "Câu người dùng là câu hỏi chen ngang, không phải câu trả lời → trả lời "
                    "riêng, không đưa vào dữ liệu, không tính lượt hỏi lại, rồi hỏi lại đúng câu đang chờ.",
                )
                return TurnResult(
                    conversation_id=conversation_id,
                    status="waiting_user",
                    interrupt=pending.model_copy(update={"message": f"{reply}\n\n{pending.message}"}),
                    steps=[note],
                )
        steps = await self._run(Command(resume=decision.model_dump(mode="json")), conversation_id)
        return await self._result(conversation_id, steps)

    async def _run(self, graph_input: Any, conversation_id: str) -> list[StepTrace]:
        """Chạy graph và ghi lại từng node đã chạy (kể cả trong subgraph) để giải thích quyết định."""
        steps: list[StepTrace] = []
        async for namespace, chunk in self._graph.astream(
            graph_input, self._config(conversation_id), stream_mode="updates", subgraphs=True
        ):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    if not namespace:  # interrupt được phát cả ở subgraph lẫn graph cha -> chỉ lấy một lần
                        steps += [describe_interrupt(len(steps), i.value) for i in update]
                elif not (not namespace and node in SUBGRAPH_NODES):  # bỏ bản gộp state khi subgraph kết thúc
                    steps.append(describe_step(len(steps), namespace, node, update))
        return steps

    async def inspect_state(self, conversation_id: str) -> AgentState | None:
        """State hiện tại của hội thoại, kể cả khi đang dừng trong subgraph (debug, evaluation).

        Khi subgraph đang chờ interrupt, state của nó chưa gộp vào graph cha nên phải đọc từ task.
        """
        snapshot = await self._graph.aget_state(self._config(conversation_id), subgraphs=True)
        if not snapshot or not snapshot.values:
            return None
        values = dict(snapshot.values)
        for task in snapshot.tasks or ():
            sub = getattr(task, "state", None)
            if sub is not None and getattr(sub, "values", None):
                values.update(sub.values)
        return AgentState.model_validate(values)

    async def pending_question(self, conversation_id: str) -> InterruptPayload | None:
        """Câu hỏi đang chờ người dùng trả lời (nếu có) — để mở lại hội thoại cũ và trả lời tiếp."""
        snapshot = await self._graph.aget_state(self._config(conversation_id))
        pending = _pending(snapshot) if snapshot else []
        return InterruptPayload.model_validate(pending[0]) if pending else None

    async def _result(self, conversation_id: str, steps: list[StepTrace]) -> TurnResult:
        snapshot = await self._graph.aget_state(self._config(conversation_id))
        pending = _pending(snapshot)
        if pending:
            return TurnResult(
                conversation_id=conversation_id,
                status="waiting_user",
                interrupt=InterruptPayload.model_validate(pending[0]),
                steps=steps,
            )
        response = snapshot.values.get("response")
        if isinstance(response, dict):
            response = AgentResponse.model_validate(response)
        return TurnResult(conversation_id=conversation_id, status="done", response=response, steps=steps)


def _step(node: str, summary: str) -> StepTrace:
    title, kind = NODES[node]
    return StepTrace(index=0, flow="main", node=node, title=title, kind=kind, summary=summary)


def _carryover(previous: AgentState | None, patient_id: str) -> Carryover | None:
    """Việc nhiều bước của lượt trước còn dang dở: bị bỏ giữa chừng (đổi việc) hoặc chuyển nhân viên vì
    hỏi lại quá nhiều. Việc đã lưu, đã hủy hay lỗi thì không nối lại. Qua các lượt việc khác thì vẫn giữ."""
    if previous is None or previous.patient_id != patient_id:
        return None
    if previous.intent in FLOW_INTENTS:
        if previous.outcome is None or previous.outcome.kind == OutcomeKind.HANDOFF:
            # Lượt trước bắt đầu lại cùng việc mà chưa nối ngữ cảnh cũ -> giữ cả lời cũ lẫn lời mới.
            same_task = previous.carryover is not None and previous.carryover.intent == previous.intent
            older = previous.carryover.sources if previous.carryover and same_task else []
            return Carryover(intent=previous.intent, sources=list(dict.fromkeys([*older, *previous.sources()])))
        return None
    return previous.carryover.model_copy(update={"just_left": False}) if previous.carryover else None


def _memory(previous: AgentState | None, patient_id: str) -> list[TurnNote]:
    """Ghi chú các lượt trước + lượt vừa kết thúc. Lượt bị bỏ dở (chưa tới respond) thì code tự ghi "chưa xong"."""
    if previous is None or previous.patient_id != patient_id:
        return []
    note = previous.turn_note or note_for_turn(
        " ".join([previous.raw_input, *previous.user_replies]), previous.intent, None, None, [], [], []
    )
    return append_note(previous.memory, note)


def _pending(snapshot: Any) -> list[Any]:
    """Giá trị interrupt đang chờ, kể cả interrupt nằm trong subgraph."""
    values = [i.value for i in getattr(snapshot, "interrupts", ()) or ()]
    if values:
        return values
    for task in getattr(snapshot, "tasks", ()) or ():
        values += [i.value for i in getattr(task, "interrupts", ()) or ()]
    return values
