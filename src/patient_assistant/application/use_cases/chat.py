"""Use case hội thoại: bọc graph, ẩn chi tiết LangGraph (config, Command, interrupt) khỏi lớp API."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel, Field

from patient_assistant.application.agent.state import (
    AgentResponse,
    AgentState,
    InterruptPayload,
    UserDecision,
)
from patient_assistant.application.agent.trace import StepTrace, describe_interrupt, describe_step
from patient_assistant.application.ports.clock import Clock
from patient_assistant.domain.common import SttFlag


class TurnResult(BaseModel):
    thread_id: str
    status: Literal["waiting_user", "done"]
    interrupt: InterruptPayload | None = None
    response: AgentResponse | None = None
    steps: list[StepTrace] = Field(default_factory=list)  # các node đã chạy trong lượt này, kèm diễn giải


SUBGRAPH_NODES = ("report_flow", "booking_flow")


class NoPendingQuestion(Exception):
    """Gọi resume khi thread không có câu hỏi nào đang chờ."""


class ChatUseCase:
    def __init__(self, graph: Any, clock: Clock, recursion_limit: int) -> None:
        self._graph = graph
        self._clock = clock
        self._recursion_limit = recursion_limit

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self._recursion_limit}

    async def send_message(
        self,
        thread_id: str,
        patient_id: str,
        text: str,
        input_source: Literal["text", "voice"] = "text",
        stt_flags: list[SttFlag] | None = None,
    ) -> TurnResult:
        """Bắt đầu một lượt mới. Toàn bộ state của lượt được đặt lại (mỗi lượt một intent chính)."""
        fresh = AgentState(
            thread_id=thread_id,
            patient_id=patient_id,
            now=self._clock.now(),
            input_source=input_source,
            raw_input=text,
            stt_flags=stt_flags or [],
        )
        steps = await self._run(fresh.model_dump(), thread_id)
        return await self._result(thread_id, steps)

    async def resume(self, thread_id: str, patient_id: str, decision: UserDecision) -> TurnResult:
        config = self._config(thread_id)
        snapshot = await self._graph.aget_state(config)
        if not snapshot or not _pending(snapshot) or snapshot.values.get("patient_id") != patient_id:
            raise NoPendingQuestion(thread_id)
        steps = await self._run(Command(resume=decision.model_dump(mode="json")), thread_id)
        return await self._result(thread_id, steps)

    async def _run(self, graph_input: Any, thread_id: str) -> list[StepTrace]:
        """Chạy graph và ghi lại từng node đã chạy (kể cả trong subgraph) để giải thích quyết định."""
        steps: list[StepTrace] = []
        async for namespace, chunk in self._graph.astream(
            graph_input, self._config(thread_id), stream_mode="updates", subgraphs=True
        ):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    if not namespace:  # interrupt được phát cả ở subgraph lẫn graph cha -> chỉ lấy một lần
                        steps += [describe_interrupt(len(steps), i.value) for i in update]
                elif not (not namespace and node in SUBGRAPH_NODES):  # bỏ bản gộp state khi subgraph kết thúc
                    steps.append(describe_step(len(steps), namespace, node, update))
        return steps

    async def inspect_state(self, thread_id: str) -> AgentState | None:
        """State hiện tại của thread, kể cả khi đang dừng trong subgraph (debug, evaluation).

        Khi subgraph đang chờ interrupt, state của nó chưa gộp vào graph cha nên phải đọc từ task.
        """
        snapshot = await self._graph.aget_state(self._config(thread_id), subgraphs=True)
        if not snapshot or not snapshot.values:
            return None
        values = dict(snapshot.values)
        for task in snapshot.tasks or ():
            sub = getattr(task, "state", None)
            if sub is not None and getattr(sub, "values", None):
                values.update(sub.values)
        return AgentState.model_validate(values)

    async def _result(self, thread_id: str, steps: list[StepTrace]) -> TurnResult:
        snapshot = await self._graph.aget_state(self._config(thread_id))
        pending = _pending(snapshot)
        if pending:
            return TurnResult(
                thread_id=thread_id,
                status="waiting_user",
                interrupt=InterruptPayload.model_validate(pending[0]),
                steps=steps,
            )
        response = snapshot.values.get("response")
        if isinstance(response, dict):
            response = AgentResponse.model_validate(response)
        return TurnResult(thread_id=thread_id, status="done", response=response, steps=steps)


def _pending(snapshot: Any) -> list[Any]:
    """Giá trị interrupt đang chờ, kể cả interrupt nằm trong subgraph."""
    values = [i.value for i in getattr(snapshot, "interrupts", ()) or ()]
    if values:
        return values
    for task in getattr(snapshot, "tasks", ()) or ():
        values += [i.value for i in getattr(task, "interrupts", ()) or ()]
    return values
