"""Use case hội thoại: bọc graph, ẩn chi tiết LangGraph (config, Command, interrupt) khỏi lớp API."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel

from patient_assistant.application.agent.state import (
    AgentResponse,
    AgentState,
    InterruptPayload,
    UserDecision,
)
from patient_assistant.application.ports.clock import Clock
from patient_assistant.domain.common import SttFlag


class TurnResult(BaseModel):
    thread_id: str
    status: Literal["waiting_user", "done"]
    interrupt: InterruptPayload | None = None
    response: AgentResponse | None = None


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
        await self._graph.ainvoke(fresh.model_dump(), self._config(thread_id))
        return await self._result(thread_id)

    async def resume(self, thread_id: str, patient_id: str, decision: UserDecision) -> TurnResult:
        config = self._config(thread_id)
        snapshot = await self._graph.aget_state(config)
        if not snapshot or not _pending(snapshot) or snapshot.values.get("patient_id") != patient_id:
            raise NoPendingQuestion(thread_id)
        await self._graph.ainvoke(Command(resume=decision.model_dump(mode="json")), config)
        return await self._result(thread_id)

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

    async def _result(self, thread_id: str) -> TurnResult:
        snapshot = await self._graph.aget_state(self._config(thread_id))
        pending = _pending(snapshot)
        if pending:
            return TurnResult(
                thread_id=thread_id, status="waiting_user", interrupt=InterruptPayload.model_validate(pending[0])
            )
        response = snapshot.values.get("response")
        if isinstance(response, dict):
            response = AgentResponse.model_validate(response)
        return TurnResult(thread_id=thread_id, status="done", response=response)


def _pending(snapshot: Any) -> list[Any]:
    """Giá trị interrupt đang chờ, kể cả interrupt nằm trong subgraph."""
    values = [i.value for i in getattr(snapshot, "interrupts", ()) or ()]
    if values:
        return values
    for task in getattr(snapshot, "tasks", ()) or ():
        values += [i.value for i in getattr(task, "interrupts", ()) or ()]
    return values
