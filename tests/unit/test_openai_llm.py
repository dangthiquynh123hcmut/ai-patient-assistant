"""Adapter OpenAI: structured output, vòng tool-use (dừng ở tool `terminal`, lỗi trả lại cho model), chuẩn hóa lỗi."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace as NS
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from patient_assistant.application.ports.llm import AgentMessage, LLMError, ToolSpecification
from patient_assistant.domain.conversation import Intent
from patient_assistant.infrastructure.llm.openai_chat import OpenAIChatLLM, supports_reasoning_effort


def _response(
    content: str | None = None, *tool_calls: NS, finish_reason: str = "stop", refusal: str | None = None
) -> NS:
    message = NS(content=content, tool_calls=list(tool_calls) or None, refusal=refusal)
    return NS(
        choices=[NS(message=message, finish_reason=finish_reason)],
        usage=NS(prompt_tokens=3, completion_tokens=1),
    )


def _call(id_: str, name: str, **arguments: Any) -> NS:
    return NS(id=id_, function=NS(name=name, arguments=json.dumps(arguments)))


class _Completions:
    def __init__(self, responses: list[NS]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(self, **parameters: Any) -> NS:
        self.requests.append(copy.deepcopy(parameters))
        return self.responses.pop(0)


def _llm(responses: list[NS], model: str = "gpt-5-mini") -> tuple[OpenAIChatLLM, _Completions]:
    completions = _Completions(responses)
    return OpenAIChatLLM(NS(chat=NS(completions=completions)), model), completions


class _Pick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_id: str


async def test_structured_output_uses_strict_json_schema_and_reasoning_effort() -> None:
    llm, completions = _llm([_response(json.dumps({"intents": [Intent.BOOK_APPOINTMENT.value]}))])
    assert await llm.classify_intent("muốn đặt lịch") == [Intent.BOOK_APPOINTMENT]
    request = completions.requests[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["reasoning_effort"] == "low"
    assert request["messages"][0]["role"] == "system"


async def test_non_reasoning_model_gets_no_reasoning_effort() -> None:
    llm, completions = _llm([_response("ok")], model="gpt-4o-mini")
    assert (await llm.ping())[0] == "ok"
    assert "reasoning_effort" not in completions.requests[0]


async def test_terminal_tool_stops_the_loop_and_errors_go_back_to_the_model() -> None:
    proposed: list[str] = []

    async def propose(input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, _Pick)
        if input_data.slot_id != "S1":
            return {"status": "error", "error": {"message": "slot lạ"}}
        proposed.append(input_data.slot_id)
        return {"status": "ok"}

    llm, completions = _llm(
        [
            _response(None, _call("t1", "propose_booking", slot_id="BIA"), finish_reason="tool_calls"),
            _response("Mình đề xuất S1.", _call("t2", "propose_booking", slot_id="S1"), finish_reason="tool_calls"),
        ]
    )
    reply = await llm.run_agent(
        "booking",
        "Today: Wednesday",
        [AgentMessage(role="user", text="đặt lịch")],
        [ToolSpecification("propose_booking", "đề xuất", _Pick, propose, terminal=True)],
        max_tool_calls=5,
    )
    assert reply.stopped_by == "propose_booking"
    assert reply.text == "Mình đề xuất S1."
    assert proposed == ["S1"]
    assert len(completions.requests) == 2
    second = completions.requests[1]
    assert second["messages"][-2]["tool_calls"][0]["id"] == "t1"
    assert second["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": json.dumps({"status": "error", "error": {"message": "slot lạ"}}, ensure_ascii=False),
    }
    assert second["tools"][0]["function"]["strict"] is True
    assert second["parallel_tool_calls"] is False
    assert "<context>\nToday: Wednesday\n</context>" in completions.requests[0]["messages"][0]["content"]


async def test_tool_budget_exhausted_forces_text_answer() -> None:
    async def lookup(_: BaseModel) -> dict[str, Any]:
        return {"status": "ok", "source_ids": ["doc-1"]}

    llm, completions = _llm(
        [
            _response(None, _call("t1", "lookup", slot_id="x"), finish_reason="tool_calls"),
            _response("Trả lời."),
        ]
    )
    answer = await llm.answer_question("hỏi", [ToolSpecification("lookup", "tra", _Pick, lookup)], max_tool_calls=1)
    assert answer.text == "Trả lời." and answer.source_ids == ["doc-1"] and answer.tool_calls == 1
    assert completions.requests[1]["tool_choice"] == "none"


@pytest.mark.parametrize(
    ("response", "code"),
    [(_response("x", finish_reason="length"), "max_tokens"), (_response(None, refusal="no"), "refusal")],
)
async def test_finish_reasons_are_normalized(response: NS, code: str) -> None:
    llm, _ = _llm([response])
    with pytest.raises(LLMError) as error:
        await llm.converse("chào")
    assert error.value.code == code


def test_reasoning_model_detection() -> None:
    assert supports_reasoning_effort("gpt-5-mini")
    assert supports_reasoning_effort("o4-mini")
    assert not supports_reasoning_effort("gpt-5-chat-latest")
    assert not supports_reasoning_effort("gpt-4.1-mini")
