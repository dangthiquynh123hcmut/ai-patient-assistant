"""Vòng tool-use của adapter Bedrock cho agent: dừng ngay ở tool `terminal`, lỗi tool trả lại cho LLM sửa."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict

from patient_assistant.application.ports.llm import AgentMessage, ToolSpecification
from patient_assistant.infrastructure.llm.bedrock import AnthropicBedrockLLM, _agent_messages


class _Block:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)
        self._fields = fields

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return dict(self._fields)


class _Response:
    def __init__(self, stop_reason: str, *blocks: _Block) -> None:
        self.stop_reason = stop_reason
        self.content = list(blocks)


def _tool_use(id_: str, name: str, **arguments: Any) -> _Block:
    return _Block(type="tool_use", id=id_, name=name, input=arguments)


def _text(text: str) -> _Block:
    return _Block(type="text", text=text)


class _Messages:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def create(self, **parameters: Any) -> _Response:
        self.requests.append(copy.deepcopy(parameters))  # vòng lặp còn append vào list messages sau lời gọi này
        return self.responses.pop(0)


class _Client:
    def __init__(self, responses: list[_Response]) -> None:
        self.messages = _Messages(responses)


class _Pick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_id: str


def _llm(responses: list[_Response]) -> tuple[AnthropicBedrockLLM, _Messages]:
    client = _Client(responses)
    return AnthropicBedrockLLM(lambda: client, "anthropic.claude-test"), client.messages


async def test_terminal_tool_stops_the_loop_and_errors_go_back_to_the_model() -> None:
    proposed: list[str] = []

    async def propose(input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, _Pick)
        if input_data.slot_id != "S1":
            return {"tool": "propose_booking", "status": "error", "error": {"message": "slot lạ"}}
        proposed.append(input_data.slot_id)
        return {"tool": "propose_booking", "status": "ok"}

    llm, messages = _llm(
        [
            _Response("tool_use", _tool_use("t1", "propose_booking", slot_id="BIA")),
            _Response("tool_use", _text("Mình đề xuất S1."), _tool_use("t2", "propose_booking", slot_id="S1")),
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
    assert len(messages.requests) == 2  # dừng ngay sau tool terminal, không gọi LLM thêm
    retry = messages.requests[1]["messages"][-1]["content"][0]
    assert retry["type"] == "tool_result" and "slot lạ" in retry["content"]
    assert "<context>\nToday: Wednesday\n</context>" in messages.requests[0]["system"][0]["text"]


async def test_terminal_tool_error_with_stop_ends_the_loop() -> None:
    async def refuse(_: BaseModel) -> dict[str, Any]:
        return {"tool": "propose_report", "status": "error", "error": {"message": "quá số lần"}, "stop": True}

    llm, messages = _llm([_Response("tool_use", _tool_use("t1", "propose_report", slot_id="x"))])
    reply = await llm.run_agent(
        "assistant",
        "",
        [AgentMessage(role="user", text="báo cáo")],
        [ToolSpecification("propose_report", "đề xuất", _Pick, refuse, terminal=True)],
        max_tool_calls=5,
    )
    assert reply.stopped_by == "propose_report"
    assert len(messages.requests) == 1  # không đưa lỗi lại cho LLM thử tiếp


def test_agent_history_starts_with_user_and_alternates() -> None:
    history = [
        AgentMessage(role="assistant", text="(bị cắt từ lượt cũ)"),
        AgentMessage(role="user", text="đặt lịch"),
        AgentMessage(role="assistant", text="Khoa nào?"),
        AgentMessage(role="assistant", text="Đã đặt xong."),
        AgentMessage(role="user", text="  "),
        AgentMessage(role="user", text="cảm ơn"),
    ]
    assert _agent_messages(history) == [
        {"role": "user", "content": "đặt lịch"},
        {"role": "assistant", "content": "Khoa nào?\n\nĐã đặt xong."},
        {"role": "user", "content": "cảm ơn"},
    ]
