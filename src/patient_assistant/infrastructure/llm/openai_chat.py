"""Adapter LLMPort gọi OpenAI (Chat Completions) bằng API key — thay cho Bedrock khi PA_LLM_PROVIDER=openai.

Dùng chung prompt, schema strict và cách vòng tool-use dừng ở tool `terminal` với adapter Bedrock.
- Structured output: `response_format` json_schema strict; tool: function calling strict, tắt parallel tool calls
  (OpenAI không bảo đảm schema strict khi gọi song song).
- `reasoning_effort` chỉ gửi cho model reasoning (gpt-5*, o1/o3/o4*) hoặc khi PA_OPENAI_REASONING=true.
- `base_url`: dùng được với dịch vụ tương thích OpenAI, miễn là hỗ trợ json_schema strict và tool calling.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import openai
from pydantic import ValidationError

from patient_assistant.application.ports.llm import (
    AgentMessage,
    AgentReply,
    BookingPrompt,
    GroundedAnswer,
    LLMError,
    ReportPrompt,
    ToolSpecification,
)
from patient_assistant.domain.booking.models import BookingRequest, SlotOption
from patient_assistant.domain.conversation import Intent
from patient_assistant.domain.report.models import ReportExtraction
from patient_assistant.infrastructure.llm.bedrock import (
    _agent_messages,
    _IntentList,
    _logged,
    _LoopResult,
    _prompt,
    _render_feedback,
    _render_sources,
    _root_cause,
    _SlotChoice,
    _with_memory,
)
from patient_assistant.infrastructure.llm.schema import strict_schema

# Effort của dự án (theo Claude) -> reasoning_effort của OpenAI.
_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def supports_reasoning_effort(model: str) -> bool:
    name = model.rsplit("/", 1)[-1]
    return name.startswith(_REASONING_PREFIXES) and "chat" not in name


class OpenAIChatLLM:
    def __init__(
        self,
        client: Any,
        model: str,
        effort_extract: str = "medium",
        effort_light: str = "low",
        effort_agent: str = "low",
        reasoning: bool | None = None,
        max_tokens: int = 8000,
    ) -> None:
        self._client = client
        self._model = model
        self._effort_extract = effort_extract
        self._effort_light = effort_light
        self._effort_agent = effort_agent
        self._reasoning = supports_reasoning_effort(model) if reasoning is None else reasoning
        self._max_tokens = max_tokens

    @classmethod
    def from_api_key(cls, api_key: str | None, model: str, base_url: str | None = None, **kwargs: Any) -> OpenAIChatLLM:
        """`api_key` trống thì SDK tự đọc biến môi trường OPENAI_API_KEY."""
        return cls(openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120), model, **kwargs)

    # ------------------------------------------------------------------ hạ tầng gọi API

    async def _create(self, effort: str, **parameters: Any) -> Any:
        if self._reasoning:
            parameters["reasoning_effort"] = _EFFORT.get(effort, "medium")
        try:
            response = await self._client.chat.completions.create(
                model=self._model, max_completion_tokens=self._max_tokens, **parameters
            )
        except openai.RateLimitError as exception:
            raise _logged(LLMError("unavailable", f"rate limited: {_root_cause(exception)}", retryable=True)) from (
                exception
            )
        except openai.APIConnectionError as exception:
            raise _logged(
                LLMError("unavailable", f"connection error: {_root_cause(exception)}", retryable=True)
            ) from exception
        except openai.APIStatusError as exception:
            retryable = exception.status_code >= 500
            raise _logged(
                LLMError(
                    "unavailable" if retryable else "bad_request",
                    f"HTTP {exception.status_code}: {_root_cause(exception)}",
                    retryable=retryable,
                )
            ) from exception
        choice = response.choices[0]
        if choice.finish_reason == "content_filter" or getattr(choice.message, "refusal", None):
            raise _logged(LLMError("refusal", "model declined the request"))
        if choice.finish_reason == "length":
            raise _logged(LLMError("max_tokens", f"output truncated at max_completion_tokens={self._max_tokens}"))
        return response

    @staticmethod
    def _text(response: Any) -> str:
        return (response.choices[0].message.content or "").strip()

    async def _structured(self, system: str, user: str, schema: dict[str, Any], effort: str) -> dict[str, Any]:
        response = await self._create(
            effort,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True},
            },
        )
        try:
            value = json.loads(self._text(response))
        except json.JSONDecodeError as exception:
            raise _logged(
                LLMError("invalid_output", f"structured output is not valid JSON: {exception}")
            ) from exception
        if not isinstance(value, dict):
            raise _logged(LLMError("invalid_output", f"structured output is not an object: {type(value).__name__}"))
        return value

    async def ping(self) -> tuple[str, int, int]:
        """Gọi thử tối thiểu để kiểm tra API key/model. Trả (text, input_tokens, output_tokens)."""
        response = await self._create("low", messages=[{"role": "user", "content": "Reply with exactly one word: ok"}])
        usage = response.usage
        return self._text(response), getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

    # ------------------------------------------------------------------ LLMPort

    async def converse(self, text: str, context: str | None = None, memory: str | None = None) -> str:
        content = f"<context>\n{context}\n</context>\n\n{text}" if context else text
        response = await self._create(
            self._effort_light,
            messages=[
                {"role": "system", "content": _prompt("converse")},
                {"role": "user", "content": _with_memory(content, memory)},
            ],
        )
        return self._text(response)

    async def classify_intent(self, text: str, memory: str | None = None) -> list[Intent]:
        raw = await self._structured(
            _prompt("classify_intent"), _with_memory(text, memory), strict_schema(_IntentList), self._effort_light
        )
        try:
            return _IntentList.model_validate(raw).intents
        except ValidationError as exception:
            raise _logged(LLMError("invalid_output", f"invalid intent list: {exception}")) from exception

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        user = _render_sources(prompt.sources, prompt.now) + _render_feedback(prompt.feedback)
        return await self._structured(
            _prompt("extract_report"), user, strict_schema(ReportExtraction), self._effort_extract
        )

    async def extract_booking(self, prompt: BookingPrompt) -> dict[str, Any]:
        catalog = "\n".join(
            f"- {department.code}: {department.name} (also called: {', '.join(department.aliases) or '—'})"
            for department in prompt.departments
        )
        user = (
            _render_sources(prompt.sources, prompt.now)
            + f"\n\nDepartment catalog:\n{catalog}"
            + _render_feedback(prompt.feedback)
        )
        codes = [department.code for department in prompt.departments]
        schema = strict_schema(
            BookingRequest,
            overrides={"department_code": {"anyOf": [{"type": "string", "enum": codes}, {"type": "null"}]}},
        )
        return await self._structured(_prompt("extract_booking"), user, schema, self._effort_extract)

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        listing = "\n".join(f"{option.option_no}. {option.slot.label()}" for option in options)
        raw = await self._structured(
            _prompt("choose_slot"),
            f"Options:\n{listing}\n\nUser's reply: {reply}",
            strict_schema(_SlotChoice),
            self._effort_light,
        )
        return _SlotChoice.model_validate(raw).option_no

    async def answer_question(
        self, question: str, tools: list[ToolSpecification], max_tool_calls: int, memory: str | None = None
    ) -> GroundedAnswer:
        loop = await self._tool_loop(
            _prompt("answer_question"),
            [{"role": "user", "content": _with_memory(question, memory)}],
            tools,
            max_tool_calls,
            self._effort_light,
        )
        return GroundedAnswer(text=loop.text, source_ids=loop.source_ids, tool_calls=loop.calls)

    async def run_agent(
        self,
        task: Literal["booking", "assistant"],
        context: str,
        history: list[AgentMessage],
        tools: list[ToolSpecification],
        max_tool_calls: int,
    ) -> AgentReply:
        system = _prompt(f"{task}_agent") + f"\n\n<context>\n{context}\n</context>"
        loop = await self._tool_loop(system, _agent_messages(history), tools, max_tool_calls, self._effort_agent)
        return AgentReply(text=loop.text, tool_calls=loop.calls, stopped_by=loop.stopped_by)

    async def _tool_loop(
        self,
        system_text: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpecification],
        max_tool_calls: int,
        effort: str,
    ) -> _LoopResult:
        by_name = {tool.name: tool for tool in tools}
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": strict_schema(tool.input_model),
                    "strict": True,
                },
            }
            for tool in tools
        ]
        conversation: list[dict[str, Any]] = [{"role": "system", "content": system_text}, *messages]
        source_ids: list[str] = []
        calls = 0
        while True:
            parameters: dict[str, Any] = {"messages": conversation}
            if tool_defs:
                parameters |= {
                    "tools": tool_defs,
                    "tool_choice": "auto" if calls < max_tool_calls else "none",
                    "parallel_tool_calls": False,
                }
            response = await self._create(effort, **parameters)
            message = response.choices[0].message
            tool_calls = list(message.tool_calls or [])
            text = (message.content or "").strip()
            if not tool_calls:
                return _LoopResult(text=text, source_ids=source_ids, calls=calls)
            conversation.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments},
                        }
                        for call in tool_calls
                    ],
                }
            )
            stopped_by: str | None = None
            for call in tool_calls:
                calls += 1
                specification = by_name.get(call.function.name)
                try:
                    if specification is None:
                        raise ValueError(f"unknown tool {call.function.name}")
                    arguments = json.loads(call.function.arguments or "{}")
                    result = await specification.handler(specification.input_model.model_validate(arguments))
                    source_ids += [
                        source_id for source_id in result.get("source_ids", []) if source_id not in source_ids
                    ]
                    if specification.terminal and (result.get("status") == "ok" or result.get("stop")):
                        stopped_by = specification.name
                    content = json.dumps(result, ensure_ascii=False)
                except (ValueError, ValidationError) as exception:  # JSONDecodeError cũng là ValueError
                    content = f"Invalid arguments: {exception}"
                # Mỗi tool_call phải có đúng một tin nhắn `tool` trả lời, kể cả khi lỗi.
                conversation.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if stopped_by:
                return _LoopResult(text=text, source_ids=source_ids, calls=calls, stopped_by=stopped_by)
