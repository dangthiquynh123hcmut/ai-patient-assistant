"""Adapter LLMPort gọi Claude trên Amazon Bedrock qua Anthropic SDK (client Bedrock Mantle, xác thực IAM).

Ghi chú cho Claude Opus 5.5: không gửi temperature/top_p; không tắt được thinking — điều khiển bằng
`output_config.effort`; không dùng tool_choice any/tool; luôn kiểm tra stop_reason trước khi đọc nội dung.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from importlib import resources
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from patient_assistant.application.ports.llm import (
    BookingPrompt,
    CorrectionFeedback,
    GroundedAnswer,
    LLMError,
    ReportPrompt,
    ToolSpec,
)
from patient_assistant.domain.booking.models import BookingRequest, SlotOption
from patient_assistant.domain.conversation import Intent
from patient_assistant.domain.report.models import ReportExtraction
from patient_assistant.domain.text.vi_dates import WEEKDAY_NAMES
from patient_assistant.infrastructure.llm.schema import strict_schema


def _prompt(name: str) -> str:
    return resources.files("patient_assistant.infrastructure.llm").joinpath(f"prompts/{name}.md").read_text("utf-8")


def _render_sources(sources: list[str], now: datetime) -> str:
    lines = [f"Hôm nay: {WEEKDAY_NAMES[now.weekday()]}, {now:%d/%m/%Y}.", "", "Lời người dùng (theo thứ tự thời gian):"]
    lines += [f"[{i}] {text}" for i, text in enumerate(sources, start=1)]
    return "\n".join(lines)


def _render_feedback(feedback: CorrectionFeedback | None) -> str:
    if feedback is None:
        return ""
    issues = "\n".join(
        f"- {i.field}: {i.message}" + (f" Gợi ý: {i.suggestion}" if i.suggestion else "") for i in feedback.issues
    )
    return (
        "\n\nĐầu ra trước của bạn bị hệ thống kiểm tra từ chối:\n"
        f"{json.dumps(feedback.previous, ensure_ascii=False)}\n\nLỗi cần sửa:\n{issues}\n\n"
        "Hãy sửa đúng các lỗi trên. Không bịa thêm dữ liệu; nếu không có bằng chứng thì để null."
    )


class _IntentList(BaseModel):
    intents: list[Intent]


class _SlotChoice(BaseModel):
    option_no: int | None


class AnthropicBedrockLLM:
    def __init__(
        self,
        client_factory: Callable[[], Any],
        model_id: str,
        effort_extract: str = "medium",
        effort_light: str = "low",
        max_tokens: int = 8000,
    ) -> None:
        self._client = client_factory()
        self._model = model_id
        self._effort_extract = effort_extract
        self._effort_light = effort_light
        self._max_tokens = max_tokens

    @classmethod
    def from_region(cls, aws_region: str, model_id: str, **kwargs: Any) -> AnthropicBedrockLLM:
        return cls(lambda: anthropic.AnthropicBedrockMantle(aws_region=aws_region), model_id, **kwargs)

    # ------------------------------------------------------------------ hạ tầng gọi API

    async def _create(self, **params: Any) -> Any:
        try:
            response = await asyncio.to_thread(
                self._client.messages.create, model=self._model, max_tokens=self._max_tokens, **params
            )
        except anthropic.RateLimitError as exc:
            raise LLMError("unavailable", "rate limited", retryable=True) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("unavailable", "connection error", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code >= 500
            raise LLMError("unavailable" if retryable else "bad_request", str(exc), retryable=retryable) from exc
        if response.stop_reason == "refusal":
            raise LLMError("refusal", "model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError("max_tokens", "output truncated")
        return response

    async def _structured(self, system: str, user: str, schema: dict[str, Any], effort: str) -> dict[str, Any]:
        response = await self._create(
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("invalid_output", "structured output is not valid JSON") from exc
        if not isinstance(value, dict):
            raise LLMError("invalid_output", "structured output is not an object")
        return value

    # ------------------------------------------------------------------ LLMPort

    async def classify_intent(self, text: str) -> list[Intent]:
        raw = await self._structured(_prompt("classify_intent"), text, strict_schema(_IntentList), self._effort_light)
        try:
            return _IntentList.model_validate(raw).intents
        except ValidationError as exc:
            raise LLMError("invalid_output", "invalid intent list") from exc

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        user = _render_sources(prompt.sources, prompt.now) + _render_feedback(prompt.feedback)
        return await self._structured(
            _prompt("extract_report"), user, strict_schema(ReportExtraction), self._effort_extract
        )

    async def extract_booking(self, prompt: BookingPrompt) -> dict[str, Any]:
        catalog = "\n".join(
            f"- {d.code}: {d.name} (còn gọi: {', '.join(d.aliases) or '—'})" for d in prompt.departments
        )
        user = (
            _render_sources(prompt.sources, prompt.now)
            + f"\n\nDanh mục khoa:\n{catalog}"
            + _render_feedback(prompt.feedback)
        )
        codes = [d.code for d in prompt.departments]
        schema = strict_schema(
            BookingRequest,
            overrides={"department_code": {"anyOf": [{"type": "string", "enum": codes}, {"type": "null"}]}},
        )
        return await self._structured(_prompt("extract_booking"), user, schema, self._effort_extract)

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        listing = "\n".join(f"{o.option_no}. {o.slot.label()}" for o in options)
        raw = await self._structured(
            _prompt("choose_slot"),
            f"Danh sách:\n{listing}\n\nNgười dùng trả lời: {reply}",
            strict_schema(_SlotChoice),
            self._effort_light,
        )
        return _SlotChoice.model_validate(raw).option_no

    async def answer_question(self, question: str, tools: list[ToolSpec], max_tool_calls: int) -> GroundedAnswer:
        by_name = {t.name: t for t in tools}
        tool_defs = [
            {"name": t.name, "description": t.description, "input_schema": strict_schema(t.input_model), "strict": True}
            for t in tools
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        source_ids: list[str] = []
        calls = 0
        system = [{"type": "text", "text": _prompt("answer_question"), "cache_control": {"type": "ephemeral"}}]
        while True:
            choice = {"type": "auto"} if calls < max_tool_calls else {"type": "none"}
            response = await self._create(
                system=system,
                messages=messages,
                tools=tool_defs,
                tool_choice=choice,
                output_config={"effort": self._effort_light},
            )
            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                return GroundedAnswer(text=text, source_ids=source_ids, tool_calls=calls)
            # Giữ nguyên toàn bộ content (kể cả thinking) khi gửi lại.
            messages.append(
                {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in response.content]}
            )
            results = []
            for block in (b for b in response.content if b.type == "tool_use"):
                calls += 1
                spec = by_name.get(block.name)
                try:
                    if spec is None:
                        raise ValueError(f"unknown tool {block.name}")
                    result = await spec.handler(spec.input_model.model_validate(block.input))
                    source_ids += [s for s in result.get("source_ids", []) if s not in source_ids]
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                except (ValueError, ValidationError) as exc:
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": f"Tham số không hợp lệ: {exc}",
                        }
                    )
            messages.append({"role": "user", "content": results})
