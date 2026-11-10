"""Adapter LLMPort gọi Claude trên Amazon Bedrock qua Anthropic SDK (xác thực IAM).

Hai endpoint, chọn theo dạng model ID:
- Mantle (Messages API): `anthropic.claude-opus-5-5`.
- bedrock-runtime (InvokeModel): ID có hậu tố phiên bản hoặc tiền tố inference profile,
  ví dụ `global.anthropic.claude-opus-4-6-v1`, `anthropic.claude-haiku-4-5-20251001-v1:0`.

Ghi chú: không gửi temperature/top_p; không dùng tool_choice any/tool; effort `xhigh` chỉ có từ Opus 4.7;
cache_control đặt trên từng block (bedrock-runtime với Opus 4.6 trở về trước từ chối cache_control cấp
request); luôn kiểm tra stop_reason trước khi đọc nội dung.

SDK mặc định gửi API version `2026-10-21`; trên bedrock-runtime version này chỉ nhận Claude 4.7 trở lên,
nên đường bedrock-runtime ghim `anthropic_version=bedrock-2023-05-31` (mọi model đều nhận). Code này không gửi
temperature/top_p/top_k nên hai version cho cùng hành vi.
"""

from __future__ import annotations

import asyncio
import json
import re
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


_RUNTIME_API_VERSION = "bedrock-2023-05-31"
_RUNTIME_MODEL_ID = re.compile(r"^(global|us|eu|apac|jp|au|ca)\.|-v\d+(:\d+)?$")


def uses_bedrock_runtime(model_id: str) -> bool:
    """ID dạng inference profile / có phiên bản chỉ chạy trên bedrock-runtime, không chạy trên Mantle."""
    return bool(_RUNTIME_MODEL_ID.search(model_id))


def _with_memory(text: str, memory: str | None) -> str:
    """Ghi chú lượt trước (code soạn) đặt trong khối riêng, tách khỏi lời người dùng hiện tại."""
    return f"<lich_su_hoi_thoai>\n{memory}\n</lich_su_hoi_thoai>\n\n{text}" if memory else text


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
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._client = client_factory()
        self._model = model_id
        self._extra_body = extra_body
        self._effort_extract = effort_extract
        self._effort_light = effort_light
        self._max_tokens = max_tokens

    @classmethod
    def from_region(
        cls, aws_region: str | None, model_id: str, aws_profile: str | None = None, **kwargs: Any
    ) -> AnthropicBedrockLLM:
        """Xác thực bằng IAM qua chuỗi credential chuẩn của AWS (profile, SSO, assumed role, biến môi trường).

        Quyền IAM cần có: `bedrock:InvokeModel` (bedrock-runtime) hoặc `bedrock-mantle:CreateInference` (Mantle)
        trên model được phép; model phải được bật trong Amazon Bedrock > Model access.
        """
        if uses_bedrock_runtime(model_id):
            client_cls: Any = anthropic.AnthropicBedrock
            kwargs.setdefault("extra_body", {"anthropic_version": _RUNTIME_API_VERSION})
        else:
            client_cls = anthropic.AnthropicBedrockMantle
        return cls(lambda: client_cls(aws_region=aws_region, aws_profile=aws_profile), model_id, **kwargs)

    # ------------------------------------------------------------------ hạ tầng gọi API

    async def _create(self, **params: Any) -> Any:
        try:
            response = await asyncio.to_thread(
                self._client.messages.create,
                model=self._model,
                max_tokens=self._max_tokens,
                extra_body=self._extra_body,
                **params,
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

    async def ping(self) -> tuple[str, int, int]:
        """Gọi thử tối thiểu để kiểm tra credential/quyền/model. Trả (text, input_tokens, output_tokens)."""
        response = await self._create(
            messages=[{"role": "user", "content": "Trả lời đúng một từ: ok"}], output_config={"effort": "low"}
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text, response.usage.input_tokens, response.usage.output_tokens

    # ------------------------------------------------------------------ LLMPort

    async def converse(self, text: str, context: str | None = None, memory: str | None = None) -> str:
        content = f"<boi_canh>\n{context}\n</boi_canh>\n\n{text}" if context else text
        content = _with_memory(content, memory)
        response = await self._create(
            system=[{"type": "text", "text": _prompt("converse"), "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self._effort_light},
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    async def classify_intent(self, text: str, memory: str | None = None) -> list[Intent]:
        raw = await self._structured(
            _prompt("classify_intent"), _with_memory(text, memory), strict_schema(_IntentList), self._effort_light
        )
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

    async def answer_question(
        self, question: str, tools: list[ToolSpec], max_tool_calls: int, memory: str | None = None
    ) -> GroundedAnswer:
        by_name = {t.name: t for t in tools}
        tool_defs = [
            {"name": t.name, "description": t.description, "input_schema": strict_schema(t.input_model), "strict": True}
            for t in tools
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": _with_memory(question, memory)}]
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
