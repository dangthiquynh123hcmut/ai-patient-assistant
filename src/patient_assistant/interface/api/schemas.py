"""DTO của HTTP API — tách khỏi state nội bộ để đổi API mà không đụng agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from patient_assistant.domain.common import SttFlag


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    input_source: Literal["text", "voice"] = "text"
    stt_flags: list[SttFlag] = Field(
        default_factory=list, description="Cờ từ /stt/transcribe, sau khi người dùng duyệt"
    )


class ResumeIn(BaseModel):
    action: Literal["reply", "confirm", "edit", "cancel", "choose", "other_time"]
    text: str | None = Field(None, max_length=2000)
    fields: dict[str, Any] = Field(default_factory=dict)
    option_no: int | None = None
