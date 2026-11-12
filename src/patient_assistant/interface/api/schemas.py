"""DTO của HTTP API — tách khỏi state nội bộ để đổi API mà không đụng agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from patient_assistant.domain.account import MIN_PASSWORD_LENGTH, User
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


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenOut(BaseModel):
    """Trả về khi đăng nhập / đăng ký / refresh. Refresh token KHÔNG nằm ở đây mà ở cookie HttpOnly."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int  # giây tới khi access token hết hạn
    user: User


class UserUpdateIn(BaseModel):
    active: bool | None = None
    password: str | None = Field(None, min_length=MIN_PASSWORD_LENGTH, max_length=128)
