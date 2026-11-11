from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class SttWord(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class SttResult(BaseModel):
    text: str
    words: list[SttWord]
    language: str


class STTError(Exception):
    pass


STT_LANGUAGES = ("en", "vi", "auto")


class STTPort(Protocol):
    async def transcribe(self, audio: bytes, language: str | None) -> SttResult:
        """`language`: mã ngôn ngữ ("en", "vi"); None = để model tự nhận ngôn ngữ."""
        ...
