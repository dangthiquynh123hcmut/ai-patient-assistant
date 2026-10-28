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


class STTPort(Protocol):
    async def transcribe(self, audio: bytes, language: str = "vi") -> SttResult: ...
