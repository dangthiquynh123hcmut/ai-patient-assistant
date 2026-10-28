"""Use case STT: audio -> transcript + cờ độ tin cậy. Chạy ngoài graph để người dùng duyệt trước (FR-01.3)."""

from __future__ import annotations

from pydantic import BaseModel

from patient_assistant.application.ports.stt import STTPort, SttWord
from patient_assistant.domain.common import SttFlag
from patient_assistant.domain.conversation import stt_flags_from_words


class TranscriptReview(BaseModel):
    text: str
    words: list[SttWord]
    flags: list[SttFlag]


class TranscribeUseCase:
    def __init__(self, stt: STTPort, confidence_threshold: float) -> None:
        self._stt = stt
        self._threshold = confidence_threshold

    async def transcribe(self, audio: bytes, language: str = "vi") -> TranscriptReview:
        result = await self._stt.transcribe(audio, language)
        flags = stt_flags_from_words([(w.word, w.probability) for w in result.words], self._threshold)
        return TranscriptReview(text=result.text, words=result.words, flags=flags)
