"""Use case STT: audio -> transcript + cờ độ tin cậy. Chạy ngoài graph để người dùng duyệt trước (FR-01.3)."""

from __future__ import annotations

from pydantic import BaseModel

from patient_assistant.application.ports.stt import STT_LANGUAGES, STTPort, SttWord
from patient_assistant.domain.common import SttFlag
from patient_assistant.domain.conversation import stt_flags_from_words


class TranscriptReview(BaseModel):
    text: str
    words: list[SttWord]
    flags: list[SttFlag]
    language: str  # ngôn ngữ đã dùng/nhận ra — hiển thị để người dùng biết model hiểu là tiếng gì


class TranscribeUseCase:
    def __init__(self, stt: STTPort, confidence_threshold: float, default_language: str = "en") -> None:
        self._stt = stt
        self._threshold = confidence_threshold
        self._default_language = default_language

    async def transcribe(self, audio: bytes, language: str | None = None) -> TranscriptReview:
        """`language`: "en" | "vi" | "auto"; None = mặc định trong cấu hình."""
        chosen = language or self._default_language
        if chosen not in STT_LANGUAGES:
            raise ValueError(f"unsupported STT language: {chosen}")
        result = await self._stt.transcribe(audio, None if chosen == "auto" else chosen)
        flags = stt_flags_from_words([(w.word, w.probability) for w in result.words], self._threshold)
        return TranscriptReview(text=result.text, words=result.words, flags=flags, language=result.language)
