"""Adapter STTPort dùng faster-whisper chạy local trên CPU (FR-01.6). Model đổi qua cấu hình."""

from __future__ import annotations

import asyncio
import io
from typing import Any

from patient_assistant.application.ports.stt import STTError, SttResult, SttWord


class FasterWhisperSTT:
    def __init__(
        self, model: str, device: str = "cpu", compute_type: str = "int8", initial_prompt: str | None = None
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._initial_prompt = initial_prompt  # danh mục khoa/bác sĩ để nhận đúng tên riêng
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _load(self) -> Any:
        async with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel  # import muộn: chỉ cần khi bật STT

                self._model = await asyncio.to_thread(
                    WhisperModel, self._model_name, device=self._device, compute_type=self._compute_type
                )
        return self._model

    def _run(self, model: Any, audio: bytes, language: str) -> SttResult:
        segments, info = model.transcribe(
            io.BytesIO(audio),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=self._initial_prompt,
        )
        words: list[SttWord] = []
        texts: list[str] = []
        for segment in segments:
            texts.append(segment.text.strip())
            words += [
                SttWord(word=w.word.strip(), start=w.start, end=w.end, probability=w.probability)
                for w in segment.words or []
            ]
        return SttResult(text=" ".join(texts).strip(), words=words, language=info.language)

    async def transcribe(self, audio: bytes, language: str = "vi") -> SttResult:
        try:
            model = await self._load()
            return await asyncio.to_thread(self._run, model, audio, language)
        except Exception as exc:  # noqa: BLE001 - chuẩn hóa mọi lỗi hạ tầng STT
            raise STTError(f"STT failed: {type(exc).__name__}") from exc


class UnavailableSTT:
    """Dùng khi chưa cài faster-whisper: endpoint STT trả lỗi rõ ràng thay vì crash lúc khởi động."""

    async def transcribe(self, audio: bytes, language: str = "vi") -> SttResult:
        raise STTError("STT chưa được cài đặt (pip install -e '.[stt]')")
