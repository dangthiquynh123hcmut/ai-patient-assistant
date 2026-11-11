"""STT dùng đúng ngôn ngữ người dùng chọn (mặc định tiếng Anh), prompt cùng ngôn ngữ, "auto" để model tự nhận."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from patient_assistant.application.ports.stt import SttResult, SttWord
from patient_assistant.application.use_cases.transcribe import TranscribeUseCase
from patient_assistant.config import Settings
from patient_assistant.container import stt_prompts
from patient_assistant.domain.conversation import stt_flags_from_words
from patient_assistant.infrastructure.stt.faster_whisper import FasterWhisperSTT


class _FakeWhisper:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def transcribe(self, _audio: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        self.kwargs = kwargs
        word = SimpleNamespace(word=" four", start=0.0, end=0.5, probability=0.3)
        segment = SimpleNamespace(text=" I exercised four days", words=[word])
        return [segment], SimpleNamespace(language=kwargs["language"] or "en")


class _RecordingSTT:
    def __init__(self) -> None:
        self.languages: list[str | None] = []

    async def transcribe(self, audio: bytes, language: str | None) -> SttResult:
        self.languages.append(language)
        words = [SttWord(word="four", start=0, end=1, probability=0.3)]
        return SttResult(text="four days", words=words, language=language or "en")


def test_default_language_is_english() -> None:
    assert Settings(_env_file=None).stt_language == "en"  # type: ignore[call-arg]


@pytest.mark.parametrize(("language", "prompt_lang"), [("en", "en"), ("vi", "vi"), (None, None)])
def test_adapter_uses_prompt_in_the_same_language(language: str | None, prompt_lang: str | None) -> None:
    prompts = stt_prompts()
    stt = FasterWhisperSTT("small", initial_prompts=prompts)
    fake = _FakeWhisper()
    result = stt._run(fake, b"audio", language)
    assert fake.kwargs["language"] == language
    assert fake.kwargs["initial_prompt"] == (prompts[prompt_lang] if prompt_lang else None)
    assert result.text == "I exercised four days"


def test_english_prompt_is_english_and_keeps_doctor_names() -> None:
    prompts = stt_prompts()
    assert prompts["en"].startswith("A patient talks to")
    assert "Khoa" not in prompts["en"] and "Trần Văn An" in prompts["en"]
    assert prompts["vi"].startswith("Khoa: ")


async def test_use_case_default_choice_and_auto() -> None:
    stt = _RecordingSTT()
    use_case = TranscribeUseCase(stt, 0.6, default_language="en")
    review = await use_case.transcribe(b"x")
    assert stt.languages[-1] == "en" and review.language == "en"
    await use_case.transcribe(b"x", "vi")
    assert stt.languages[-1] == "vi"
    review = await use_case.transcribe(b"x", "auto")
    assert stt.languages[-1] is None  # model tự nhận ngôn ngữ
    with pytest.raises(ValueError):
        await use_case.transcribe(b"x", "fr")


def test_english_key_words_are_flagged_when_uncertain() -> None:
    flags = stt_flags_from_words([("I", 0.2), ("exercised", 0.5), ("four", 0.4), ("Tuesday", 0.3), ("the", 0.1)])
    assert [f.text for f in flags] == ["exercised", "four", "Tuesday"]
