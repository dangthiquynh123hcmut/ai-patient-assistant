"""Nhận diện cách người dùng mô tả triệu chứng và việc tập — dùng chung cho validator và adapter offline."""

from __future__ import annotations

import re
from typing import Literal

from patient_assistant.domain.text.normalize import contains_phrase, normalize
from patient_assistant.domain.text.vi_numbers import ZERO_PHRASES, digitize

Direction = Literal["better", "worse", "same"]

# "đỡ đau hơn" chứa "đau hơn" -> phải xét cụm "đỡ" trước rồi bỏ đi, kẻo hiểu thành xấu đi.
BETTER_PHRASES = ("đỡ đau hơn", "đỡ đau", "đỡ hơn", "đau ít hơn", "giảm đau", "khá hơn", "bớt đau")
WORSE_PHRASES = ("đau tăng", "đau nhiều hơn", "đau hơn", "nặng hơn", "tệ hơn")
SAME_PHRASES = ("như cũ", "không thay đổi", "vẫn vậy", "vẫn thế")
_POSITIVE_COUNT = re.compile(r"\btập\D{0,10}?([1-7]) ?(?:ngày|buổi|hôm|lần)")


def symptom_direction(text: str) -> Direction | None:
    norm = f" {normalize(text)} "
    better = False
    for phrase in BETTER_PHRASES:
        token = f" {normalize(phrase)} "
        if token in norm:
            better = True
            norm = norm.replace(token, " ")
    worse = contains_phrase(norm, WORSE_PHRASES) is not None
    if better and worse:
        return None  # vừa đỡ vừa tệ hơn -> không kết luận
    if worse:
        return "worse"
    if better:
        return "better"
    return "same" if contains_phrase(norm, SAME_PHRASES) else None


def latest_direction(sources: list[str]) -> Direction | None:
    """Lời mô tả gần nhất có nói về chiều hướng triệu chứng."""
    for source in reversed(sources):
        direction = symptom_direction(source)
        if direction:
            return direction
    return None


def mentions_no_exercise(text: str) -> bool:
    return contains_phrase(text, ZERO_PHRASES) is not None


def mentions_positive_exercise(text: str) -> bool:
    return _POSITIVE_COUNT.search(digitize(text)) is not None
