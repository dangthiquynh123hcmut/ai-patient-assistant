"""Chuẩn hóa văn bản tiếng Việt cho việc so khớp (giữ nguyên dấu thanh)."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s:/]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    norm = normalize(text)
    return norm.split(" ") if norm else []


def contains_phrase(text: str, phrases: tuple[str, ...] | list[str]) -> str | None:
    """Trả về cụm đầu tiên xuất hiện trọn từ trong `text`, hoặc None."""
    padded = f" {normalize(text)} "
    for phrase in phrases:
        if f" {normalize(phrase)} " in padded:
            return phrase
    return None
