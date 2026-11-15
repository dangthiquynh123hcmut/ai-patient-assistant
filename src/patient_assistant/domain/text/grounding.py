"""Source-grounding: kiểm tra một trích dẫn có thật sự xuất hiện trong nguồn hay không."""

from __future__ import annotations

from difflib import SequenceMatcher

from patient_assistant.domain.common import SttFlag
from patient_assistant.domain.text.normalize import normalize
from patient_assistant.domain.text.vi_numbers import extract_numbers

DEFAULT_THRESHOLD = 0.85


def quote_in_sources(quote: str, sources: list[str], threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Đúng nếu `quote` xuất hiện (chính xác hoặc gần đúng) trong ít nhất một nguồn.

    Gần đúng = có một cửa sổ token trong nguồn, dài xấp xỉ quote, đạt tỉ lệ giống >= threshold.
    Cho phép sai khác nhỏ về dấu câu / chính tả mà vẫn chặn được trích dẫn bịa.
    """
    normalized_quote = normalize(quote)
    if not normalized_quote:
        return False
    q_tokens = normalized_quote.split(" ")
    for source in sources:
        normalized_source = normalize(source)
        if f" {normalized_quote} " in f" {normalized_source} ":
            return True
        s_tokens = normalized_source.split(" ")
        token_count = len(q_tokens)
        for size in (token_count - 1, token_count, token_count + 1):
            if size <= 0 or size > len(s_tokens):
                continue
            for start in range(len(s_tokens) - size + 1):
                window = " ".join(s_tokens[start : start + size])
                # Gần đúng về chữ nhưng phải giữ nguyên các con số: "tập 5 ngày" không khớp "tập 4 ngày".
                if SequenceMatcher(None, normalized_quote, window).ratio() >= threshold and sorted(
                    extract_numbers(normalized_quote)
                ) == sorted(extract_numbers(window)):
                    return True
    return False


def overlapping_flags(quote: str, flags: list[SttFlag]) -> list[SttFlag]:
    """Các cờ STT có từ nằm trong trích dẫn — trường dùng trích dẫn này cần hỏi lại."""
    normalized_quote = f" {normalize(quote)} "
    return [flag for flag in flags if f" {normalize(flag.text)} " in normalized_quote]
