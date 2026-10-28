"""Đọc số tiếng Việt viết bằng chữ hoặc chữ số ("bốn" -> 4, "hai mươi mốt" -> 21, "5/10" -> [5, 10])."""

from __future__ import annotations

import re

from patient_assistant.domain.text.normalize import contains_phrase, normalize

DIGITS: dict[str, int] = {
    "không": 0,
    "một": 1,
    "mốt": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "lăm": 5,
    "sáu": 6,
    "bảy": 7,
    "bẩy": 7,
    "tám": 8,
    "chín": 9,
}
_TEN_WORDS = {"mười", "mươi"}
_NUMBER_TOKEN = re.compile(r"^\d+$")

# Cụm biểu thị "0 lần/ngày" — dùng cho grounding và kiểm tra mâu thuẫn.
ZERO_PHRASES = (
    "chưa tập",
    "không tập",
    "chưa tập lần nào",
    "không tập ngày nào",
    "chưa tập được",
    "không tập được",
    "nghỉ tập cả tuần",
)


def _parse_word_number(words: list[str]) -> int | None:
    """Ghép một dãy từ số thành số nguyên (0..99). Trả None nếu không hợp lệ."""
    if not words:
        return None
    if len(words) == 1:
        if words[0] == "mười":
            return 10
        return DIGITS.get(words[0])
    if words[0] == "mười":  # mười một .. mười chín
        unit = DIGITS.get(words[1]) if len(words) == 2 else None
        return 10 + unit if unit is not None and unit != 0 else None
    tens = DIGITS.get(words[0])
    if tens is None or words[1] != "mươi":
        return None
    if len(words) == 2:
        return tens * 10
    unit = DIGITS.get(words[2]) if len(words) == 3 else None
    return tens * 10 + unit if unit is not None else None


def extract_numbers(text: str) -> list[int]:
    """Tất cả số xuất hiện trong văn bản, theo thứ tự."""
    norm = normalize(text).replace("/", " / ")
    out: list[int] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            value = _parse_word_number(buffer)
            if value is not None:
                out.append(value)
            buffer.clear()

    for tok in norm.split(" "):
        if _NUMBER_TOKEN.match(tok):
            flush()
            out.append(int(tok))
        elif tok in DIGITS or tok in _TEN_WORDS:
            buffer.append(tok)
            if len(buffer) == 3:
                flush()
        else:
            flush()
    flush()
    return out


def digitize(text: str) -> str:
    """Chuẩn hóa rồi thay các cụm số bằng chữ thành chữ số: "tháng mười một" -> "tháng 11"."""
    out: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        while buffer:
            # ghép tham lam cụm dài nhất (3 -> 1 từ) parse được
            for size in (3, 2, 1):
                value = _parse_word_number(buffer[:size]) if len(buffer) >= size else None
                if value is not None:
                    out.append(str(value))
                    del buffer[:size]
                    break
            else:
                out.append(buffer.pop(0))

    for tok in normalize(text).split(" "):
        if tok in DIGITS or tok in _TEN_WORDS:
            buffer.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(t for t in out if t)


def number_supported(value: int, quote: str) -> bool:
    """Giá trị `value` có được hỗ trợ bởi trích dẫn không (bằng số hoặc cụm biểu thị 0)."""
    if value in extract_numbers(quote):
        return True
    return value == 0 and contains_phrase(quote, ZERO_PHRASES) is not None
