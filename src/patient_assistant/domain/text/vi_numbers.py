"""Đọc số viết bằng chữ hoặc chữ số ("bốn" -> 4, "hai mươi mốt" -> 21, "five" -> 5, "5/10" -> [5, 10]).

Chủ yếu là tiếng Việt; tiếng Anh chỉ có số đếm nhỏ và cụm "không tập" thường gặp. Danh sách cụm từ không thể phủ hết
ngôn ngữ tự nhiên, nên validator chỉ dùng module này để bắt trích dẫn *nói rõ một con số khác* (`number_contradicted`).
Trích dẫn không chứa con số nào thì code không kết luận, người dùng kiểm tra trên thẻ xác nhận.
"""

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
# Số đếm tiếng Anh. Tách khỏi DIGITS vì DIGITS còn dùng để ghép số tiếng Việt và chọn từ quan trọng cho cờ STT.
EN_NUMBERS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_EN_ZERO_PHRASES = tuple(
    f"{negation} {verb}"
    for negation in ("don't", "didn't", "do not", "did not", "haven't", "have not", "never", "no", "not")
    for verb in ("exercise", "exercised", "exercises", "work out", "worked out", "workout", "train", "trained")
)

# Cụm biểu thị "0 lần/ngày" — dùng cho grounding và kiểm tra mâu thuẫn.
ZERO_PHRASES = (
    "chưa tập",
    "không tập",
    "chưa tập lần nào",
    "không tập ngày nào",
    "chưa tập được",
    "không tập được",
    "nghỉ tập cả tuần",
    *_EN_ZERO_PHRASES,
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
        elif tok in EN_NUMBERS:
            flush()
            out.append(EN_NUMBERS[tok])
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
    return " ".join(token for token in out if token)


def number_supported(value: int, quote: str) -> bool:
    """Giá trị `value` có được hỗ trợ bởi trích dẫn không (bằng số hoặc cụm biểu thị 0)."""
    if value in extract_numbers(quote):
        return True
    return value == 0 and contains_phrase(quote, ZERO_PHRASES) is not None


def number_contradicted(value: int, quote: str) -> bool:
    """Trích dẫn nói rõ một con số khác `value` ("tập 4 ngày" mà value=5, "không tập" mà value=3).

    Trích dẫn không chứa con số nào ("I walk most days", "dạo này lười") thì code không kết luận được -> không coi là
    mâu thuẫn; trích dẫn vẫn phải có thật trong lời người dùng và giá trị được hiện trên thẻ xác nhận trước khi lưu.
    """
    if number_supported(value, quote):
        return False
    return bool(extract_numbers(quote)) or contains_phrase(quote, ZERO_PHRASES) is not None
