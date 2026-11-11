"""Đưa cụm ngày giờ tiếng Anh về dạng tiếng Việt chuẩn mà `vi_dates` đã hiểu.

Chỉ dịch từ vựng, không tính ngày: "next Tuesday morning" -> "thứ 3 tuần sau sáng", "3 pm" -> "3 giờ chiều".
Mọi phép tính và mọi luật hỏi lại (ngày đã qua, thứ không khớp ngày, cụm mơ hồ, ngoài giờ) vẫn nằm ở
`vi_dates`, nên tiếng Anh và tiếng Việt đi qua cùng một bộ luật.

Quy ước: "next <thứ>" = thứ đó của TUẦN SAU (giống "thứ ba tuần sau"); "this <thứ>" = tuần này; "<thứ>" đứng
một mình = thứ đó gần nhất sắp tới. Ngày dạng số "5/11" theo thứ tự ngày/tháng như tiếng Việt.
Đầu vào đã qua `normalize` (chữ thường, bỏ dấu câu trừ ":" và "/").
"""

from __future__ import annotations

import re

_WEEKDAYS = {
    "monday": "thứ 2",
    "tuesday": "thứ 3",
    "tues": "thứ 3",
    "tue": "thứ 3",
    "wednesday": "thứ 4",
    "wed": "thứ 4",
    "thursday": "thứ 5",
    "thurs": "thứ 5",
    "thur": "thứ 5",
    "friday": "thứ 6",
    "fri": "thứ 6",
    "saturday": "thứ 7",
    "sunday": "chủ nhật",
}
_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_NUMBERS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "fifteen": "15",
    "twenty": "20",
    "thirty": "30",
    "forty five": "45",
    "forty": "40",
    "fifty": "50",
}
# Thứ tự quan trọng: cụm dài trước cụm ngắn ("day after tomorrow" trước "tomorrow",
# "early next week" trước "next week").
_PHRASES: tuple[tuple[str, str], ...] = (
    ("the day after tomorrow", "ngày mốt"),
    ("day after tomorrow", "ngày mốt"),
    ("tomorrow", "ngày mai"),
    ("today", "hôm nay"),
    ("tonight", "hôm nay tối"),
    # cụm mơ hồ -> cụm mơ hồ tiếng Việt để bị hỏi lại
    ("early next week", "đầu tuần"),
    ("beginning of the week", "đầu tuần"),
    ("start of the week", "đầu tuần"),
    ("middle of the week", "giữa tuần"),
    ("midweek", "giữa tuần"),
    ("mid week", "giữa tuần"),
    ("end of the week", "cuối tuần"),
    ("weekend", "cuối tuần"),
    ("end of the month", "cuối tháng"),
    ("beginning of the month", "đầu tháng"),
    ("early next month", "đầu tháng"),
    ("next month", "tháng sau"),
    ("some day", "hôm nào đó"),
    ("sometime", "hôm nào đó"),
    ("in a few days", "mấy hôm nữa"),
    # sớm nhất
    ("as soon as possible", "sớm nhất"),
    ("asap", "sớm nhất"),
    ("earliest", "sớm nhất"),
    ("any day", "sớm nhất"),
    ("anytime", "sớm nhất"),
    ("any time", "sớm nhất"),
    ("whenever", "sớm nhất"),
    ("next week", "tuần sau"),
    ("this week", "tuần này"),
    # buổi
    ("in the morning", "sáng"),
    ("morning", "sáng"),
    ("in the afternoon", "chiều"),
    ("afternoon", "chiều"),
    ("in the evening", "tối"),
    ("evening", "tối"),
    ("at night", "tối"),
    ("midday", "trưa"),
    ("noon", "trưa"),
    ("lunchtime", "trưa"),
    ("lunch time", "trưa"),
)
_ORDINAL = r"(\d{1,2})(?:st|nd|rd|th)?"
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))


def _sub_phrases(text: str) -> str:
    for en, vi in _PHRASES:
        text = re.sub(rf"\b{en}\b", vi, text)
    return text


def english_to_vi(text: str) -> str:
    """Dịch từ vựng ngày giờ tiếng Anh trong `text` (đã normalize). Câu tiếng Việt đi qua gần như nguyên vẹn."""
    if not re.search(r"[a-z]", text):
        return text
    out = f" {text} "
    for word, digit in _NUMBERS.items():
        out = re.sub(rf"\b{word}\b", digit, out)
    # Ngày có tên tháng: "november 5th 2026", "5 nov", "the 5th of november".
    out = re.sub(
        rf"\b({_MONTH}) (?:the )?{_ORDINAL}(?: (\d{{4}}))?\b",
        lambda m: f"ngày {m.group(2)} tháng {_MONTHS[m.group(1)]}" + (f" năm {m.group(3)}" if m.group(3) else ""),
        out,
    )
    out = re.sub(
        rf"\b(?:the )?{_ORDINAL} (?:of )?({_MONTH})(?: (\d{{4}}))?\b",
        lambda m: f"ngày {m.group(1)} tháng {_MONTHS[m.group(2)]}" + (f" năm {m.group(3)}" if m.group(3) else ""),
        out,
    )
    out = re.sub(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b", r"ngày \1", out)
    # Thứ: "next tuesday" / "tuesday next week" -> tuần sau; "this tuesday" -> tuần này.
    out = re.sub(rf"\bnext ({_WEEKDAY})\b", lambda m: f"{_WEEKDAYS[m.group(1)]} tuần sau", out)
    out = re.sub(rf"\bthis ({_WEEKDAY})\b", lambda m: f"{_WEEKDAYS[m.group(1)]} tuần này", out)
    out = re.sub(rf"\b({_WEEKDAY})s?\b", lambda m: _WEEKDAYS[m.group(1)], out)
    # Giờ: "half past 9", "quarter past 9", "9:30 am", "9 30 pm", "9 am", "9 o clock", "at 9".
    out = re.sub(r"\bhalf past (\d{1,2})\b", r"\1 giờ rưỡi", out)
    out = re.sub(r"\bquarter past (\d{1,2})\b", r"\1 giờ 15", out)
    out = re.sub(r"\b(\d{1,2})(?::| )(\d{2}) ?(?:am|a m)\b", r"\1:\2 sáng", out)
    out = re.sub(r"\b(\d{1,2})(?::| )(\d{2}) ?(?:pm|p m)\b", r"\1:\2 chiều", out)
    out = re.sub(r"\b(\d{1,2}) ?(?:am|a m)\b", r"\1 giờ sáng", out)
    out = re.sub(r"\b(\d{1,2}) ?(?:pm|p m)\b", r"\1 giờ chiều", out)
    out = re.sub(r"\b(\d{1,2}) o ?clock\b", r"\1 giờ", out)
    out = re.sub(r"\bat (\d{1,2}) (\d{2})\b", r"\1:\2", out)
    out = re.sub(r"\bat (\d{1,2})\b(?! ?(?:giờ|:))", r"\1 giờ", out)
    out = _sub_phrases(out)
    return " ".join(out.split())
