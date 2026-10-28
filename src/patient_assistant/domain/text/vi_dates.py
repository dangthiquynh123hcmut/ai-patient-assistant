"""Quy đổi cụm ngày giờ tiếng Việt thành khoảng thời gian cụ thể.

Nguyên tắc: LLM chỉ trích nguyên văn cụm thời gian; mọi phép tính ngày nằm ở đây, là code thuần,
nhận `now` từ bên ngoài để test/eval tái lập được. Cụm mơ hồ hoặc mâu thuẫn -> trả issue `ask_user`,
không tự đoán.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from pydantic import BaseModel, ConfigDict

from patient_assistant.domain.common import Layer, Resolution, ValidationIssue
from patient_assistant.domain.text.normalize import contains_phrase, normalize
from patient_assistant.domain.text.vi_numbers import digitize

WEEKDAY_NAMES = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")

# Sau digitize: "thứ hai" -> "thứ 2", "thứ năm" -> "thứ 5"; "thứ tư" -> "thứ 4".
_WEEKDAY_PATTERN = re.compile(r"\bthứ (2|3|4|5|6|7)\b|\bchủ nhật\b|\bcn\b")
_DMY_WORDS = re.compile(r"\bngày (\d{1,2}) tháng (\d{1,2})(?: (?:năm|5) (\d{4}))?")
_DMY_SLASH = re.compile(r"(?<![\d:])(\d{1,2})/(\d{1,2})(?:/(\d{4}))?(?![\d:])")
_DAY_ONLY = re.compile(r"\bngày (\d{1,2})\b(?! tháng)")
_HOUR_MIN = re.compile(r"\b(\d{1,2}) ?(?:giờ|h|g)(?: ?(rưỡi|\d{1,2}))?")
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")

_NEXT_WEEK = ("tuần sau", "tuần tới")
_THIS_WEEK = ("tuần này",)
_EARLIEST = ("sớm nhất", "càng sớm càng tốt", "lúc nào cũng được", "khi nào cũng được", "ngày nào cũng được")
_VAGUE_DATE = (
    "cuối tuần",
    "đầu tuần",
    "giữa tuần",
    "đầu tháng",
    "giữa tháng",
    "cuối tháng",
    "tháng sau",
    "tháng tới",
    "dạo này",
    "hôm nào đó",
    "mấy hôm nữa",
)
EARLIEST_SPAN_DAYS = 14
EXACT_TIME_TOLERANCE = timedelta(minutes=60)


@dataclass(frozen=True)
class DayParts:
    morning: tuple[time, time] = (time(7, 30), time(11, 30))
    afternoon: tuple[time, time] = (time(13, 30), time(16, 30))


class DateWindow(BaseModel):
    """Khoảng ngày/giờ đã quy đổi. `rules` ghi luật đã áp dụng, dùng cho phần giải thích."""

    model_config = ConfigDict(frozen=True)

    date_from: date
    date_to: date
    time_from: time | None = None
    time_to: time | None = None
    rules: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.date_from == self.date_to:
            days = f"{WEEKDAY_NAMES[self.date_from.weekday()]}, {self.date_from:%d/%m/%Y}"
        else:
            days = f"{self.date_from:%d/%m/%Y} – {self.date_to:%d/%m/%Y}"
        if self.time_from and self.time_to:
            return f"{days}, {self.time_from:%H:%M}–{self.time_to:%H:%M}"
        return days


@dataclass
class DateResolution:
    window: DateWindow | None
    issues: list[ValidationIssue] = field(default_factory=list)


def _issue(field_name: str, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        field=field_name, layer=Layer.BUSINESS, code=code, message=message, resolution=Resolution.ASK_USER
    )


def _weekday_mentioned(text: str) -> int | None:
    m = _WEEKDAY_PATTERN.search(text)
    if not m:
        return None
    return int(m.group(1)) - 2 if m.group(1) else 6


def _explicit_date(text: str, today: date) -> tuple[date | None, ValidationIssue | None]:
    for pattern in (_DMY_WORDS, _DMY_SLASH):
        m = pattern.search(text)
        if m:
            day, month = int(m.group(1)), int(m.group(2))
            year = int(m.group(3)) if m.group(3) else today.year
            try:
                return date(year, month, day), None
            except ValueError:
                return None, _issue("date_expression", "invalid_date", f"Ngày {day}/{month} không tồn tại.")
    m = _DAY_ONLY.search(text)
    if m:
        try:
            return today.replace(day=int(m.group(1))), None
        except ValueError:
            return None, _issue("date_expression", "invalid_date", "Ngày được nêu không tồn tại trong tháng này.")
    return None, None


def _resolve_date(raw: str, today: date) -> tuple[DateWindow | None, list[ValidationIssue]]:
    norm = normalize(raw)
    if not norm:
        return None, [_issue("date_expression", "date_missing", "Chưa có ngày muốn khám.")]

    # Cụm phải kiểm tra trước khi digitize ("mốt" bị đọc thành số 1).
    relative: tuple[date, str] | None = None
    if contains_phrase(norm, ("ngày kia", "ngày mốt", "mốt")):
        relative = (today + timedelta(days=2), "day_after_tomorrow")
    elif contains_phrase(norm, ("ngày mai", "mai")):
        relative = (today + timedelta(days=1), "tomorrow")
    elif contains_phrase(norm, ("hôm nay",)):
        relative = (today, "today")

    text = digitize(norm)
    weekday = _weekday_mentioned(text)
    explicit, err = _explicit_date(text, today)
    if err:
        return None, [err]

    if explicit:
        if weekday is not None and explicit.weekday() != weekday:
            return None, [
                _issue(
                    "date_expression",
                    "weekday_mismatch",
                    f"Ngày {explicit:%d/%m/%Y} là {WEEKDAY_NAMES[explicit.weekday()]}, "
                    f"không phải {WEEKDAY_NAMES[weekday]}.",
                )
            ]
        if relative and relative[0] != explicit:
            return None, [_issue("date_expression", "date_conflict", "Hai mốc ngày được nêu không khớp nhau.")]
        if explicit < today:
            return None, [_issue("date_expression", "date_in_past", f"Ngày {explicit:%d/%m/%Y} đã qua.")]
        return DateWindow(date_from=explicit, date_to=explicit, rules=("explicit_date",)), []

    if relative:
        day, rule = relative
        if weekday is not None and day.weekday() != weekday:
            return None, [_issue("date_expression", "weekday_mismatch", "Thứ được nêu không khớp với ngày.")]
        return DateWindow(date_from=day, date_to=day, rules=(rule,)), []

    next_week_monday = today + timedelta(days=7 - today.weekday())
    if weekday is not None:
        if contains_phrase(text, _NEXT_WEEK):
            day = next_week_monday + timedelta(days=weekday)
            return DateWindow(date_from=day, date_to=day, rules=("next_week_weekday",)), []
        if contains_phrase(text, _THIS_WEEK) or contains_phrase(text, ("này",)):
            day = today + timedelta(days=weekday - today.weekday())
            if day < today:
                return None, [_issue("date_expression", "date_in_past", f"{WEEKDAY_NAMES[weekday]} tuần này đã qua.")]
            return DateWindow(date_from=day, date_to=day, rules=("this_week_weekday",)), []
        delta = (weekday - today.weekday()) % 7 or 7
        day = today + timedelta(days=delta)
        return DateWindow(date_from=day, date_to=day, rules=("upcoming_weekday",)), []

    if contains_phrase(text, _VAGUE_DATE):
        return None, [_issue("date_expression", "vague_date", "Mốc thời gian chưa đủ cụ thể để tìm lịch.")]
    if contains_phrase(text, _NEXT_WEEK):
        return DateWindow(
            date_from=next_week_monday, date_to=next_week_monday + timedelta(days=6), rules=("next_week",)
        ), []
    if contains_phrase(text, _THIS_WEEK):
        sunday = today + timedelta(days=6 - today.weekday())
        return DateWindow(date_from=today, date_to=sunday, rules=("this_week",)), []
    if contains_phrase(text, _EARLIEST):
        return DateWindow(
            date_from=today, date_to=today + timedelta(days=EARLIEST_SPAN_DAYS), rules=("earliest_available",)
        ), []
    return None, [_issue("date_expression", "date_unparsed", "Chưa hiểu được ngày muốn khám.")]


def _resolve_time(raw: str, parts: DayParts) -> tuple[tuple[time, time] | None, str | None, list[ValidationIssue]]:
    text = digitize(raw)
    if not text:
        return None, None, []
    part: str | None = None
    if contains_phrase(text, ("sáng",)) and contains_phrase(text, ("chiều", "tối")):
        return (
            None,
            None,
            [_issue("time_expression", "time_part_conflict", "Cùng lúc nêu buổi sáng và buổi chiều/tối.")],
        )
    if contains_phrase(text, ("sáng",)):
        part = "morning"
    elif contains_phrase(text, ("chiều",)):
        part = "afternoon"
    elif contains_phrase(text, ("trưa",)):
        return None, None, [_issue("time_expression", "vague_time", "Buổi trưa là giờ nghỉ, cần chọn sáng hoặc chiều.")]
    elif contains_phrase(text, ("tối", "đêm")):
        return None, None, [_issue("time_expression", "outside_hours", "Phòng khám không làm việc buổi tối.")]

    hour: int | None = None
    minute = 0
    m = _CLOCK.search(text)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        m = _HOUR_MIN.search(text)
        if m:
            hour = int(m.group(1))
            if m.group(2) == "rưỡi":
                minute = 30
            elif m.group(2):
                minute = int(m.group(2))

    if hour is None:
        if part == "morning":
            return parts.morning, "morning", []
        if part == "afternoon":
            return parts.afternoon, "afternoon", []
        return None, None, []

    rule = "exact_time"
    if part == "afternoon" and hour < 12:
        hour += 12
    elif part == "morning" and hour >= 12:
        return None, None, [_issue("time_expression", "time_part_conflict", "Giờ được nêu không thuộc buổi sáng.")]
    elif part is None and 1 <= hour <= 5:
        hour += 12
        rule = "hour_assumed_afternoon"
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None, None, [_issue("time_expression", "invalid_time", "Giờ được nêu không hợp lệ.")]

    center = datetime.combine(date.min, time(hour, minute))
    lo = max(center - EXACT_TIME_TOLERANCE, datetime.combine(date.min, time(0, 0)))
    hi = min(center + EXACT_TIME_TOLERANCE, datetime.combine(date.min, time(23, 59)))
    return (lo.time(), hi.time()), rule, []


def resolve(
    date_expression: str | None,
    time_expression: str | None,
    now: datetime,
    parts: DayParts | None = None,
) -> DateResolution:
    """Quy đổi cụm ngày + giờ thành `DateWindow`. Cụm giờ có thể nằm lẫn trong cụm ngày và ngược lại."""
    parts = parts or DayParts()
    combined = f"{date_expression or ''} {time_expression or ''}"
    window, issues = _resolve_date(combined, now.date())
    times, time_rule, time_issues = _resolve_time(combined, parts)
    issues += time_issues
    if window is None or issues:
        return DateResolution(None, issues)
    rules = window.rules + ((time_rule,) if time_rule else ())
    if times:
        window = DateWindow(
            date_from=window.date_from, date_to=window.date_to, time_from=times[0], time_to=times[1], rules=rules
        )
    else:
        window = DateWindow(date_from=window.date_from, date_to=window.date_to, rules=rules)
    return DateResolution(window, [])
