"""Ghép câu trả lời chọn slot ("cái thứ hai", "chín giờ rưỡi", "bác sĩ An") với danh sách đã đề xuất."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patient_assistant.domain.booking.models import SlotOption
from patient_assistant.domain.text.normalize import contains_phrase, normalize
from patient_assistant.domain.text.vi_numbers import digitize

_ORDINAL = re.compile(r"\b(?:cái|số|phương án|lựa chọn|option|slot|lịch)(?: thứ)? (\d{1,2})\b")
_BARE_NUMBER = re.compile(r"^(?:chọn |lấy )?(\d{1,2})$")
_TIME = re.compile(r"\b(\d{1,2}) ?(?:giờ|h|:)(?: ?(rưỡi|\d{2}))?")
_WEEKDAY = re.compile(r"\bthứ (2|3|4|5|6|7)\b|\bchủ nhật\b")


@dataclass
class SelectionResult:
    option_no: int | None
    candidates: list[int] = field(default_factory=list)
    rule: str | None = None

    @property
    def ambiguous(self) -> bool:
        return self.option_no is None and len(self.candidates) > 1


def match_selection(reply: str, options: list[SlotOption]) -> SelectionResult:
    if not options:
        return SelectionResult(None)
    numbers = {o.option_no for o in options}
    raw = normalize(reply)
    text = digitize(raw)

    if contains_phrase(raw, ("đầu tiên", "cái đầu", "cái trên cùng")):
        return SelectionResult(min(numbers), rule="first")
    if contains_phrase(raw, ("cuối cùng", "cái cuối")):
        return SelectionResult(max(numbers), rule="last")
    m = _ORDINAL.search(text) or _BARE_NUMBER.match(text)
    if m and int(m.group(1)) in numbers:
        return SelectionResult(int(m.group(1)), rule="ordinal")

    candidates = list(options)
    rule_parts: list[str] = []
    tm = _TIME.search(text)
    if tm:
        hour = int(tm.group(1))
        minute = 30 if tm.group(2) == "rưỡi" else int(tm.group(2) or 0)
        if contains_phrase(raw, ("chiều",)) and hour < 12:
            hour += 12
        candidates = [
            o
            for o in candidates
            if (o.slot.start_at.hour, o.slot.start_at.minute) == (hour, minute)
            or (o.slot.start_at.hour == hour + 12 and 1 <= hour <= 5 and o.slot.start_at.minute == minute)
        ]
        rule_parts.append("time")
    wm = _WEEKDAY.search(text)
    if wm:
        weekday = int(wm.group(1)) - 2 if wm.group(1) else 6
        candidates = [o for o in candidates if o.slot.start_at.weekday() == weekday]
        rule_parts.append("weekday")
    by_doctor = [o for o in candidates if f" {normalize(o.slot.doctor_name).split(' ')[-1]} " in f" {raw} "]
    if by_doctor:
        candidates = by_doctor
        rule_parts.append("doctor")

    if not rule_parts:
        return SelectionResult(None)
    if len(candidates) == 1:
        return SelectionResult(candidates[0].option_no, rule="+".join(rule_parts))
    return SelectionResult(None, candidates=[o.option_no for o in candidates], rule="+".join(rule_parts))
