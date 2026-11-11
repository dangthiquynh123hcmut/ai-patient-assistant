"""Ghép câu trả lời chọn slot ("cái thứ hai", "chín giờ rưỡi", "bác sĩ An") với danh sách đã đề xuất."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patient_assistant.domain.booking.models import SlotOption
from patient_assistant.domain.text.en_dates import english_to_vi
from patient_assistant.domain.text.normalize import contains_phrase, normalize, unaccent
from patient_assistant.domain.text.vi_dates import expand_weekday_abbrev
from patient_assistant.domain.text.vi_numbers import digitize

_ORDINAL = re.compile(r"\b(?:cái|số|phương án|lựa chọn|option|slot|lịch|number|choice)(?: thứ)? (\d{1,2})\b")
_EN_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
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
    # Tiếng Anh ("the 9 am one", "Thursday", "doctor Binh") đưa về cùng dạng với tiếng Việt trước khi ghép.
    raw = expand_weekday_abbrev(english_to_vi(normalize(reply)))
    text = digitize(raw)

    if contains_phrase(raw, ("đầu tiên", "cái đầu", "cái trên cùng", "the first", "first one", "earliest one")):
        return SelectionResult(min(numbers), rule="first")
    if contains_phrase(raw, ("cuối cùng", "cái cuối", "the last", "last one")):
        return SelectionResult(max(numbers), rule="last")
    for word, no in _EN_ORDINAL.items():
        if contains_phrase(raw, (f"the {word}", f"{word} one", f"{word} option")) and no in numbers:
            return SelectionResult(no, rule="ordinal")
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
    plain = f" {unaccent(raw)} "
    by_doctor = [o for o in candidates if f" {unaccent(o.slot.doctor_name).split(' ')[-1]} " in plain]
    if by_doctor:
        candidates = by_doctor
        rule_parts.append("doctor")

    if not rule_parts:
        return SelectionResult(None)
    if len(candidates) == 1:
        return SelectionResult(candidates[0].option_no, rule="+".join(rule_parts))
    return SelectionResult(None, candidates=[o.option_no for o in candidates], rule="+".join(rule_parts))
