"""Hội thoại tiếng Anh đi qua cùng bộ luật code với tiếng Việt: ngày giờ, xác nhận, chọn slot, an toàn, intent."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from patient_assistant.domain.booking.models import Doctor, Slot, SlotOption
from patient_assistant.domain.booking.selection import match_selection
from patient_assistant.domain.booking.validators import departments_matching, find_doctor
from patient_assistant.domain.conversation import (
    ConfirmIntent,
    Intent,
    SafetyKind,
    check_input_safety,
    classify_by_rules,
    interpret_confirmation,
    output_is_safe,
)
from patient_assistant.domain.text.vi_dates import mentions_when, resolve
from patient_assistant.infrastructure.persistence.seed import DEPARTMENTS

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
NOW = datetime(2026, 10, 28, 10, 0, tzinfo=TZ)  # Thứ Tư


@pytest.mark.parametrize(
    ("date_expr", "time_expr", "day", "hours"),
    [
        ("next Tuesday", "morning", date(2026, 11, 3), (time(7, 30), time(11, 30))),
        ("Tuesday next week", None, date(2026, 11, 3), None),
        ("this Friday", "3 pm", date(2026, 10, 30), (time(14, 0), time(16, 0))),
        ("tomorrow", "at 9:30 am", date(2026, 10, 29), (time(8, 30), time(10, 30))),
        ("Thursday", "nine thirty am", date(2026, 10, 29), (time(8, 30), time(10, 30))),
        ("November 5th", None, date(2026, 11, 5), None),
        ("the 5th of November", "half past two in the afternoon", date(2026, 11, 5), (time(13, 30), time(15, 30))),
        ("the day after tomorrow", "9 o'clock", date(2026, 10, 30), (time(8, 0), time(10, 0))),
        ("5/11", None, date(2026, 11, 5), None),
    ],
)
def test_english_dates_and_times_are_resolved_by_code(
    date_expr: str, time_expr: str | None, day: date, hours: tuple[time, time] | None
) -> None:
    window = resolve(date_expr, time_expr, NOW).window
    assert window is not None and window.date_from == window.date_to == day
    assert (window.time_from, window.time_to) == (hours or (None, None))


def test_english_ranges() -> None:
    week = resolve("next week", None, NOW).window
    assert week and (week.date_from, week.date_to) == (date(2026, 11, 2), date(2026, 11, 8))
    soon = resolve("as soon as possible", None, NOW).window
    assert soon and soon.date_from == NOW.date() and soon.date_to == NOW.date() + timedelta(days=14)


@pytest.mark.parametrize(
    ("date_expr", "time_expr", "code"),
    [
        ("this weekend", None, "vague_date"),
        ("early next week", None, "vague_date"),
        ("next month", None, "vague_date"),
        ("October 20", None, "date_in_past"),
        ("Monday", "noon", "vague_time"),
        ("tomorrow", "in the evening", "outside_hours"),
        ("tomorrow", "morning or afternoon", "time_part_conflict"),
        ("Friday November 5", None, "weekday_mismatch"),
        ("February 30", None, "invalid_date"),
    ],
)
def test_english_ambiguous_or_invalid_is_asked(date_expr: str, time_expr: str | None, code: str) -> None:
    result = resolve(date_expr, time_expr, NOW)
    assert result.window is None and code in [i.code for i in result.issues]


def test_vietnamese_still_works_and_mentions_when_understands_english() -> None:
    vi = resolve("thứ ba tuần sau", "buổi sáng", NOW).window
    assert vi and vi.date_from == date(2026, 11, 3)
    # Lỗi cũ: "mốt" bị đổi thành số 1 -> "ngày 1" -> báo mâu thuẫn ngày.
    after = resolve("ngày mốt", "9 giờ", NOW).window
    assert after and after.date_from == date(2026, 10, 30)
    assert [i.code for i in resolve("ngày mai ngày 30", None, NOW).issues] == ["date_conflict"]
    assert mentions_when("something in the afternoon please", NOW)
    assert mentions_when("how about Thursday", NOW)
    assert not mentions_when("the second one", NOW)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("confirm", ConfirmIntent.CONFIRM),
        ("Yes, book it", ConfirmIntent.CONFIRM),
        ("go ahead", ConfirmIntent.CONFIRM),
        ("cancel", ConfirmIntent.CANCEL),
        ("never mind", ConfirmIntent.CANCEL),
        ("don't book it", ConfirmIntent.UNCLEAR),  # có cả "book it" lẫn "don t book" -> hỏi lại
        ("yes", ConfirmIntent.UNCLEAR),  # như "ừ": chưa đủ rõ (FR-09.7)
        ("ok", ConfirmIntent.UNCLEAR),
        ("the pain stops when I rest", ConfirmIntent.UNCLEAR),
    ],
)
def test_english_spoken_confirmation(text: str, expected: ConfirmIntent) -> None:
    assert interpret_confirmation(text) == expected


def _options() -> list[SlotOption]:
    def slot(i: int, day: int, hh: int, doctor: str) -> SlotOption:
        start = datetime(2026, 11, day, hh, 0, tzinfo=TZ)
        return SlotOption(
            option_no=i,
            slot=Slot(
                slot_id=f"s{i}",
                department_code="PHCN",
                doctor_id=doctor,
                doctor_name="BS. Trần Văn An" if doctor == "doc-an" else "BS. Lê Thị Bình",
                start_at=start,
                end_at=start + timedelta(minutes=30),
            ),
        )

    return [slot(1, 3, 8, "doc-an"), slot(2, 3, 9, "doc-binh"), slot(3, 5, 14, "doc-an")]


@pytest.mark.parametrize(
    ("reply", "option_no"),
    [
        ("the first one", 1),
        ("the second one please", 2),
        ("the last one", 3),
        ("option 2", 2),
        ("number 3", 3),
        ("the 9 am one", 2),
        ("2 pm on Thursday", 3),
        ("doctor Binh", 2),
    ],
)
def test_english_slot_selection(reply: str, option_no: int) -> None:
    assert match_selection(reply, _options()).option_no == option_no


def test_english_ambiguous_selection_asks() -> None:
    result = match_selection("doctor An", _options())
    assert result.option_no is None and result.ambiguous


@pytest.mark.parametrize(
    "text",
    ["I have chest pain", "I can't breathe", "my father fainted", "I think it's a stroke"],
)
def test_english_emergency_is_caught(text: str) -> None:
    assert check_input_safety(text).kind == SafetyKind.EMERGENCY


def test_english_medical_advice_is_blocked_both_ways() -> None:
    assert check_input_safety("which medicine should I take").kind == SafetyKind.MEDICAL_ADVICE
    assert check_input_safety("can you diagnose my knee").kind == SafetyKind.MEDICAL_ADVICE
    assert not output_is_safe("You should take 2 pills, that is the usual dosage.")
    assert output_is_safe("You have an appointment on Tuesday.")


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("I want to book an appointment next Tuesday", Intent.BOOK_APPOINTMENT),
        ("show my appointments", Intent.LIST_APPOINTMENTS),
        ("my pain level is 5 this week", Intent.CREATE_REPORT),
        ("which doctors work in rehabilitation", Intent.GET_DOCTOR_INFO),
        ("show me my previous reports", Intent.GET_REPORT_HISTORY),
        ("who are you", Intent.SMALL_TALK),
    ],
)
def test_english_intent_rules(text: str, intent: Intent) -> None:
    assert classify_by_rules(text).primary == intent


def test_english_department_and_doctor_names() -> None:
    assert [d.code for d in departments_matching("rehabilitation", list(DEPARTMENTS))] == ["PHCN"]
    assert [d.code for d in departments_matching("cardiology", list(DEPARTMENTS))] == ["TM"]
    doctors = [
        Doctor(id="doc-an", name="BS. Trần Văn An", department_code="PHCN"),
        Doctor(id="doc-binh", name="BS. Lê Thị Bình", department_code="PHCN"),
    ]
    found = find_doctor("doctor Binh", "PHCN", doctors)
    assert found is not None and found.id == "doc-binh"
    found = find_doctor("Dr. An", None, doctors)
    assert found is not None and found.id == "doc-an"
