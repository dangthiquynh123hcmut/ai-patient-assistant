from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from patient_assistant.domain.text.grounding import quote_in_sources
from patient_assistant.domain.text.vi_dates import resolve
from patient_assistant.domain.text.vi_numbers import digitize, extract_numbers, number_supported
from tests.conftest import NOW, TZ


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tôi đau mức 5 trên 10", [5, 10]),
        ("đau mức năm trên mười", [5, 10]),
        ("đã tập bốn ngày", [4]),
        ("hai mươi mốt", [21]),
        ("mười lăm phút", [15]),
        ("5/10", [5, 10]),
    ],
)
def test_extract_numbers(text: str, expected: list[int]) -> None:
    assert extract_numbers(text) == expected


def test_digitize_and_zero_phrases() -> None:
    assert digitize("ngày năm tháng mười một") == "ngày 5 tháng 11"
    assert number_supported(0, "tuần này tôi chưa tập lần nào")
    assert not number_supported(5, "đã tập 4 ngày")


def test_quote_in_sources_tolerates_punctuation_but_rejects_fabrication() -> None:
    sources = ["Tuần này tôi đau mức 5 trên 10, đã tập 4 ngày."]
    assert quote_in_sources("đau mức 5 trên 10", sources)
    assert quote_in_sources("Đã tập 4 ngày!", sources)
    assert not quote_in_sources("đã tập 6 ngày liền", sources)


@pytest.mark.parametrize(
    ("date_expr", "time_expr", "day"),
    [
        ("ngày mai", None, date(2026, 10, 29)),
        ("thứ ba tuần sau", None, date(2026, 11, 3)),
        ("thứ Sáu này", None, date(2026, 10, 30)),
        ("ngày 5 tháng 11", None, date(2026, 11, 5)),
        ("ngày năm tháng mười một", None, date(2026, 11, 5)),
        ("5/11", None, date(2026, 11, 5)),
        ("thứ năm", None, date(2026, 10, 29)),
        ("sáng mai", None, date(2026, 10, 29)),
    ],
)
def test_resolve_dates(date_expr: str, time_expr: str | None, day: date) -> None:
    result = resolve(date_expr, time_expr, NOW)
    assert result.issues == []
    assert result.window is not None
    assert (result.window.date_from, result.window.date_to) == (day, day)


@pytest.mark.parametrize(
    ("date_expr", "time_expr", "code"),
    [
        ("thứ hai ngày 5/11", None, "weekday_mismatch"),  # 05/11/2026 là Thứ Năm
        ("thứ hai này", None, "date_in_past"),
        ("ngày 5/1", None, "date_in_past"),  # không tự cộng năm
        ("cuối tuần", None, "vague_date"),
        (None, None, "date_missing"),
        ("ngày 31/11", None, "invalid_date"),
        ("ngày mai", "buổi tối", "outside_hours"),
        ("ngày mai", "sáng 3 giờ chiều", "time_part_conflict"),
    ],
)
def test_resolve_rejects_ambiguous_or_conflicting(date_expr: str | None, time_expr: str | None, code: str) -> None:
    result = resolve(date_expr, time_expr, NOW)
    assert result.window is None
    assert code in [i.code for i in result.issues]


def test_resolve_times() -> None:
    morning = resolve("ngày mai", "buổi sáng", NOW).window
    assert morning and (morning.time_from, morning.time_to) == (time(7, 30), time(11, 30))
    afternoon = resolve("thứ năm", "2 giờ chiều", NOW).window
    assert afternoon and (afternoon.time_from, afternoon.time_to) == (time(13, 0), time(15, 0))
    assumed = resolve("ngày mai", "3 giờ", NOW).window
    assert assumed and assumed.time_from == time(14, 0) and "hour_assumed_afternoon" in assumed.rules
    half = resolve("ngày mai", "chín giờ rưỡi", NOW).window
    assert half and (half.time_from, half.time_to) == (time(8, 30), time(10, 30))


def test_next_week_range() -> None:
    window = resolve("tuần sau", None, NOW).window
    assert window and (window.date_from, window.date_to) == (date(2026, 11, 2), date(2026, 11, 8))


_WEEKDAYS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]


@given(offset=st.integers(min_value=0, max_value=800), weekday=st.integers(min_value=0, max_value=6))
def test_next_week_weekday_invariants(offset: int, weekday: int) -> None:
    now = datetime(2026, 1, 1, 9, 0, tzinfo=TZ) + timedelta(days=offset)
    window = resolve(f"{_WEEKDAYS[weekday]} tuần sau", None, now).window
    assert window is not None
    assert window.date_from.weekday() == weekday
    assert window.date_from > now.date()
    assert window.date_from.isocalendar().week != now.date().isocalendar().week


def test_fuzzy_grounding_rejects_changed_numbers() -> None:
    sources = ["Tuần này tôi đau mức 5 trên 10, đã tập 4 ngày."]
    assert not quote_in_sources("đã tập 5 ngày", sources)
    assert not quote_in_sources("đau mức 6 trên 10", sources)
