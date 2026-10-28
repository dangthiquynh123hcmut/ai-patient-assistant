"""Lịch nghỉ lễ Việt Nam qua thư viện `holidays` — tiêm vào BookingPolicy để domain không phụ thuộc thư viện."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date


def vietnam_holiday_checker() -> Callable[[date], bool]:
    import holidays

    calendar = holidays.country_holidays("VN")
    return lambda day: day in calendar
