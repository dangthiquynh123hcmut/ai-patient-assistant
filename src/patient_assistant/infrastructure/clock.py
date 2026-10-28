from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


class SystemClock:
    def __init__(self, tz: str) -> None:
        self._tz = ZoneInfo(tz)

    def now(self) -> datetime:
        return datetime.now(self._tz)


class FixedClock:
    """Đồng hồ cố định cho test/eval tái lập được."""

    def __init__(self, at: datetime) -> None:
        assert at.tzinfo is not None, "FixedClock cần datetime có timezone"
        self._at = at

    def now(self) -> datetime:
        return self._at
