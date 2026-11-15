from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


class SystemClock:
    def __init__(self, timezone: str) -> None:
        self._timezone = ZoneInfo(timezone)

    def now(self) -> datetime:
        return datetime.now(self._timezone)


class FixedClock:
    """Đồng hồ cố định cho test/eval tái lập được."""

    def __init__(self, at: datetime) -> None:
        assert at.tzinfo is not None, "FixedClock cần datetime có timezone"
        self._at = at

    def now(self) -> datetime:
        return self._at
