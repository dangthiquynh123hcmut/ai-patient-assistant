"""Audit log qua structlog (JSON). Chỉ ghi metadata, không ghi lời người dùng hay dữ liệu sức khỏe (FR-08.8)."""

from __future__ import annotations

from typing import Any

import structlog

_REDACT_KEYS = {"output", "text", "raw_input", "transcript", "reason", "evidence"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<redacted>" if k in _REDACT_KEYS else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


class StructlogAuditLog:
    def __init__(self) -> None:
        self._log = structlog.get_logger("patient_assistant.audit")

    async def record(self, kind: str, thread_id: str, payload: dict[str, Any]) -> None:
        self._log.info(kind, thread_id=thread_id, **_redact(payload))
