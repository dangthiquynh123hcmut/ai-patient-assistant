"""Hợp đồng kết quả tool chuẩn hóa (FR-03.3, FR-03.7)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolErrorCode(StrEnum):
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    INVALID_ARGS = "invalid_args"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    INTERNAL = "internal"


class ToolError(BaseModel):
    code: ToolErrorCode
    message: str
    retryable: bool = False
    details: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    tool: str
    status: str  # "ok" | "error"
    data: Any = None
    error: ToolError | None = None
    source_ids: list[str] = Field(default_factory=list)
    latency_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @classmethod
    def success(cls, tool: str, data: Any, source_ids: list[str] | None = None) -> ToolResult:
        return cls(tool=tool, status="ok", data=data, source_ids=source_ids or [])

    @classmethod
    def failure(
        cls, tool: str, code: ToolErrorCode, message: str, retryable: bool = False, details: list[str] | None = None
    ) -> ToolResult:
        return cls(
            tool=tool,
            status="error",
            error=ToolError(code=code, message=message, retryable=retryable, details=details or []),
        )


class ToolCallRecord(BaseModel):
    tool: str
    called_by: str  # "code" | "llm"
    status: str
    error_code: str | None = None
    latency_ms: int
    source_ids: list[str] = Field(default_factory=list)
