"""Lỗi LLM phải nói rõ nguyên nhân gốc trong trace và log, không chỉ "llm_unavailable"."""

from __future__ import annotations

from patient_assistant.application.agent.trace import describe_step
from patient_assistant.infrastructure.llm.bedrock import _root_cause


def test_root_cause_follows_exception_chain() -> None:
    try:
        try:
            raise PermissionError("The security token included in the request is expired")
        except PermissionError as inner:
            raise ConnectionError("Connection error.") from inner
    except ConnectionError as outer:
        text = _root_cause(outer)
    assert text == (
        "ConnectionError: Connection error. <- PermissionError: The security token included in the request is expired"
    )


def test_trace_summary_shows_error_detail() -> None:
    update = {
        "outcome": {
            "kind": "error",
            "code": "llm_unavailable",
            "detail": "connection error: APIConnectionError: Connection error. <- ExpiredTokenException: token expired",
        }
    }
    step = describe_step(3, ("agent_flow:abc",), "agent", update)
    assert "llm_unavailable" in step.summary
    assert "ExpiredTokenException: token expired" in step.summary
    assert step.details["outcome"]["detail"].startswith("connection error")
