from __future__ import annotations

import pytest

from patient_assistant.application.agent import routing
from patient_assistant.application.agent.deps import AgentLimits
from patient_assistant.application.agent.state import AgentState, UserDecision
from patient_assistant.domain.common import Layer, Resolution, ValidationIssue
from patient_assistant.domain.conversation import (
    ConfirmIntent,
    Intent,
    SafetyKind,
    check_input_safety,
    classify_by_rules,
    interpret_confirmation,
    stt_flags_from_words,
)
from tests.conftest import NOW


@pytest.mark.parametrize(
    ("text", "primary", "secondary"),
    [
        ("Tuần này tôi đau mức 5, đã tập 4 ngày", Intent.CREATE_REPORT, []),
        ("Cho tôi đặt lịch khoa phục hồi chức năng", Intent.BOOK_APPOINTMENT, []),
        ("Xem lịch hẹn của tôi", Intent.LIST_APPOINTMENTS, []),
        ("Cho xem các báo cáo trước", Intent.GET_REPORT_HISTORY, []),
        ("Báo cáo tuần: đau mức 3, và đặt lịch tái khám", Intent.CREATE_REPORT, [Intent.BOOK_APPOINTMENT]),
        ("Chườm lạnh bao lâu là đủ", Intent.ANSWER_FAQ, []),
        ("xin chào", None, []),
    ],
)
def test_rule_intents(text: str, primary: Intent | None, secondary: list[Intent]) -> None:
    guess = classify_by_rules(text)
    assert guess.primary == primary
    assert guess.secondary == secondary


def test_safety_rules() -> None:
    assert check_input_safety("tôi đau ngực, khó thở").kind == SafetyKind.EMERGENCY
    assert check_input_safety("tôi bị bệnh gì vậy").kind == SafetyKind.MEDICAL_ADVICE
    assert check_input_safety("đau gối thì nên khám khoa nào").kind == SafetyKind.MEDICAL_ADVICE
    assert check_input_safety("liệt kê lịch hẹn của tôi").kind == SafetyKind.OK


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("xác nhận", ConfirmIntent.CONFIRM),
        ("đồng ý, đặt đi", ConfirmIntent.CONFIRM),
        ("ờ", ConfirmIntent.UNCLEAR),
        ("ừ", ConfirmIntent.UNCLEAR),
        ("đặt đi, à mà hủy", ConfirmIntent.UNCLEAR),
        ("tập 5 ngày thôi", ConfirmIntent.UNCLEAR),
        ("hủy giúp tôi", ConfirmIntent.CANCEL),
    ],
)
def test_spoken_confirmation_requires_clear_phrase(text: str, expected: ConfirmIntent) -> None:
    assert interpret_confirmation(text) == expected


def test_stt_flags_only_low_confidence_key_tokens() -> None:
    flags = stt_flags_from_words([("tôi", 0.2), ("đau", 0.9), ("bốn", 0.4), ("ngày", 0.95), ("9h", 0.5)])
    assert [f.text for f in flags] == ["bốn", "9h"]


def _state(**kw: object) -> AgentState:
    return AgentState(thread_id="t", patient_id="p", now=NOW, raw_input="x", **kw)  # type: ignore[arg-type]


def _issue(resolution: Resolution) -> ValidationIssue:
    return ValidationIssue(field="f", layer=Layer.BUSINESS, code="c", message="m", resolution=resolution)


def test_after_validate_decision_table() -> None:
    limits = AgentLimits(max_corrections=2)
    assert routing.after_validate(_state(), limits) == "accept"
    assert routing.after_validate(_state(issues=[_issue(Resolution.LLM_FIXABLE)]), limits) == "correct"
    assert (
        routing.after_validate(_state(issues=[_issue(Resolution.LLM_FIXABLE)], correction_count=2), limits) == "handoff"
    )
    mixed = [_issue(Resolution.LLM_FIXABLE), _issue(Resolution.ASK_USER)]
    assert routing.after_validate(_state(issues=mixed), limits) == "clarify"
    assert (
        routing.after_validate(_state(issues=[_issue(Resolution.BLOCK), _issue(Resolution.ASK_USER)]), limits)
        == "handoff"
    )


def test_present_report_routing() -> None:
    assert routing.after_present_report(_state(decision=UserDecision(action="confirm"))) == "save"
    assert routing.after_present_report(_state(decision=UserDecision(action="reply", text="xác nhận"))) == "save"
    assert routing.after_present_report(_state(decision=UserDecision(action="reply", text="ờ"))) == "present_draft"
    assert (
        routing.after_present_report(_state(decision=UserDecision(action="edit", fields={"pain_level": 4})))
        == "apply_edits"
    )
