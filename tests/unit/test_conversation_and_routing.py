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
        ("xin chào", Intent.SMALL_TALK, []),
        ("hello", Intent.SMALL_TALK, []),
        ("cảm ơn bạn nhiều", Intent.SMALL_TALK, []),
        ("bạn là ai vậy", Intent.SMALL_TALK, []),
        ("chào bạn, cho tôi đặt lịch khám", Intent.BOOK_APPOINTMENT, []),
        ("hi, chườm lạnh bao lâu là đủ", Intent.ANSWER_FAQ, []),
        ("abc xyz", None, []),
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
    return AgentState(conversation_id="t", patient_id="p", now=NOW, raw_input="x", **kw)  # type: ignore[arg-type]


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


@pytest.mark.parametrize(
    ("text", "current", "kind"),
    [
        ("tập được 4 ngày", Intent.CREATE_REPORT, "answer"),
        ("thứ năm", Intent.BOOK_APPOINTMENT, "answer"),
        ("xác nhận", Intent.CREATE_REPORT, "answer"),
        ("à nhầm, đau mức 8", Intent.CREATE_REPORT, "answer"),
        ("bạn có biết tôi là ai đâu mà hỏi rồi lưu dữ liệu vào đâu", Intent.CREATE_REPORT, "aside"),
        ("cảm ơn nhé", Intent.BOOK_APPOINTMENT, "aside"),
        ("tôi muốn tra cứu thông tin bệnh nhân", Intent.CREATE_REPORT, "switch"),
        ("xem lịch hẹn của tôi", Intent.CREATE_REPORT, "switch"),
        ("đặt lịch khám khác", Intent.BOOK_APPOINTMENT, "answer"),
        ("xem lịch hẹn của tôi", Intent.UNKNOWN, "answer"),  # đang hỏi rõ yêu cầu: để classify xử lý
    ],
)
async def test_reply_triage(text: str, current: Intent, kind: str) -> None:
    from patient_assistant.application.agent.side_talk import SideTalk
    from tests.conftest import ScriptedLLM

    assert (await SideTalk(ScriptedLLM(), None).triage(text, current)).value == kind  # type: ignore[arg-type]


def test_continuation_and_context_phrases() -> None:
    from patient_assistant.domain.conversation import asks_about_assistant_context, is_continuation

    assert is_continuation("vậy giờ bạn lưu đi")
    assert is_continuation("làm tiếp báo cáo lúc nãy")
    assert not is_continuation("tập 4 ngày")
    assert asks_about_assistant_context("dữ liệu của tôi lưu ở đâu")
    assert not asks_about_assistant_context("tôi đau mức 3")


def test_next_ask_counts_only_repeats_of_the_same_question() -> None:
    from types import SimpleNamespace

    from patient_assistant.application.agent.deps import AgentLimits
    from patient_assistant.application.agent.nodes.shared import next_ask

    deps = SimpleNamespace(limits=AgentLimits(max_clarify_rounds=2, max_questions=5))
    state = _state()
    asked = []
    for question in ["a", "a", "b", "b", "b"]:  # đổi câu hỏi = người dùng đã trả lời được -> đếm lại
        counters = next_ask(state, question, deps)  # type: ignore[arg-type]
        assert counters is not None
        asked.append(counters["clarify_rounds"])
        state = state.model_copy(update=counters)
    assert asked == [0, 1, 0, 1, 2]
    assert next_ask(state, "b", deps) is None  # hỏi lại "b" lần thứ 3 > max_clarify_rounds  # type: ignore[arg-type]
    assert next_ask(state, "c", deps) is None  # đã hỏi đủ max_questions câu  # type: ignore[arg-type]


async def test_reference_to_earlier_answers_is_an_answer_not_another_task() -> None:
    from patient_assistant.application.agent.side_talk import SideTalk
    from tests.conftest import ScriptedLLM

    llm = ScriptedLLM(intents=[[Intent.GET_REPORT_HISTORY]])  # LLM sẽ đoán sai nếu được hỏi
    kind = await SideTalk(llm, None).triage("lấy thông tin cũ tôi đã trả lời phía trên", Intent.BOOK_APPOINTMENT)  # type: ignore[arg-type]
    assert kind.value == "answer"


def test_carryover_accumulates_when_the_same_task_is_restarted() -> None:
    from patient_assistant.application.agent.state import Carryover
    from patient_assistant.application.use_cases.chat import _carryover

    older = Carryover(intent=Intent.BOOK_APPOINTMENT, sources=["đặt lịch khám mới", "nội tổng quát", "mai"])
    restarted = _state(intent=Intent.BOOK_APPOINTMENT, user_replies=["buổi chiều"], carryover=older).model_copy(
        update={"raw_input": "đặt lịch khám"}
    )
    carry = _carryover(restarted, "p")
    assert carry is not None
    assert carry.sources == ["đặt lịch khám mới", "nội tổng quát", "mai", "đặt lịch khám", "buổi chiều"]
