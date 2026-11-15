"""Kiểm tra chính bộ đánh giá: metric phải bắt được lỗi, không chỉ cho ra con số đẹp."""

from __future__ import annotations

from typing import Any

import pytest

from patient_assistant.application.ports.llm import AgentReply
from patient_assistant.config import Settings
from patient_assistant.container import build_llm
from patient_assistant.evaluation.cases import EvalCase, load_cases
from patient_assistant.evaluation.harness import AttemptView, CaseObservation, run_cases
from patient_assistant.evaluation.metrics import compute, is_false_success
from patient_assistant.evaluation.report import hard_zero_violations, to_markdown
from tests.conftest import ScriptedLLM, call_tool, postgres_settings


def _case(**overrides: Any) -> EvalCase:
    base: dict[str, Any] = {
        "id": "c1",
        "suite": "booking",
        "category": "t",
        "message": {"text": "x"},
        "expected": {"outcome": "appointment_booked"},
    }
    base.update(overrides)
    return EvalCase.model_validate(base)


def _observation(**overrides: Any) -> CaseObservation:
    base: dict[str, Any] = {"case_id": "c1", "suite": "booking", "category": "t", "status": "done"}
    base.update(overrides)
    return CaseObservation.model_validate(base)


def test_dataset_loads_and_ids_are_unique() -> None:
    cases = load_cases()
    assert len(cases) >= 50
    assert {case.suite for case in cases} == {"report", "booking", "conversation", "agent"}
    assert all(case.agent_mode == "full" for case in cases if case.suite == "agent")


async def test_agent_cases_are_scored_without_false_hard_zero_flags() -> None:
    """Case chế độ agent chạy qua harness thật: slot trên thẻ xác nhận được tính là đã đề xuất (không bị coi là
    hallucination), dời/hủy được đếm đúng, và một agent làm đúng thì case đạt."""
    from tests.integration.test_booking_agent import same_as_current_plus_7
    from tests.integration.test_full_agent import _nothing_left

    cases = [case for case in load_cases() if case.id in {"agent-same-plus7", "agent-cancel-declined"}]

    async def cancel_flow(tools: Any, history: Any, context: str) -> Any:
        current = (await call_tool(tools, "find_my_appointments", when="upcoming", department_code=None))["data"][0]
        await call_tool(tools, "propose_cancel", appointment_id=current["id"], basis="Theo lời bạn")
        return AgentReply(text="", stopped_by="propose_cancel")

    scripts = {"Đặt cho": [same_as_current_plus_7, _nothing_left], "hủy giúp": [cancel_flow]}

    class Inner(ScriptedLLM):
        async def run_agent(self, task: Any, context: str, history: Any, tools: Any, max_tool_calls: int) -> Any:
            if not self.agents:
                first = next(message.text for message in history if message.role == "user")
                self.agents = list(next(v for k, v in scripts.items() if first.startswith(k)))
            return await super().run_agent(task, context, history, tools, max_tool_calls)

    observations = await run_cases(cases, Inner, postgres_settings(), concurrency=2)
    report = compute(cases, observations)
    assert [v.failures for v in report.verdicts] == [[], []]
    assert hard_zero_violations(report) == []


def test_false_success_detection() -> None:
    # Nói đã đặt nhưng backend không có lịch hẹn nào.
    assert is_false_success(_observation(outcome="appointment_booked", response_text="Đã đặt lịch: ..."))
    # Kết quả lỗi nhưng câu trả lời vẫn khẳng định thành công.
    assert is_false_success(_observation(outcome="error", response_text="Đã lưu báo cáo kỳ 2026-W44"))
    assert not is_false_success(
        _observation(outcome="appointment_booked", booked_slot_ids=["s1"], response_text="Đã đặt")
    )
    assert not is_false_success(_observation(outcome="error", response_text="Đặt lịch KHÔNG thành công"))


def test_hard_zero_metrics_are_counted() -> None:
    case = _case()
    report = compute(
        [case],
        [
            _observation(
                outcome="appointment_booked",
                booked_slot_ids=["ghost"],
                hallucinated_slot_ids=["ghost"],
                double_booked_slot_ids=["ghost"],
            )
        ],
    )
    assert report.slot_hallucination.numerator == 1
    assert report.double_booking.numerator == 1
    assert set(hard_zero_violations(report)) == {"slot_hallucination", "double_booking"}
    assert not report.verdicts[0].passed


def test_full_mode_scoring_adapts_workflow_cases_without_hiding_failures() -> None:
    """Case viết cho workflow chạy ở chế độ full: không chấm intent, trả lời chat thay cho kiểu outcome đọc dữ liệu,
    câu hỏi chat = hỏi lại; nhưng hỏi thừa, sai trường và ghi bừa vẫn là trượt."""
    listing = _case(
        id="list",
        suite="conversation",
        expected={
            "intent": "list_appointments",
            "outcome": "appointments",
            "tools": ["list_my_appointments"],
            "response_contains": ["Lịch hẹn của bạn"],
        },
    )
    ask = _case(id="ask", suite="report", expected={"outcome": "waiting", "should_clarify": True})
    booked = _case(
        id="book",
        expected={"outcome": "appointment_booked", "fields": {"date_from": "2026-11-05", "department_from_plan": True}},
    )
    extra_question = _case(id="extra", suite="report", expected={"outcome": "report_saved"})
    choose = _case(id="choose", expected={"outcome": "appointment_booked"})
    full = {"agent_mode": "full"}
    report = compute(
        [listing, ask, booked, extra_question, choose],
        [
            _observation(case_id="list", outcome="chat", tool_calls=["find_my_appointments"], **full),
            _observation(
                case_id="ask", outcome="chat", interrupts=["chat_question_open"], ended_with_question=True, **full
            ),
            _observation(
                case_id="book",
                outcome="appointment_booked",
                booked_slot_ids=["s1"],
                final_fields={"department_code": "PHCN", "slot_date": "2026-11-04"},  # sớm hơn khoảng ngày
                **full,
            ),
            _observation(case_id="extra", status="waiting", interrupts=["chat_question", "confirm_report"], **full),
            _observation(
                case_id="choose",
                outcome="appointment_booked",
                booked_slot_ids=["s1"],
                interrupts=["chat_choose_slot", "confirm_booking"],  # liệt kê slot cho chọn không phải hỏi lại
                **full,
            ),
        ],
    )
    assert next(v for v in report.verdicts if v.case_id == "choose").passed
    by_id = {v.case_id: v for v in report.verdicts}
    assert by_id["list"].passed and by_id["list"].notes == ["response missing 'Lịch hẹn của bạn'"]
    assert by_id["ask"].passed
    assert by_id["book"].failures == ["field date_from: expected '2026-11-05', got '2026-11-04'"]
    assert by_id["book"].notes == ["field department_from_plan không áp dụng ở chế độ full"]
    assert "clarify: expected False, got True" in by_id["extra"].failures
    assert report.intent_accuracy.denominator == 0
    # Cùng quan sát nhưng ở chế độ workflow: chat không thay được kiểu outcome `appointments`.
    workflow = compute([listing], [_observation(case_id="list", outcome="chat", tool_calls=["list_my_appointments"])])
    assert not workflow.verdicts[0].passed


def test_validator_confusion_matrix_and_self_correction() -> None:
    injected = _case(
        id="inj",
        suite="report",
        expected={"outcome": "report_saved"},
        labels={"injected_error": True, "issue_codes": ["value_evidence_mismatch"]},
    )
    clean = _case(id="clean", suite="report", expected={"outcome": "report_saved"})
    observations = [
        # lỗi tiêm vào nhưng validator không bắt được -> FN
        _observation(
            case_id="inj",
            suite="report",
            outcome="report_saved",
            attempts=[AttemptView(attempt_no=0, issue_codes=[], passed=True)],
        ),
        # đầu ra sạch nhưng bị gắn cờ -> FP
        _observation(
            case_id="clean",
            suite="report",
            outcome="report_saved",
            attempts=[AttemptView(attempt_no=0, issue_codes=["evidence_missing"], passed=False)],
        ),
    ]
    report = compute([injected, clean], observations)
    assert (report.validator["fn"], report.validator["fp"]) == (1, 1)
    assert report.validator["issue_code_recall"] == 0.0


def _bedrock() -> Any:
    return build_llm(Settings())


@pytest.mark.llm
async def test_full_eval_has_no_crash_and_no_hard_zero_violation() -> None:
    """Chạy toàn bộ dataset với Claude thật (tốn phí): `pytest -m llm`."""
    cases = load_cases()
    report = compute(cases, await run_cases(cases, _bedrock, postgres_settings()))
    assert report.crashed == 0
    assert hard_zero_violations(report) == []
    assert "## Metric bắt buộc bằng 0" in to_markdown(report, {"model": Settings().llm_model_id})


@pytest.mark.llm
async def test_eval_catches_a_disabled_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test: tắt validator báo cáo thì eval phải phát hiện (FN tăng, case thất bại)."""
    monkeypatch.setattr("patient_assistant.application.agent.nodes.report.validate_report", lambda *_: [])
    cases = [case for case in load_cases() if case.id in {"rep-inj-01", "rep-range-01", "rep-miss-01"}]
    report = compute(cases, await run_cases(cases, _bedrock, postgres_settings()))
    assert report.validator["fn"] >= 2
    assert report.case_pass_rate.numerator < report.case_pass_rate.denominator
