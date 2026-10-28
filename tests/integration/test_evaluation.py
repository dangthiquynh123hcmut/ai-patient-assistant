"""Kiểm tra chính bộ đánh giá: metric phải bắt được lỗi, không chỉ cho ra con số đẹp."""

from __future__ import annotations

from typing import Any

import pytest

from patient_assistant.evaluation.cases import EvalCase, load_cases
from patient_assistant.evaluation.harness import AttemptView, CaseObservation, run_cases
from patient_assistant.evaluation.metrics import compute, is_false_success
from patient_assistant.evaluation.report import hard_zero_violations, to_markdown
from patient_assistant.infrastructure.llm.offline import OfflineRuleLLM


def _case(**kw: Any) -> EvalCase:
    base: dict[str, Any] = {
        "id": "c1",
        "suite": "booking",
        "category": "t",
        "message": {"text": "x"},
        "expected": {"outcome": "appointment_booked"},
    }
    base.update(kw)
    return EvalCase.model_validate(base)


def _obs(**kw: Any) -> CaseObservation:
    base: dict[str, Any] = {"case_id": "c1", "suite": "booking", "category": "t", "status": "done"}
    base.update(kw)
    return CaseObservation.model_validate(base)


def test_dataset_loads_and_ids_are_unique() -> None:
    cases = load_cases()
    assert len(cases) >= 50
    assert {c.suite for c in cases} == {"report", "booking", "conversation"}


def test_false_success_detection() -> None:
    # Nói đã đặt nhưng backend không có lịch hẹn nào.
    assert is_false_success(_obs(outcome="appointment_booked", response_text="Đã đặt lịch: ..."))
    # Kết quả lỗi nhưng câu trả lời vẫn khẳng định thành công.
    assert is_false_success(_obs(outcome="error", response_text="Đã lưu báo cáo kỳ 2026-W44"))
    assert not is_false_success(_obs(outcome="appointment_booked", booked_slot_ids=["s1"], response_text="Đã đặt"))
    assert not is_false_success(_obs(outcome="error", response_text="Đặt lịch KHÔNG thành công"))


def test_hard_zero_metrics_are_counted() -> None:
    case = _case()
    report = compute(
        [case],
        [
            _obs(
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
        _obs(
            case_id="inj",
            suite="report",
            outcome="report_saved",
            attempts=[AttemptView(attempt_no=0, issue_codes=[], passed=True)],
        ),
        # đầu ra sạch nhưng bị gắn cờ -> FP
        _obs(
            case_id="clean",
            suite="report",
            outcome="report_saved",
            attempts=[AttemptView(attempt_no=0, issue_codes=["evidence_missing"], passed=False)],
        ),
    ]
    report = compute([injected, clean], observations)
    assert (report.validator["fn"], report.validator["fp"]) == (1, 1)
    assert report.validator["issue_code_recall"] == 0.0


async def test_full_offline_eval_has_no_crash_and_no_hard_zero_violation() -> None:
    cases = load_cases()
    observations = await run_cases(cases, OfflineRuleLLM)
    report = compute(cases, observations)
    assert report.crashed == 0
    assert hard_zero_violations(report) == []
    assert report.case_pass_rate.numerator == report.case_pass_rate.denominator, [
        (v.case_id, v.failures) for v in report.verdicts if not v.passed
    ]
    assert "## Metric bắt buộc bằng 0" in to_markdown(report, {"llm": "offline"})


async def test_eval_catches_a_disabled_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test: tắt validator báo cáo thì eval phải phát hiện (FN tăng, case thất bại)."""
    monkeypatch.setattr("patient_assistant.application.agent.nodes.report.validate_report", lambda *_: [])
    cases = [c for c in load_cases() if c.id in {"rep-inj-01", "rep-range-01", "rep-miss-01"}]
    report = compute(cases, await run_cases(cases, OfflineRuleLLM))
    assert report.validator["fn"] >= 2
    assert report.case_pass_rate.numerator < report.case_pass_rate.denominator
