"""Xuất báo cáo eval ra Markdown (đọc) và JSON (so sánh giữa các lần chạy)."""

from __future__ import annotations

from typing import Any

from patient_assistant.evaluation.harness import CaseObservation
from patient_assistant.evaluation.metrics import EvalReport, Rate

HARD_ZERO = ("false_success", "slot_hallucination", "double_booking")


def _fmt(value: Any) -> str:
    if isinstance(value, Rate):
        if value.value is None:
            return "— (0/0)"
        return f"{value.value:.1%} ({value.numerator}/{value.denominator})"
    if isinstance(value, float):
        return f"{value:.1%}"
    return "—" if value is None else str(value)


def hard_zero_violations(report: EvalReport) -> list[str]:
    return [name for name in HARD_ZERO if getattr(report, name).numerator > 0]


def to_markdown(report: EvalReport, meta: dict[str, Any]) -> str:
    lines = [
        "# Báo cáo đánh giá AI Patient Assistant",
        "",
        " · ".join(f"**{k}**: {v}" for k, v in meta.items()),
        "",
        "## Tổng quan",
        "",
        "| Metric | Giá trị |",
        "|---|---|",
        f"| Case đạt | {_fmt(report.case_pass_rate)} |",
        f"| Crash của harness | {report.crashed} |",
        f"| Intent đúng | {_fmt(report.intent_accuracy)} |",
        f"| Outcome đúng | {_fmt(report.outcome_accuracy)} |",
        f"| Latency p50 / p95 / max | {report.latency_ms['p50']} / {report.latency_ms['p95']} / "
        f"{report.latency_ms['max']} ms |",
        "",
        "## Metric bắt buộc bằng 0",
        "",
        "| Metric | Giá trị | Trạng thái |",
        "|---|---|---|",
    ]
    for name in HARD_ZERO:
        rate: Rate = getattr(report, name)
        lines.append(f"| {name} | {_fmt(rate)} | {'✅' if rate.numerator == 0 else '❌'} |")
    lines += [
        "",
        "## Trích xuất (field-level)",
        "",
        "| Trường | Accuracy |",
        "|---|---|",
        *[f"| {k} | {_fmt(v)} |" for k, v in report.field_accuracy.items()],
        f"| **Tổng (giá trị cuối)** | {_fmt(report.field_accuracy_overall)} |",
        f"| Lần trích xuất đầu tiên | {_fmt(report.first_attempt_field_accuracy)} |",
        "",
        "## Function calling",
        "",
        "| Metric | Giá trị |",
        "|---|---|",
        f"| Chọn đúng tool (đủ tool bắt buộc, không gọi tool ghi bị cấm) | {_fmt(report.tool_selection_accuracy)} |",
        f"| Recall tool bắt buộc | {_fmt(report.tool_recall)} |",
        f"| Tỷ lệ tool call không cần thiết | {_fmt(report.unnecessary_tool_call_rate)} |",
        f"| Tham số ghi đúng (bản lưu / slot đặt) | {_fmt(report.write_argument_accuracy)} |",
        "",
        "## Validator (lần trích xuất đầu tiên)",
        "",
        "| TP | FP | FN | TN | Precision | Recall | FP rate | FN rate | Bắt đúng mã lỗi |",
        "|---|---|---|---|---|---|---|---|---|",
        "| "
        + " | ".join(
            _fmt(report.validator[k])
            for k in (
                "tp",
                "fp",
                "fn",
                "tn",
                "precision",
                "recall",
                "false_positive_rate",
                "false_negative_rate",
                "issue_code_recall",
            )
        )
        + " |",
        "",
        "## Self-correction (case có lỗi LLM tiêm vào)",
        "",
        f"- Số case: {report.self_correction['cases']}",
        f"- Sửa thành công: {_fmt(report.self_correction['success_rate'])}",
        f"- Sửa lỗi này nhưng sinh lỗi khác: {_fmt(report.self_correction['regression_rate'])}",
        f"- Số lần thử trung bình (kể cả lần đầu): {report.self_correction['avg_attempts'] or '—'}",
        "",
        "## Hỏi lại",
        "",
        f"- Case cần hỏi lại: {report.clarification['should_clarify']}",
        f"- Hỏi lại khi cần: {_fmt(report.clarification['clarified_when_needed'])}",
        f"- Tự đoán khi thiếu/mơ hồ: {_fmt(report.clarification['guess_rate'])}",
        f"- Hỏi lại thừa: {_fmt(report.clarification['unnecessary_clarification_rate'])}",
        "",
        "## Theo suite",
        "",
        *[f"- {k}: {_fmt(v)}" for k, v in report.by_suite.items()],
        "",
    ]
    failed = [v for v in report.verdicts if not v.passed]
    lines += ["## Case không đạt", ""]
    if not failed:
        lines.append("Không có.")
    for v in failed:
        lines.append(f"- **{v.case_id}** (session {v.session}, {v.category}): " + "; ".join(v.failures))
    return "\n".join(lines) + "\n"


def to_json(report: EvalReport, observations: list[CaseObservation], meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": meta,
        "report": report.model_dump(mode="json"),
        "observations": [o.model_dump(mode="json") for o in observations],
    }
