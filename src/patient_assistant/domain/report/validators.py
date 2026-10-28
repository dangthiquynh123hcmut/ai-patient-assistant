"""Validator nhiều lớp cho báo cáo. Thuần code, không gọi LLM (FR-04)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from patient_assistant.domain.common import (
    Layer,
    Resolution,
    SttFlag,
    ValidationIssue,
    evidence_for,
)
from patient_assistant.domain.report.models import REPORT_FIELDS, ReportExtraction, SymptomChange
from patient_assistant.domain.report.phrases import latest_direction, mentions_no_exercise, mentions_positive_exercise
from patient_assistant.domain.report.rules import (
    EXERCISE_DAYS_RANGE,
    PAIN_RANGE,
    REQUIRED_FIELDS,
    period_issue,
    range_issue,
)
from patient_assistant.domain.text.grounding import overlapping_flags, quote_in_sources
from patient_assistant.domain.text.vi_numbers import number_supported

NUMERIC_BOUNDS = {"pain_level": PAIN_RANGE, "exercise_days": EXERCISE_DAYS_RANGE}


@dataclass(frozen=True)
class ReportValidationContext:
    sources: list[str]
    now: datetime
    stt_flags: list[SttFlag] = field(default_factory=list)
    edited_fields: frozenset[str] = frozenset()


def parse_report(raw: dict[str, Any]) -> tuple[ReportExtraction | None, list[ValidationIssue]]:
    """Lớp 1 — schema. Chuyển lỗi pydantic thành issue có trường và gợi ý sửa."""
    try:
        return ReportExtraction.model_validate(raw), []
    except ValidationError as exc:
        return None, [
            ValidationIssue(
                field=str(err["loc"][0]) if err["loc"] else None,
                layer=Layer.SCHEMA,
                code="invalid_schema",
                message=err["msg"],
                suggestion="Trả đúng kiểu dữ liệu theo schema; dùng null nếu không có thông tin.",
                resolution=Resolution.LLM_FIXABLE,
            )
            for err in exc.errors()
        ]


def _grounding_issues(draft: ReportExtraction, ctx: ReportValidationContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name in REPORT_FIELDS:
        value = getattr(draft, name)
        if value is None:
            continue
        if name in ctx.edited_fields:
            # Người dùng sửa trực tiếp: không cần trích dẫn, nhưng vẫn phải đúng phạm vi.
            if name in NUMERIC_BOUNDS and (
                out := range_issue(name, int(value), NUMERIC_BOUNDS[name], Resolution.ASK_USER)
            ):
                issues.append(out)
            continue
        quote = evidence_for(draft.evidence, name)
        if not quote:
            issues.append(
                ValidationIssue(
                    field=name,
                    layer=Layer.GROUNDING,
                    code="evidence_missing",
                    message=f"Trường {name} có giá trị nhưng không có trích dẫn.",
                    suggestion="Thêm trích dẫn nguyên văn từ lời người dùng, hoặc đặt giá trị null.",
                    resolution=Resolution.LLM_FIXABLE,
                )
            )
            continue
        if not quote_in_sources(quote, ctx.sources):
            issues.append(
                ValidationIssue(
                    field=name,
                    layer=Layer.GROUNDING,
                    code="evidence_not_found",
                    message=f"Trích dẫn '{quote}' không có trong lời người dùng.",
                    suggestion="Chỉ trích nguyên văn đoạn có trong transcript.",
                    resolution=Resolution.LLM_FIXABLE,
                )
            )
            continue
        flags = overlapping_flags(quote, ctx.stt_flags)
        if flags:
            issues.append(
                ValidationIssue(
                    field=name,
                    layer=Layer.GROUNDING,
                    code="stt_uncertain",
                    message=f"Đoạn '{quote}' có thể bị nhận dạng giọng nói sai ({flags[0].text}).",
                    resolution=Resolution.ASK_USER,
                )
            )
            continue
        if name in NUMERIC_BOUNDS:
            if not number_supported(int(value), quote):
                issues.append(
                    ValidationIssue(
                        field=name,
                        layer=Layer.GROUNDING,
                        code="value_evidence_mismatch",
                        message=f"Giá trị {name}={value} không khớp với trích dẫn '{quote}'.",
                        suggestion="Lấy đúng con số trong trích dẫn.",
                        resolution=Resolution.LLM_FIXABLE,
                    )
                )
                continue
            # Giá trị đúng như lời người dùng nhưng ngoài phạm vi -> hỏi lại, không "sửa" cho qua.
            out = range_issue(name, int(value), NUMERIC_BOUNDS[name], Resolution.ASK_USER)
            if out:
                issues.append(out)
    return issues


def _consistency_issues(draft: ReportExtraction, ctx: ReportValidationContext) -> list[ValidationIssue]:
    """Lời nói gần nhất thắng: người dùng đính chính ở câu trả lời sau thì không coi là mâu thuẫn nữa."""
    issues: list[ValidationIssue] = []
    latest = ctx.sources[-1] if ctx.sources else ""
    if (
        "exercise_days" not in ctx.edited_fields
        and mentions_no_exercise(latest)
        and (draft.exercise_days or mentions_positive_exercise(latest))
    ):
        issues.append(
            ValidationIssue(
                field="exercise_days",
                layer=Layer.CONSISTENCY,
                code="contradiction",
                message="Lời bạn vừa nói có cả ý chưa tập và có số ngày tập.",
                resolution=Resolution.ASK_USER,
            )
        )
    direction = latest_direction(ctx.sources)
    if "symptom_change" not in ctx.edited_fields and draft.symptom_change is not None and direction:
        worse = draft.symptom_change in (SymptomChange.WORSE, SymptomChange.WORSE_DURING_ACTIVITY)
        expected_worse = direction == "worse"
        if worse != expected_worse or (direction == "better" and draft.symptom_change != SymptomChange.BETTER):
            issues.append(
                ValidationIssue(
                    field="symptom_change",
                    layer=Layer.CONSISTENCY,
                    code="contradicts_source",
                    message=f"Lời người dùng cho thấy triệu chứng '{direction}' nhưng symptom_change="
                    f"{draft.symptom_change.value}.",
                    suggestion="Chọn symptom_change đúng với mô tả của người dùng.",
                    resolution=Resolution.LLM_FIXABLE,
                )
            )
    stale = [f for f in draft.missing_fields if f in REPORT_FIELDS and getattr(draft, f) is not None]
    if stale:
        issues.append(
            ValidationIssue(
                field=stale[0],
                layer=Layer.CONSISTENCY,
                code="missing_fields_inconsistent",
                message=f"Trường {', '.join(stale)} vừa có giá trị vừa nằm trong missing_fields.",
                resolution=Resolution.LLM_FIXABLE,
            )
        )
    return issues


def validate_report(draft: ReportExtraction, ctx: ReportValidationContext) -> list[ValidationIssue]:
    """Lớp 2–4 (business, grounding, consistency). Lớp 1 là `parse_report`; lớp 6 ở `domain.safety`."""
    issues: list[ValidationIssue] = []
    for name in REQUIRED_FIELDS:
        if getattr(draft, name) is None:
            issues.append(
                ValidationIssue(
                    field=name,
                    layer=Layer.BUSINESS,
                    code="missing_field",
                    message=f"Chưa có thông tin {name}.",
                    resolution=Resolution.ASK_USER,
                )
            )
    period = period_issue(draft.reporting_period, ctx.now)
    if period:
        issues.append(period)
    issues += _grounding_issues(draft, ctx)
    issues += _consistency_issues(draft, ctx)
    return issues
