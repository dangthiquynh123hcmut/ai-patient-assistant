"""Quy tắc nghiệp vụ của báo cáo. Dùng chung cho validator của agent và service lưu ở backend (FR-04.7)."""

from __future__ import annotations

import re
from datetime import date, datetime

from patient_assistant.domain.common import Layer, Resolution, ValidationIssue
from patient_assistant.domain.report.models import ReportExtraction

PAIN_RANGE = (0, 10)
EXERCISE_DAYS_RANGE = (0, 7)
REQUIRED_FIELDS = ("pain_level", "exercise_days")
_ISO_WEEK = re.compile(r"^(\d{4})-W(\d{2})$")


def current_period(now: datetime) -> str:
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def range_issue(
    field: str, value: int | None, bounds: tuple[int, int], resolution: Resolution
) -> ValidationIssue | None:
    if value is None or bounds[0] <= value <= bounds[1]:
        return None
    return ValidationIssue(
        field=field,
        layer=Layer.BUSINESS,
        code="out_of_range",
        message=f"{field}={value} nằm ngoài phạm vi {bounds[0]}–{bounds[1]}.",
        resolution=resolution,
    )


def period_issue(period: str | None, now: datetime) -> ValidationIssue | None:
    if period is None:
        return None
    m = _ISO_WEEK.match(period)
    if not m:
        return ValidationIssue(
            field="reporting_period",
            layer=Layer.SCHEMA,
            code="invalid_format",
            message=f"reporting_period '{period}' không đúng định dạng YYYY-Www.",
            suggestion="Dùng ISO week, ví dụ 2026-W44, hoặc null nếu người dùng không nêu.",
            resolution=Resolution.LLM_FIXABLE,
        )
    try:
        start = date.fromisocalendar(int(m.group(1)), int(m.group(2)), 1)
    except ValueError:
        return ValidationIssue(
            field="reporting_period",
            layer=Layer.SCHEMA,
            code="invalid_format",
            message=f"Tuần '{period}' không tồn tại.",
            resolution=Resolution.LLM_FIXABLE,
        )
    if start > now.date():
        return ValidationIssue(
            field="reporting_period",
            layer=Layer.BUSINESS,
            code="period_in_future",
            message="Kỳ báo cáo ở tương lai.",
            resolution=Resolution.ASK_USER,
        )
    return None


def backend_issues(draft: ReportExtraction, now: datetime) -> list[ValidationIssue]:
    """Kiểm tra tối thiểu backend luôn chạy lại khi lưu, bất kể agent đã validate gì."""
    issues: list[ValidationIssue] = []
    for name in REQUIRED_FIELDS:
        if getattr(draft, name) is None:
            issues.append(
                ValidationIssue(
                    field=name,
                    layer=Layer.BUSINESS,
                    code="missing_field",
                    message=f"Thiếu {name}.",
                    resolution=Resolution.ASK_USER,
                )
            )
    for issue in (
        range_issue("pain_level", draft.pain_level, PAIN_RANGE, Resolution.ASK_USER),
        range_issue("exercise_days", draft.exercise_days, EXERCISE_DAYS_RANGE, Resolution.ASK_USER),
        period_issue(draft.reporting_period, now),
    ):
        if issue:
            issues.append(issue)
    return issues
