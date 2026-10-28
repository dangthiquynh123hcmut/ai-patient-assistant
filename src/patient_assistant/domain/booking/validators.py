"""Validator cho yêu cầu đặt lịch. Thuần code, không gọi LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from patient_assistant.domain.booking.models import BookingRequest, Department, Doctor
from patient_assistant.domain.booking.policy import BookingPolicy, window_issues
from patient_assistant.domain.common import Layer, Resolution, SttFlag, ValidationIssue
from patient_assistant.domain.text.grounding import overlapping_flags, quote_in_sources
from patient_assistant.domain.text.normalize import contains_phrase, normalize
from patient_assistant.domain.text.vi_dates import DateResolution

VERBATIM_FIELDS = ("doctor_name", "date_expression", "time_expression", "reason")


@dataclass(frozen=True)
class BookingValidationContext:
    sources: list[str]
    now: datetime
    policy: BookingPolicy
    departments: list[Department]
    doctors: list[Doctor]
    stt_flags: list[SttFlag] = field(default_factory=list)
    department_from_plan: bool = False


def parse_booking(raw: dict[str, Any]) -> tuple[BookingRequest | None, list[ValidationIssue]]:
    try:
        return BookingRequest.model_validate(raw), []
    except ValidationError as exc:
        return None, [
            ValidationIssue(
                field=str(err["loc"][0]) if err["loc"] else None,
                layer=Layer.SCHEMA,
                code="invalid_schema",
                message=err["msg"],
                suggestion="Trả đúng schema; dùng null nếu không có thông tin.",
                resolution=Resolution.LLM_FIXABLE,
            )
            for err in exc.errors()
        ]


def find_department(code: str | None, departments: list[Department]) -> Department | None:
    return next((d for d in departments if d.code == code), None)


def find_doctor(name: str, department_code: str | None, doctors: list[Doctor]) -> Doctor | None:
    """Khớp tên bác sĩ theo chuỗi con đã chuẩn hóa; chỉ trả về khi khớp đúng một người."""
    wanted = normalize(name).removeprefix("bác sĩ ").removeprefix("bs ").strip()
    if not wanted:
        return None
    pool = [d for d in doctors if department_code is None or d.department_code == department_code]
    matches = [d for d in pool if f" {wanted} " in f" {normalize(d.name)} "]
    return matches[0] if len(matches) == 1 else None


def _department_issues(req: BookingRequest, ctx: BookingValidationContext) -> list[ValidationIssue]:
    if req.department_code is None:
        return [
            ValidationIssue(
                field="department_code",
                layer=Layer.BUSINESS,
                code="missing_field",
                message="Chưa có khoa muốn khám.",
                resolution=Resolution.ASK_USER,
            )
        ]
    dept = find_department(req.department_code, ctx.departments)
    if dept is None:
        return [
            ValidationIssue(
                field="department_code",
                layer=Layer.SCHEMA,
                code="unknown_department",
                message=f"Mã khoa '{req.department_code}' không có trong danh mục.",
                suggestion="Chỉ dùng mã khoa trong danh mục, hoặc null nếu người dùng không nêu khoa.",
                resolution=Resolution.LLM_FIXABLE,
            )
        ]
    if ctx.department_from_plan:
        return []
    # Khoa phải do người dùng nêu — không được suy ra từ triệu chứng (FR-09.11).
    if not contains_phrase(" ".join(ctx.sources), dept.names()):
        return [
            ValidationIssue(
                field="department_code",
                layer=Layer.GROUNDING,
                code="department_not_in_source",
                message=f"Người dùng không nêu khoa '{dept.name}'.",
                suggestion="Không suy ra khoa từ triệu chứng; đặt department_code = null.",
                resolution=Resolution.LLM_FIXABLE,
            )
        ]
    return []


def _verbatim_issues(req: BookingRequest, ctx: BookingValidationContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name in VERBATIM_FIELDS:
        value = getattr(req, name)
        if not value:
            continue
        if not quote_in_sources(value, ctx.sources):
            issues.append(
                ValidationIssue(
                    field=name,
                    layer=Layer.GROUNDING,
                    code="expression_not_in_source",
                    message=f"'{value}' không phải nguyên văn lời người dùng.",
                    suggestion="Trích nguyên văn cụm trong transcript, không diễn giải hay quy đổi.",
                    resolution=Resolution.LLM_FIXABLE,
                )
            )
            continue
        flags = overlapping_flags(value, ctx.stt_flags)
        if flags:
            issues.append(
                ValidationIssue(
                    field=name,
                    layer=Layer.GROUNDING,
                    code="stt_uncertain",
                    message=f"Đoạn '{value}' có thể bị nhận dạng giọng nói sai ({flags[0].text}).",
                    resolution=Resolution.ASK_USER,
                )
            )
    return issues


def validate_booking(
    req: BookingRequest, resolution: DateResolution, ctx: BookingValidationContext
) -> list[ValidationIssue]:
    issues = _department_issues(req, ctx) + _verbatim_issues(req, ctx)
    # Lỗi do LLM trích sai phải sửa trước; chưa cần xét ngày khi cụm ngày còn chưa đúng nguyên văn.
    if any(i.resolution == Resolution.LLM_FIXABLE for i in issues):
        return issues
    if req.date_expression is None and resolution.window is None:
        issues.append(
            ValidationIssue(
                field="date_expression",
                layer=Layer.BUSINESS,
                code="missing_field",
                message="Chưa có ngày muốn khám.",
                resolution=Resolution.ASK_USER,
            )
        )
    else:
        issues += resolution.issues
        if resolution.window:
            issues += window_issues(resolution.window, ctx.now, ctx.policy)
    if req.doctor_name and find_doctor(req.doctor_name, req.department_code, ctx.doctors) is None:
        issues.append(
            ValidationIssue(
                field="doctor_name",
                layer=Layer.CONSISTENCY,
                code="doctor_not_found",
                message=f"Không tìm thấy duy nhất một bác sĩ '{req.doctor_name}' trong khoa đã chọn.",
                resolution=Resolution.ASK_USER,
            )
        )
    return issues
