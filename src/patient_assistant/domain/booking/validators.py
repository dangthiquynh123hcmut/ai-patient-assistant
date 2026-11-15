"""Validator cho yêu cầu đặt lịch. Thuần code, không gọi LLM."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from patient_assistant.domain.booking.models import BookingRequest, Department, Doctor
from patient_assistant.domain.booking.policy import BookingPolicy, window_issues
from patient_assistant.domain.common import Layer, Resolution, SttFlag, ValidationIssue
from patient_assistant.domain.text.grounding import overlapping_flags, quote_in_sources
from patient_assistant.domain.text.normalize import contains_phrase, normalize, unaccent
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
    department_from_memory: bool = False  # khoa lấy từ lịch hẹn/lượt trước theo cụm nhắc lại, code đã kiểm


def parse_booking(raw: dict[str, Any]) -> tuple[BookingRequest | None, list[ValidationIssue]]:
    try:
        return BookingRequest.model_validate(raw), []
    except ValidationError as exception:
        return None, [
            ValidationIssue(
                field=str(error["loc"][0]) if error["loc"] else None,
                layer=Layer.SCHEMA,
                code="invalid_schema",
                message=error["msg"],
                suggestion="Trả đúng schema; dùng null nếu không có thông tin.",
                resolution=Resolution.LLM_FIXABLE,
            )
            for error in exception.errors()
        ]


def find_department(code: str | None, departments: list[Department]) -> Department | None:
    return next((department for department in departments if department.code == code), None)


_DEPARTMENT_LEAD_WORDS = {"khoa", "chuyen", "kham", "phong"}  # "khoa nội", "chuyên khoa mắt", "khám mắt" (đã bỏ dấu)


def _fold(text: str) -> list[str]:
    """Chữ thường, bỏ dấu, bỏ chữ dẫn ("khoa", "khám"...): "Khoa Nội tổng" -> ["noi", "tong"]."""
    decomposed = unicodedata.normalize("NFD", normalize(text).replace("đ", "d"))
    words = "".join(character for character in decomposed if unicodedata.category(character) != "Mn").split()
    while words and words[0] in _DEPARTMENT_LEAD_WORDS:
        words = words[1:]
    return words


def _names_department(said: list[str], name: list[str]) -> bool:
    if not said or not name:
        return False
    # Gõ thiếu: "noi", "noi tong", "noi tong qu" là phần đầu của "noi tong quat".
    if len(said) <= len(name) and said[:-1] == name[: len(said) - 1] and name[len(said) - 1].startswith(said[-1]):
        return True
    # Gõ thừa: "noi tong quat a" có chứa trọn tên khoa.
    return any(said[i : i + len(name)] == name for i in range(len(said) - len(name) + 1))


def departments_matching(mention: str, departments: list[Department]) -> list[Department]:
    """Các khoa mà cụm người dùng nói có thể chỉ tới: đúng tên/cách gọi khác, gõ thiếu ("nội tổng"), không dấu.

    Đây vẫn là khoa NGƯỜI DÙNG NÊU (so với tên khoa), không phải suy ra từ triệu chứng.
    """
    said = _fold(mention)
    if len("".join(said)) < 2:
        return []
    return [
        department
        for department in departments
        if any(_names_department(said, _fold(department_name)) for department_name in department.names())
    ]


def department_in_text(text: str, departments: list[Department]) -> Department | None:
    """Khoa người dùng nêu trong câu; nhiều khoa cùng khớp thì lấy cụm dài nhất
    ("khoa nội thần kinh" -> Nội thần kinh, không phải Nội tổng quát qua "khoa nội")."""
    hits = [(len(hit), department) for department in departments if (hit := contains_phrase(text, department.names()))]
    return max(hits, key=lambda scored_hit: scored_hit[0])[1] if hits else None


def find_doctor(name: str, department_code: str | None, doctors: list[Doctor]) -> Doctor | None:
    """Khớp tên bác sĩ theo chuỗi con đã chuẩn hóa; chỉ trả về khi khớp đúng một người."""
    wanted = normalize(name)
    for prefix in ("bác sĩ ", "bs ", "doctor ", "dr "):
        wanted = wanted.removeprefix(prefix)
    wanted = wanted.strip()
    if not wanted:
        return None
    pool = [doctor for doctor in doctors if department_code is None or doctor.department_code == department_code]
    matches = [doctor for doctor in pool if f" {wanted} " in f" {normalize(doctor.name)} "]
    if not matches:  # gõ/nói không dấu hoặc tiếng Anh: "doctor Binh"
        matches = [doctor for doctor in pool if f" {unaccent(wanted)} " in f" {unaccent(doctor.name)} "]
    return matches[0] if len(matches) == 1 else None


def _ambiguous_department(mention: str, candidates: list[Department]) -> ValidationIssue:
    return ValidationIssue(
        field="department_code",
        layer=Layer.BUSINESS,
        code="department_ambiguous",
        message=f"“{mention}” có thể là khoa {' hoặc '.join(department.name for department in candidates)}.",
        resolution=Resolution.ASK_USER,
    )


def _department_issues(request: BookingRequest, context: BookingValidationContext) -> list[ValidationIssue]:
    if request.department_code is None and request.department_mentioned:
        candidates = departments_matching(request.department_mentioned, context.departments)
        if len(candidates) > 1:
            return [_ambiguous_department(request.department_mentioned, candidates)]
        # Người dùng nêu một khoa phòng khám không có -> nói rõ là không có, mời chọn lại (không hỏi chung chung).
        return [
            ValidationIssue(
                field="department_code",
                layer=Layer.BUSINESS,
                code="department_not_offered",
                message=f"Hiện phòng khám chưa có “{request.department_mentioned}”.",
                resolution=Resolution.ASK_USER,
            )
        ]
    if request.department_code is None:
        return [
            ValidationIssue(
                field="department_code",
                layer=Layer.BUSINESS,
                code="missing_field",
                message="Chưa có khoa muốn khám.",
                resolution=Resolution.ASK_USER,
            )
        ]
    department = find_department(request.department_code, context.departments)
    if department is None:
        return [
            ValidationIssue(
                field="department_code",
                layer=Layer.SCHEMA,
                code="unknown_department",
                message=f"Mã khoa '{request.department_code}' không có trong danh mục.",
                suggestion="Chỉ dùng mã khoa trong danh mục, hoặc null nếu người dùng không nêu khoa.",
                resolution=Resolution.LLM_FIXABLE,
            )
        ]
    if context.department_from_plan or context.department_from_memory:
        return []
    # Khoa phải do người dùng nêu — không được suy ra từ triệu chứng (FR-09.11). Người dùng có thể gõ tắt/không
    # dấu ("nội tổng"): chấp nhận khi cụm khoa họ nói (nguyên văn trong lời họ) chỉ đúng khoa này.
    mention = request.department_mentioned
    if mention and quote_in_sources(mention, context.sources):
        candidates = departments_matching(mention, context.departments)
        if (
            department in candidates
            and len(candidates) > 1
            and not contains_phrase(" ".join(context.sources), department.names())
        ):
            return [_ambiguous_department(mention, candidates)]
        if department in candidates:
            return []
    if not contains_phrase(" ".join(context.sources), department.names()):
        return [
            ValidationIssue(
                field="department_code",
                layer=Layer.GROUNDING,
                code="department_not_in_source",
                message=f"Người dùng không nêu khoa '{department.name}'.",
                suggestion="Không suy ra khoa từ triệu chứng; đặt department_code = null.",
                resolution=Resolution.LLM_FIXABLE,
            )
        ]
    return []


def _verbatim_issues(request: BookingRequest, context: BookingValidationContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name in VERBATIM_FIELDS:
        value = getattr(request, name)
        if not value:
            continue
        if not quote_in_sources(value, context.sources):
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
        flags = overlapping_flags(value, context.stt_flags)
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
    request: BookingRequest, resolution: DateResolution, context: BookingValidationContext
) -> list[ValidationIssue]:
    issues = _department_issues(request, context) + _verbatim_issues(request, context)
    # Lỗi do LLM trích sai phải sửa trước; chưa cần xét ngày khi cụm ngày còn chưa đúng nguyên văn.
    if any(issue.resolution == Resolution.LLM_FIXABLE for issue in issues):
        return issues
    if request.date_expression is None and resolution.window is None:
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
            issues += window_issues(resolution.window, context.now, context.policy)
    if request.doctor_name and find_doctor(request.doctor_name, request.department_code, context.doctors) is None:
        issues.append(
            ValidationIssue(
                field="doctor_name",
                layer=Layer.CONSISTENCY,
                code="doctor_not_found",
                message=f"Không tìm thấy duy nhất một bác sĩ '{request.doctor_name}' trong khoa đã chọn.",
                resolution=Resolution.ASK_USER,
            )
        )
    return issues
