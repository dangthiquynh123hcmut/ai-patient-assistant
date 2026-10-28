"""Kiểu dữ liệu dùng chung của domain: lỗi validation, evidence, nguồn gốc dữ liệu.

Domain chỉ phụ thuộc pydantic (để định nghĩa schema) và thư viện chuẩn.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Layer(StrEnum):
    SCHEMA = "schema"
    BUSINESS = "business"
    GROUNDING = "grounding"
    CONSISTENCY = "consistency"
    TOOL_RESULT = "tool_result"
    SAFETY = "safety"


class Resolution(StrEnum):
    """Ai sửa được lỗi này. Do validator (code) gán, không do LLM gán."""

    LLM_FIXABLE = "llm_fixable"  # LLM trích xuất sai -> gửi lỗi về để sửa
    ASK_USER = "ask_user"  # dữ liệu nguồn thiếu/mơ hồ/mâu thuẫn -> hỏi người dùng
    BLOCK = "block"  # ngoài phạm vi / không an toàn -> dừng


class ValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str | None
    layer: Layer
    code: str
    message: str
    resolution: Resolution
    suggestion: str | None = None


class Evidence(BaseModel):
    """Trích dẫn nguyên văn hỗ trợ một trường. `quote=None` nghĩa là không có bằng chứng."""

    model_config = ConfigDict(frozen=True)

    field: str
    quote: str | None


class Origin(StrEnum):
    """Phân biệt nguồn gốc dữ kiện (FR-05.4)."""

    USER_SAID = "user_said"
    USER_EDIT = "user_edit"
    BACKEND = "backend"
    SYSTEM_DEFAULT = "system_default"
    AI_SUMMARY = "ai_summary"


class SttFlag(BaseModel):
    """Một từ STT nhận dạng với độ tin cậy thấp, nằm ở vị trí quan trọng (số, ngày, tên riêng...)."""

    model_config = ConfigDict(frozen=True)

    text: str
    probability: float
    reason: str


def evidence_for(evidence: list[Evidence], field: str) -> str | None:
    for item in evidence:
        if item.field == field:
            return item.quote
    return None
