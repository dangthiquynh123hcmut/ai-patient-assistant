from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from patient_assistant.domain.common import Evidence
from patient_assistant.domain.text.vi_dates import WEEKDAY_NAMES

BOOKING_FIELDS = ("department_code", "doctor_name", "date_expression", "time_expression", "reason")


class Department(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    name: str
    aliases: tuple[str, ...] = ()

    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


class Doctor(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    department_code: str


class PatientPlan(BaseModel):
    plan_id: str
    patient_id: str
    department_code: str
    summary: str


class BookingRequest(BaseModel):
    """Thứ LLM được phép sinh ra. Không có trường ngày dạng `date`: LLM không tự quy đổi ngày."""

    model_config = ConfigDict(extra="forbid")

    department_code: str | None = Field(None, description="Mã khoa trong danh mục; null nếu người dùng không nêu")
    doctor_name: str | None = None
    date_expression: str | None = Field(None, description="Cụm ngày NGUYÊN VĂN, ví dụ 'thứ Ba tuần sau'")
    time_expression: str | None = Field(None, description="Cụm giờ NGUYÊN VĂN, ví dụ 'buổi sáng'")
    reason: str | None = Field(None, description="Lý do khám ngắn, nguyên văn; không suy diễn")
    missing_fields: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class Slot(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot_id: str
    department_code: str
    doctor_id: str
    doctor_name: str
    start_at: datetime
    end_at: datetime

    def label(self, department_name: str | None = None) -> str:
        dept = f" – {department_name}" if department_name else ""
        return (
            f"{WEEKDAY_NAMES[self.start_at.weekday()]}, {self.start_at:%d/%m/%Y}, "
            f"{self.start_at:%H:%M} – {self.doctor_name}{dept}"
        )


class SlotOption(BaseModel):
    """Slot đã đưa cho người dùng chọn, đánh số 1..N để chọn bằng giọng nói."""

    model_config = ConfigDict(frozen=True)

    option_no: int
    slot: Slot


class AppointmentStatus(StrEnum):
    BOOKED = "booked"
    CANCELLED = "cancelled"


class Appointment(BaseModel):
    """`id` và `created_at` do backend gán."""

    id: str
    patient_id: str
    slot: Slot
    status: AppointmentStatus
    reason: str | None
    idempotency_key: str
    created_at: datetime
