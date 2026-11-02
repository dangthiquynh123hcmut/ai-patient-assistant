"""Tài khoản đăng nhập và phân quyền. Quyền được quyết định ở đây bằng code, không bao giờ do LLM."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

USERNAME = re.compile(r"^[a-z0-9._-]{3,32}$")
MIN_PASSWORD_LENGTH = 8


class Role(StrEnum):
    PATIENT = "patient"  # chat với trợ lý, chỉ thấy dữ liệu của chính mình
    STAFF = "staff"  # bác sĩ / nhân viên: xem hội thoại và câu hỏi của bệnh nhân, không chat thay
    ADMIN = "admin"  # như staff + quản lý tài khoản


class User(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    username: str
    display_name: str
    role: Role
    patient_id: str | None = None  # bắt buộc với role patient, rỗng với staff/admin
    active: bool = True
    created_at: datetime
    # Khóa tạm sau nhiều lần sai mật khẩu — lưu ở DB để mọi worker/instance thấy cùng một trạng thái.
    failed_logins: int = 0
    locked_until: datetime | None = None

    @property
    def can_chat(self) -> bool:
        return self.active and self.role == Role.PATIENT and self.patient_id is not None

    @property
    def can_view_conversations(self) -> bool:
        return self.active and self.role in (Role.STAFF, Role.ADMIN)

    @property
    def can_manage_users(self) -> bool:
        return self.active and self.role == Role.ADMIN


class NewUser(BaseModel):
    username: str
    display_name: str = Field(min_length=1, max_length=128)
    role: Role
    patient_id: str | None = None
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=128)

    @field_validator("username")
    @classmethod
    def _username(cls, value: str) -> str:
        value = value.strip().lower()
        if not USERNAME.match(value):
            raise ValueError("username: 3–32 ký tự a-z, 0-9, dấu chấm, gạch dưới, gạch ngang")
        return value

    @model_validator(mode="after")
    def _patient_link(self) -> NewUser:
        if (self.role == Role.PATIENT) != (self.patient_id is not None):
            raise ValueError("Tài khoản patient phải gắn đúng một mã bệnh nhân; staff/admin thì không")
        return self


class ConversationSummary(BaseModel):
    thread_id: str
    patient_id: str
    user_id: str
    title: str
    message_count: int
    last_kind: str | None
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    """Một dòng trong lịch sử: lời người dùng (`role=user`) hoặc câu agent trả lời/hỏi lại (`role=assistant`)."""

    id: str
    thread_id: str
    role: str  # user | assistant
    action: str  # user: message|reply|confirm|choose|edit|cancel|other_time; assistant: kind của câu trả lời
    text: str
    intent: str | None = None
    created_at: datetime
