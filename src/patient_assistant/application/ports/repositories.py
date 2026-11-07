"""Port truy cập dữ liệu. Implementation: in-memory (test/demo) và SQLAlchemy (Postgres)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel

from patient_assistant.domain.account import ChatMessage, ConversationSummary, User
from patient_assistant.domain.booking.models import Appointment, Department, Doctor, PatientPlan, Slot
from patient_assistant.domain.common import Evidence
from patient_assistant.domain.patient import PatientProfile
from patient_assistant.domain.report.models import Report, SymptomChange


class SlotTakenError(Exception):
    """Slot đã có lịch hẹn hiệu lực (ràng buộc unique ở DB)."""


class NewReport(BaseModel):
    patient_id: str
    reporting_period: str
    pain_level: int
    exercise_days: int
    symptom_change: SymptomChange | None
    triggering_activity: str | None
    evidence: list[Evidence]
    idempotency_key: str


class KnowledgeDoc(BaseModel):
    doc_id: str
    title: str
    content: str


class ReportRepository(Protocol):
    async def add(self, report: NewReport) -> Report: ...
    async def get(self, report_id: str) -> Report | None: ...
    async def get_by_idempotency_key(self, key: str) -> Report | None: ...
    async def list_by_patient(self, patient_id: str, limit: int) -> list[Report]: ...


class CatalogRepository(Protocol):
    async def list_departments(self) -> list[Department]: ...
    async def list_doctors(self) -> list[Doctor]: ...


class PatientRepository(Protocol):
    async def get_profile(self, patient_id: str) -> PatientProfile | None: ...

    async def list_profiles(self) -> list[PatientProfile]: ...

    async def add(self, profile: PatientProfile) -> None: ...

    async def remove(self, patient_id: str) -> None:
        """Chỉ dùng để hoàn tác hồ sơ vừa tạo khi đăng ký thất bại (hồ sơ chưa có dữ liệu nào)."""
        ...


class PlanRepository(Protocol):
    async def get_active_plan(self, patient_id: str) -> PatientPlan | None: ...


class SlotRepository(Protocol):
    async def find_available(
        self, department_code: str, start: datetime, end: datetime, doctor_id: str | None, limit: int
    ) -> list[Slot]: ...
    async def get(self, slot_id: str) -> Slot | None: ...


class AppointmentRepository(Protocol):
    async def book(self, patient_id: str, slot: Slot, reason: str | None, idempotency_key: str) -> Appointment:
        """Ghi nguyên tử. Ném `SlotTakenError` nếu slot đã có lịch hiệu lực."""
        ...

    async def get(self, appointment_id: str) -> Appointment | None: ...
    async def get_by_idempotency_key(self, key: str) -> Appointment | None: ...
    async def list_for_patient(self, patient_id: str) -> list[Appointment]: ...


class KnowledgeRepository(Protocol):
    async def search(self, query: str, limit: int) -> list[KnowledgeDoc]: ...


class UsernameTakenError(Exception):
    """Username đã tồn tại (ràng buộc unique)."""


class UserRepository(Protocol):
    async def add(self, user: User, password_hash: str) -> User:
        """Ném `UsernameTakenError` nếu username đã có."""
        ...

    async def get(self, user_id: str) -> User | None: ...
    async def get_credentials(self, username: str) -> tuple[User, str] | None:
        """(user, password_hash) theo username, hoặc None."""
        ...

    async def list(self) -> list[User]: ...
    async def update(
        self, user_id: str, *, active: bool | None = None, password_hash: str | None = None
    ) -> User | None: ...
    async def count(self) -> int: ...
    async def set_login_state(self, user_id: str, failed_logins: int, locked_until: datetime | None) -> None: ...


class SessionRepository(Protocol):
    """Phiên đăng nhập. Chỉ lưu hash của token — lộ DB không dùng được token."""

    async def add(self, token_hash: str, user_id: str, expires_at: datetime) -> None: ...
    async def get(self, token_hash: str) -> tuple[str, datetime] | None:
        """(user_id, expires_at) hoặc None."""
        ...

    async def delete(self, token_hash: str) -> None: ...
    async def delete_for_user(self, user_id: str) -> None: ...


class ConversationLogRepository(Protocol):
    """Lưu lại mọi câu người dùng gửi và câu agent trả lời, theo hội thoại."""

    async def open(self, thread_id: str, user_id: str, patient_id: str, title: str, now: datetime) -> str:
        """Tạo hội thoại nếu chưa có. Trả patient_id chủ sở hữu (của hội thoại đã có, nếu có)."""
        ...

    async def owner(self, thread_id: str) -> str | None: ...
    async def add_message(self, message: ChatMessage) -> None:
        """Thêm tin nhắn và cập nhật số tin nhắn / thời điểm / kind cuối của hội thoại."""
        ...

    async def list_conversations(self, patient_id: str | None, limit: int) -> list[ConversationSummary]: ...
    async def list_messages(self, thread_id: str) -> list[ChatMessage]: ...


class AuditLog(Protocol):
    """Ghi metadata phục vụ đánh giá (FR-06.6, FR-08). Không ghi dữ liệu nhận dạng."""

    async def record(self, kind: str, thread_id: str, payload: dict[str, Any]) -> None: ...
