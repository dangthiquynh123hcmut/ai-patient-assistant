"""Port truy cập dữ liệu. Implementation: in-memory (test/demo) và SQLAlchemy (Postgres)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel

from patient_assistant.domain.booking.models import Appointment, Department, Doctor, PatientPlan, Slot
from patient_assistant.domain.common import Evidence
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


class AuditLog(Protocol):
    """Ghi metadata phục vụ đánh giá (FR-06.6, FR-08). Không ghi dữ liệu nhận dạng."""

    async def record(self, kind: str, thread_id: str, payload: dict[str, Any]) -> None: ...
