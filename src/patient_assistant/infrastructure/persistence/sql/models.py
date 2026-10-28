"""Bảng SQLAlchemy. Chạy được trên Postgres (production) và SQLite (test)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, TypeDecorator, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UTCDateTime(TypeDecorator[datetime]):
    """Lưu UTC không kèm tz, đọc ra luôn có tz=UTC — hành vi giống nhau trên Postgres và SQLite."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Cần datetime có timezone")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DepartmentRow(Base):
    __tablename__ = "departments"
    code: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)


class DoctorRow(Base):
    __tablename__ = "doctors"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    department_code: Mapped[str] = mapped_column(ForeignKey("departments.code"))


class PatientPlanRow(Base):
    __tablename__ = "patient_plans"
    plan_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    department_code: Mapped[str] = mapped_column(ForeignKey("departments.code"))
    summary: Mapped[str] = mapped_column(Text)


class SlotRow(Base):
    __tablename__ = "slots"
    slot_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    department_code: Mapped[str] = mapped_column(ForeignKey("departments.code"), index=True)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.id"))
    start_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    end_at: Mapped[datetime] = mapped_column(UTCDateTime)


class AppointmentRow(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        # Một slot chỉ có một lịch hẹn hiệu lực (FR-09.9).
        Index(
            "uq_appointments_active_slot",
            "slot_id",
            unique=True,
            postgresql_where=text("status = 'booked'"),
            sqlite_where=text("status = 'booked'"),
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    slot_id: Mapped[str] = mapped_column(ForeignKey("slots.slot_id"))
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)


class ReportRow(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    reporting_period: Mapped[str] = mapped_column(String(16))
    pain_level: Mapped[int] = mapped_column(Integer)
    exercise_days: Mapped[int] = mapped_column(Integer)
    symptom_change: Mapped[str | None] = mapped_column(String(32), nullable=True)
    triggering_activity: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)


class KnowledgeDocRow(Base):
    __tablename__ = "knowledge_docs"
    doc_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    content: Mapped[str] = mapped_column(Text)
