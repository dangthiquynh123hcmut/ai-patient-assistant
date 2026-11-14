from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from patient_assistant.domain.common import Evidence


class SymptomChange(StrEnum):
    BETTER = "better"
    NO_CHANGE = "no_change"
    WORSE = "worse"
    WORSE_DURING_ACTIVITY = "worse_during_activity"


REPORT_FIELDS = ("reporting_period", "pain_level", "exercise_days", "symptom_change", "triggering_activity")


class ReportExtraction(BaseModel):
    """Thứ LLM được phép sinh ra. Không có validation_status/ID/timestamp — những trường đó do code gán."""

    model_config = ConfigDict(extra="forbid")

    reporting_period: str | None = Field(None, description="ISO week, e.g. 2026-W44; null if not stated")
    pain_level: int | None = Field(None, description="Pain level 0-10 exactly as the user said")
    exercise_days: int | None = Field(None, description="Number of days exercised in the period")
    symptom_change: SymptomChange | None = None
    triggering_activity: str | None = Field(None, description="Activity that changed the symptoms")
    missing_fields: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(
        default_factory=list, description="Verbatim quote of the user's words for each field"
    )


class Report(BaseModel):
    """Báo cáo đã lưu. `id` và `created_at` do backend gán (FR-07.4)."""

    id: str
    patient_id: str
    reporting_period: str
    pain_level: int
    exercise_days: int
    symptom_change: SymptomChange | None
    triggering_activity: str | None
    evidence: list[Evidence]
    idempotency_key: str
    created_at: datetime
