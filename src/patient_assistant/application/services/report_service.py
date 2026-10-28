"""Nghiệp vụ lưu báo cáo phía backend: kiểm tra lại quy tắc, chống ghi trùng, gán ID/timestamp."""

from __future__ import annotations

from datetime import datetime

from patient_assistant.application.ports.repositories import NewReport, ReportRepository
from patient_assistant.application.tools.result import ToolErrorCode, ToolResult
from patient_assistant.domain.report.models import ReportExtraction
from patient_assistant.domain.report.rules import backend_issues, current_period

TOOL = "save_patient_report"


class ReportService:
    def __init__(self, reports: ReportRepository) -> None:
        self._reports = reports

    async def save(self, patient_id: str, draft: ReportExtraction, idempotency_key: str, now: datetime) -> ToolResult:
        existing = await self._reports.get_by_idempotency_key(idempotency_key)
        if existing:
            return ToolResult.success(TOOL, existing.model_dump(mode="json"), [existing.id])
        issues = backend_issues(draft, now)
        if issues:
            return ToolResult.failure(
                TOOL,
                ToolErrorCode.INVALID_ARGS,
                "Báo cáo không đạt quy tắc nghiệp vụ.",
                details=[i.message for i in issues],
            )
        assert draft.pain_level is not None and draft.exercise_days is not None
        report = await self._reports.add(
            NewReport(
                patient_id=patient_id,
                reporting_period=draft.reporting_period or current_period(now),
                pain_level=draft.pain_level,
                exercise_days=draft.exercise_days,
                symptom_change=draft.symptom_change,
                triggering_activity=draft.triggering_activity,
                evidence=draft.evidence,
                idempotency_key=idempotency_key,
            )
        )
        return ToolResult.success(TOOL, report.model_dump(mode="json"), [report.id])

    async def get(self, patient_id: str, report_id: str) -> ToolResult:
        report = await self._reports.get(report_id)
        if report is None or report.patient_id != patient_id:
            return ToolResult.failure("get_report", ToolErrorCode.NOT_FOUND, "Không tìm thấy báo cáo.")
        return ToolResult.success("get_report", report.model_dump(mode="json"), [report.id])

    async def history(self, patient_id: str, limit: int) -> ToolResult:
        reports = await self._reports.list_by_patient(patient_id, limit)
        return ToolResult.success(
            "get_report_history", [r.model_dump(mode="json") for r in reports], [r.id for r in reports]
        )
