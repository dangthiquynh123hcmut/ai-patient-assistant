"""Kiểm chứng các case eval mới (trùng lịch, gợi ý cùng ngày, bộ nhớ hội thoại, tiếng Anh) KHÔNG cần Bedrock.

LLM thay thế trả đúng đầu ra trích xuất mà một LLM tốt sẽ trả cho từng câu. Như vậy test này kiểm tra
dataset (kỳ vọng có đúng với hệ thống không) và harness (setup, history, lỗi tiêm, metric) — còn chất lượng
LLM thật được đo khi chạy `pa-eval`.
"""

from __future__ import annotations

from typing import Any

import pytest

from patient_assistant.application.ports.llm import BookingPrompt, GroundedAnswer, ReportPrompt, ToolSpec
from patient_assistant.domain.booking.models import SlotOption
from patient_assistant.domain.conversation import Intent
from patient_assistant.evaluation.cases import load_cases
from patient_assistant.evaluation.harness import run_cases
from patient_assistant.evaluation.metrics import compute
from patient_assistant.evaluation.report import hard_zero_violations
from tests.conftest import booking_out, pg_settings

NEW_CASES = {
    *("bk-own-01", "bk-rec-01", "bk-rec-02", "bk-rec-03", "bk-clash-01", "bk-max-01", "bk-mot-01"),
    *("bk-en-01", "bk-en-02", "bk-en-03", "bk-en-04"),
    *("conv-mem-01", "conv-mem-02", "conv-mem-03", "conv-mem-04", "conv-mem-05"),
    *("conv-en-emerg-01", "conv-en-emerg-02", "conv-en-advice-01", "conv-en-list-01"),
}

PHCN = ("PHCN", "khoa phục hồi chức năng")
REHAB = ("PHCN", "rehabilitation")
TM = ("TM", "khoa tim mạch")

# Câu người dùng mới nhất -> đầu ra trích xuất (nguyên văn như prompt yêu cầu).
EXTRACTIONS: dict[str, dict[str, Any]] = {
    "đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng": booking_out(PHCN, "thứ ba tuần sau", "buổi sáng"),
    "đặt lịch khoa phục hồi chức năng tuần sau": booking_out(PHCN, "tuần sau"),
    "đặt lịch khoa phục hồi chức năng tuần sau buổi sáng": booking_out(PHCN, "tuần sau", "buổi sáng"),
    "đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi chiều": booking_out(PHCN, "thứ ba tuần sau", "buổi chiều"),
    "đặt lịch khoa tim mạch ngày mốt lúc 9 giờ": booking_out(TM, "ngày mốt", "9 giờ"),
    "đặt lịch khoa tim mạch thứ tư tuần sau buổi sáng": booking_out(TM, "thứ tư tuần sau", "buổi sáng"),
    "I'd like to book an appointment with rehabilitation next Tuesday morning": booking_out(
        REHAB, "next Tuesday", "morning"
    ),
    "book an appointment with cardiology the day after tomorrow at 9 o'clock": booking_out(
        ("TM", "cardiology"), "the day after tomorrow", "at 9 o'clock"
    ),
    "book an appointment with rehabilitation this weekend": booking_out(REHAB, "this weekend"),
    "next Wednesday at 9 am": booking_out(REHAB, "next Wednesday", "at 9 am"),
    "book rehabilitation next Tuesday morning": booking_out(REHAB, "next Tuesday", "morning"),
    "something in the afternoon please": booking_out(REHAB, "next Tuesday", "in the afternoon"),
    "book an appointment with rehabilitation next Tuesday morning": booking_out(REHAB, "next Tuesday", "morning"),
    # Bộ nhớ: người dùng không nêu khoa; LLM có thể chép nhầm cụm nhắc lại thành tên -> code phải bỏ.
    "đặt thêm cho tôi một lịch như vậy vào thứ năm tuần sau buổi sáng": booking_out(
        None, "thứ năm tuần sau", "buổi sáng"
    ),
    "đặt thêm một lịch như vậy thứ năm tuần sau": booking_out(None, "thứ năm tuần sau"),
    "cho tôi đặt lịch với bác sĩ đó thứ ba tuần sau buổi sáng": booking_out(
        None, "thứ ba tuần sau", "buổi sáng", doctor="bác sĩ đó"
    ),
    "vậy đặt lịch khoa đó thứ tư tuần sau buổi sáng": booking_out(
        None, "thứ tư tuần sau", "buổi sáng", mentioned="khoa đó"
    ),
    "book another one the same as before next Thursday morning": booking_out(None, "next Thursday", "morning"),
}


class StandInLLM:
    """Không có luật hiểu ngôn ngữ: chỉ trả đầu ra trích xuất định sẵn theo câu mới nhất."""

    async def classify_intent(self, text: str, memory: str | None = None) -> list[Intent]:
        # Chỉ được gọi khi luật không kết luận được ("book rehabilitation ..."): trả như một LLM tốt.
        return [Intent.BOOK_APPOINTMENT] if "book" in text.lower() else []

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        return {"missing_fields": [], "evidence": []}

    async def extract_booking(self, prompt: BookingPrompt) -> dict[str, Any]:
        for source in reversed(prompt.sources):
            if source in EXTRACTIONS:
                return EXTRACTIONS[source]
        return {"missing_fields": ["department_code", "date_expression"], "evidence": []}

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        return None

    async def answer_question(
        self, question: str, tools: list[ToolSpec], max_tool_calls: int, memory: str | None = None
    ) -> GroundedAnswer:
        return GroundedAnswer(text="")

    async def converse(self, text: str, context: str | None = None, memory: str | None = None) -> str:
        return "Chào bạn"


async def test_new_eval_cases_pass_with_a_correct_extractor() -> None:
    cases = [c for c in load_cases() if c.id in NEW_CASES]
    assert {c.id for c in cases} == NEW_CASES
    observations = await run_cases(cases, StandInLLM, pg_settings(), concurrency=4)
    report = compute(cases, observations)
    failed = {v.case_id: v.failures for v in report.verdicts if not v.passed}
    assert not failed, failed
    assert hard_zero_violations(report) == []
    assert report.own_schedule_overlap.numerator == 0 and report.own_schedule_overlap.denominator > 0


async def test_eval_catches_offers_that_ignore_the_patients_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation test: bỏ bộ lọc/gợi ý theo lịch của bệnh nhân thì eval phải báo vi phạm."""
    from patient_assistant.domain.booking.ranking import SlotFit

    monkeypatch.setattr(
        "patient_assistant.application.services.booking_service.fit_to_schedule",
        lambda slots, *_: [SlotFit(s) for s in slots],
    )
    cases = [c for c in load_cases() if c.id in {"bk-own-01", "bk-rec-01"}]
    report = compute(cases, await run_cases(cases, StandInLLM, pg_settings(), concurrency=2))
    assert "own_schedule_overlap" in hard_zero_violations(report)
    failed = {v.case_id for v in report.verdicts if not v.passed}
    assert failed == {"bk-own-01", "bk-rec-01"}
