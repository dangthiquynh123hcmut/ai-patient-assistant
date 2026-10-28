"""LLM offline dựa trên luật — chạy demo/test không cần AWS.

Không phải mô hình ngôn ngữ: chỉ đủ để đi hết các luồng với câu nói mẫu. Mọi đầu ra vẫn đi qua
cùng validator như LLM thật, nên khi luật sai thì validator bắt được giống hệt.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from patient_assistant.application.ports.llm import (
    BookingPrompt,
    GroundedAnswer,
    ReportPrompt,
    ToolSpec,
)
from patient_assistant.domain.booking.models import SlotOption
from patient_assistant.domain.conversation import Intent
from patient_assistant.domain.report.models import SymptomChange
from patient_assistant.domain.report.phrases import symptom_direction
from patient_assistant.domain.report.rules import REQUIRED_FIELDS
from patient_assistant.domain.text.normalize import contains_phrase, normalize
from patient_assistant.domain.text.vi_numbers import ZERO_PHRASES, digitize

_CLAUSE = re.compile(r"[,.;!?]| và | nhưng ")
_PAIN = re.compile(r"(?:đau|mức đau)\D{0,15}?(\d{1,2})")
_DAYS = [re.compile(r"tập\D{0,20}?(\d)\s*(?:ngày|buổi|hôm|lần)"), re.compile(r"(\d)\s*(?:ngày|buổi|hôm)\D{0,10}tập")]
_ACTIVITIES = ("gập gối", "duỗi gối", "leo cầu thang", "đi bộ", "ngồi xổm")

_NUM_WORD = r"(?:\d{1,2}|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười(?: một| hai)?)"
_DATE_PATTERNS = [
    re.compile(r"(?:thứ (?:hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật)(?: (?:tuần )?(?:sau|tới|này))?"),
    re.compile(rf"ngày {_NUM_WORD}(?: tháng {_NUM_WORD})?(?: năm \d{{4}})?"),
    re.compile(r"\d{1,2}/\d{1,2}(?:/\d{4})?"),
    re.compile(
        r"ngày mai|ngày kia|ngày mốt|hôm nay|tuần sau|tuần tới|tuần này|sớm nhất|cuối tuần|đầu tháng"
        r"|cuối tháng|tháng sau|\bmai\b"
    ),
]
_TIME_PATTERNS = [
    re.compile(r"(?:buổi )?(?:sáng|chiều|trưa|tối)"),
    re.compile(rf"(?:khoảng )?{_NUM_WORD} ?(?:giờ|h)(?: ?(?:rưỡi|\d{{1,2}}))?"),
]
_DOCTOR = re.compile(r"(?:bác sĩ|bs) ([a-zà-ỹ]+)")
_REASON = re.compile(r"(?:vì|do|để) ([^,.;]+)")


def _clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE.split(text) if c.strip()]


def _span(text: str, patterns: list[re.Pattern[str]]) -> str | None:
    """Đoạn nguyên văn (đã chuẩn hóa) từ match đầu đến match cuối — giữ được cụm như 'thứ hai ngày 5/11'."""
    matches = [m for p in patterns for m in p.finditer(text)]
    if not matches:
        return None
    start, end = min(m.start() for m in matches), max(m.end() for m in matches)
    return text[start:end].strip()


class OfflineRuleLLM:
    async def classify_intent(self, text: str) -> list[Intent]:
        return []

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        out: dict[str, Any] = {
            "reporting_period": None,
            "pain_level": None,
            "exercise_days": None,
            "symptom_change": None,
            "triggering_activity": None,
        }
        evidence: dict[str, str] = {}
        for source in reversed(prompt.sources):  # nguồn mới nhất được ưu tiên
            for clause in _clauses(source):
                digits = digitize(clause)
                if out["pain_level"] is None and (m := _PAIN.search(digits)):
                    out["pain_level"], evidence["pain_level"] = int(m.group(1)), clause
                if out["exercise_days"] is None:
                    if contains_phrase(clause, ZERO_PHRASES):
                        out["exercise_days"], evidence["exercise_days"] = 0, clause
                    elif m := next((p.search(digits) for p in _DAYS if p.search(digits)), None):
                        out["exercise_days"], evidence["exercise_days"] = int(m.group(1)), clause
                if out["symptom_change"] is None and (direction := symptom_direction(clause)):
                    if direction == "worse":
                        during = contains_phrase(clause, ("khi", "lúc", "trong lúc"))
                        change = SymptomChange.WORSE_DURING_ACTIVITY if during else SymptomChange.WORSE
                    else:
                        change = SymptomChange.BETTER if direction == "better" else SymptomChange.NO_CHANGE
                    out["symptom_change"], evidence["symptom_change"] = change.value, clause
                if out["triggering_activity"] is None and (act := contains_phrase(clause, _ACTIVITIES)):
                    out["triggering_activity"], evidence["triggering_activity"] = act, act
        out["missing_fields"] = [f for f in REQUIRED_FIELDS if out[f] is None]
        out["evidence"] = [{"field": k, "quote": v} for k, v in evidence.items()]
        return out

    async def extract_booking(self, prompt: BookingPrompt) -> dict[str, Any]:
        found: dict[str, str | None] = {
            "department_code": None,
            "doctor_name": None,
            "date_expression": None,
            "time_expression": None,
            "reason": None,
        }
        evidence: dict[str, str] = {}
        for source in reversed(prompt.sources):
            text = normalize(source)
            if found["department_code"] is None:
                for dept in prompt.departments:
                    hit = contains_phrase(text, dept.names())
                    if hit:
                        found["department_code"], evidence["department_code"] = dept.code, hit
                        break
            if found["date_expression"] is None:
                found["date_expression"] = _span(text, _DATE_PATTERNS)
            if found["time_expression"] is None:
                found["time_expression"] = _span(text, _TIME_PATTERNS)
            if found["doctor_name"] is None and (m := _DOCTOR.search(text)):
                found["doctor_name"] = m.group(0)
            if found["reason"] is None and (m := _REASON.search(text)):
                found["reason"] = m.group(1).strip()
        for key in ("date_expression", "time_expression", "doctor_name", "reason"):
            if found[key]:
                evidence[key] = found[key]  # type: ignore[assignment]
        return {
            **found,
            "missing_fields": [k for k in ("department_code", "date_expression") if not found[k]],
            "evidence": [{"field": k, "quote": v} for k, v in evidence.items()],
        }

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        return None

    async def answer_question(self, question: str, tools: list[ToolSpec], max_tool_calls: int) -> GroundedAnswer:
        search = next((t for t in tools if t.name == "search_approved_knowledge"), None)
        if search is None or max_tool_calls < 1:
            return GroundedAnswer(text="")
        inp: BaseModel = search.input_model.model_validate({"query": question})
        result = await search.handler(inp)
        docs = result.get("data") or []
        if result.get("status") != "ok" or not docs:
            return GroundedAnswer(text="", tool_calls=1)
        top = docs[0]
        return GroundedAnswer(text=f"{top['content']} ({top['doc_id']})", source_ids=[top["doc_id"]], tool_calls=1)
