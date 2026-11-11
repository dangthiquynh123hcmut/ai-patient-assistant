"""Bộ nhớ ngắn hạn của MỘT cuộc hội thoại: ghi chú các lượt đã xong, do code soạn từ kết quả có cấu trúc.

- Ghi chú chỉ dùng để HIỂU câu nói (phân loại việc, trả lời câu hỏi, trò chuyện, giải nghĩa "như vậy",
  "bác sĩ đó", "khoa đó"). Không bao giờ là evidence cho dữ liệu sẽ ghi: giá trị lấy từ bộ nhớ được ghi rõ nguồn
  (lịch hẹn đã đặt / lượt trước) và vẫn phải qua xác nhận.
- Không dán nguyên văn hội thoại; lời người dùng được cắt ngắn, kết quả chỉ gồm dữ kiện backend trả về.
- Chỉ sống trong checkpoint của `conversation_id` này và chỉ nối khi cùng bệnh nhân.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from patient_assistant.domain.booking.models import Department, Doctor
from patient_assistant.domain.booking.validators import department_in_text
from patient_assistant.domain.conversation import Intent
from patient_assistant.domain.text.normalize import contains_phrase, normalize, unaccent
from patient_assistant.domain.text.vi_dates import WEEKDAY_NAMES

MAX_TURNS = 8
MAX_SAID_CHARS = 200

# Cụm nhắc lại thứ đã có ở lượt trước. Code giải nghĩa; LLM không tự điền khoa/bác sĩ từ lịch sử.
SAME_AS_BEFORE = (
    "như vậy",
    "như thế",
    "y như vậy",
    "giống vậy",
    "giống như vậy",
    "như lần trước",
    "giống lần trước",
    "như lịch vừa đặt",
    "giống lịch vừa đặt",
    "như lịch trước",
    "như lịch đó",
    "giống lịch đó",
    "như lịch vừa rồi",
    "the same as before",
    "same as before",
    "same as last time",
    "like last time",
    "like that one",
    "same one",
)
DOCTOR_REF = (
    "bác sĩ đó",
    "bác sĩ đấy",
    "bác sĩ ấy",
    "bác sĩ này",
    "bác sĩ vừa rồi",
    "bác sĩ lúc nãy",
    "bác sĩ vừa nói",
    "cùng bác sĩ",
    "bs đó",
    "bs ấy",
    "that doctor",
    "the same doctor",
    "same doctor",
    "this doctor",
)
DEPARTMENT_REF = (
    "khoa đó",
    "khoa đấy",
    "khoa ấy",
    "khoa này",
    "khoa vừa rồi",
    "khoa lúc nãy",
    "cùng khoa",
    "that department",
    "the same department",
    "same department",
    "this department",
)


class MemoryAppointment(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    department_code: str
    department_name: str
    doctor_id: str
    doctor_name: str
    start_at: datetime

    def describe(self) -> str:
        return (
            f"{self.department_name}, {self.doctor_name}, {self.start_at:%H:%M} "
            f"{WEEKDAY_NAMES[self.start_at.weekday()]} {self.start_at:%d/%m/%Y} (mã {self.id})"
        )


class TurnNote(BaseModel):
    """Một lượt đã kết thúc (xong, hủy, lỗi hoặc bỏ dở)."""

    model_config = ConfigDict(frozen=True)

    said: str
    intent: Intent | None = None
    result: str
    appointment: MemoryAppointment | None = None
    doctor_ids: list[str] = Field(default_factory=list)  # bác sĩ người dùng nhắc tên trong lượt này
    department_codes: list[str] = Field(default_factory=list)  # khoa người dùng nhắc trong lượt này


class MemoryRef(BaseModel):
    """Khoa/bác sĩ code lấy lại từ lượt trước theo cụm nhắc lại của người dùng."""

    model_config = ConfigDict(frozen=True)

    phrase: str  # cụm người dùng nói, ví dụ "như lần trước"
    department_code: str
    doctor_id: str | None = None
    appointment_id: str | None = None
    explanation: str


# ---------------------------------------------------------------- soạn ghi chú


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_SAID_CHARS else text[: MAX_SAID_CHARS - 1] + "…"


def doctors_mentioned(text: str, doctors: list[Doctor]) -> list[str]:
    """Bác sĩ người dùng gọi tên ("bác sĩ An", "BS. Trần Văn An", "doctor Binh"); chỉ nhận tên khớp đúng một người.
    So khớp không dấu để nhận cả người gõ/nói không dấu hoặc tiếng Anh."""
    plain = f" {unaccent(text)} "
    found: list[str] = []
    for d in doctors:
        full = unaccent(d.name).removeprefix("bs ").strip()
        given = full.split(" ")[-1]
        if f" {full} " in plain:
            found.append(d.id)
            continue
        same_given = [o for o in doctors if unaccent(o.name).split(" ")[-1] == given]
        titles = ("bac si", "bs", "doctor", "dr")
        if len(same_given) == 1 and any(f" {t} {given} " in plain for t in titles):
            found.append(d.id)
    return found


def _appointment(data: Any, departments: list[Department]) -> MemoryAppointment | None:
    if not isinstance(data, dict) or not isinstance(data.get("slot"), dict):
        return None
    slot = data["slot"]
    names = {d.code: d.name for d in departments}
    try:
        return MemoryAppointment(
            id=str(data["id"]),
            department_code=slot["department_code"],
            department_name=names.get(slot["department_code"], slot["department_code"]),
            doctor_id=slot["doctor_id"],
            doctor_name=slot["doctor_name"],
            start_at=slot["start_at"],
        )
    except (KeyError, ValueError):
        return None


_RESULT_TEXT = {
    "report_saved": "đã lưu báo cáo theo dõi",
    "history": "đã xem lịch sử báo cáo",
    "appointments": "đã xem danh sách lịch hẹn",
    "patient_info": "đã xem thông tin và kế hoạch theo dõi",
    "answer": "đã được trả lời",
    "chat": "trò chuyện",
    "cancelled": "người dùng đã hủy, chưa lưu gì",
    "handoff": "chưa xong, đã chuyển nhân viên",
    "safety": "đã dừng vì lý do an toàn",
    "error": "lỗi hệ thống, chưa lưu gì",
}


def note_for_turn(
    said: str,
    intent: Intent | None,
    outcome_kind: str | None,
    outcome_data: Any,
    source_ids: list[str],
    departments: list[Department],
    doctors: list[Doctor],
) -> TurnNote:
    appointment = _appointment(outcome_data, departments) if outcome_kind == "appointment_booked" else None
    if appointment:
        result = f"đã đặt lịch: {appointment.describe()}"
    elif outcome_kind == "report_saved" and source_ids:
        result = f"{_RESULT_TEXT['report_saved']} (mã {source_ids[0]})"
    elif outcome_kind is None:
        result = "chưa xong, người dùng chuyển sang việc khác"
    else:
        result = _RESULT_TEXT.get(outcome_kind, outcome_kind)
    dept = department_in_text(said, departments)
    return TurnNote(
        said=_short(said),
        intent=intent,
        result=result,
        appointment=appointment,
        doctor_ids=doctors_mentioned(said, doctors),
        department_codes=[dept.code] if dept else [],
    )


def append_note(memory: list[TurnNote], note: TurnNote) -> list[TurnNote]:
    return [*memory, note][-MAX_TURNS:]


def render_memory(memory: list[TurnNote]) -> str | None:
    """Ghi chú gửi kèm cho LLM — chỉ để hiểu câu nói, không phải dữ liệu để lưu."""
    if not memory:
        return None
    lines = []
    for k, note in enumerate(memory):
        back = len(memory) - k
        label = "lượt vừa rồi" if back == 1 else f"{back} lượt trước"
        lines.append(f"- {label}: người dùng nói «{note.said}» → {note.result}")
    return "\n".join(lines)


# ---------------------------------------------------------------- giải nghĩa cụm nhắc lại


def _distinct_appointments(memory: list[TurnNote]) -> list[MemoryAppointment]:
    seen: dict[tuple[str, str], MemoryAppointment] = {}
    for note in memory:
        if note.appointment:
            seen[(note.appointment.department_code, note.appointment.doctor_id)] = note.appointment
    return list(seen.values())


def mentions_reference(text: str) -> bool:
    return contains_phrase(text, (*SAME_AS_BEFORE, *DOCTOR_REF, *DEPARTMENT_REF)) is not None


def is_reference(text: str) -> bool:
    """Cụm chỉ là lời nhắc lại ("khoa đó", "bác sĩ ấy", "như vậy"), không phải tên khoa/bác sĩ."""
    norm = normalize(text)
    return any(norm == normalize(p) for p in (*SAME_AS_BEFORE, *DOCTOR_REF, *DEPARTMENT_REF))


def resolve_reference(text: str, memory: list[TurnNote], doctors: list[Doctor]) -> MemoryRef | None:
    """Khoa/bác sĩ mà cụm "như vậy", "bác sĩ đó", "khoa đó" trỏ tới. Không chắc (0 hoặc nhiều ứng viên) -> None,
    để luồng hỏi lại thay vì đoán."""
    if not memory:
        return None
    by_id = {d.id: d for d in doctors}

    if phrase := contains_phrase(text, SAME_AS_BEFORE):
        booked = _distinct_appointments(memory)
        if len(booked) != 1:
            return None
        appt = booked[0]
        return MemoryRef(
            phrase=phrase,
            department_code=appt.department_code,
            doctor_id=appt.doctor_id,
            appointment_id=appt.id,
            explanation=f"Bạn nói “{phrase}”: lấy theo lịch hẹn đã đặt ở lượt trước ({appt.describe()}).",
        )

    if phrase := contains_phrase(text, DOCTOR_REF):
        for note in reversed(memory):
            ids = list(dict.fromkeys([*note.doctor_ids, *([note.appointment.doctor_id] if note.appointment else [])]))
            if not ids:
                continue
            doctor = by_id.get(ids[0]) if len(ids) == 1 else None
            if doctor is None:
                return None  # lượt gần nhất nhắc nhiều bác sĩ -> không đoán
            return MemoryRef(
                phrase=phrase,
                department_code=doctor.department_code,
                doctor_id=doctor.id,
                explanation=f"Bạn nói “{phrase}”: là {doctor.name}, người bạn nhắc tới ở lượt trước.",
            )
        return None

    if phrase := contains_phrase(text, DEPARTMENT_REF):
        for note in reversed(memory):
            codes = list(
                dict.fromkeys(
                    [*note.department_codes, *([note.appointment.department_code] if note.appointment else [])]
                )
            )
            if not codes:
                continue
            if len(codes) > 1:
                return None
            return MemoryRef(
                phrase=phrase,
                department_code=codes[0],
                explanation=f"Bạn nói “{phrase}”: là khoa bạn nhắc tới ở lượt trước.",
            )
    return None
