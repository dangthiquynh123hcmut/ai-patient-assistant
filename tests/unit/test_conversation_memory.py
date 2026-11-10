from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from patient_assistant.application.agent.memory import (
    MAX_TURNS,
    TurnNote,
    append_note,
    doctors_mentioned,
    is_reference,
    note_for_turn,
    render_memory,
    resolve_reference,
)
from patient_assistant.domain.booking.models import Department, Doctor
from patient_assistant.domain.conversation import Intent

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEPTS = [Department(code="PHCN", name="Phục hồi chức năng"), Department(code="TM", name="Tim mạch")]
DOCTORS = [
    Doctor(id="doc-an", name="BS. Trần Văn An", department_code="PHCN"),
    Doctor(id="doc-binh", name="BS. Lê Thị Bình", department_code="PHCN"),
    Doctor(id="doc-hai", name="BS. Hoàng Minh Hải", department_code="TM"),
]


def _booked(appt_id: str, dept: str, doctor: Doctor, hour: int = 9) -> TurnNote:
    start = datetime(2026, 11, 3, hour, 0, tzinfo=TZ)
    data = {
        "id": appt_id,
        "slot": {
            "slot_id": "s",
            "department_code": dept,
            "doctor_id": doctor.id,
            "doctor_name": doctor.name,
            "start_at": start.isoformat(),
            "end_at": start.isoformat(),
        },
    }
    return note_for_turn("đặt lịch", Intent.BOOK_APPOINTMENT, "appointment_booked", data, [appt_id], DEPTS, DOCTORS)


def test_booked_turn_is_summarised_from_backend_data() -> None:
    note = _booked("LH-1", "PHCN", DOCTORS[0])
    assert note.appointment is not None
    assert note.result == "đã đặt lịch: Phục hồi chức năng, BS. Trần Văn An, 09:00 Thứ Ba 03/11/2026 (mã LH-1)"


def test_doctors_and_departments_mentioned_are_recognised() -> None:
    assert doctors_mentioned("bác sĩ An khám thứ mấy", DOCTORS) == ["doc-an"]
    assert doctors_mentioned("BS Trần Văn An", DOCTORS) == ["doc-an"]
    assert doctors_mentioned("bác sĩ nào cũng được", DOCTORS) == []
    note = note_for_turn("khoa tim mạch có ai", None, "answer", None, [], DEPTS, DOCTORS)
    assert note.department_codes == ["TM"]


def test_same_as_before_needs_exactly_one_distinct_booking() -> None:
    one = [_booked("LH-1", "PHCN", DOCTORS[0])]
    ref = resolve_reference("đặt thêm một lịch như vậy", one, DOCTORS)
    assert ref and ref.department_code == "PHCN" and ref.doctor_id == "doc-an" and ref.appointment_id == "LH-1"

    two = [*one, _booked("LH-2", "TM", DOCTORS[2])]
    assert resolve_reference("đặt thêm một lịch như vậy", two, DOCTORS) is None
    # Hai lịch cùng khoa, cùng bác sĩ -> vẫn rõ.
    same = [*one, _booked("LH-3", "PHCN", DOCTORS[0], hour=10)]
    assert resolve_reference("như lần trước nhé", same, DOCTORS) is not None


def test_doctor_reference_uses_most_recent_turn_naming_one_doctor() -> None:
    asked = note_for_turn("bác sĩ Bình khám thứ mấy", None, "answer", None, [], DEPTS, DOCTORS)
    ref = resolve_reference("đặt với bác sĩ đó", [_booked("LH-1", "PHCN", DOCTORS[0]), asked], DOCTORS)
    assert ref and ref.doctor_id == "doc-binh" and ref.department_code == "PHCN"

    both = note_for_turn("bác sĩ An hay bác sĩ Bình", None, "answer", None, [], DEPTS, DOCTORS)
    assert resolve_reference("đặt với bác sĩ đó", [both], DOCTORS) is None


def test_department_reference_and_no_reference() -> None:
    asked = note_for_turn("khoa tim mạch làm thứ bảy không", None, "answer", None, [], DEPTS, DOCTORS)
    ref = resolve_reference("vậy đặt khoa đó cho tôi", [asked], DOCTORS)
    assert ref and ref.department_code == "TM" and ref.doctor_id is None
    assert resolve_reference("đặt lịch thứ ba", [asked], DOCTORS) is None
    assert resolve_reference("như vậy", [], DOCTORS) is None


def test_reference_phrases_are_not_names() -> None:
    assert is_reference("khoa đó") and is_reference("Bác sĩ ấy") and is_reference("như vậy")
    assert not is_reference("khoa tim mạch") and not is_reference("bác sĩ An")


def test_memory_is_bounded_and_rendered_oldest_first() -> None:
    memory: list[TurnNote] = []
    for i in range(MAX_TURNS + 3):
        memory = append_note(memory, TurnNote(said=f"câu {i}", result="trò chuyện"))
    assert len(memory) == MAX_TURNS and memory[-1].said == f"câu {MAX_TURNS + 2}"
    text = render_memory(memory) or ""
    assert text.splitlines()[-1].startswith("- lượt vừa rồi: người dùng nói «câu")
    assert render_memory([]) is None


def test_long_user_text_is_truncated() -> None:
    note = note_for_turn("a" * 1000, None, "chat", None, [], DEPTS, DOCTORS)
    assert len(note.said) <= 200
