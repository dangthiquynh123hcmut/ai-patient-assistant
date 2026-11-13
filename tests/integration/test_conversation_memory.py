"""Bộ nhớ trong MỘT cuộc hội thoại: lượt sau hiểu "như vậy", "bác sĩ đó", "khoa đó" dựa trên lượt trước."""

from __future__ import annotations

from datetime import date
from typing import Any

from patient_assistant.application.agent.state import UserDecision
from patient_assistant.domain.common import Origin
from patient_assistant.domain.conversation import Intent
from tests.conftest import NOW, PATIENT, ScriptedLLM, booking_out, done, waiting

PHCN = ("PHCN", "phục hồi chức năng")
TM = ("TM", "tim mạch")
FIRST = "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng"
FIRST_OUT = booking_out(department=PHCN, date="thứ ba tuần sau", time="buổi sáng")


async def _book(app: Any, conv: str, text: str, option_no: int = 2) -> dict[str, Any]:
    waiting(await app.chat().send_message(conv, PATIENT, text), "choose_slot")
    waiting(
        await app.chat().resume(conv, PATIENT, UserDecision(action="choose", option_no=option_no)), "confirm_booking"
    )
    final = done(await app.chat().resume(conv, PATIENT, UserDecision(action="confirm")), "appointment_booked")
    return final.response.data


async def test_same_as_before_reuses_department_and_doctor_of_booked_appointment(make_app: Any) -> None:
    again = booking_out(date="thứ năm tuần sau")  # LLM không trích được khoa: người dùng chỉ nói "như vậy"
    llm = ScriptedLLM(bookings=[FIRST_OUT, again])
    app = await make_app(llm=llm)
    first = await _book(app, "m1", FIRST)
    doctor_id = first["slot"]["doctor_id"]

    options = waiting(
        await app.chat().send_message("m1", PATIENT, "Đặt thêm cho tôi một lịch như vậy vào thứ năm tuần sau"),
        "choose_slot",
    ).interrupt.options
    assert options
    assert all(o.slot.department_code == "PHCN" and o.slot.doctor_id == doctor_id for o in options)
    assert all(o.slot.start_at.date() == date(2026, 11, 5) for o in options)

    confirm = waiting(
        await app.chat().resume("m1", PATIENT, UserDecision(action="choose", option_no=1)), "confirm_booking"
    )
    explain = {e.field: e for e in confirm.interrupt.explanations}
    assert explain["department_code"].origin == Origin.BACKEND
    assert explain["department_code"].evidence == "như vậy"
    assert first["id"] in explain["department_code"].explanation
    assert explain["doctor_name"].origin == Origin.BACKEND


async def test_reference_with_two_different_bookings_asks_instead_of_guessing(make_app: Any) -> None:
    tm_out = booking_out(department=TM, date="thứ tư tuần sau")
    vague = booking_out(date="thứ năm tuần sau")
    app = await make_app(llm=ScriptedLLM(bookings=[FIRST_OUT, tm_out, vague]))
    await _book(app, "m2", FIRST)
    await _book(app, "m2", "Đặt lịch khoa tim mạch thứ tư tuần sau")

    ask = waiting(await app.chat().send_message("m2", PATIENT, "Đặt thêm một lịch như vậy thứ năm tuần sau"), "clarify")
    # Hai lịch khác khoa -> "như vậy" không rõ; không được lấy bừa khoa theo kế hoạch.
    assert "khoa nào" in ask.interrupt.message


async def test_doctor_named_in_earlier_question_is_that_doctor(make_app: Any) -> None:
    later = booking_out(date="thứ ba tuần sau", time="buổi sáng")
    app = await make_app(llm=ScriptedLLM(intents=[[Intent.GET_DOCTOR_INFO]], bookings=[later]))
    first = await app.chat().send_message("m3", PATIENT, "Bác sĩ An khám ở khoa nào vậy?")
    assert first.status == "done"

    options = waiting(
        await app.chat().send_message("m3", PATIENT, "Cho tôi đặt lịch với bác sĩ đó thứ ba tuần sau buổi sáng"),
        "choose_slot",
    ).interrupt.options
    assert options and all(o.slot.doctor_id == "doc-an" and o.slot.department_code == "PHCN" for o in options)


async def test_earlier_turns_are_passed_to_the_llm_as_notes_only_in_same_conversation(make_app: Any) -> None:
    llm = ScriptedLLM(bookings=[FIRST_OUT], replies=["Chào bạn", "Không có gì"])
    app = await make_app(llm=llm)
    await _book(app, "m4", FIRST)
    done(await app.chat().send_message("m4", PATIENT, "cảm ơn bạn nhé"), "chat")
    memory = llm.memories[-1]
    assert memory is not None
    assert "đã đặt lịch: Phục hồi chức năng" in memory and "thứ ba tuần sau" in memory

    # Hội thoại khác: không mang lịch sử sang.
    done(await app.chat().send_message("m5", PATIENT, "cảm ơn bạn nhé"), "chat")
    assert llm.memories[-1] is None


async def test_memory_is_not_shared_with_another_patient_in_same_conversation_id(make_app: Any) -> None:
    llm = ScriptedLLM(bookings=[FIRST_OUT], replies=["Chào", "Chào"])
    app = await make_app(llm=llm)
    await _book(app, "m6", FIRST)
    state = await app.chat().inspect_state("m6")
    assert state is not None and state.turn_note is not None and state.turn_note.appointment is not None
    # Cùng conversation_id nhưng bệnh nhân khác -> bắt đầu không có ghi chú.
    from patient_assistant.application.use_cases.chat import _memory

    assert _memory(state, "patient-someone-else") == []
    assert len(_memory(state, PATIENT)) == 1


async def test_unfinished_turn_is_noted_as_not_done(make_app: Any) -> None:
    llm = ScriptedLLM(bookings=[FIRST_OUT], replies=["Chào"])
    app = await make_app(llm=llm)
    waiting(await app.chat().send_message("m7", PATIENT, FIRST), "choose_slot")
    done(await app.chat().send_message("m7", PATIENT, "xin chào"), "chat")
    memory = llm.memories[-1] or ""
    assert "chưa xong" in memory
    assert NOW  # giữ import dùng chung với các test khác
