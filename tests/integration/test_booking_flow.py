from __future__ import annotations

from datetime import date
from typing import Any

from patient_assistant.application.agent.state import UserDecision
from tests.conftest import NOW, PATIENT, ScriptedLLM, done, waiting

REQUEST = "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng"


async def _appointments(app: Any, patient: str = PATIENT) -> list[dict[str, Any]]:
    return (await app.toolbox.list_my_appointments(patient, NOW)).data


async def test_voice_style_booking_happy_path(make_app: Any) -> None:
    app = make_app()
    options = waiting(await app.chat.send_message("b1", PATIENT, REQUEST, "voice"), "choose_slot")
    slots = options.interrupt.options
    assert len(slots) == 3
    assert all(o.slot.start_at.date() == date(2026, 11, 3) and o.slot.start_at.hour < 12 for o in slots)
    assert "1. Thứ Ba, 03/11/2026" in options.interrupt.message

    confirm = waiting(
        await app.chat.resume("b1", PATIENT, UserDecision(action="reply", text="cái thứ hai")), "confirm_booking"
    )
    explain = {e.field: e for e in confirm.interrupt.explanations}
    assert explain["slot"].origin.value == "backend"
    assert "tuần kế tiếp" in explain["date_expression"].explanation
    assert await _appointments(app) == []

    final = done(
        await app.chat.resume("b1", PATIENT, UserDecision(action="reply", text="xác nhận")), "appointment_booked"
    )
    booked = await _appointments(app)
    assert len(booked) == 1 and booked[0]["slot"]["slot_id"] == slots[1].slot.slot_id
    assert booked[0]["id"] in final.response.text


async def test_weekday_date_conflict_is_asked(make_app: Any) -> None:
    app = make_app()
    ask = waiting(
        await app.chat.send_message("b2", PATIENT, "đặt lịch khoa phục hồi chức năng thứ hai ngày 5/11"), "clarify"
    )
    assert "Thứ Năm" in ask.interrupt.message
    options = waiting(
        await app.chat.resume("b2", PATIENT, UserDecision(action="reply", text="thứ năm ngày 5/11")), "choose_slot"
    )
    assert all(o.slot.start_at.date() == date(2026, 11, 5) for o in options.interrupt.options)


async def test_department_defaults_to_plan_from_backend(make_app: Any) -> None:
    app = make_app()
    options = waiting(
        await app.chat.send_message("b3", PATIENT, "đặt lịch tái khám ngày mai buổi chiều"), "choose_slot"
    )
    assert {o.slot.department_code for o in options.interrupt.options} == {"PHCN"}
    assert all(o.slot.start_at.hour >= 13 for o in options.interrupt.options)
    confirm = waiting(
        await app.chat.resume("b3", PATIENT, UserDecision(action="choose", option_no=1)), "confirm_booking"
    )
    dept = {e.field: e for e in confirm.interrupt.explanations}["department_code"]
    assert dept.origin.value == "backend"


async def test_department_guessed_from_symptoms_is_rejected(make_app: Any) -> None:
    guessed = {
        "department_code": "CXK",
        "doctor_name": None,
        "date_expression": "ngày mai",
        "time_expression": None,
        "reason": None,
        "missing_fields": [],
        "evidence": [],
    }
    app = make_app(llm=ScriptedLLM(bookings=[guessed]), repos=None)
    # Bệnh nhân không có kế hoạch -> sau khi LLM sửa về null, phải hỏi khoa thay vì đoán.
    ask = waiting(
        await app.chat.send_message("b4", "patient-no-plan", "gối tôi đau, đặt lịch khám ngày mai"), "clarify"
    )
    assert "khoa nào" in ask.interrupt.message


async def test_slot_taken_at_confirmation_offers_new_slots(make_app: Any) -> None:
    app = make_app()
    options = waiting(await app.chat.send_message("b5", PATIENT, REQUEST), "choose_slot").interrupt.options
    waiting(await app.chat.resume("b5", PATIENT, UserDecision(action="choose", option_no=1)), "confirm_booking")
    # Người khác đặt mất slot trong lúc bệnh nhân đang xác nhận.
    taken = await app.toolbox.create_appointment("someone-else", options[0].slot.slot_id, None, "other-key", NOW)
    assert taken.ok
    again = waiting(await app.chat.resume("b5", PATIENT, UserDecision(action="confirm")), "choose_slot")
    assert "vừa được người khác đặt" in again.interrupt.message
    assert options[0].slot.slot_id not in {o.slot.slot_id for o in again.interrupt.options}
    assert await _appointments(app) == []


async def test_ambiguous_choice_and_other_time(make_app: Any) -> None:
    app = make_app()
    waiting(await app.chat.send_message("b6", PATIENT, REQUEST), "choose_slot")
    # 2 slot đầu cùng 08:00 với hai bác sĩ khác nhau -> "8 giờ" là mơ hồ.
    again = waiting(await app.chat.resume("b6", PATIENT, UserDecision(action="reply", text="8 giờ")), "choose_slot")
    assert "nhiều lịch khớp" in again.interrupt.message
    later = waiting(
        await app.chat.resume("b6", PATIENT, UserDecision(action="reply", text="giờ khác, buổi chiều")), "choose_slot"
    )
    assert all(o.slot.start_at.hour >= 13 for o in later.interrupt.options)


async def test_sunday_and_cancel(make_app: Any) -> None:
    app = make_app()
    ask = waiting(
        await app.chat.send_message("b7", PATIENT, "đặt lịch khoa phục hồi chức năng chủ nhật tuần sau"), "clarify"
    )
    assert "không làm việc" in ask.interrupt.message
    done(await app.chat.resume("b7", PATIENT, UserDecision(action="reply", text="thôi hủy")), "cancelled")


async def test_list_appointments_and_multi_intent_notice(make_app: Any) -> None:
    app = make_app()
    done(await app.chat.send_message("b8", PATIENT, "xem lịch hẹn của tôi"), "appointments")
    waiting(
        await app.chat.send_message("b9", PATIENT, "tuần này đau mức 4, tập 5 ngày, và đặt lịch tái khám"),
        "confirm_report",
    )
    final = done(await app.chat.resume("b9", PATIENT, UserDecision(action="confirm")), "report_saved")
    assert "đặt lịch khám" in final.response.text
