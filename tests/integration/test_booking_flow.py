from __future__ import annotations

from datetime import date
from typing import Any

from patient_assistant.application.agent.state import UserDecision
from tests.conftest import NOW, PATIENT, ScriptedLLM, booking_out, done, report_out, waiting

REQUEST = "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng"
PHCN = ("PHCN", "phục hồi chức năng")
REQUEST_OUT = booking_out(department=PHCN, date="thứ ba tuần sau", time="buổi sáng")


def _reply(text: str) -> UserDecision:
    return UserDecision(action="reply", text=text)


async def _request_app(make_app: Any) -> Any:
    return await make_app(llm=ScriptedLLM(bookings=[REQUEST_OUT]))


async def _appointments(app: Any, patient: str = PATIENT) -> list[dict[str, Any]]:
    return (await app.toolbox().list_my_appointments(patient, NOW)).data


async def test_voice_style_booking_happy_path(make_app: Any) -> None:
    app = await _request_app(make_app)
    options = waiting(await app.chat().send_message("b1", PATIENT, REQUEST, "voice"), "choose_slot")
    slots = options.interrupt.options
    assert len(slots) == 3
    assert all(o.slot.start_at.date() == date(2026, 11, 3) and o.slot.start_at.hour < 12 for o in slots)
    assert "1. Thứ Ba, 03/11/2026" in options.interrupt.message

    confirm = waiting(
        await app.chat().resume("b1", PATIENT, UserDecision(action="reply", text="cái thứ hai")), "confirm_booking"
    )
    explain = {e.field: e for e in confirm.interrupt.explanations}
    assert explain["slot"].origin.value == "backend"
    assert "tuần kế tiếp" in explain["date_expression"].explanation
    assert await _appointments(app) == []

    final = done(
        await app.chat().resume("b1", PATIENT, UserDecision(action="reply", text="xác nhận")), "appointment_booked"
    )
    booked = await _appointments(app)
    assert len(booked) == 1 and booked[0]["slot"]["slot_id"] == slots[1].slot.slot_id
    assert booked[0]["id"] in final.response.text


async def test_weekday_date_conflict_is_asked(make_app: Any) -> None:
    llm = ScriptedLLM(
        bookings=[
            booking_out(department=PHCN, date="thứ hai ngày 5/11"),
            booking_out(department=PHCN, date="thứ năm ngày 5/11"),
        ]
    )
    app = await make_app(llm=llm)
    ask = waiting(
        await app.chat().send_message("b2", PATIENT, "đặt lịch khoa phục hồi chức năng thứ hai ngày 5/11"), "clarify"
    )
    assert "Thứ Năm" in ask.interrupt.message
    options = waiting(
        await app.chat().resume("b2", PATIENT, UserDecision(action="reply", text="thứ năm ngày 5/11")), "choose_slot"
    )
    assert all(o.slot.start_at.date() == date(2026, 11, 5) for o in options.interrupt.options)


async def test_department_defaults_to_plan_from_backend(make_app: Any) -> None:
    app = await make_app(llm=ScriptedLLM(bookings=[booking_out(date="ngày mai", time="buổi chiều")]))
    options = waiting(
        await app.chat().send_message("b3", PATIENT, "đặt lịch tái khám ngày mai buổi chiều"), "choose_slot"
    )
    assert {o.slot.department_code for o in options.interrupt.options} == {"PHCN"}
    assert all(o.slot.start_at.hour >= 13 for o in options.interrupt.options)
    confirm = waiting(
        await app.chat().resume("b3", PATIENT, UserDecision(action="choose", option_no=1)), "confirm_booking"
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
    app = await make_app(llm=ScriptedLLM(bookings=[guessed]))
    # Bệnh nhân không có kế hoạch -> sau khi LLM sửa về null, phải hỏi khoa thay vì đoán.
    ask = waiting(
        await app.chat().send_message("b4", "patient-no-plan", "gối tôi đau, đặt lịch khám ngày mai"), "clarify"
    )
    assert "khoa nào" in ask.interrupt.message


async def test_slot_taken_at_confirmation_offers_new_slots(make_app: Any) -> None:
    app = await _request_app(make_app)
    options = waiting(await app.chat().send_message("b5", PATIENT, REQUEST), "choose_slot").interrupt.options
    waiting(await app.chat().resume("b5", PATIENT, UserDecision(action="choose", option_no=1)), "confirm_booking")
    # Người khác đặt mất slot trong lúc bệnh nhân đang xác nhận.
    taken = await app.toolbox().create_appointment("someone-else", options[0].slot.slot_id, None, "other-key", NOW)
    assert taken.ok
    again = waiting(await app.chat().resume("b5", PATIENT, UserDecision(action="confirm")), "choose_slot")
    assert "vừa được người khác đặt" in again.interrupt.message
    assert options[0].slot.slot_id not in {o.slot.slot_id for o in again.interrupt.options}
    assert await _appointments(app) == []


async def test_ambiguous_choice_and_other_time(make_app: Any) -> None:
    later = booking_out(department=PHCN, date="thứ ba tuần sau", time="buổi chiều")
    app = await make_app(llm=ScriptedLLM(bookings=[REQUEST_OUT, later]))
    waiting(await app.chat().send_message("b6", PATIENT, REQUEST), "choose_slot")
    # 2 slot đầu cùng 08:00 với hai bác sĩ khác nhau -> "8 giờ" là mơ hồ.
    again = waiting(await app.chat().resume("b6", PATIENT, UserDecision(action="reply", text="8 giờ")), "choose_slot")
    assert "nhiều lịch khớp" in again.interrupt.message
    later = waiting(
        await app.chat().resume("b6", PATIENT, UserDecision(action="reply", text="giờ khác, buổi chiều")), "choose_slot"
    )
    assert all(o.slot.start_at.hour >= 13 for o in later.interrupt.options)


async def test_sunday_and_cancel(make_app: Any) -> None:
    app = await make_app(llm=ScriptedLLM(bookings=[booking_out(department=PHCN, date="chủ nhật tuần sau")]))
    ask = waiting(
        await app.chat().send_message("b7", PATIENT, "đặt lịch khoa phục hồi chức năng chủ nhật tuần sau"), "clarify"
    )
    assert "không làm việc" in ask.interrupt.message
    done(await app.chat().resume("b7", PATIENT, UserDecision(action="reply", text="thôi hủy")), "cancelled")


async def test_list_appointments_and_multi_intent_notice(make_app: Any) -> None:
    report = report_out(pain=(4, "tuần này đau mức 4"), days=(5, "tập 5 ngày"))
    app = await make_app(llm=ScriptedLLM(reports=[report]))
    done(await app.chat().send_message("b8", PATIENT, "xem lịch hẹn của tôi"), "appointments")
    waiting(
        await app.chat().send_message("b9", PATIENT, "tuần này đau mức 4, tập 5 ngày, và đặt lịch tái khám"),
        "confirm_report",
    )
    final = done(await app.chat().resume("b9", PATIENT, UserDecision(action="confirm")), "report_saved")
    assert "đặt lịch khám" in final.response.text


async def test_progress_never_hands_off_and_time_of_day_reply_searches_again(make_app: Any) -> None:
    """Tái hiện hội thoại thật: trả lời đúng mọi lượt (khoa, ngày, "buổi chiều") mà vẫn bị chuyển nhân viên."""
    llm = ScriptedLLM(
        bookings=[
            booking_out(),  # "đặt lịch khám mới": chưa có khoa, chưa có ngày
            booking_out(department=PHCN),
            booking_out(department=PHCN, date="thứ ba tuần sau"),
            booking_out(department=PHCN, date="thứ ba tuần sau", time="buổi chiều"),
        ]
    )
    app = await make_app(llm=llm)
    waiting(await app.chat().send_message("b10", "patient-no-plan", "đặt lịch khám mới"), "clarify")
    waiting(await app.chat().resume("b10", "patient-no-plan", _reply("phục hồi chức năng")), "clarify")
    first = waiting(await app.chat().resume("b10", "patient-no-plan", _reply("thứ ba tuần sau")), "choose_slot")
    assert any(o.slot.start_at.hour < 12 for o in first.interrupt.options)

    # "buổi chiều" không khớp lịch nào đang hiện -> hiểu là muốn tìm lại buổi chiều, không chuyển nhân viên.
    later = waiting(await app.chat().resume("b10", "patient-no-plan", _reply("buổi chiều")), "choose_slot")
    assert later.interrupt.options and all(o.slot.start_at.hour >= 13 for o in later.interrupt.options)
    assert all(o.slot.start_at.date() == date(2026, 11, 3) for o in later.interrupt.options)
    waiting(await app.chat().resume("b10", "patient-no-plan", _reply("cái đầu tiên")), "confirm_booking")


async def test_same_unanswerable_slot_question_still_hands_off(make_app: Any) -> None:
    app = await _request_app(make_app)
    waiting(await app.chat().send_message("b11", PATIENT, REQUEST), "choose_slot")
    for _ in range(3):  # hỏi lại đúng câu đó tối đa 3 lần
        again = waiting(await app.chat().resume("b11", PATIENT, _reply("abc xyz")), "choose_slot")
        assert "chưa xác định được" in again.interrupt.message
    final = done(await app.chat().resume("b11", PATIENT, _reply("abc xyz")), "handoff")
    assert "nhân viên" in final.response.text


async def test_own_appointment_clash_at_confirmation_keeps_request_and_offers_other_times(make_app: Any) -> None:
    app = await _request_app(make_app)  # chỉ có MỘT đầu ra extract: hỏi lại từ đầu sẽ không có dữ liệu
    options = waiting(await app.chat().send_message("b-own", PATIENT, REQUEST), "choose_slot").interrupt.options
    chosen = options[0].slot
    waiting(await app.chat().resume("b-own", PATIENT, UserDecision(action="choose", option_no=1)), "confirm_booking")
    # Trong lúc chờ xác nhận, bệnh nhân tự đặt một lịch Tim mạch trùng giờ (tab khác).
    tm_slot = f"doc-hai-{chosen.start_at:%Y%m%d-%H%M}"
    assert (await app.toolbox().create_appointment(PATIENT, tm_slot, None, "tab-2", NOW)).ok

    again = waiting(await app.chat().resume("b-own", PATIENT, UserDecision(action="confirm")), "choose_slot")
    assert "Trùng giờ" in again.interrupt.message
    new = [o.slot for o in again.interrupt.options]
    assert new and all(s.department_code == "PHCN" and s.start_at.date() == date(2026, 11, 3) for s in new)
    assert all(s.start_at.hour < 12 for s in new)  # vẫn giữ "buổi sáng" người dùng đã nói
    assert all(abs((s.start_at - chosen.start_at).total_seconds()) >= 3600 for s in new)  # 30' slot + 30' đệm
    assert all(o.hint and "Tim mạch" in o.hint for o in again.interrupt.options)

    waiting(await app.chat().resume("b-own", PATIENT, UserDecision(action="choose", option_no=1)), "confirm_booking")
    done(await app.chat().resume("b-own", PATIENT, UserDecision(action="confirm")), "appointment_booked")
    booked = {a["slot"]["slot_id"] for a in await _appointments(app)}
    assert booked == {tm_slot, new[0].slot_id}


async def test_same_day_slots_are_recommended_next_to_existing_appointment(make_app: Any) -> None:
    week = booking_out(department=PHCN, date="tuần sau")
    app = await make_app(llm=ScriptedLLM(bookings=[week]))
    # Đã có lịch Tim mạch chiều thứ Năm 05/11 lúc 14:00.
    assert (await app.toolbox().create_appointment(PATIENT, "doc-hai-20261105-1400", None, "tm", NOW)).ok

    options = waiting(
        await app.chat().send_message("b-rec", PATIENT, "đặt lịch phục hồi chức năng tuần sau"), "choose_slot"
    )
    slots = [o.slot for o in options.interrupt.options]
    assert all(s.start_at.date() == date(2026, 11, 5) for s in slots)
    assert slots[0].start_at.hour == 15 and slots[0].start_at.minute == 0  # sát nhất mà vẫn đủ 30' đi lại
    assert all(not (14 <= s.start_at.hour < 15) for s in slots)  # 14:00 trùng, 14:30 không kịp di chuyển
    assert all(o.hint and "14:00" in o.hint for o in options.interrupt.options)
    assert "gợi ý: cùng ngày với lịch khám Tim mạch lúc 14:00" in options.interrupt.message


async def test_no_existing_appointments_means_plain_time_order_and_no_hint(make_app: Any) -> None:
    app = await _request_app(make_app)
    options = waiting(await app.chat().send_message("b-plain", PATIENT, REQUEST), "choose_slot").interrupt.options
    starts = [o.slot.start_at for o in options]
    assert starts == sorted(starts)
    assert all(o.hint is None for o in options)
