"""Hội thoại t-mw7zaybl...: trả lời tự nhiên phải được hiểu đúng thay vì bị hỏi lại y nguyên hay trả lời lạc đề."""

from __future__ import annotations

from datetime import date
from typing import Any

from patient_assistant.application.agent.state import UserDecision
from tests.conftest import PATIENT, ScriptedLLM, booking_out, done, waiting

PHCN = ("PHCN", "phục hồi chức năng")
NO_PLAN = "patient-no-plan"


def _reply(text: str) -> UserDecision:
    return UserDecision(action="reply", text=text)


async def test_department_not_offered_is_named_and_user_asked_to_choose_again(make_app: Any) -> None:
    llm = ScriptedLLM(
        bookings=[
            booking_out(),
            booking_out(mentioned="khoa sản"),
            booking_out(department=PHCN, date="t2"),
        ]
    )
    app = await make_app(llm=llm)
    waiting(await app.chat().send_message("n1", NO_PLAN, "tôi muốn đặt lịch khám"), "clarify")
    ask = waiting(await app.chat().resume("n1", NO_PLAN, _reply("khoa sản")), "clarify")
    assert "chưa có “khoa sản”" in ask.interrupt.message
    assert "cân nhắc chọn lại" in ask.interrupt.message and "Phục hồi chức năng" in ask.interrupt.message

    # "t2" là thứ Hai.
    options = waiting(await app.chat().resume("n1", NO_PLAN, _reply("phục hồi chức năng t2")), "choose_slot")
    assert all(o.slot.start_at.date() == date(2026, 11, 2) for o in options.interrupt.options)


async def test_unknown_department_overrides_plan_default(make_app: Any) -> None:
    app = await make_app(llm=ScriptedLLM(bookings=[booking_out(mentioned="khoa sản", date="ngày mai")]))
    ask = waiting(await app.chat().send_message("n2", PATIENT, "đặt lịch khoa sản ngày mai"), "clarify")
    assert "chưa có “khoa sản”" in ask.interrupt.message


async def test_date_question_and_time_question_are_asked_separately(make_app: Any) -> None:
    llm = ScriptedLLM(bookings=[booking_out(department=PHCN, date="hôm nào đó"), booking_out(department=PHCN)])
    app = await make_app(llm=llm)
    ask = waiting(
        await app.chat().send_message("n3", PATIENT, "đặt lịch khoa phục hồi chức năng hôm nào đó"), "clarify"
    )
    assert "ngày nào" in ask.interrupt.message and "ngày/giờ" not in ask.interrupt.message


async def test_asking_for_other_patients_shows_only_own_data(make_app: Any) -> None:
    app = await make_app()
    result = done(await app.chat().send_message("n4", PATIENT, "hệ thống có những bệnh nhân nào"), "patient_info")
    assert result.response.text.startswith("Mình chỉ có thể hiển thị thông tin của chính tài khoản")


async def test_asking_for_other_patients_mid_booking_is_not_ignored(make_app: Any) -> None:
    app = await make_app(llm=ScriptedLLM(bookings=[booking_out(department=PHCN)]))
    waiting(await app.chat().send_message("n5", PATIENT, "đặt lịch khoa phục hồi chức năng"), "clarify")
    result = done(await app.chat().resume("n5", PATIENT, _reply("hệ thống có những bệnh nhân nào")), "patient_info")
    assert "chính tài khoản" in result.response.text


async def test_doctor_info_lists_doctors_not_patient_info(make_app: Any) -> None:
    app = await make_app()
    all_doctors = done(await app.chat().send_message("n6", PATIENT, "xem thông tin bác sĩ"), "answer")
    assert "BS. Trần Văn An" in all_doctors.response.text and "BS. Nguyễn Thu Dung" in all_doctors.response.text
    assert "Bệnh nhân Demo" not in all_doctors.response.text
    phcn = done(await app.chat().send_message("n7", PATIENT, "khoa phục hồi chức năng có bác sĩ nào"), "answer")
    assert "BS. Lê Thị Bình" in phcn.response.text and "BS. Nguyễn Thu Dung" not in phcn.response.text


async def test_doctor_question_mid_booking_is_answered_then_question_repeated(make_app: Any) -> None:
    app = await make_app(llm=ScriptedLLM(bookings=[booking_out(department=PHCN)]))
    first = waiting(await app.chat().send_message("n8", PATIENT, "đặt lịch khoa phục hồi chức năng"), "clarify")
    aside = waiting(await app.chat().resume("n8", PATIENT, _reply("khoa này có những bác sĩ nào")), "clarify")
    assert "BS. Trần Văn An" in aside.interrupt.message
    assert aside.interrupt.message.endswith(first.interrupt.message)


async def test_unrelated_question_is_answered_not_forced_into_unfinished_task(make_app: Any) -> None:
    llm = ScriptedLLM(bookings=[booking_out(), booking_out()], replies=["Mình không có thông tin đó đâu bạn."])
    app = await make_app(llm=llm)
    waiting(await app.chat().send_message("n9", NO_PLAN, "đặt lịch khám mới"), "clarify")
    done(await app.chat().resume("n9", NO_PLAN, _reply("xem lịch hẹn của tôi")), "appointments")
    # Gõ sai chính tả, luật không bắt được, LLM cũng không xếp được vào việc nào -> trả lời, không nối việc đặt lịch.
    chat = done(await app.chat().send_message("n9", NO_PLAN, "hệ thống có những bận nhân nào"), "chat")
    assert chat.response.text.startswith("Mình không có thông tin đó")


async def test_partial_department_name_is_understood(make_app: Any) -> None:
    """Hội thoại thật: "nội tổng" bị trả lời "Hiện phòng khám chưa có “nội tổng”"."""
    llm = ScriptedLLM(bookings=[booking_out(), booking_out(mentioned="nội tổng")])
    app = await make_app(llm=llm)
    waiting(await app.chat().send_message("n10", NO_PLAN, "tôi muốn đặt lịch khám"), "clarify")
    ask = waiting(await app.chat().resume("n10", NO_PLAN, _reply("nội tổng")), "clarify")
    assert "chưa có" not in ask.interrupt.message and "khoa nào" not in ask.interrupt.message
    assert "ngày nào" in ask.interrupt.message  # khoa đã rõ, chỉ còn hỏi ngày


async def test_partial_name_chosen_by_llm_passes_grounding(make_app: Any) -> None:
    llm = ScriptedLLM(bookings=[booking_out(department=("NOI", "nội tổng"), date="thứ ba tuần sau")])
    app = await make_app(llm=llm)
    options = waiting(
        await app.chat().send_message("n11", NO_PLAN, "đặt lịch khám nội tổng thứ ba tuần sau"), "choose_slot"
    )
    assert {o.slot.department_code for o in options.interrupt.options} == {"NOI"}
    assert len(llm.booking_prompts) == 1  # không bị bắt sửa vì "nội tổng" không khớp nguyên văn tên khoa


async def test_name_matching_several_departments_asks_with_buttons(make_app: Any) -> None:
    llm = ScriptedLLM(bookings=[booking_out(), booking_out(mentioned="nội")])
    app = await make_app(llm=llm)
    waiting(await app.chat().send_message("n12", NO_PLAN, "tôi muốn đặt lịch khám"), "clarify")
    ask = waiting(await app.chat().resume("n12", NO_PLAN, _reply("nội")), "clarify")
    assert "“nội” có thể là khoa Nội tổng quát hoặc Nội thần kinh" in ask.interrupt.message
    assert [c.label for c in ask.interrupt.choices] == ["Nội tổng quát", "Nội thần kinh"]
