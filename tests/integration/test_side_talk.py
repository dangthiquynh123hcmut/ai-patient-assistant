"""Câu người dùng gõ khi agent đang chờ: hỏi chen ngang, đổi việc, rồi quay lại việc dang dở."""

from __future__ import annotations

from typing import Any

from patient_assistant.application.agent.state import UserDecision
from patient_assistant.domain.conversation import Intent
from tests.conftest import PATIENT, ScriptedLLM, booking_out, done, report_out, waiting

PAIN = (3, "tuần này tôi đau mức 3")


def _reply(text: str) -> UserDecision:
    return UserDecision(action="reply", text=text)


async def _report_app(make_app: Any, days: tuple[int, str], **script: Any) -> Any:
    """Lượt đầu chỉ có mức đau; lượt trả lời bổ sung số ngày tập."""
    reports = [report_out(pain=PAIN), report_out(pain=PAIN, days=days)]
    return await make_app(llm=ScriptedLLM(reports=reports, **script))


async def test_aside_question_is_answered_and_same_question_asked_again(make_app: Any) -> None:
    app = await _report_app(make_app, (4, "tập được 4 ngày"))
    first = waiting(await app.chat().send_message("s1", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    aside = waiting(
        await app.chat().resume("s1", PATIENT, _reply("bạn có biết tôi là ai đâu mà hỏi rồi lưu dữ liệu vào đâu")),
        "clarify",
    )
    assert "patient-demo-001" in aside.interrupt.message  # biết người dùng là ai: từ phiên đăng nhập
    assert "chưa có gì được lưu" in aside.interrupt.message
    assert aside.interrupt.message.endswith(first.interrupt.message)  # hỏi lại đúng câu đang chờ
    assert [s.node for s in aside.steps] == ["side_reply"]
    # Câu hỏi chen ngang không vào dữ liệu và không tính lượt hỏi lại.
    state = await app.chat().inspect_state("s1")
    assert state.user_replies == [] and state.clarify_rounds == 0
    waiting(await app.chat().resume("s1", PATIENT, _reply("tập được 4 ngày")), "confirm_report")


async def test_small_talk_while_waiting_does_not_break_flow(make_app: Any) -> None:
    app = await _report_app(make_app, (2, "tập được 2 ngày"), replies=["Không có gì đâu bạn!"])
    waiting(await app.chat().send_message("s2", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    thanks = waiting(await app.chat().resume("s2", PATIENT, _reply("cảm ơn nhé")), "clarify")
    assert "Không có gì" in thanks.interrupt.message
    waiting(await app.chat().resume("s2", PATIENT, _reply("tập được 2 ngày")), "confirm_report")


async def test_switch_task_then_continue_unfinished_report(make_app: Any) -> None:
    # Khi làm tiếp, báo cáo được trích xuất lại từ câu cũ trước khi hỏi tiếp.
    reports = [report_out(pain=PAIN), report_out(pain=PAIN), report_out(pain=PAIN, days=(4, "tập được 4 ngày"))]
    app = await make_app(llm=ScriptedLLM(reports=reports))
    waiting(await app.chat().send_message("s3", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    info = done(await app.chat().resume("s3", PATIENT, _reply("tôi muốn tra cứu thông tin bệnh nhân")), "patient_info")
    assert info.steps[0].node == "switch_task"
    assert "nhắn “làm tiếp”" in info.response.text

    resumed = waiting(await app.chat().send_message("s3", PATIENT, "vậy giờ bạn lưu đi"), "clarify")
    assert resumed.interrupt.message.startswith("Mình tiếp tục việc gửi báo cáo theo dõi")
    draft = waiting(await app.chat().resume("s3", PATIENT, _reply("tập được 4 ngày")), "confirm_report")
    assert [e.value for e in draft.interrupt.explanations if e.field == "pain_level"] == [3]  # nhớ câu cũ
    saved = done(await app.chat().resume("s3", PATIENT, UserDecision(action="confirm")), "report_saved")
    assert "Đã lưu" in saved.response.text


async def test_unfinished_task_survives_unrelated_turns_but_not_a_new_report(make_app: Any) -> None:
    app = await make_app()
    waiting(await app.chat().send_message("s4", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    done(await app.chat().resume("s4", PATIENT, _reply("xem lịch hẹn của tôi")), "appointments")
    later = done(await app.chat().send_message("s4", PATIENT, "xin chào"), "chat")
    assert "làm tiếp" not in later.response.text  # chỉ nhắc một lần
    waiting(await app.chat().send_message("s4", PATIENT, "làm tiếp đi"), "clarify")


async def test_continuation_without_unfinished_task_is_not_invented(make_app: Any) -> None:
    app = await make_app()
    waiting(await app.chat().send_message("s5", PATIENT, "vậy giờ bạn lưu đi"), "clarify")


async def test_booking_reply_with_a_different_request_switches(make_app: Any) -> None:
    request = "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng"
    out = booking_out(department=("PHCN", "phục hồi chức năng"), date="thứ ba tuần sau", time="buổi sáng")
    app = await make_app(llm=ScriptedLLM(bookings=[out, out]))
    waiting(await app.chat().send_message("s6", PATIENT, request), "choose_slot")
    done(await app.chat().resume("s6", PATIENT, _reply("cho tôi xem các báo cáo trước")), "history")
    back = waiting(await app.chat().send_message("s6", PATIENT, "đặt tiếp lịch lúc nãy"), "choose_slot")
    assert back.interrupt.options


# ---------------------------------------------------------------- dùng lại ngữ cảnh đã nói (hội thoại t-mw76ehck...)

PHCN = ("PHCN", "phục hồi chức năng")
NO_PLAN = "patient-no-plan"  # không có kế hoạch theo dõi -> phải hỏi khoa


async def _abandoned_booking(app: Any, conversation_id: str) -> None:
    """Đã nói khoa, đang được hỏi ngày thì đổi sang việc khác -> việc đặt lịch còn dang dở."""
    waiting(await app.chat().send_message(conversation_id, NO_PLAN, "đặt lịch khám mới"), "clarify")
    waiting(await app.chat().resume(conversation_id, NO_PLAN, _reply("phục hồi chức năng")), "clarify")
    done(await app.chat().resume(conversation_id, NO_PLAN, _reply("xem lịch hẹn của tôi")), "appointments")


async def test_asking_for_the_same_task_again_reuses_what_was_said(make_app: Any) -> None:
    llm = ScriptedLLM(
        bookings=[booking_out(), booking_out(department=PHCN), booking_out(department=PHCN, date="thứ ba tuần sau")]
    )
    app = await make_app(llm=llm)
    await _abandoned_booking(app, "c1")

    # Không nói "làm tiếp": chỉ yêu cầu lại việc đặt lịch -> không hỏi lại khoa, dùng luôn câu trả lời cũ.
    result = await app.chat().send_message("c1", NO_PLAN, "tôi muốn đặt lịch khám thứ ba tuần sau")
    sources = llm.booking_prompts[-1].sources
    assert "phục hồi chức năng" in sources and sources[-1] == "tôi muốn đặt lịch khám thứ ba tuần sau"
    options = waiting(result, "choose_slot")
    assert "dùng lại thông tin bạn đã nói" in options.interrupt.message
    assert {o.slot.department_code for o in options.interrupt.options} == {"PHCN"}


async def test_vague_request_with_unfinished_task_continues_it(make_app: Any) -> None:
    llm = ScriptedLLM(bookings=[booking_out(), booking_out(department=PHCN), booking_out(department=PHCN)])
    app = await make_app(llm=llm)
    await _abandoned_booking(app, "c2")

    # "tôi muốn xử lí ngay bây giờ": LLM không xác định được việc gì -> làm tiếp việc đặt lịch đang dở.
    ask = waiting(await app.chat().send_message("c2", NO_PLAN, "tôi muốn xử lí ngay bây giờ"), "clarify")
    assert "ngày nào" in ask.interrupt.message and "khoa nào" not in ask.interrupt.message
    assert "phục hồi chức năng" in llm.booking_prompts[-1].sources


async def test_reference_to_earlier_answers_reuses_them(make_app: Any) -> None:
    # LLM được cài sẵn để phân loại sai thành "xem báo cáo" — câu nhắc lại thông tin cũ không được hỏi LLM.
    llm = ScriptedLLM(
        bookings=[booking_out(), booking_out(department=PHCN), booking_out(department=PHCN)],
        intents=[[Intent.GET_REPORT_HISTORY]],
    )
    app = await make_app(llm=llm)
    await _abandoned_booking(app, "c3")
    ask = waiting(await app.chat().send_message("c3", NO_PLAN, "lấy thông tin cũ tôi đã trả lời phía trên"), "clarify")
    assert "ngày nào" in ask.interrupt.message
    assert "phục hồi chức năng" in llm.booking_prompts[-1].sources
