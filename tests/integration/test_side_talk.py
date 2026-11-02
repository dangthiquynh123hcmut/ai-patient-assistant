"""Câu người dùng gõ khi agent đang chờ: hỏi chen ngang, đổi việc, rồi quay lại việc dang dở."""

from __future__ import annotations

from typing import Any

from patient_assistant.application.agent.state import UserDecision
from tests.conftest import PATIENT, done, waiting


def _reply(text: str) -> UserDecision:
    return UserDecision(action="reply", text=text)


async def test_aside_question_is_answered_and_same_question_asked_again(make_app: Any) -> None:
    app = make_app()
    first = waiting(await app.chat.send_message("s1", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    aside = waiting(
        await app.chat.resume("s1", PATIENT, _reply("bạn có biết tôi là ai đâu mà hỏi rồi lưu dữ liệu vào đâu")),
        "clarify",
    )
    assert "patient-demo-001" in aside.interrupt.message  # biết người dùng là ai: từ phiên đăng nhập
    assert "chưa có gì được lưu" in aside.interrupt.message
    assert aside.interrupt.message.endswith(first.interrupt.message)  # hỏi lại đúng câu đang chờ
    assert [s.node for s in aside.steps] == ["side_reply"]
    # Câu hỏi chen ngang không vào dữ liệu và không tính lượt hỏi lại.
    state = await app.chat.inspect_state("s1")
    assert state.user_replies == [] and state.clarify_rounds == 0
    waiting(await app.chat.resume("s1", PATIENT, _reply("tập được 4 ngày")), "confirm_report")


async def test_small_talk_while_waiting_does_not_break_flow(make_app: Any) -> None:
    app = make_app()
    waiting(await app.chat.send_message("s2", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    thanks = waiting(await app.chat.resume("s2", PATIENT, _reply("cảm ơn nhé")), "clarify")
    assert "Không có gì" in thanks.interrupt.message
    waiting(await app.chat.resume("s2", PATIENT, _reply("tập được 2 ngày")), "confirm_report")


async def test_switch_task_then_continue_unfinished_report(make_app: Any) -> None:
    app = make_app()
    waiting(await app.chat.send_message("s3", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    info = done(await app.chat.resume("s3", PATIENT, _reply("tôi muốn tra cứu thông tin bệnh nhân")), "patient_info")
    assert info.steps[0].node == "switch_task"
    assert "nhắn “làm tiếp”" in info.response.text

    resumed = waiting(await app.chat.send_message("s3", PATIENT, "vậy giờ bạn lưu đi"), "clarify")
    assert resumed.interrupt.message.startswith("Mình tiếp tục việc gửi báo cáo theo dõi")
    draft = waiting(await app.chat.resume("s3", PATIENT, _reply("tập được 4 ngày")), "confirm_report")
    assert [e.value for e in draft.interrupt.explanations if e.field == "pain_level"] == [3]  # nhớ câu cũ
    saved = done(await app.chat.resume("s3", PATIENT, UserDecision(action="confirm")), "report_saved")
    assert "Đã lưu" in saved.response.text


async def test_unfinished_task_survives_unrelated_turns_but_not_a_new_report(make_app: Any) -> None:
    app = make_app()
    waiting(await app.chat.send_message("s4", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    done(await app.chat.resume("s4", PATIENT, _reply("xem lịch hẹn của tôi")), "appointments")
    later = done(await app.chat.send_message("s4", PATIENT, "xin chào"), "chat")
    assert "làm tiếp" not in later.response.text  # chỉ nhắc một lần
    waiting(await app.chat.send_message("s4", PATIENT, "làm tiếp đi"), "clarify")


async def test_continuation_without_unfinished_task_is_not_invented(make_app: Any) -> None:
    app = make_app()
    waiting(await app.chat.send_message("s5", PATIENT, "vậy giờ bạn lưu đi"), "clarify")


async def test_booking_reply_with_a_different_request_switches(make_app: Any) -> None:
    app = make_app()
    request = "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng"
    waiting(await app.chat.send_message("s6", PATIENT, request), "choose_slot")
    done(await app.chat.resume("s6", PATIENT, _reply("cho tôi xem các báo cáo trước")), "history")
    back = waiting(await app.chat.send_message("s6", PATIENT, "đặt tiếp lịch lúc nãy"), "choose_slot")
    assert back.interrupt.options
