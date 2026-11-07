"""Chạy graph thật (LangGraph + checkpointer Postgres) qua các interrupt, với repo Postgres và LLM giả."""

from __future__ import annotations

from typing import Any

from patient_assistant.application.agent.state import UserDecision
from patient_assistant.application.ports.llm import LLMError
from tests.conftest import PATIENT, FailingLLM, ScriptedLLM, done, report_out, waiting

GOOD = "Tuần này tôi đau mức 5 trên 10, đã tập 4 ngày, thấy đau tăng khi gập gối"
GOOD_OUT = report_out(
    pain=(5, "Tuần này tôi đau mức 5 trên 10"),
    days=(4, "đã tập 4 ngày"),
    change=("worse_during_activity", "thấy đau tăng khi gập gối"),
    activity="gập gối",
)


async def _good_app(make_app: Any, times: int = 1) -> Any:
    return await make_app(llm=ScriptedLLM(reports=[GOOD_OUT] * times))


async def _history(app: Any) -> list[dict[str, Any]]:
    return (await app.toolbox.get_report_history(PATIENT)).data


async def test_happy_path_requires_confirmation_then_saves(make_app: Any) -> None:
    app = await _good_app(make_app)
    first = waiting(await app.chat.send_message("t1", PATIENT, GOOD), "confirm_report")
    fields = {e.field: e for e in first.interrupt.explanations}
    assert fields["pain_level"].value == 5 and fields["pain_level"].evidence == "Tuần này tôi đau mức 5 trên 10"
    assert fields["pain_level"].origin.value == "user_said"
    assert fields["reporting_period"].origin.value == "system_default"
    assert await _history(app) == []  # chưa lưu khi chưa xác nhận

    final = done(await app.chat.resume("t1", PATIENT, UserDecision(action="confirm")), "report_saved")
    saved = await _history(app)
    assert len(saved) == 1 and saved[0]["pain_level"] == 5 and saved[0]["exercise_days"] == 4
    assert saved[0]["id"] in final.response.text


async def test_missing_field_asks_user_then_continues(make_app: Any) -> None:
    pain = (3, "tuần này tôi đau mức 3")
    llm = ScriptedLLM(reports=[report_out(pain=pain), report_out(pain=pain, days=(2, "tập được 2 ngày"))])
    app = await make_app(llm=llm)
    ask = waiting(await app.chat.send_message("t2", PATIENT, "tuần này tôi đau mức 3"), "clarify")
    assert "bao nhiêu ngày" in ask.interrupt.message
    confirm = waiting(
        await app.chat.resume("t2", PATIENT, UserDecision(action="reply", text="tập được 2 ngày")), "confirm_report"
    )
    assert {e.field: e.value for e in confirm.interrupt.explanations}["exercise_days"] == 2
    done(await app.chat.resume("t2", PATIENT, UserDecision(action="reply", text="xác nhận")), "report_saved")


async def test_out_of_range_value_is_asked_not_silently_fixed(make_app: Any) -> None:
    days = (3, "tập 3 ngày")
    llm = ScriptedLLM(
        reports=[
            report_out(pain=(12, "tôi đau 12/10"), days=days),
            report_out(pain=(8, "à nhầm, đau mức 8"), days=days),
        ]
    )
    app = await make_app(llm=llm)
    ask = waiting(await app.chat.send_message("t3", PATIENT, "tôi đau 12/10, tập 3 ngày"), "clarify")
    assert "0–10" in ask.interrupt.message
    confirm = waiting(
        await app.chat.resume("t3", PATIENT, UserDecision(action="reply", text="à nhầm, đau mức 8")), "confirm_report"
    )
    assert {e.field: e.value for e in confirm.interrupt.explanations}["pain_level"] == 8


def _wrong_then_right() -> list[dict[str, Any]]:
    evidence = [
        {"field": "pain_level", "quote": "đau mức 5 trên 10"},
        {"field": "exercise_days", "quote": "đã tập 4 ngày"},
    ]
    base = {
        "reporting_period": None,
        "symptom_change": None,
        "triggering_activity": None,
        "missing_fields": [],
        "evidence": evidence,
        "pain_level": 5,
    }
    return [{**base, "exercise_days": 5}, {**base, "exercise_days": 4}]


async def test_self_correction_fixes_llm_error_and_is_recorded(make_app: Any) -> None:
    llm = ScriptedLLM(reports=_wrong_then_right())
    app = await make_app(llm=llm)
    confirm = waiting(await app.chat.send_message("t4", PATIENT, GOOD), "confirm_report")
    assert {e.field: e.value for e in confirm.interrupt.explanations}["exercise_days"] == 4
    # Lần sửa nhận đúng lỗi cụ thể từ validator (FR-06.1).
    feedback = llm.report_prompts[1].feedback
    assert feedback and [i.code for i in feedback.issues] == ["value_evidence_mismatch"]
    # State của subgraph chỉ gộp vào graph cha khi subgraph kết thúc -> kiểm tra sau khi lưu xong.
    done(await app.chat.resume("t4", PATIENT, UserDecision(action="confirm")), "report_saved")
    snapshot = await app.chat._graph.aget_state({"configurable": {"thread_id": "t4"}})
    attempts = snapshot.values["attempts"]
    assert [(a.attempt_no, a.passed) for a in attempts] == [(0, False), (1, True)]


async def test_correction_limit_hands_off_and_never_saves(make_app: Any) -> None:
    wrong = _wrong_then_right()[0]
    app = await make_app(llm=ScriptedLLM(reports=[wrong, wrong, wrong]))
    final = done(await app.chat.send_message("t5", PATIENT, GOOD), "handoff")
    assert "nhân viên" in final.response.text
    assert await _history(app) == []


async def test_cancel_and_unclear_confirmation(make_app: Any) -> None:
    app = await _good_app(make_app)
    waiting(await app.chat.send_message("t6", PATIENT, GOOD), "confirm_report")
    again = waiting(await app.chat.resume("t6", PATIENT, UserDecision(action="reply", text="ờ")), "confirm_report")
    assert "chưa rõ" in again.interrupt.message
    done(await app.chat.resume("t6", PATIENT, UserDecision(action="cancel")), "cancelled")
    assert await _history(app) == []


async def test_user_edit_is_revalidated(make_app: Any) -> None:
    app = await _good_app(make_app)
    waiting(await app.chat.send_message("t7", PATIENT, GOOD), "confirm_report")
    bad = waiting(
        await app.chat.resume("t7", PATIENT, UserDecision(action="edit", fields={"pain_level": 15})), "clarify"
    )
    assert "0–10" in bad.interrupt.message or "phạm vi" in bad.interrupt.message


async def test_llm_failure_is_reported_not_faked(make_app: Any) -> None:
    app = await make_app(llm=FailingLLM())
    final = done(await app.chat.send_message("t8", PATIENT, GOOD), "error")
    assert "Chưa có dữ liệu nào được lưu" in final.response.text


async def test_backend_failure_on_save_is_not_reported_as_success(make_app: Any) -> None:
    app = await _good_app(make_app)

    async def broken(*_: Any, **__: Any) -> Any:
        raise ConnectionError("db down")

    app.toolbox._reports._reports.add = broken  # type: ignore[method-assign]
    waiting(await app.chat.send_message("t9", PATIENT, GOOD), "confirm_report")
    final = done(await app.chat.resume("t9", PATIENT, UserDecision(action="confirm")), "error")
    assert "thành công" not in final.response.text.replace("KHÔNG thành công", "")


async def test_retry_with_same_content_does_not_duplicate(make_app: Any) -> None:
    app = await _good_app(make_app, times=2)
    for _ in range(2):
        waiting(await app.chat.send_message("t10", PATIENT, GOOD), "confirm_report")
        done(await app.chat.resume("t10", PATIENT, UserDecision(action="confirm")), "report_saved")
    assert len(await _history(app)) == 1


async def test_safety_and_other_intents(make_app: Any) -> None:
    llm = ScriptedLLM(
        searches=["chườm lạnh"],
        replies=["Chào bạn! Mình là trợ lý ảo của phòng khám.", "Không có gì đâu bạn!"],
    )
    app = await make_app(llm=llm)
    emergency = done(await app.chat.send_message("t11", PATIENT, "tôi đau ngực và khó thở"), "safety")
    assert "115" in emergency.response.text
    advice = done(await app.chat.send_message("t12", PATIENT, "tôi bị bệnh gì, có nên đổi thuốc không"), "safety")
    assert "chẩn đoán" in advice.response.text
    faq = done(await app.chat.send_message("t13", PATIENT, "sau khi tập gập gối có nên chườm lạnh bao lâu"), "answer")
    assert faq.response.source_ids
    hello = done(await app.chat.send_message("t14a", PATIENT, "hello"), "chat")
    assert hello.response.text.startswith("Chào bạn")
    thanks = done(await app.chat.send_message("t14b", PATIENT, "cảm ơn nhé"), "chat")
    assert "Không có gì" in thanks.response.text
    unknown = waiting(await app.chat.send_message("t14", PATIENT, "abc xyz"), "clarify")
    assert "abc xyz" in unknown.interrupt.message  # hỏi lại có nhắc đúng câu người dùng
    assert "Xem thông tin của tôi" in [c.label for c in unknown.interrupt.choices]
    done(await app.chat.resume("t14", PATIENT, UserDecision(action="reply", text="xem các báo cáo trước")), "history")


async def test_llm_errors_do_not_leak(make_app: Any) -> None:
    llm = ScriptedLLM(reports=[LLMError("max_tokens", "cut")])
    final = done(await (await make_app(llm=llm)).chat.send_message("t15", PATIENT, GOOD), "error")
    assert final.response.kind.value == "error"
