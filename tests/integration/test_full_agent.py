"""Chế độ agent chung (PA_AGENT_MODE=full): một agent cho mọi việc — đặt/dời/hủy lịch, báo cáo, tra cứu, tài liệu,
nhiều việc trong một câu. LLM giả gọi tool thật qua schema thật, nên test kiểm tra đúng các hàng rào của code."""

from __future__ import annotations

from datetime import date
from typing import Any

from patient_assistant.application.agent.nodes.assistant_agent import SYSTEM_NOTE
from patient_assistant.application.agent.state import UserDecision
from patient_assistant.application.ports.llm import AgentMessage, AgentReply, ToolSpec
from patient_assistant.domain.common import SttFlag
from patient_assistant.domain.text.vi_dates import DateWindow
from tests.conftest import NOW, PATIENT, ScriptedLLM, call_tool, done, waiting

Tools = dict[str, ToolSpec]
REPORT_SAID = (
    "Tuần này tôi đau mức 5, tập được 4 ngày, rồi đặt giúp tôi lịch tái khám phục hồi chức năng thứ ba tuần sau"
)


async def _make(make_app: Any, llm: ScriptedLLM) -> Any:
    return await make_app(llm=llm, agent_mode="full")


def _report(pain: tuple[int, str], days: tuple[int, str]) -> dict[str, Any]:
    return {
        "reporting_period": None,
        "pain_level": pain[0],
        "exercise_days": days[0],
        "symptom_change": None,
        "triggering_activity": None,
        "missing_fields": [],
        "evidence": [{"field": "pain_level", "quote": pain[1]}, {"field": "exercise_days", "quote": days[1]}],
    }


async def _propose_report(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
    result = await call_tool(tools, "propose_report", **_report((5, "đau mức 5"), (4, "tập được 4 ngày")))
    assert result["status"] == "ok", result
    return AgentReply(text="", stopped_by="propose_report")


async def _nothing_left(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
    assert history[-1].text.startswith(SYSTEM_NOTE)
    return AgentReply(text="")


async def _slots(tools: Tools, department: str, day: date) -> list[dict[str, Any]]:
    found = await call_tool(
        tools,
        "search_slots",
        department_code=department,
        doctor_id=None,
        date_from=day.isoformat(),
        date_to=day.isoformat(),
        time_from=None,
        time_to=None,
    )
    assert found["status"] == "ok", found
    return found["data"]


async def _history(app: Any) -> list[dict[str, Any]]:
    return (await app.toolbox().get_report_history(PATIENT)).data


async def _booked(app: Any) -> list[dict[str, Any]]:
    return (await app.toolbox().list_my_appointments(PATIENT, NOW)).data


async def _book(app: Any, day: date) -> dict[str, Any]:
    tools = app.toolbox()
    slots = await tools.get_available_slots(PATIENT, "PHCN", DateWindow(date_from=day, date_to=day), None, NOW)
    booked = await tools.create_appointment(PATIENT, slots.data[0]["slot_id"], "tái khám", f"setup-{day}", NOW)
    assert booked.ok
    return booked.data


async def test_report_and_booking_in_one_message(make_app: Any) -> None:
    async def then_booking(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        # Hệ thống báo việc báo cáo đã xong -> agent làm tiếp việc thứ hai trong cùng câu.
        assert history[-1].text.startswith(SYSTEM_NOTE) and "Đã lưu báo cáo" in history[-1].text
        slots = await _slots(tools, "PHCN", date(2026, 11, 3))
        result = await call_tool(
            tools, "propose_booking", slot_id=slots[0]["slot_id"], reason="tái khám", basis="Theo lời bạn"
        )
        assert result["status"] == "ok"
        return AgentReply(text="", stopped_by="propose_booking")

    async def tries_report_again(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        # Lỗi thật gặp trên UI: sau khi đặt lịch, LLM đề xuất lưu lại báo cáo đã lưu. Code phải chặn.
        note = history[-1].text
        assert "1. Đã lưu báo cáo" in note and "2. Đã đặt lịch" in note  # thông báo liệt kê đủ việc đã xong
        again = await call_tool(tools, "propose_report", **_report((5, "đau mức 5"), (4, "tập được 4 ngày")))
        assert again["status"] == "error" and "already completed" in again["error"]["message"]
        return AgentReply(text="")

    llm = ScriptedLLM(agents=[_propose_report, then_booking, tries_report_again])
    app = await _make(make_app, llm)
    chat = app.chat()

    card = waiting(await chat.send_message("f1", PATIENT, REPORT_SAID), "confirm_report")
    assert {e.field: e.value for e in card.interrupt.explanations}["pain_level"] == 5
    assert llm.intents == [] and llm.memories == []  # không qua bước phân loại

    second = waiting(await chat.resume("f1", PATIENT, UserDecision(action="confirm")), "confirm_booking")
    assert "Đã lưu báo cáo" in second.interrupt.message  # thẻ thứ hai cho thấy việc thứ nhất đã xong
    assert len(await _history(app)) == 1 and await _booked(app) == []

    final = done(await chat.resume("f1", PATIENT, UserDecision(action="confirm")), "appointment_booked")
    assert "Đã lưu báo cáo" in final.response.text and "Đã đặt lịch" in final.response.text
    assert len(await _booked(app)) == 1
    assert len(llm.agent_calls) == 3


async def test_cancelling_the_second_task_keeps_the_first_reported_as_saved(make_app: Any) -> None:
    async def then_booking(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        slots = await _slots(tools, "PHCN", date(2026, 11, 3))
        await call_tool(tools, "propose_booking", slot_id=slots[0]["slot_id"], reason=None, basis="Theo lời bạn")
        return AgentReply(text="", stopped_by="propose_booking")

    app = await _make(make_app, ScriptedLLM(agents=[_propose_report, then_booking]))
    chat = app.chat()
    waiting(await chat.send_message("f8", PATIENT, REPORT_SAID), "confirm_report")
    waiting(await chat.resume("f8", PATIENT, UserDecision(action="confirm")), "confirm_booking")
    final = done(await chat.resume("f8", PATIENT, UserDecision(action="cancel")), "cancelled")
    assert "Đã lưu báo cáo" in final.response.text
    assert "Chưa có dữ liệu nào được lưu" not in final.response.text
    assert len(await _history(app)) == 1 and await _booked(app) == []


async def test_report_needs_verbatim_evidence(make_app: Any) -> None:
    async def invents_quote(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        result = await call_tool(tools, "propose_report", **_report((5, "đau mức năm"), (4, "tập 4 buổi")))
        assert result["status"] == "error" and result["error"]["details"]
        return AgentReply(text="Bạn tập mấy ngày trong tuần này?")

    app = await _make(make_app, ScriptedLLM(agents=[invents_quote]))
    done(await app.chat().send_message("f2", PATIENT, "tuần này tôi đau mức 5"), "chat")
    assert await _history(app) == []


async def test_report_edit_on_card_is_revalidated_by_code(make_app: Any) -> None:
    llm = ScriptedLLM(agents=[_propose_report, _nothing_left])
    app = await _make(make_app, llm)
    chat = app.chat()
    waiting(await chat.send_message("f3", PATIENT, "Tuần này tôi đau mức 5, tập được 4 ngày"), "confirm_report")
    edited = waiting(
        await chat.resume("f3", PATIENT, UserDecision(action="edit", fields={"pain_level": 4})), "confirm_report"
    )
    row = {e.field: e for e in edited.interrupt.explanations}["pain_level"]
    assert row.value == 4 and row.origin.value == "user_edit"
    assert len(llm.agent_calls) == 1  # sửa trên form không cần gọi LLM
    done(await chat.resume("f3", PATIENT, UserDecision(action="confirm")), "report_saved")
    assert [r["pain_level"] for r in await _history(app)] == [4]


async def test_current_week_without_quote_is_not_a_rejection(make_app: Any) -> None:
    """LLM điền tuần hiện tại cho "tuần này" mà không trích dẫn: code bỏ đi (mặc định đã là tuần hiện tại)."""

    async def fills_current_week(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        report = _report((5, "đau mức 5"), (4, "tập được 4 ngày")) | {"reporting_period": f"{NOW:%G-W%V}"}
        result = await call_tool(tools, "propose_report", **report)
        assert result["status"] == "ok", result
        return AgentReply(text="", stopped_by="propose_report")

    app = await _make(make_app, ScriptedLLM(agents=[fills_current_week, _nothing_left]))
    chat = app.chat()
    waiting(await chat.send_message("f9", PATIENT, "Tuần này tôi đau mức 5, tập được 4 ngày"), "confirm_report")
    done(await chat.resume("f9", PATIENT, UserDecision(action="confirm")), "report_saved")
    assert [r["reporting_period"] for r in await _history(app)] == [f"{NOW:%G-W%V}"]


async def test_report_rejected_too_many_times_is_handed_to_staff(make_app: Any) -> None:
    """Cùng trần sửa với workflow: lần đề xuất sai thứ (max_corrections + 1) -> chuyển nhân viên, không lưu, không
    cho LLM thử tiếp. Lỗi phải hỏi người dùng (thiếu trường) không bị tính."""
    results: list[dict[str, Any]] = []

    async def keeps_inventing(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        missing = _report((5, "đau mức 5"), (4, "tập được 4 ngày")) | {"exercise_days": None, "evidence": []}
        missing["evidence"] = [{"field": "pain_level", "quote": "đau mức 5"}]
        results.append(await call_tool(tools, "propose_report", **missing))
        for _ in range(3):
            results.append(await call_tool(tools, "propose_report", **_report((5, "đau mức 5"), (5, "tập 5 ngày"))))
            if results[-1].get("stop"):
                return AgentReply(text="", stopped_by="propose_report")
        return AgentReply(text="unreachable")

    app = await _make(make_app, ScriptedLLM(agents=[keeps_inventing]))
    turn = done(await app.chat().send_message("f8", PATIENT, "Tuần này tôi đau mức 5, tập được 4 ngày"), "handoff")
    assert "nhân viên" in turn.response.text
    assert [r["status"] for r in results] == ["error"] * 4 and [bool(r.get("stop")) for r in results] == [
        False,
        False,
        False,
        True,
    ]
    assert await _history(app) == []


async def test_filler_reply_on_card_shows_the_same_card_without_writing(make_app: Any) -> None:
    llm = ScriptedLLM(agents=[_propose_report, _nothing_left])
    app = await _make(make_app, llm)
    chat = app.chat()
    waiting(await chat.send_message("f6", PATIENT, "Tuần này tôi đau mức 5, tập được 4 ngày"), "confirm_report")
    again = waiting(await chat.resume("f6", PATIENT, UserDecision(action="reply", text="ờ")), "confirm_report")
    assert {e.field: e.value for e in again.interrupt.explanations}["pain_level"] == 5
    assert await _history(app) == [] and len(llm.agent_calls) == 1  # không ghi, không quay lại LLM
    done(await chat.resume("f6", PATIENT, UserDecision(action="reply", text="đồng ý lưu")), "report_saved")
    assert [r["pain_level"] for r in await _history(app)] == [5]


async def test_low_confidence_voice_words_reach_the_agent(make_app: Any) -> None:
    seen: list[str] = []

    async def asks_back(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        seen.append(context)
        return AgentReply(text="Bạn tập 4 ngày đúng không?")

    app = await _make(make_app, ScriptedLLM(agents=[asks_back]))
    flag = SttFlag(text="4", probability=0.31, reason="low_confidence_key_token")
    await app.chat().send_message("f7", PATIENT, "tôi đau mức 5, đã tập 4 ngày", "voice", [flag])
    assert '"4" (confidence 0.31)' in seen[0]


async def test_reschedule_moves_the_appointment(make_app: Any) -> None:
    async def reschedule(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        current = (await call_tool(tools, "find_my_appointments", when="upcoming", department_code=None))["data"][0]
        slots = await _slots(tools, "PHCN", date(2026, 11, 5))
        result = await call_tool(
            tools,
            "propose_reschedule",
            appointment_id=current["id"],
            slot_id=slots[0]["slot_id"],
            basis="Dời sang thứ năm",
        )
        assert result["status"] == "ok", result
        return AgentReply(text="", stopped_by="propose_reschedule")

    app = await _make(make_app, ScriptedLLM(agents=[reschedule, _nothing_left]))
    old = await _book(app, date(2026, 11, 3))
    card = waiting(
        await app.chat().send_message("f4", PATIENT, "dời lịch thứ ba sang thứ năm giúp tôi"), "confirm_booking"
    )
    assert {e.field for e in card.interrupt.explanations} >= {"appointment", "slot"}
    assert [a["id"] for a in await _booked(app)] == [old["id"]]

    final = done(await app.chat().resume("f4", PATIENT, UserDecision(action="confirm")), "appointment_rescheduled")
    booked = await _booked(app)
    assert len(booked) == 1 and booked[0]["id"] != old["id"]
    assert booked[0]["slot"]["start_at"].startswith("2026-11-05")
    assert "Đã dời lịch" in final.response.text
    back = await app.toolbox().read_back_appointment(PATIENT, old["id"])
    assert back.data["status"] == "cancelled"


async def test_cancel_only_own_upcoming_appointment(make_app: Any) -> None:
    async def cancel(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        wrong = await call_tool(tools, "propose_cancel", appointment_id="khong-co", basis="đoán")
        assert wrong["status"] == "error"
        current = (await call_tool(tools, "find_my_appointments", when="upcoming", department_code=None))["data"][0]
        result = await call_tool(tools, "propose_cancel", appointment_id=current["id"], basis="Theo lời bạn")
        assert result["status"] == "ok"
        return AgentReply(text="", stopped_by="propose_cancel")

    app = await _make(make_app, ScriptedLLM(agents=[cancel, _nothing_left]))
    await _book(app, date(2026, 11, 3))
    waiting(await app.chat().send_message("f5", PATIENT, "hủy lịch khám thứ ba của tôi"), "confirm_booking")
    done(await app.chat().resume("f5", PATIENT, UserDecision(action="confirm")), "appointment_cancelled")
    assert await _booked(app) == []


async def test_knowledge_answer_keeps_sources_and_follow_up_goes_to_agent(make_app: Any) -> None:
    async def answers(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        found = await call_tool(tools, "search_approved_knowledge", query="chườm lạnh sau khi tập gập gối")
        doc = found["data"][0]
        return AgentReply(text=f"{doc['content']} ({doc['doc_id']})")

    async def sees_previous_turn(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        assert [m.role for m in history] == ["user", "assistant", "user"]
        return AgentReply(text="Không có gì đâu bạn!")

    app = await _make(make_app, ScriptedLLM(agents=[answers, sees_previous_turn]))
    faq = done(await app.chat().send_message("f6", PATIENT, "sau khi tập gập gối có nên chườm lạnh bao lâu"), "chat")
    assert faq.response.source_ids
    done(await app.chat().send_message("f6", PATIENT, "cảm ơn nhé"), "chat")


async def test_safety_still_blocks_before_the_agent(make_app: Any) -> None:
    llm = ScriptedLLM()
    app = await _make(make_app, llm)
    result = done(await app.chat().send_message("f7", PATIENT, "tôi đau ngực và khó thở"), "safety")
    assert "115" in result.response.text
    assert llm.agent_calls == []
