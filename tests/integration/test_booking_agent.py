"""Đặt lịch kiểu agent (PA_BOOKING_MODE=agent): LLM tự tra dữ liệu, code giữ cổng xác nhận trước khi ghi.

LLM giả ở đây đóng vai Claude: gọi tool thật (qua schema thật) theo kịch bản, nên test kiểm tra đúng các hàng rào
của code — slot phải do backend trả, chỉ ghi sau khi xác nhận, câu tiếp theo đi thẳng vào agent đang hỏi dở.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from patient_assistant.application.agent.nodes.assistant_agent import SYSTEM_NOTE
from patient_assistant.application.agent.state import UserDecision
from patient_assistant.application.ports.llm import AgentMessage, AgentReply, ToolSpec
from patient_assistant.domain.conversation import Intent
from patient_assistant.domain.text.vi_dates import DateWindow
from tests.conftest import NOW, PATIENT, TZ, ScriptedLLM, call_tool, done, waiting

SAME_AS_CURRENT = "Đặt cho tôi một lịch khám giống lịch hiện tại nhưng cách 7 ngày"
Tools = dict[str, ToolSpec]


async def _make(make_app: Any, llm: ScriptedLLM) -> Any:
    return await make_app(llm=llm, booking_mode="agent")


async def _book_current(app: Any, day: date = date(2026, 11, 3)) -> dict[str, Any]:
    """Lịch hiện tại của bệnh nhân: slot đầu tiên của khoa PHCN ngày `day`."""
    tools = app.toolbox()
    slots = await tools.get_available_slots(PATIENT, "PHCN", DateWindow(date_from=day, date_to=day), None, NOW)
    booked = await tools.create_appointment(PATIENT, slots.data[0]["slot_id"], "tái khám", "setup-current", NOW)
    assert booked.ok
    return booked.data


async def _appointments(app: Any) -> list[dict[str, Any]]:
    return (await app.toolbox().list_my_appointments(PATIENT, NOW)).data


async def _search(tools: Tools, department: str, day: date, doctor: str | None = None, at: str | None = None) -> Any:
    until = None
    if at:
        until = (datetime.strptime(at, "%H:%M") + timedelta(minutes=30)).strftime("%H:%M")
    found = await call_tool(
        tools,
        "search_slots",
        department_code=department,
        doctor_id=doctor,
        date_from=day.isoformat(),
        date_to=day.isoformat(),
        time_from=at,
        time_to=until,
    )
    assert found["status"] == "ok", found
    return found["data"]


async def _propose(
    tools: Tools, slot_id: str, basis: str = "Theo yêu cầu của bạn", reason: str | None = None
) -> AgentReply:
    result = await call_tool(tools, "propose_booking", slot_id=slot_id, reason=reason, basis=basis)
    assert result["status"] == "ok", result
    return AgentReply(text="", stopped_by="propose_booking")


async def same_as_current_plus_7(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
    """Cách Claude xử lý câu "giống lịch hiện tại, cách 7 ngày": tra lịch, tự cộng ngày, tìm đúng bác sĩ/giờ."""
    current = (await call_tool(tools, "find_my_appointments", when="upcoming", department_code=None))["data"][0]
    start = datetime.fromisoformat(current["start_at"]).astimezone(TZ)
    hhmm = f"{start:%H:%M}"
    slots = await _search(
        tools, current["department_code"], start.date() + timedelta(days=7), current["doctor_id"], hhmm
    )
    basis = f"Giống lịch {current['department_name']} {hhmm} ngày {start:%d/%m}, cách 7 ngày"
    # Claude thật hay chép lý do khám của lịch cũ sang; thẻ phải ghi đúng nguồn (không phải "người dùng nói").
    return await _propose(tools, slots[0]["slot_id"], basis, reason=current["reason"])


async def test_same_as_current_appointment_seven_days_later_without_questions(make_app: Any) -> None:
    # Claude thật xếp "xem lịch" trước vì câu nhắc "lịch hiện tại"; ở chế độ agent vẫn phải vào thẳng agent.
    llm = ScriptedLLM(intents=[[Intent.LIST_APPOINTMENTS, Intent.BOOK_APPOINTMENT]], agents=[same_as_current_plus_7])
    app = await _make(make_app, llm)
    current = await _book_current(app)

    confirm = waiting(await app.chat().send_message("a1", PATIENT, SAME_AS_CURRENT), "confirm_booking")
    # Không hỏi khoa/ngày/giờ: đi thẳng tới thẻ xác nhận, kèm căn cứ để người dùng kiểm tra.
    assert "cách 7 ngày" in confirm.interrupt.message
    explain = {e.field: e for e in confirm.interrupt.explanations}
    assert set(explain) >= {"slot", "basis", "reason"}
    assert explain["reason"].value == "tái khám" and explain["reason"].origin.value == "backend"
    assert llm.booking_prompts == []  # không dùng bộ trích xuất của luồng workflow
    assert [s.node for s in confirm.steps if s.node == "agent"] == ["agent"]
    assert len(await _appointments(app)) == 1  # chưa ghi gì trước khi xác nhận

    final = done(await app.chat().resume("a1", PATIENT, UserDecision(action="confirm")), "appointment_booked")
    booked = await _appointments(app)
    assert len(booked) == 2
    new = next(a for a in booked if a["id"] != current["id"])
    old_start = datetime.fromisoformat(current["slot"]["start_at"])
    new_start = datetime.fromisoformat(new["slot"]["start_at"])
    assert new_start - old_start == timedelta(days=7)
    assert new["slot"]["doctor_id"] == current["slot"]["doctor_id"]
    assert new["id"] in final.response.text


async def test_agent_cannot_propose_a_slot_backend_did_not_return(make_app: Any) -> None:
    async def invents_slot(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        result = await call_tool(tools, "propose_booking", slot_id="BS-BIA-20261110-0900", reason=None, basis="đoán")
        assert result["status"] == "error"
        assert "search_slots" in result["error"]["message"]
        return AgentReply(text="Mình cần tìm slot còn trống trước. Bạn muốn khám ngày nào?", tool_calls=1)

    llm = ScriptedLLM(intents=[[Intent.BOOK_APPOINTMENT]], agents=[invents_slot])
    app = await _make(make_app, llm)
    result = done(await app.chat().send_message("a2", PATIENT, "đặt lịch khám cho tôi"), "chat")
    assert "tìm slot" in result.response.text
    assert await _appointments(app) == []
    state = await app.chat().inspect_state("a2")
    assert state is not None and state.agent_active


async def test_search_outside_policy_returns_error_to_the_agent(make_app: Any) -> None:
    async def asks_past_date(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        result = await call_tool(
            tools,
            "search_slots",
            department_code="PHCN",
            doctor_id=None,
            date_from="2026-10-01",
            date_to="2026-10-02",
            time_from=None,
            time_to=None,
        )
        assert result["status"] == "error"
        assert result["error"]["details"]
        return AgentReply(text="Ngày đó đã qua rồi, bạn chọn ngày khác nhé.")

    app = await _make(make_app, ScriptedLLM(intents=[[Intent.BOOK_APPOINTMENT]], agents=[asks_past_date]))
    done(await app.chat().send_message("a3", PATIENT, "đặt lịch khám ngày 1/10"), "chat")


async def test_follow_up_goes_to_agent_and_change_at_confirm_card_is_understood(make_app: Any) -> None:
    tuesday = date(2026, 11, 3)

    async def asks(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        return AgentReply(text="Bạn muốn khám khoa nào, ngày nào?")

    async def first_slot(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        # Lượt 2 thấy cả lượt 1: câu hỏi của agent và câu người dùng trả lời.
        assert [m.role for m in history] == ["user", "assistant", "user"]
        assert history[1].text == "Bạn muốn khám khoa nào, ngày nào?"
        slots = await _search(tools, "PHCN", tuesday)
        return await _propose(tools, slots[0]["slot_id"])

    later: list[str] = []

    async def later_slot(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        assert history[-1].text == "cho mình giờ muộn hơn"
        slots = await _search(tools, "PHCN", tuesday)
        later.append(slots[1]["slot_id"])
        return await _propose(tools, slots[1]["slot_id"])

    llm = ScriptedLLM(intents=[[Intent.BOOK_APPOINTMENT]], agents=[asks, first_slot, later_slot])
    app = await _make(make_app, llm)
    chat = app.chat()
    done(await chat.send_message("a4", PATIENT, "tôi muốn đặt lịch khám"), "chat")

    # "khoa phục hồi chức năng thứ ba" không cần phân loại lại: agent đang hỏi dở nên nhận luôn.
    first = waiting(
        await chat.send_message("a4", PATIENT, "khoa phục hồi chức năng thứ ba tuần sau"), "confirm_booking"
    )
    second = waiting(
        await chat.resume("a4", PATIENT, UserDecision(action="reply", text="cho mình giờ muộn hơn")), "confirm_booking"
    )
    assert first.interrupt.message != second.interrupt.message
    assert await _appointments(app) == []

    done(await chat.resume("a4", PATIENT, UserDecision(action="reply", text="xác nhận")), "appointment_booked")
    booked = await _appointments(app)
    assert [a["slot"]["slot_id"] for a in booked] == later  # đặt đúng slot của đề xuất sau cùng

    # Việc đặt lịch đã xong: câu tiếp theo không còn bị dồn vào agent.
    calls = len(llm.agent_calls)
    done(await chat.send_message("a4", PATIENT, "cảm ơn"), "chat")
    assert len(llm.agent_calls) == calls
    state = await chat.inspect_state("a4")
    assert state is not None and not state.agent_active
    assert state.agent_history[-1].text.startswith("Đã đặt lịch:")


async def test_other_task_while_agent_is_asking_leaves_the_agent(make_app: Any) -> None:
    async def asks(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        return AgentReply(text="Bạn muốn khám ngày nào?")

    llm = ScriptedLLM(intents=[[Intent.BOOK_APPOINTMENT]], agents=[asks])
    app = await _make(make_app, llm)
    done(await app.chat().send_message("a5", PATIENT, "tôi muốn đặt lịch khám"), "chat")
    done(await app.chat().send_message("a5", PATIENT, "cho tôi xem lịch sử báo cáo"), "history")
    assert len(llm.agent_calls) == 1
    state = await app.chat().inspect_state("a5")
    assert state is not None and not state.agent_active


async def test_slot_taken_before_write_goes_back_to_agent(make_app: Any) -> None:
    proposed: list[str] = []

    async def propose_first(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        slots = await _search(tools, "PHCN", date(2026, 11, 4))
        proposed.append(slots[0]["slot_id"])
        return await _propose(tools, slots[0]["slot_id"])

    async def told_about_conflict(tools: Tools, history: list[AgentMessage], context: str) -> AgentReply:
        assert history[-1].text.startswith(SYSTEM_NOTE)
        return AgentReply(text="Lịch đó vừa có người đặt mất, bạn muốn chọn giờ khác không?")

    llm = ScriptedLLM(intents=[[Intent.BOOK_APPOINTMENT]], agents=[propose_first, told_about_conflict])
    app = await _make(make_app, llm)
    waiting(await app.chat().send_message("a6", PATIENT, "đặt lịch khám thứ tư tuần sau"), "confirm_booking")
    # Trong lúc người dùng còn xem thẻ, chính slot đó bị đặt (ở đây: bởi một yêu cầu khác của cùng bệnh nhân).
    taken = await app.toolbox().create_appointment(PATIENT, proposed[0], None, "other-request", NOW)
    assert taken.ok

    result = done(await app.chat().resume("a6", PATIENT, UserDecision(action="confirm")), "chat")
    assert "đặt mất" in result.response.text
    assert len(await _appointments(app)) == 1  # chỉ có lịch của yêu cầu kia, không ghi trùng


async def test_workflow_mode_is_still_the_default(make_app: Any) -> None:
    llm = ScriptedLLM(intents=[[Intent.BOOK_APPOINTMENT]])
    app = await make_app(llm=llm)
    waiting(await app.chat().send_message("a7", PATIENT, SAME_AS_CURRENT), "clarify")
    assert llm.agent_calls == []
