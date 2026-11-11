"""Đặt lịch trọn vẹn bằng tiếng Anh: cùng workflow, cùng validator và chốt chặn như tiếng Việt."""

from __future__ import annotations

from datetime import date
from typing import Any

from patient_assistant.application.agent.state import UserDecision
from patient_assistant.domain.common import Origin
from tests.conftest import NOW, PATIENT, ScriptedLLM, booking_out, done, waiting

REQUEST = "I'd like to book an appointment with rehabilitation next Tuesday morning"
REQUEST_OUT = booking_out(department=("PHCN", "rehabilitation"), date="next Tuesday", time="morning")


def _reply(text: str) -> UserDecision:
    return UserDecision(action="reply", text=text)


async def test_english_booking_happy_path(make_app: Any) -> None:
    app = await make_app(llm=ScriptedLLM(bookings=[REQUEST_OUT]))
    options = waiting(await app.chat.send_message("en1", PATIENT, REQUEST, "voice"), "choose_slot").interrupt.options
    assert options and all(o.slot.start_at.date() == date(2026, 11, 3) and o.slot.start_at.hour < 12 for o in options)

    confirm = waiting(await app.chat.resume("en1", PATIENT, _reply("the second one")), "confirm_booking")
    explain = {e.field: e for e in confirm.interrupt.explanations}
    assert explain["department_code"].origin == Origin.USER_SAID
    # Giọng nói: "yes" chưa đủ rõ -> hỏi lại, chưa ghi gì.
    again = waiting(await app.chat.resume("en1", PATIENT, _reply("yes")), "confirm_booking")
    assert again.interrupt is not None
    assert (await app.toolbox.list_my_appointments(PATIENT, NOW)).data == []

    final = done(await app.chat.resume("en1", PATIENT, _reply("yes, book it")), "appointment_booked")
    booked = (await app.toolbox.list_my_appointments(PATIENT, NOW)).data
    assert len(booked) == 1 and booked[0]["slot"]["slot_id"] == options[1].slot.slot_id
    assert booked[0]["id"] in final.response.text


async def test_english_other_time_and_cancel(make_app: Any) -> None:
    afternoon = booking_out(department=("PHCN", "rehabilitation"), date="next Tuesday", time="afternoon")
    app = await make_app(llm=ScriptedLLM(bookings=[REQUEST_OUT, afternoon]))
    waiting(await app.chat.send_message("en2", PATIENT, REQUEST), "choose_slot")
    later = waiting(await app.chat.resume("en2", PATIENT, _reply("something in the afternoon please")), "choose_slot")
    assert all(o.slot.start_at.hour >= 12 for o in later.interrupt.options)
    done(await app.chat.resume("en2", PATIENT, _reply("never mind, cancel")), "cancelled")
    assert (await app.toolbox.list_my_appointments(PATIENT, NOW)).data == []


async def test_english_vague_date_is_asked(make_app: Any) -> None:
    vague = booking_out(department=("PHCN", "rehabilitation"), date="this weekend")
    app = await make_app(llm=ScriptedLLM(bookings=[vague]))
    ask = waiting(
        await app.chat.send_message("en3", PATIENT, "book an appointment with rehabilitation this weekend"), "clarify"
    )
    assert ask.interrupt is not None


async def test_english_emergency_stops_booking(make_app: Any) -> None:
    app = await make_app(llm=ScriptedLLM(bookings=[REQUEST_OUT]))
    result = done(
        await app.chat.send_message("en4", PATIENT, "book an appointment, I have chest pain and can't breathe"),
        "safety",
    )
    assert "115" in result.response.text
    assert (await app.toolbox.list_my_appointments(PATIENT, NOW)).data == []
