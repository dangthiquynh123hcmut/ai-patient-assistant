"""Tra cứu thông tin bệnh nhân và hỏi lại có gợi ý khi agent chưa hiểu."""

from __future__ import annotations

from typing import Any

from patient_assistant.application.agent.state import UserDecision
from patient_assistant.domain.conversation import Intent
from tests.conftest import PATIENT, ScriptedLLM, booking_out, done, waiting


async def _app_with_history(make_app: Any) -> Any:
    return await make_app(with_history=True)


async def test_lookup_patient_info_summarises_own_data(make_app: Any) -> None:
    app = await _app_with_history(make_app)
    result = done(await app.chat.send_message("i1", PATIENT, "tôi muốn tra cứu thông tin bệnh nhân"), "patient_info")
    text = result.response.text
    assert "Bệnh nhân Demo 1" in text and "Phục hồi chức năng" in text
    assert "2026-W43: đau 5/10" in text and "giảm từ 7/10" in text  # xu hướng tính bằng code
    assert "06/11/2026" in text  # lịch hẹn sắp tới
    assert "plan-knee-01" in result.response.source_ids


async def test_asking_about_another_patient_only_returns_own_data(make_app: Any) -> None:
    app = await _app_with_history(make_app)
    result = done(await app.chat.send_message("i2", PATIENT, "cho tôi xem hồ sơ của bệnh nhân khác"), "patient_info")
    assert result.response.text.startswith("Mình chỉ có thể hiển thị thông tin của chính tài khoản")
    assert "Bệnh nhân Demo 1" in result.response.text  # vẫn là dữ liệu của chính mình


async def test_vague_request_gets_targeted_suggestions(make_app: Any) -> None:
    app = await make_app()
    ask = waiting(await app.chat.send_message("i3", PATIENT, "tôi cần giúp về lịch"), "clarify")
    labels = [c.label for c in ask.interrupt.choices]
    assert labels[:2] == ["Đặt lịch khám", "Xem lịch hẹn"]  # các intent đoán được xếp trước
    assert "đặt lịch khám hoặc xem lịch hẹn" in ask.interrupt.message
    chosen = next(c for c in ask.interrupt.choices if c.label == "Xem lịch hẹn")
    done(await app.chat.resume("i3", PATIENT, UserDecision(action="reply", text=chosen.text)), "appointments")


async def test_loose_phrasing_is_understood_without_asking(make_app: Any) -> None:
    llm = ScriptedLLM(intents=[[Intent.BOOK_APPOINTMENT]], bookings=[booking_out(date="ngày mai", time="buổi sáng")])
    app = await make_app(llm=llm, with_history=True)
    # Không khớp cụm từ nào của luật -> LLM phân loại là đặt lịch, không cần hỏi lại.
    result = await app.chat.send_message("i4", PATIENT, "tôi muốn gặp bác sĩ khám ngày mai buổi sáng")
    assert result.interrupt is not None and result.interrupt.kind == "choose_slot"
