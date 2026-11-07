"""Gọi Claude THẬT trên Bedrock (tốn phí, cần credential AWS). Không chạy mặc định.

pytest -m llm -q
"""

from __future__ import annotations

import pytest

from patient_assistant.config import Settings
from patient_assistant.container import build_llm
from tests.conftest import PATIENT, done, waiting

pytestmark = pytest.mark.llm


@pytest.fixture
async def bedrock_app(make_app):  # noqa: ANN001, ANN201
    settings = Settings()
    return await make_app(llm=build_llm(settings), with_history=True)


async def test_loose_report_is_understood_and_validated(bedrock_app) -> None:  # noqa: ANN001
    result = await bedrock_app.chat.send_message(
        "live-1", PATIENT, "báo cáo tuần này: gối đau chừng 4 trên 10, tập được ba buổi, đỡ hơn tuần trước"
    )
    fields = {e.field: e.value for e in waiting(result, "confirm_report").interrupt.explanations}
    assert (fields["pain_level"], fields["exercise_days"]) == (4, 3)


async def test_vague_booking_goes_to_booking_flow(bedrock_app) -> None:  # noqa: ANN001
    result = await bedrock_app.chat.send_message(
        "live-2", PATIENT, "dạo này gối hơi nhức, cho tôi hẹn gặp bác sĩ sáng mai"
    )
    assert result.interrupt is not None and result.interrupt.kind in {"choose_slot", "clarify"}


async def test_patient_info(bedrock_app) -> None:  # noqa: ANN001
    done(
        await bedrock_app.chat.send_message("live-3", PATIENT, "cho mình xem tình hình của mình dạo này"),
        "patient_info",
    )
