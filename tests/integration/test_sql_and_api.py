from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from patient_assistant.application.services.booking_service import BookingService
from patient_assistant.domain.booking.policy import BookingPolicy
from patient_assistant.infrastructure.persistence.sql import repositories as sql
from patient_assistant.infrastructure.persistence.sql.setup import create_schema, make_engine, seed_demo_data
from patient_assistant.interface.api.app import create_app
from tests.conftest import NOW, PATIENT, TZ


@pytest.fixture
async def sessions(tmp_path: Any) -> Any:
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await create_schema(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    await seed_demo_data(factory, NOW.date(), TZ)
    yield factory
    await engine.dispose()


async def test_sql_booking_is_unique_per_slot_and_idempotent(sessions: Any) -> None:
    service = BookingService(sql.SqlSlotRepository(sessions), sql.SqlAppointmentRepository(sessions), BookingPolicy())
    slot_id = "doc-an-20261103-0900"
    results = await asyncio.gather(*[service.book(f"patient-{i}", slot_id, None, f"key-{i}", NOW) for i in range(5)])
    assert sum(r.ok for r in results) == 1
    assert {r.error.code.value for r in results if not r.ok} == {"conflict"}

    retry_a = await service.book("patient-x", "doc-an-20261103-0930", None, "same-key", NOW)
    retry_b = await service.book("patient-x", "doc-an-20261103-0930", None, "same-key", NOW)
    assert retry_a.ok and retry_b.ok and retry_a.data["id"] == retry_b.data["id"]


async def test_sql_slot_search_excludes_booked(sessions: Any) -> None:
    from patient_assistant.domain.text.vi_dates import resolve

    service = BookingService(
        sql.SqlSlotRepository(sessions, TZ), sql.SqlAppointmentRepository(sessions, TZ), BookingPolicy()
    )
    window = resolve("thứ ba tuần sau", "buổi sáng", NOW).window
    first = await service.search("PHCN", window, None, 3, NOW)
    booked_id = first.data[0]["slot_id"]
    assert (await service.book(PATIENT, booked_id, None, "k1", NOW)).ok
    second = await service.search("PHCN", window, None, 3, NOW)
    assert booked_id not in [s["slot_id"] for s in second.data]
    assert all(s["start_at"].startswith("2026-11-03T0") and s["start_at"].endswith("+07:00") for s in second.data)


async def test_http_api_round_trip(make_app: Any) -> None:
    container = make_app()
    app = create_app()
    app.state.container = container  # bỏ qua lifespan, tiêm container đã lắp sẵn
    headers = {"X-Patient-Id": PATIENT}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/chat/api-1/messages",
            headers=headers,
            json={
                "text": "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng",
                "input_source": "voice",
            },
        )
        assert r.status_code == 200 and r.json()["interrupt"]["kind"] == "choose_slot"
        r = await client.post("/v1/chat/api-1/resume", headers=headers, json={"action": "choose", "option_no": 2})
        assert r.json()["interrupt"]["kind"] == "confirm_booking"
        r = await client.post("/v1/chat/api-1/resume", headers=headers, json={"action": "confirm"})
        assert r.json()["status"] == "done" and r.json()["response"]["kind"] == "appointment_booked"
        assert len((await client.get("/v1/me/appointments", headers=headers)).json()) == 1

        # Không có câu hỏi đang chờ / người khác resume -> 409
        assert (
            await client.post("/v1/chat/api-1/resume", headers=headers, json={"action": "confirm"})
        ).status_code == 409
        assert (await client.post("/v1/chat/api-1/messages", json={"text": "hi"})).status_code == 422
        stt = await client.post("/v1/stt/transcribe", headers=headers, files={"audio": ("a.wav", b"123")})
        assert stt.status_code == 503


async def test_ui_and_decision_trace_endpoints(make_app: Any) -> None:
    container = make_app()
    app = create_app()
    app.state.container = container
    headers = {"X-Patient-Id": PATIENT}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/")).headers["location"] == "/ui/"
        page = await client.get("/ui/")
        assert page.status_code == 200 and "Cách agent ra quyết định" in page.text
        assert (await client.get("/ui/app.js")).status_code == 200

        meta = (await client.get("/v1/meta")).json()
        assert meta["llm_model"] == "offline-rules" and meta["limits"]["max_corrections"] == 2
        assert {d["code"] for d in meta["departments"]} == {"PHCN", "CXK", "NOI"}

        turn = (
            await client.post(
                "/v1/chat/ui-1/messages",
                headers=headers,
                json={"text": "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng"},
            )
        ).json()
        nodes = [(s["flow"], s["node"], s["kind"]) for s in turn["steps"]]
        assert nodes[:3] == [("main", "ingest", "code"), ("main", "guard_input", "code"), ("main", "classify", "code")]
        assert ("booking_flow", "extract_booking", "llm") in nodes
        assert nodes[-1] == ("main", "interrupt", "hitl")  # interrupt chỉ xuất hiện một lần
        resolve_step = next(s for s in turn["steps"] if s["node"] == "resolve_datetime")
        assert "2026-11-03" in resolve_step["summary"] and "next_week_weekday" in resolve_step["summary"]

        state = (await client.get("/v1/chat/ui-1/state", headers=headers)).json()
        assert state["intent"] == "book_appointment" and len(state["offered_slots"]) == 3
        assert state["attempts"][0]["passed"] is True
        # Người khác không xem được state của hội thoại này.
        assert (await client.get("/v1/chat/ui-1/state", headers={"X-Patient-Id": "other"})).status_code == 404
