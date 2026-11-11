from __future__ import annotations

import asyncio
from typing import Any

import pytest

from patient_assistant.application.services.booking_service import BookingService
from patient_assistant.domain.booking.policy import BookingPolicy
from patient_assistant.infrastructure.persistence.sql import repositories as sql
from patient_assistant.infrastructure.persistence.sql.setup import seed_demo_data
from tests.conftest import NOW, PATIENT, TZ, ScriptedLLM, api_client, booking_out, login


def _booking_llm() -> ScriptedLLM:
    out = booking_out(department=("PHCN", "phục hồi chức năng"), date="thứ ba tuần sau", time="buổi sáng")
    return ScriptedLLM(bookings=[out])


@pytest.fixture
async def sessions(make_storage: Any) -> Any:
    return (await make_storage()).sessions


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
    first = await service.search("patient-x", "PHCN", window, None, 3, NOW)
    booked_id = first.data[0]["slot_id"]
    assert (await service.book(PATIENT, booked_id, None, "k1", NOW)).ok
    second = await service.search("patient-x", "PHCN", window, None, 3, NOW)
    assert booked_id not in [s["slot_id"] for s in second.data]
    assert all(s["start_at"].startswith("2026-11-03T0") and s["start_at"].endswith("+07:00") for s in second.data)


async def test_http_api_round_trip(make_app: Any) -> None:
    async with api_client(await make_app(llm=_booking_llm())) as client:
        headers = await login(client, "benhnhan1")
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
        assert (await client.post("/v1/chat/api-1/messages", headers=headers, json={})).status_code == 422
        assert (await client.post("/v1/chat/api-1/messages", json={"text": "hi"})).status_code == 401
        stt = await client.post("/v1/stt/transcribe", headers=headers, files={"audio": ("a.wav", b"123")})
        assert stt.status_code == 503
        for lang, code in (("vi", 503), ("auto", 503), ("fr", 422)):
            r = await client.post(
                "/v1/stt/transcribe", headers=headers, files={"audio": ("a.wav", b"123")}, data={"language": lang}
            )
            assert r.status_code == code, lang
        assert (await client.get("/v1/meta", headers=headers)).json()["stt_language"] == "en"


async def test_meta_and_decision_trace_endpoints(make_app: Any) -> None:
    async with api_client(await make_app(llm=_booking_llm())) as client:
        headers = await login(client, "benhnhan1")
        assert (await client.get("/health")).json() == {"status": "ok"}

        meta = (await client.get("/v1/meta", headers=headers)).json()
        assert meta["user"]["patient_id"] == PATIENT and meta["patients"] == []  # bệnh nhân không thấy danh sách
        assert meta["llm_model"] == "global.anthropic.claude-opus-4-6-v1" and meta["limits"]["max_corrections"] == 2
        assert {d["code"] for d in meta["departments"]} == {"PHCN", "CXK", "NOI", "DL", "TM", "TK", "TMH", "MAT"}

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
        other = await login(client, "benhnhan2")
        assert (await client.get("/v1/chat/ui-1/state", headers=other)).status_code == 404


async def test_sql_seed_with_history(make_storage: Any) -> None:
    factory = (await make_storage(with_history=True)).sessions
    await seed_demo_data(factory, NOW.date(), TZ, with_history=True)  # chạy lại không nhân đôi dữ liệu
    reports = await sql.SqlReportRepository(factory).list_by_patient(PATIENT, 10)
    appointments = await sql.SqlAppointmentRepository(factory, TZ).list_for_patient(PATIENT)
    assert [r.pain_level for r in reports] == [5, 6, 7]  # mới nhất trước
    assert len(appointments) == 1 and appointments[0].slot.start_at.tzinfo is not None


async def test_reseed_adds_departments_and_doctors_added_to_json(make_storage: Any) -> None:
    """DB đã có dữ liệu: chạy lại seed chỉ bổ sung khoa/bác sĩ mới (và slot của họ), không đụng dữ liệu cũ."""
    from sqlalchemy import delete, func, select, update

    from patient_assistant.infrastructure.persistence.sql.models import DepartmentRow, DoctorRow, PatientRow, SlotRow

    sessions = (await make_storage(with_history=True)).sessions
    async with sessions.begin() as session:  # giả lập DB cũ, từ trước khi có khoa Da liễu và ngày sinh/SĐT
        await session.execute(update(PatientRow).values(date_of_birth=None, phone=None))
        await session.execute(delete(SlotRow).where(SlotRow.department_code == "DL"))
        await session.execute(delete(DoctorRow).where(DoctorRow.department_code == "DL"))
        await session.execute(delete(DepartmentRow).where(DepartmentRow.code == "DL"))

    added = await seed_demo_data(sessions, NOW.date(), TZ, with_history=True)
    assert added > 0
    async with sessions() as session:
        assert await session.scalar(select(DepartmentRow.name).where(DepartmentRow.code == "DL")) == "Da liễu"
        assert await session.scalar(select(func.count()).select_from(SlotRow).where(SlotRow.doctor_id == "doc-giang"))
        assert await session.scalar(select(PatientRow.phone).where(PatientRow.id == PATIENT)) == "0901000001"
    reports = await sql.SqlReportRepository(sessions).list_by_patient(PATIENT, 10)
    assert [r.pain_level for r in reports] == [5, 6, 7]  # lịch sử không bị nạp lại / nhân đôi
    assert await seed_demo_data(sessions, NOW.date(), TZ, with_history=True) == 0  # chạy lại: không thêm gì
