from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from patient_assistant.application.ports.llm import (
    BookingPrompt,
    GroundedAnswer,
    LLMError,
    ReportPrompt,
    ToolSpec,
)
from patient_assistant.application.use_cases.chat import TurnResult
from patient_assistant.config import Settings
from patient_assistant.container import Container, Storage, assemble, open_storage, seed_demo_users
from patient_assistant.domain.booking.models import SlotOption
from patient_assistant.domain.conversation import Intent
from patient_assistant.infrastructure.clock import FixedClock
from patient_assistant.infrastructure.persistence.seed import DEMO_PATIENT_ID
from patient_assistant.infrastructure.stt.faster_whisper import UnavailableSTT

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
NOW = datetime(2026, 10, 28, 10, 0, tzinfo=TZ)  # Thứ Tư
PATIENT = DEMO_PATIENT_ID

# Test chạy với Postgres thật: `docker compose run --rm --build test` (compose đặt sẵn biến này).
TEST_DATABASE_URL = os.environ.get("PA_TEST_DATABASE_URL", "")


def pytest_configure(config: pytest.Config) -> None:
    if not TEST_DATABASE_URL:
        pytest.exit("Đặt PA_TEST_DATABASE_URL (Postgres) hoặc chạy: docker compose run --rm --build test", 2)
    asyncio.run(_ensure_database(TEST_DATABASE_URL))


async def _ensure_database(url: str) -> None:
    """Tạo database test nếu chưa có (kết nối qua database `postgres` mặc định của server)."""
    import asyncpg

    dsn = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    base, name = dsn.rsplit("/", 1)
    conn = await asyncpg.connect(f"{base}/postgres")
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


def pg_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, database_url=TEST_DATABASE_URL, **overrides)  # type: ignore[call-arg]


def _next(queue: list[Any], default: Any) -> Any:
    if not queue:
        return default
    out = queue.pop(0)
    if isinstance(out, Exception):
        raise out
    return out


class ScriptedLLM:
    """LLM giả, không có luật hiểu ngôn ngữ: trả lần lượt các đầu ra định sẵn trong test.

    Hết kịch bản thì trả kết quả "không hiểu" (không intent, không trích được trường nào).
    """

    def __init__(
        self,
        reports: list[Any] | None = None,
        bookings: list[Any] | None = None,
        intents: list[Any] | None = None,
        slots: list[Any] | None = None,
        searches: list[Any] | None = None,
        replies: list[Any] | None = None,
    ) -> None:
        self.reports = list(reports or [])
        self.bookings = list(bookings or [])
        self.intents = list(intents or [])
        self.slots = list(slots or [])
        self.searches = list(searches or [])
        self.replies = list(replies or [])
        self.report_prompts: list[ReportPrompt] = []
        self.booking_prompts: list[BookingPrompt] = []
        self.memories: list[str | None] = []  # ghi chú lịch sử nhận được ở mỗi lần gọi classify/answer/converse

    async def classify_intent(self, text: str, memory: str | None = None) -> list[Intent]:
        self.memories.append(memory)
        return _next(self.intents, [])

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        self.report_prompts.append(prompt)
        return _next(self.reports, {"missing_fields": [], "evidence": []})

    async def extract_booking(self, prompt: BookingPrompt) -> dict[str, Any]:
        self.booking_prompts.append(prompt)
        return _next(self.bookings, {"missing_fields": ["department_code", "date_expression"], "evidence": []})

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        return _next(self.slots, None)

    async def answer_question(
        self, question: str, tools: list[ToolSpec], max_tool_calls: int, memory: str | None = None
    ) -> GroundedAnswer:
        """Kịch bản là câu truy vấn: gọi tool tìm tài liệu thật rồi trả lời bằng tài liệu đầu tiên."""
        self.memories.append(memory)
        query = _next(self.searches, None)
        search = next((t for t in tools if t.name == "search_approved_knowledge"), None)
        if query is None or search is None:
            return GroundedAnswer(text="")
        result = await search.handler(search.input_model.model_validate({"query": query}))
        docs = result.get("data") or []
        if not docs:
            return GroundedAnswer(text="", tool_calls=1)
        top = docs[0]
        return GroundedAnswer(text=f"{top['content']} ({top['doc_id']})", source_ids=[top["doc_id"]], tool_calls=1)

    async def converse(self, text: str, context: str | None = None, memory: str | None = None) -> str:
        self.memories.append(memory)
        return _next(self.replies, "Chào bạn! Mình là trợ lý ảo của phòng khám.")


class FailingLLM(ScriptedLLM):
    async def classify_intent(self, text: str, memory: str | None = None) -> list[Intent]:
        raise LLMError("unavailable", "down", retryable=True)

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        raise LLMError("refusal", "declined")

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        raise LLMError("unavailable", "down")

    async def answer_question(
        self, question: str, tools: list[ToolSpec], max_tool_calls: int, memory: str | None = None
    ) -> GroundedAnswer:
        raise LLMError("unavailable", "down")

    async def converse(self, text: str, context: str | None = None, memory: str | None = None) -> str:
        raise LLMError("unavailable", "down")


def report_out(
    pain: tuple[int, str] | None = None,
    days: tuple[int, str] | None = None,
    change: tuple[str, str] | None = None,
    activity: str | None = None,
) -> dict[str, Any]:
    """Đầu ra extract_report như LLM thật trả: (giá trị, trích dẫn nguyên văn) cho từng trường."""
    fields = {"pain_level": pain, "exercise_days": days, "symptom_change": change}
    out: dict[str, Any] = {k: v[0] if v else None for k, v in fields.items()}
    out |= {"reporting_period": None, "triggering_activity": activity}
    out["missing_fields"] = [f for f in ("pain_level", "exercise_days") if out[f] is None]
    out["evidence"] = [{"field": k, "quote": v[1]} for k, v in fields.items() if v]
    if activity:
        out["evidence"].append({"field": "triggering_activity", "quote": activity})
    return out


def booking_out(
    department: tuple[str, str] | None = None,
    date: str | None = None,
    time: str | None = None,
    doctor: str | None = None,
    reason: str | None = None,
    mentioned: str | None = None,
) -> dict[str, Any]:
    """Đầu ra extract_booking: khoa kèm trích dẫn, các cụm còn lại là nguyên văn lời người dùng.

    `mentioned`: tên khoa người dùng nói nhưng không có trong danh mục (department_code để null).
    """
    found = {"date_expression": date, "time_expression": time, "doctor_name": doctor, "reason": reason}
    evidence = [{"field": k, "quote": v} for k, v in found.items() if v]
    if department:
        evidence.insert(0, {"field": "department_code", "quote": department[1]})
    return {
        "department_code": department[0] if department else None,
        "department_mentioned": mentioned or (department[1] if department else None),
        **found,
        "missing_fields": [k for k, v in (("department_code", department), ("date_expression", date)) if not v],
        "evidence": evidence,
    }


@pytest.fixture
async def make_storage() -> AsyncIterator[Callable[..., Awaitable[Storage]]]:
    """Mỗi lần gọi: một schema Postgres mới (bảng + dữ liệu demo + tài khoản demo), xóa khi test kết thúc."""
    async with AsyncExitStack() as stack:

        async def _make(with_history: bool = False, settings: Settings | None = None) -> Storage:
            settings = settings or pg_settings()
            schema = f"t_{uuid.uuid4().hex[:12]}"
            storage = await open_storage(stack, settings, FixedClock(NOW), schema=schema, with_history=with_history)
            await seed_demo_users(storage.sessions, settings.seed_password, NOW)
            return storage

        yield _make


@pytest.fixture
async def make_app(make_storage: Callable[..., Awaitable[Storage]]) -> Callable[..., Awaitable[Container]]:
    async def _make(llm: Any = None, with_history: bool = False, **overrides: Any) -> Container:
        settings = pg_settings(**overrides)
        storage = await make_storage(with_history=with_history, settings=settings)
        clock = FixedClock(NOW)
        return assemble(settings, storage.repos, llm or ScriptedLLM(), UnavailableSTT(), clock, storage.checkpointer)

    return _make


def waiting(result: TurnResult, kind: str) -> TurnResult:
    assert result.status == "waiting_user", result
    assert result.interrupt is not None and result.interrupt.kind == kind, result.interrupt
    return result


def done(result: TurnResult, kind: str) -> TurnResult:
    assert result.status == "done", result
    assert result.response is not None and result.response.kind.value == kind, result.response
    return result


DEMO_PASSWORD = "demo1234"


def api_client(container: Container) -> Any:
    """Client HTTP gọi thẳng app ASGI, dùng container đã lắp sẵn (bỏ qua lifespan)."""
    import httpx

    from patient_assistant.interface.api.app import create_app

    app = create_app()
    app.state.container = container
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def login(client: Any, username: str, password: str = DEMO_PASSWORD) -> dict[str, str]:
    """Đăng nhập, trả header Bearer cho các request sau."""
    r = await client.post("/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    token = r.cookies.get("pa_session")
    client.cookies.clear()  # mỗi request tự chọn danh tính qua header, không dính cookie của lần đăng nhập trước
    return {"Authorization": f"Bearer {token}"}
