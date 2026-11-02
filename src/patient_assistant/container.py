"""Composition root: nơi DUY NHẤT biết implementation cụ thể và lắp chúng vào port."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from patient_assistant.application.agent.deps import AgentDeps, AgentLimits
from patient_assistant.application.agent.graph import build_agent_graph
from patient_assistant.application.agent.side_talk import SideTalk
from patient_assistant.application.agent.state import checkpoint_types
from patient_assistant.application.ports.clock import Clock
from patient_assistant.application.ports.llm import LLMPort
from patient_assistant.application.ports.repositories import ConversationLogRepository
from patient_assistant.application.ports.stt import STTPort
from patient_assistant.application.services.auth_service import AuthService, hash_password
from patient_assistant.application.services.booking_service import BookingService
from patient_assistant.application.services.report_service import ReportService
from patient_assistant.application.tools.toolbox import Toolbox
from patient_assistant.application.use_cases.chat import ChatUseCase
from patient_assistant.application.use_cases.conversation import ConversationService
from patient_assistant.application.use_cases.transcribe import TranscribeUseCase
from patient_assistant.config import Settings
from patient_assistant.domain.account import User
from patient_assistant.domain.booking.policy import BookingPolicy
from patient_assistant.infrastructure.audit import StructlogAuditLog
from patient_assistant.infrastructure.calendar.vn_holidays import vietnam_holiday_checker
from patient_assistant.infrastructure.clock import SystemClock
from patient_assistant.infrastructure.persistence import memory, seed


@dataclass
class Container:
    chat: ChatUseCase
    conversations: ConversationService
    auth: AuthService
    log: ConversationLogRepository
    transcribe: TranscribeUseCase
    toolbox: Toolbox
    clock: Clock
    settings: Settings
    patients: list[dict[str, str]] = field(default_factory=list)


def checkpoint_serde() -> JsonPlusSerializer:
    """Serializer chỉ giải mã các kiểu của AgentState (an toàn hơn mặc định "cho phép mọi kiểu")."""
    return JsonPlusSerializer(allowed_msgpack_modules=sorted(checkpoint_types()))


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=checkpoint_serde())


def build_llm(settings: Settings) -> LLMPort:
    if settings.llm_provider == "bedrock":
        from patient_assistant.infrastructure.llm.bedrock import AnthropicBedrockLLM

        return AnthropicBedrockLLM.from_region(
            settings.aws_region,
            settings.llm_model_id,
            aws_profile=settings.aws_profile,
            effort_extract=settings.llm_effort_extract,
            effort_light=settings.llm_effort_light,
        )
    from patient_assistant.infrastructure.llm.offline import OfflineRuleLLM

    return OfflineRuleLLM()


def build_stt(settings: Settings) -> STTPort:
    from patient_assistant.infrastructure.stt.faster_whisper import FasterWhisperSTT, UnavailableSTT

    if not settings.stt_enabled:
        return UnavailableSTT()
    hint = (
        "Khoa: "
        + ", ".join(d.name for d in seed.DEPARTMENTS)
        + ". Bác sĩ: "
        + ", ".join(d.name for d in seed.DOCTORS)
        + "."
    )
    return FasterWhisperSTT(settings.stt_model, compute_type=settings.stt_compute_type, initial_prompt=hint)


@cache
def _demo_users(password: str, created_at: Any) -> tuple[tuple[User, str], ...]:
    """Băm mật khẩu demo một lần cho cả tiến trình (scrypt chậm có chủ đích)."""
    return tuple(
        (
            User(id=f"user-{a.username}", created_at=created_at, **a.model_dump(exclude={"password"})),
            hash_password(a.password),
        )
        for a in seed.demo_accounts(password)
    )


def in_memory_repositories(clock: Clock, with_history: bool = False, seed_password: str = "demo1234") -> dict[str, Any]:
    """Repo in-memory nạp từ `demo_data/*.json`. Test/eval dùng `with_history=False` để bắt đầu sạch."""
    now = clock.now()
    slots = seed.generate_slots(now.date(), now.tzinfo)  # type: ignore[arg-type]
    appointments = seed.history_appointments(now.date(), slots) if with_history else []
    reports = seed.history_reports(now.date(), now.tzinfo) if with_history else []  # type: ignore[arg-type]
    store = memory.InMemoryBookingStore(slots, appointments)
    return {
        "users": memory.InMemoryUserRepository(list(_demo_users(seed_password, now))),
        "sessions": memory.InMemorySessionRepository(),
        "conversation_log": memory.InMemoryConversationLog(),
        "reports": memory.InMemoryReportRepository(reports),
        "catalog": memory.InMemoryCatalogRepository(seed.DEPARTMENTS, seed.DOCTORS),
        "plans": memory.InMemoryPlanRepository(seed.PLANS),
        "patients": memory.InMemoryPatientRepository(seed.profiles()),
        "slots": store,
        "appointments": memory.InMemoryAppointmentRepository(store),
        "knowledge": memory.InMemoryKnowledgeRepository(seed.KNOWLEDGE),
    }


async def sql_repositories(url: str, clock: Clock, create: bool = True, seed_data: bool = True) -> dict[str, Any]:
    """Repo SQL. `create`/`seed_data` tắt khi schema do Alembic quản lý và dữ liệu do CLI seed nạp."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from patient_assistant.infrastructure.persistence.sql import accounts
    from patient_assistant.infrastructure.persistence.sql import repositories as sql
    from patient_assistant.infrastructure.persistence.sql.setup import (
        create_schema,
        make_engine,
        seed_demo_data,
    )

    if url.startswith("sqlite") and ":///" in url:  # tạo thư mục chứa file DB nếu chưa có
        Path(url.split(":///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(url)
    if create:
        await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = clock.now()
    if seed_data:
        await seed_demo_data(sessions, now.date(), now.tzinfo, with_history=True)  # type: ignore[arg-type]
    return {
        "users": accounts.SqlUserRepository(sessions),
        "sessions": accounts.SqlSessionRepository(sessions),
        "conversation_log": accounts.SqlConversationLog(sessions),
        "reports": sql.SqlReportRepository(sessions),
        "catalog": sql.SqlCatalogRepository(sessions),
        "plans": sql.SqlPlanRepository(sessions),
        "patients": sql.SqlPatientRepository(sessions),
        "slots": sql.SqlSlotRepository(sessions, now.tzinfo),  # type: ignore[arg-type]
        "appointments": sql.SqlAppointmentRepository(sessions, now.tzinfo),  # type: ignore[arg-type]
        "knowledge": sql.SqlKnowledgeRepository(sessions),
    }


def assemble(
    settings: Settings,
    repos: dict[str, Any],
    llm: LLMPort,
    stt: STTPort,
    clock: Clock,
    checkpointer: Any,
    audit: Any = None,
    policy: BookingPolicy | None = None,
) -> Container:
    """Lắp ráp thuần (không I/O) — test dùng trực tiếp với repo in-memory, LLM giả và FixedClock."""
    policy = policy or BookingPolicy()
    toolbox = Toolbox(
        reports=ReportService(repos["reports"]),
        booking=BookingService(repos["slots"], repos["appointments"], policy),
        catalog=repos["catalog"],
        plans=repos["plans"],
        patients=repos["patients"],
        knowledge=repos["knowledge"],
        slot_limit=settings.slot_options,
    )
    limits = AgentLimits(
        max_corrections=settings.max_corrections,
        max_clarify_rounds=settings.max_clarify_rounds,
        max_slot_search_rounds=settings.max_slot_search_rounds,
    )
    deps = AgentDeps(llm=llm, tools=toolbox, audit=audit or StructlogAuditLog(), limits=limits)
    graph = build_agent_graph(deps, checkpointer)
    chat = ChatUseCase(graph, clock, limits.recursion_limit, SideTalk(llm, toolbox))
    auth = AuthService(
        repos["users"], repos["sessions"], repos["patients"], clock, timedelta(hours=settings.session_ttl_hours)
    )
    return Container(
        chat=chat,
        conversations=ConversationService(chat, repos["conversation_log"], clock),
        auth=auth,
        log=repos["conversation_log"],
        transcribe=TranscribeUseCase(stt, settings.stt_confidence_threshold),
        toolbox=toolbox,
        clock=clock,
        settings=settings,
        patients=[{"id": p.id, "label": p.label} for p in seed.PATIENTS],
    )


async def postgres_checkpointer(stack: AsyncExitStack, settings: Settings) -> Any:
    """Checkpointer LangGraph trên Postgres, dùng connection pool (mỗi worker một pool, đóng khi tắt app).

    State hội thoại (kể cả câu hỏi đang chờ người dùng trả lời) nằm ở DB nên sống qua restart và dùng chung
    giữa các worker/instance.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    if not settings.checkpointer_url:
        raise ValueError("PA_CHECKPOINTER_URL là bắt buộc khi dùng checkpointer postgres")
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=settings.checkpointer_url,
        min_size=1,
        max_size=settings.checkpointer_pool_size,
        open=False,
        # Theo yêu cầu của AsyncPostgresSaver: autocommit, không prepared statement (an toàn với pgbouncer), dict row.
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
    )
    await pool.open(wait=True)
    stack.push_async_callback(pool.close)
    return AsyncPostgresSaver(pool, serde=checkpoint_serde())


@asynccontextmanager
async def build_container(settings: Settings) -> AsyncIterator[Container]:
    """Container cho ứng dụng thật; quản lý vòng đời kết nối (checkpointer Postgres)."""
    clock = SystemClock(settings.timezone)
    async with AsyncExitStack() as stack:
        if settings.checkpointer == "postgres":
            checkpointer = await postgres_checkpointer(stack, settings)
            # Production: bảng checkpoint do `cli.seed` tạo trước, không để các worker tranh nhau tạo.
            if settings.auto_create_schema:
                await checkpointer.setup()
        else:
            checkpointer = memory_checkpointer()
        repos = (
            await sql_repositories(
                settings.database_url, clock, create=settings.auto_create_schema, seed_data=settings.seed_demo_data
            )
            if settings.database_url
            else in_memory_repositories(clock, with_history=True, seed_password=settings.seed_password)
        )
        policy = BookingPolicy(is_holiday=vietnam_holiday_checker())
        container = assemble(
            settings, repos, build_llm(settings), build_stt(settings), clock, checkpointer, policy=policy
        )
        if settings.seed_demo_data:
            await container.auth.seed(seed.demo_accounts(settings.seed_password))  # chỉ khi DB chưa có tài khoản
        yield container
