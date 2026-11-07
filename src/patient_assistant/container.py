"""Composition root: nơi DUY NHẤT biết implementation cụ thể và lắp chúng vào port."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cache
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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
from patient_assistant.infrastructure.llm.bedrock import AnthropicBedrockLLM
from patient_assistant.infrastructure.persistence import seed
from patient_assistant.infrastructure.persistence.sql import accounts
from patient_assistant.infrastructure.persistence.sql import repositories as sql
from patient_assistant.infrastructure.persistence.sql.setup import (
    create_db_schema,
    drop_db_schema,
    make_engine,
    migrate,
    seed_demo_data,
)


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
    # Mô tả thêm cho bệnh nhân demo (demo_data/patients.json); bệnh nhân tự đăng ký hiển thị bằng họ tên.
    patient_labels: dict[str, str] = field(default_factory=dict)


def checkpoint_serde() -> JsonPlusSerializer:
    """Serializer chỉ giải mã các kiểu của AgentState (an toàn hơn mặc định "cho phép mọi kiểu")."""
    return JsonPlusSerializer(allowed_msgpack_modules=sorted(checkpoint_types()))


def build_llm(settings: Settings) -> AnthropicBedrockLLM:
    return AnthropicBedrockLLM.from_region(
        settings.aws_region,
        settings.llm_model_id,
        aws_profile=settings.aws_profile,
        effort_extract=settings.llm_effort_extract,
        effort_light=settings.llm_effort_light,
    )


async def require_llm(llm: AnthropicBedrockLLM, settings: Settings) -> None:
    """Hệ thống chỉ chạy khi gọi được LLM: thử một lần lúc khởi động, lỗi thì dừng hẳn."""
    try:
        await llm.ping()
    except Exception as exc:  # noqa: BLE001 - thiếu credential, sai quyền, mất mạng... đều là không gọi được LLM
        raise RuntimeError(
            f"Không gọi được LLM ({settings.llm_model_id}): {type(exc).__name__}: {exc}. "
            "Kiểm tra bằng: python -m patient_assistant.interface.cli.check_llm"
        ) from exc


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


async def seed_demo_users(sessions: async_sessionmaker[AsyncSession], password: str, now: Any) -> None:
    """Tạo tài khoản demo nếu DB chưa có tài khoản nào, dùng mật khẩu đã băm sẵn (nhanh cho test/eval)."""
    users = accounts.SqlUserRepository(sessions)
    if await users.count():
        return
    for user, password_hash in _demo_users(password, now):
        await users.add(user, password_hash)


def sql_repositories(sessions: async_sessionmaker[AsyncSession], tz: Any) -> dict[str, Any]:
    return {
        "users": accounts.SqlUserRepository(sessions),
        "sessions": accounts.SqlSessionRepository(sessions),
        "conversation_log": accounts.SqlConversationLog(sessions),
        "reports": sql.SqlReportRepository(sessions),
        "catalog": sql.SqlCatalogRepository(sessions),
        "plans": sql.SqlPlanRepository(sessions),
        "patients": sql.SqlPatientRepository(sessions),
        "slots": sql.SqlSlotRepository(sessions, tz),
        "appointments": sql.SqlAppointmentRepository(sessions, tz),
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
    """Lắp ráp thuần (không I/O) — test dùng trực tiếp với repo Postgres trong schema riêng, LLM giả và FixedClock."""
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
        patient_labels={p.id: p.label for p in seed.PATIENTS},
    )


def checkpointer_dsn(database_url: str) -> str:
    """URL SQLAlchemy (`postgresql+asyncpg://...`) -> DSN libpq cho psycopg mà checkpointer LangGraph dùng."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def postgres_checkpointer(stack: AsyncExitStack, settings: Settings, schema: str | None = None) -> Any:
    """Checkpointer LangGraph trên Postgres, dùng connection pool (mỗi worker một pool, đóng khi tắt app).

    State hội thoại (kể cả câu hỏi đang chờ người dùng trả lời) nằm ở DB nên sống qua restart và dùng chung
    giữa các worker/instance.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    assert settings.database_url
    # Theo yêu cầu của AsyncPostgresSaver: autocommit, không prepared statement (an toàn với pgbouncer), dict row.
    kwargs: dict[str, Any] = {"autocommit": True, "prepare_threshold": None, "row_factory": dict_row}
    if schema:
        kwargs["options"] = f"-c search_path={schema}"
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=checkpointer_dsn(settings.database_url),
        min_size=1,
        max_size=settings.checkpointer_pool_size,
        open=False,
        kwargs=kwargs,
    )
    await pool.open(wait=True)
    stack.push_async_callback(pool.close)
    return AsyncPostgresSaver(pool, serde=checkpoint_serde())


@dataclass
class Storage:
    repos: dict[str, Any]
    sessions: async_sessionmaker[AsyncSession]
    checkpointer: Any


async def open_storage(
    stack: AsyncExitStack, settings: Settings, clock: Clock, schema: str | None = None, with_history: bool = True
) -> Storage:
    """Postgres là nơi lưu trữ duy nhất: dữ liệu nghiệp vụ (asyncpg) và state hội thoại (checkpointer, psycopg).

    `schema`: tạo một schema riêng và xóa nó khi đóng `stack`. Test và eval dùng để mỗi lượt chạy có dữ liệu
    sạch, không đụng nhau dù chạy song song trên cùng database. Schema riêng luôn mới tinh nên luôn chạy
    migration và nạp dữ liệu demo, bất kể `auto_create_schema` / `seed_demo_data` (vốn dành cho database chính).
    Bảng luôn được tạo bằng migration Alembic (`alembic upgrade head`), không dùng `create_all`.
    """
    if not settings.database_url:
        raise ValueError("PA_DATABASE_URL là bắt buộc (postgresql+asyncpg://user:pass@host:5432/db)")
    create = settings.auto_create_schema or schema is not None
    if schema:
        await create_db_schema(settings.database_url, schema)
        stack.push_async_callback(drop_db_schema, settings.database_url, schema)
    engine = make_engine(settings.database_url, schema)
    stack.push_async_callback(engine.dispose)
    if create:
        await migrate(settings.database_url, schema)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = clock.now()
    if settings.seed_demo_data or schema is not None:
        await seed_demo_data(sessions, now.date(), now.tzinfo, with_history=with_history)  # type: ignore[arg-type]
    checkpointer = await postgres_checkpointer(stack, settings, schema)
    # Production: bảng (kể cả bảng checkpoint) do migrate/`cli.seed` tạo trước, không để các worker tranh nhau tạo.
    if create:
        await checkpointer.setup()
    return Storage(sql_repositories(sessions, now.tzinfo), sessions, checkpointer)


@asynccontextmanager
async def build_container(settings: Settings) -> AsyncIterator[Container]:
    """Container cho ứng dụng thật; quản lý vòng đời kết nối Postgres."""
    clock = SystemClock(settings.timezone)
    async with AsyncExitStack() as stack:
        storage = await open_storage(stack, settings, clock)
        llm = build_llm(settings)
        await require_llm(llm, settings)
        policy = BookingPolicy(is_holiday=vietnam_holiday_checker())
        container = assemble(
            settings, storage.repos, llm, build_stt(settings), clock, storage.checkpointer, policy=policy
        )
        if settings.seed_demo_data:
            await container.auth.seed(seed.demo_accounts(settings.seed_password))  # chỉ khi DB chưa có tài khoản
        yield container
