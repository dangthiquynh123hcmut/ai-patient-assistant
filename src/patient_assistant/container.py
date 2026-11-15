"""Composition root: nơi DUY NHẤT biết implementation cụ thể và lắp chúng vào port."""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from typing import Any

from dependency_injector import containers, providers
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_assistant.application.agent.dependencies import AgentDependencies, AgentLimits
from patient_assistant.application.agent.graph import build_agent_graph
from patient_assistant.application.agent.side_talk import SideTalk
from patient_assistant.application.agent.state import checkpoint_types
from patient_assistant.application.ports.clock import Clock
from patient_assistant.application.ports.llm import LLMPort
from patient_assistant.application.ports.stt import STTPort
from patient_assistant.application.ports.tokens import TokenCodec
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
from patient_assistant.infrastructure.security.jwt_codec import JwtCodec

log = logging.getLogger("patient_assistant")


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
        effort_agent=settings.llm_effort_agent,
    )


async def require_llm(llm: AnthropicBedrockLLM, settings: Settings) -> None:
    """Hệ thống chỉ chạy khi gọi được LLM: thử một lần lúc khởi động, lỗi thì dừng hẳn."""
    try:
        await llm.ping()
    except Exception as exception:  # noqa: BLE001 - thiếu credential, sai quyền, mất mạng... đều là không gọi được LLM
        raise RuntimeError(
            f"Không gọi được LLM ({settings.llm_model_id}): {type(exception).__name__}: {exception}. "
            "Kiểm tra bằng: python -m patient_assistant.interface.cli.check_llm"
        ) from exception


def build_stt(settings: Settings) -> STTPort:
    from patient_assistant.infrastructure.stt.faster_whisper import FasterWhisperSTT, UnavailableSTT

    if not settings.stt_enabled:
        return UnavailableSTT()
    return FasterWhisperSTT(settings.stt_model, compute_type=settings.stt_compute_type, initial_prompts=stt_prompts())


def stt_prompts() -> dict[str, str]:
    """initial_prompt theo ngôn ngữ: từ vựng của phòng khám để whisper nhận đúng tên riêng."""
    doctors = ", ".join(doctor.name.removeprefix("BS. ") for doctor in seed.DOCTORS)
    return {
        "vi": (
            "Khoa: "
            + ", ".join(department.name for department in seed.DEPARTMENTS)
            + ". Bác sĩ: "
            + ", ".join(doctor.name for doctor in seed.DOCTORS)
            + "."
        ),
        "en": (
            "A patient talks to a rehabilitation clinic assistant about booking an appointment "
            "(department, doctor, day, morning or afternoon) or a weekly report "
            f"(pain level from 0 to 10, exercise days, symptoms). Doctors: {doctors}."
        ),
    }


@cache
def _demo_users(password: str, created_at: Any) -> tuple[tuple[User, str], ...]:
    """Băm mật khẩu demo một lần cho cả tiến trình (scrypt chậm có chủ đích)."""
    return tuple(
        (
            User(id=f"user-{account.username}", created_at=created_at, **account.model_dump(exclude={"password"})),
            hash_password(account.password),
        )
        for account in seed.demo_accounts(password)
    )


async def seed_demo_users(sessions: async_sessionmaker[AsyncSession], password: str, now: Any) -> None:
    """Tạo tài khoản demo cho DB (schema) mới, dùng mật khẩu đã băm sẵn (nhanh cho test/eval)."""
    users = accounts.SqlUserRepository(sessions)
    if await users.count():
        return
    for user, password_hash in _demo_users(password, now):
        await users.add(user, password_hash)


@cache
def _ephemeral_jwt_secret() -> str:
    log.warning(
        "PA_JWT_SECRET chưa đặt: dùng khóa JWT tạm sinh ngẫu nhiên cho tiến trình này. Token không dùng được giữa các "
        "worker và mất hiệu lực khi khởi động lại — đặt PA_JWT_SECRET cho môi trường thật."
    )
    return secrets.token_urlsafe(48)


def token_codec(settings: Settings) -> TokenCodec:
    return JwtCodec(settings.jwt_secret or _ephemeral_jwt_secret())


def sql_repositories(sessions: async_sessionmaker[AsyncSession], timezone: Any) -> dict[str, Any]:
    return {
        "users": accounts.SqlUserRepository(sessions),
        "sessions": accounts.SqlSessionRepository(sessions),
        "conversation_log": accounts.SqlConversationLog(sessions),
        "reports": sql.SqlReportRepository(sessions),
        "catalog": sql.SqlCatalogRepository(sessions),
        "plans": sql.SqlPlanRepository(sessions),
        "patients": sql.SqlPatientRepository(sessions),
        "slots": sql.SqlSlotRepository(sessions, timezone),
        "appointments": sql.SqlAppointmentRepository(sessions, timezone),
        "knowledge": sql.SqlKnowledgeRepository(sessions),
    }


def _side_talk(agent_mode: str, llm: LLMPort, toolbox: Toolbox) -> SideTalk | None:
    return None if agent_mode == "full" else SideTalk(llm, toolbox)


class Container(containers.DeclarativeContainer):
    """Composition root: provider của mọi service/use case (theo docs JESCO-AI-BE conventions R2/R3).

    Chỉ ở đây mới gọi constructor của service, repository, adapter. Các tầng trong chỉ nhận port qua constructor.

    Phụ thuộc từ bên ngoài (`providers.Dependency`) được truyền khi tạo container, vì chúng cần I/O hoặc
    quản lý vòng đời (kết nối Postgres, ping LLM) — xem `assemble()` và `build_container()`:
        settings, clock, repositories (dict repository SQL), checkpointer, llm, stt
    Test/eval truyền repo trong schema riêng, LLM giả, FixedClock; production truyền bản thật.

    Dùng: `container.chat()` trả instance (Singleton: một instance cho mỗi container).
    Router FastAPI nhận qua `Depends(Provide[Container.chat])` + `@inject` (container được wire trong create_app).
    """

    # ------------------------------------------------------------------ từ bên ngoài
    settings = providers.Dependency(instance_of=Settings)
    clock: providers.Dependency[Clock] = providers.Dependency()
    repositories = providers.Dependency(instance_of=dict)
    checkpointer: providers.Dependency[Any] = providers.Dependency()
    llm: providers.Dependency[LLMPort] = providers.Dependency()
    stt: providers.Dependency[STTPort] = providers.Dependency()

    # Có mặc định, ghi đè được (vd. production dùng lịch nghỉ lễ Việt Nam)
    audit = providers.Singleton(StructlogAuditLog)
    policy = providers.Singleton(BookingPolicy)

    # ------------------------------------------------------------------ repositories (adapter SQL)
    user_repository = repositories.provided["users"]
    session_repository = repositories.provided["sessions"]
    conversation_log = repositories.provided["conversation_log"]
    report_repository = repositories.provided["reports"]
    catalog_repository = repositories.provided["catalog"]
    plan_repository = repositories.provided["plans"]
    patient_repository = repositories.provided["patients"]
    slot_repository = repositories.provided["slots"]
    appointment_repository = repositories.provided["appointments"]
    knowledge_repository = repositories.provided["knowledge"]
    # Tên ngắn API dùng (lịch sử hội thoại)
    log = conversation_log

    # ------------------------------------------------------------------ services nghiệp vụ
    report_service = providers.Singleton(ReportService, report_repository)
    booking_service = providers.Singleton(BookingService, slot_repository, appointment_repository, policy)
    toolbox = providers.Singleton(
        Toolbox,
        reports=report_service,
        booking=booking_service,
        catalog=catalog_repository,
        plans=plan_repository,
        patients=patient_repository,
        knowledge=knowledge_repository,
        slot_limit=settings.provided.slot_options,
    )

    # ------------------------------------------------------------------ xác thực
    tokens = providers.Singleton(token_codec, settings)
    auth = providers.Singleton(
        AuthService,
        user_repository,
        session_repository,
        patient_repository,
        clock,
        providers.Factory(timedelta, hours=settings.provided.session_ttl_hours),
        tokens=tokens,
        access_ttl=providers.Factory(timedelta, minutes=settings.provided.access_token_ttl_minutes),
    )

    # ------------------------------------------------------------------ agent LangGraph
    limits = providers.Singleton(
        AgentLimits,
        max_corrections=settings.provided.max_corrections,
        max_clarify_rounds=settings.provided.max_clarify_rounds,
        max_questions=settings.provided.max_questions,
        max_slot_search_rounds=settings.provided.max_slot_search_rounds,
        max_agent_tool_calls=settings.provided.max_agent_tool_calls,
    )
    agent_dependencies = providers.Singleton(
        AgentDependencies,
        llm=llm,
        tools=toolbox,
        audit=audit,
        limits=limits,
        agent_mode=settings.provided.effective_agent_mode,
    )
    agent_graph = providers.Singleton(build_agent_graph, agent_dependencies, checkpointer)
    # Câu trả lời ở thẻ chờ: workflow cần SideTalk để tách câu hỏi chen ngang / đổi việc; ở chế độ full agent tự hiểu.
    side_talk = providers.Singleton(_side_talk, settings.provided.effective_agent_mode, llm, toolbox)

    # ------------------------------------------------------------------ use cases (API gọi)
    chat = providers.Singleton(ChatUseCase, agent_graph, clock, limits.provided.recursion_limit, side_talk)
    conversations = providers.Singleton(ConversationService, chat, conversation_log, clock)
    transcribe = providers.Singleton(
        TranscribeUseCase, stt, settings.provided.stt_confidence_threshold, settings.provided.stt_language
    )
    # Mô tả thêm cho bệnh nhân demo (demo_data/patients.json); bệnh nhân tự đăng ký hiển thị bằng họ tên.
    patient_labels = providers.Object({patient.id: patient.label for patient in seed.PATIENTS})


def build_auth(settings: Settings, repositories: dict[str, Any], clock: Clock) -> AuthService:
    """AuthService không cần LLM/graph (CLI seed, purge_sessions): lấy từ container với các phụ thuộc tối thiểu."""
    container = Container(
        settings=providers.Object(settings), repositories=providers.Object(repositories), clock=providers.Object(clock)
    )
    auth: AuthService = container.auth()
    return auth


def assemble(
    settings: Settings,
    repositories: dict[str, Any],
    llm: LLMPort,
    stt: STTPort,
    clock: Clock,
    checkpointer: Any,
    audit: Any = None,
    policy: BookingPolicy | None = None,
) -> Container:
    """Lắp ráp thuần (không I/O) — test dùng trực tiếp với repo Postgres trong schema riêng, LLM giả và FixedClock."""
    container = Container(
        settings=providers.Object(settings),
        repositories=providers.Object(repositories),
        llm=providers.Object(llm),
        stt=providers.Object(stt),
        clock=providers.Object(clock),
        checkpointer=providers.Object(checkpointer),
    )
    if audit is not None:
        container.audit.override(providers.Object(audit))
    if policy is not None:
        container.policy.override(providers.Object(policy))
    return container


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
    repositories: dict[str, Any]
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
            settings, storage.repositories, llm, build_stt(settings), clock, storage.checkpointer, policy=policy
        )
        if settings.seed_demo_data:
            await container.auth().seed(seed.demo_accounts(settings.seed_password))  # chỉ tạo tài khoản còn thiếu
        try:
            yield container
        finally:
            container.unwire()
