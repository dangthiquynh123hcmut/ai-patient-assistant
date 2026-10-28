"""Composition root: nơi DUY NHẤT biết implementation cụ thể và lắp chúng vào port."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from patient_assistant.application.agent.deps import AgentDeps, AgentLimits
from patient_assistant.application.agent.graph import build_agent_graph
from patient_assistant.application.agent.state import checkpoint_types
from patient_assistant.application.ports.clock import Clock
from patient_assistant.application.ports.llm import LLMPort
from patient_assistant.application.ports.stt import STTPort
from patient_assistant.application.services.booking_service import BookingService
from patient_assistant.application.services.report_service import ReportService
from patient_assistant.application.tools.toolbox import Toolbox
from patient_assistant.application.use_cases.chat import ChatUseCase
from patient_assistant.application.use_cases.transcribe import TranscribeUseCase
from patient_assistant.config import Settings
from patient_assistant.domain.booking.policy import BookingPolicy
from patient_assistant.infrastructure.audit import StructlogAuditLog
from patient_assistant.infrastructure.calendar.vn_holidays import vietnam_holiday_checker
from patient_assistant.infrastructure.clock import SystemClock
from patient_assistant.infrastructure.persistence import memory, seed


@dataclass
class Container:
    chat: ChatUseCase
    transcribe: TranscribeUseCase
    toolbox: Toolbox
    clock: Clock
    settings: Settings


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


def in_memory_repositories(clock: Clock) -> dict[str, Any]:
    now = clock.now()
    store = memory.InMemoryBookingStore(seed.generate_slots(now.date(), now.tzinfo))  # type: ignore[arg-type]
    return {
        "reports": memory.InMemoryReportRepository(),
        "catalog": memory.InMemoryCatalogRepository(seed.DEPARTMENTS, seed.DOCTORS),
        "plans": memory.InMemoryPlanRepository(seed.PLANS),
        "slots": store,
        "appointments": memory.InMemoryAppointmentRepository(store),
        "knowledge": memory.InMemoryKnowledgeRepository(seed.KNOWLEDGE),
    }


async def sql_repositories(url: str, clock: Clock) -> dict[str, Any]:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from patient_assistant.infrastructure.persistence.sql import repositories as sql
    from patient_assistant.infrastructure.persistence.sql.setup import (
        create_schema,
        make_engine,
        seed_demo_data,
    )

    engine = make_engine(url)
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = clock.now()
    await seed_demo_data(sessions, now.date(), now.tzinfo)  # type: ignore[arg-type]
    return {
        "reports": sql.SqlReportRepository(sessions),
        "catalog": sql.SqlCatalogRepository(sessions),
        "plans": sql.SqlPlanRepository(sessions),
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
    return Container(
        chat=ChatUseCase(graph, clock, limits.recursion_limit),
        transcribe=TranscribeUseCase(stt, settings.stt_confidence_threshold),
        toolbox=toolbox,
        clock=clock,
        settings=settings,
    )


@asynccontextmanager
async def build_container(settings: Settings) -> AsyncIterator[Container]:
    """Container cho ứng dụng thật; quản lý vòng đời kết nối (checkpointer Postgres)."""
    clock = SystemClock(settings.timezone)
    async with AsyncExitStack() as stack:
        if settings.checkpointer == "postgres":
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            assert settings.checkpointer_url, "PA_CHECKPOINTER_URL là bắt buộc khi dùng checkpointer postgres"
            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(settings.checkpointer_url, serde=checkpoint_serde())
            )
            await checkpointer.setup()
        else:
            checkpointer = memory_checkpointer()
        repos = (
            await sql_repositories(settings.database_url, clock)
            if settings.database_url
            else in_memory_repositories(clock)
        )
        policy = BookingPolicy(is_holiday=vietnam_holiday_checker())
        yield assemble(settings, repos, build_llm(settings), build_stt(settings), clock, checkpointer, policy=policy)
