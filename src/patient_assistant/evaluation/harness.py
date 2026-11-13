"""Chạy một case qua agent thật (graph + validator + service), ghi lại những gì quan sát được.

Mỗi case dùng container mới với schema Postgres riêng (xóa khi xong), đồng hồ cố định và (tùy chọn) lỗi tiêm sẵn.
Harness chỉ QUAN SÁT; việc chấm điểm nằm ở `metrics.py`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from datetime import time as dtime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_assistant.application.agent.state import AgentState, UserDecision
from patient_assistant.application.ports.llm import (
    BookingPrompt,
    GroundedAnswer,
    LLMError,
    LLMPort,
    ReportPrompt,
    ToolSpec,
)
from patient_assistant.application.use_cases.chat import TurnResult
from patient_assistant.config import Settings
from patient_assistant.container import Container, assemble, open_storage, seed_demo_users
from patient_assistant.domain.booking.models import AppointmentStatus, Slot, SlotOption
from patient_assistant.domain.conversation import Intent
from patient_assistant.domain.report.models import Report
from patient_assistant.evaluation.cases import EvalCase, LLMScript, Reply, UserTurn
from patient_assistant.infrastructure.clock import FixedClock
from patient_assistant.infrastructure.persistence.sql.models import AppointmentRow, ReportRow, SlotRow
from patient_assistant.infrastructure.persistence.sql.repositories import _report
from patient_assistant.infrastructure.stt.faster_whisper import UnavailableSTT

LLMFactory = Callable[[], LLMPort]


class AttemptView(BaseModel):
    attempt_no: int
    issue_codes: list[str]
    llm_fixable_codes: list[str] = Field(default_factory=list)
    passed: bool
    output: dict[str, Any] | None = None


class CaseObservation(BaseModel):
    case_id: str
    suite: str
    category: str
    session: int = 0
    status: Literal["done", "waiting", "crashed"]
    outcome: str | None = None
    outcome_code: str | None = None
    response_text: str = ""
    intent: str | None = None
    interrupts: list[str] = Field(default_factory=list)
    first_offered_slot_ids: list[str] = Field(default_factory=list)
    offered_slot_ids: list[str] = Field(default_factory=list)
    booked_slot_ids: list[str] = Field(default_factory=list)
    reports_written: int = 0
    saved_report: dict[str, Any] | None = None
    final_fields: dict[str, Any] = Field(default_factory=dict)
    attempts: list[AttemptView] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    hallucinated_slot_ids: list[str] = Field(default_factory=list)
    # Slot được đề xuất trùng giờ một lịch hẹn mà bệnh nhân ĐÃ CÓ lúc đề xuất (phải = 0).
    own_overlap_slot_ids: list[str] = Field(default_factory=list)
    # Lần đề xuất đầu tiên sau tin nhắn chính: slot_id, start_at, doctor_id, department_code, hint.
    first_offers: list[dict[str, Any]] = Field(default_factory=list)
    interrupt_messages: list[str] = Field(default_factory=list)
    double_booked_slot_ids: list[str] = Field(default_factory=list)
    replies_left: int = 0
    latency_ms: int = 0
    error: str | None = None


# ---------------------------------------------------------------- LLM bọc cho eval


class ScriptedLLM:
    """Trả lần lượt đầu ra tiêm sẵn; hết kịch bản thì chuyển cho LLM bên trong."""

    def __init__(self, inner: LLMPort, script: LLMScript) -> None:
        self._inner = inner
        self._reports = list(script.reports)
        self._bookings = list(script.bookings)

    async def classify_intent(self, text: str, memory: str | None = None) -> list[Intent]:
        return await self._inner.classify_intent(text, memory)

    async def extract_report(self, prompt: ReportPrompt) -> dict[str, Any]:
        return self._reports.pop(0) if self._reports else await self._inner.extract_report(prompt)

    async def extract_booking(self, prompt: BookingPrompt) -> dict[str, Any]:
        return self._bookings.pop(0) if self._bookings else await self._inner.extract_booking(prompt)

    async def choose_slot(self, reply: str, options: list[SlotOption]) -> int | None:
        return await self._inner.choose_slot(reply, options)

    async def answer_question(
        self, question: str, tools: list[ToolSpec], max_tool_calls: int, memory: str | None = None
    ) -> GroundedAnswer:
        return await self._inner.answer_question(question, tools, max_tool_calls, memory)

    async def converse(self, text: str, context: str | None = None, memory: str | None = None) -> str:
        return await self._inner.converse(text, context, memory)


class UnavailableLLM:
    async def _fail(self, *_: Any, **__: Any) -> Any:
        raise LLMError("unavailable", "injected fault", retryable=True)

    classify_intent = extract_report = extract_booking = choose_slot = answer_question = converse = _fail


async def _raise_unavailable(*_: Any, **__: Any) -> Any:
    raise ConnectionError("injected fault: backend unavailable")


# ---------------------------------------------------------------- chạy case


async def _prebook(repos: dict[str, Any], spec: dict[str, str], now: datetime) -> None:
    """Chiếm trước mọi slot của một khoa trong một ngày (tới giờ `until`) để buộc agent nới điều kiện."""
    day = datetime.fromisoformat(spec["date"]).date()
    until = dtime.fromisoformat(spec.get("until", "23:59"))
    start = datetime.combine(day, dtime(0, 0), now.tzinfo)
    end = datetime.combine(day, until, now.tzinfo)
    for i, slot in enumerate(await repos["slots"].find_available(spec["department"], start, end, None, 1000)):
        await repos["appointments"].book("prebook-patient", slot, None, f"prebook-{i}")


async def _setup_appointments(repos: dict[str, Any], case: EvalCase) -> None:
    for i, slot_id in enumerate(case.setup.appointments):
        slot = await repos["slots"].get(slot_id)
        if slot is None:
            raise ValueError(f"{case.id}: setup.appointments có slot không tồn tại: {slot_id}")
        await repos["appointments"].book(case.patient_id, slot, None, f"setup-{case.id}-{i}")


async def _book_own_clash(container: Container, repos: dict[str, Any], patient: str, slot_id: str, key: str) -> None:
    """Bệnh nhân tự đặt một lịch khoa khác cùng giờ với slot đang chờ xác nhận (giả lập tab khác)."""
    chosen = await repos["slots"].get(slot_id)
    if chosen is None:
        return
    departments, _ = await container.toolbox().catalog()
    for dept in departments:
        if dept.code == chosen.department_code:
            continue
        found = await repos["slots"].find_available(dept.code, chosen.start_at, chosen.start_at, None, 1)
        if found:
            await repos["appointments"].book(patient, found[0], None, key)
            return


def _overlaps(slot: Slot, others: list[Slot]) -> bool:
    return any(o.start_at < slot.end_at and slot.start_at < o.end_at for o in others)


class _Db:
    """Đọc thẳng database của case để kiểm tra kết quả (không đi qua agent)."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def booked_slot_ids(self, patient_id: str) -> list[str]:
        query = (
            select(AppointmentRow.slot_id)
            .where(AppointmentRow.status == AppointmentStatus.BOOKED.value, AppointmentRow.patient_id == patient_id)
            .order_by(AppointmentRow.created_at)
        )
        async with self._sessions() as session:
            return list(await session.scalars(query))

    async def double_booked_slot_ids(self) -> list[str]:
        query = (
            select(AppointmentRow.slot_id)
            .where(AppointmentRow.status == AppointmentStatus.BOOKED.value)
            .group_by(AppointmentRow.slot_id)
            .having(func.count() > 1)
        )
        async with self._sessions() as session:
            return sorted(await session.scalars(query))

    async def slot_ids(self) -> set[str]:
        async with self._sessions() as session:
            return set(await session.scalars(select(SlotRow.slot_id)))

    async def reports(self, patient_id: str) -> list[Report]:
        query = select(ReportRow).where(ReportRow.patient_id == patient_id).order_by(ReportRow.created_at)
        async with self._sessions() as session:
            return [_report(row) for row in await session.scalars(query)]


def _fields(state: AgentState) -> dict[str, Any]:
    if state.report_draft is not None:
        return state.report_draft.model_dump(mode="json", exclude={"evidence", "missing_fields"})
    if state.booking_request is not None:
        out: dict[str, Any] = {
            "department_code": state.booking_request.department_code,
            "department_from_plan": state.department_from_plan,
        }
        if state.resolved_window is not None:
            out |= state.resolved_window.model_dump(mode="json", exclude={"rules"})
        return out
    return {}


class CaseRunner:
    def __init__(self, llm_factory: LLMFactory, settings: Settings) -> None:
        self._llm_factory = llm_factory
        self._settings = settings

    @asynccontextmanager
    async def _container(self, case: EvalCase) -> AsyncIterator[tuple[Container, dict[str, Any], _Db]]:
        clock = FixedClock(case.now)
        schema = f"eval_{uuid.uuid4().hex[:12]}"
        async with AsyncExitStack() as stack:
            storage = await open_storage(stack, self._settings, clock, schema=schema, with_history=False)
            await seed_demo_users(storage.sessions, self._settings.seed_password, case.now)
            repos = storage.repos
            if case.faults.save_unavailable:
                repos["reports"].add = _raise_unavailable
            if case.faults.booking_unavailable:
                repos["appointments"].book = _raise_unavailable
            llm: LLMPort = (
                UnavailableLLM() if case.faults.llm_unavailable else ScriptedLLM(self._llm_factory(), case.llm_script)
            )
            container = assemble(self._settings, repos, llm, UnavailableSTT(), clock, storage.checkpointer)
            yield container, repos, _Db(storage.sessions)

    async def run(self, case: EvalCase) -> list[CaseObservation]:
        async with self._container(case) as (container, repos, db):
            if case.faults.prebook_window:
                await _prebook(repos, case.faults.prebook_window, case.now)
            await _setup_appointments(repos, case)
            sessions = case.concurrency.sessions if case.concurrency else 1
            barrier = asyncio.Barrier(sessions) if sessions > 1 else None
            observations = await asyncio.gather(
                *[self._drive(case, container, repos, db, i, barrier) for i in range(sessions)]
            )
            # Kiểm tra toàn cục sau khi mọi phiên kết thúc.
            doubled = await db.double_booked_slot_ids()
            for obs in observations:
                obs.double_booked_slot_ids = doubled
            return list(observations)

    async def _drive(
        self,
        case: EvalCase,
        container: Container,
        repos: dict[str, Any],
        db: _Db,
        session: int,
        barrier: asyncio.Barrier | None,
    ) -> CaseObservation:
        conversation_id = f"{case.id}-{session}"
        patient = case.patient_id if barrier is None else f"{case.patient_id}-{session}"
        obs = CaseObservation(
            case_id=case.id, suite=case.suite, category=case.category, session=session, status="waiting"
        )
        started = time.perf_counter()
        offered: list[str] = []
        # Lịch hẹn có từ trước tin nhắn chính (setup, lượt history, lịch tự đặt ở tab khác): không tính là
        # kết quả của case, nhưng mọi slot đề xuất sau đó không được trùng giờ với chúng.
        existing: set[str] = set()
        stolen = False
        replies: list[Reply] = list(case.replies)

        async def converse(message: UserTurn, turn_replies: list[Reply], scored: bool) -> TurnResult:
            nonlocal stolen
            result: TurnResult = await container.chat().send_message(
                conversation_id, patient, message.text, message.input_source, message.stt_flags
            )
            while result.status == "waiting_user" and result.interrupt is not None:
                payload = result.interrupt
                if scored:
                    obs.interrupts.append(payload.kind)
                    obs.interrupt_messages.append(payload.message)
                if payload.kind == "choose_slot":
                    ids = [o.slot.slot_id for o in payload.options]
                    offered.extend(ids)
                    if scored:
                        mine = [s for s in [await repos["slots"].get(i) for i in sorted(existing)] if s]
                        obs.own_overlap_slot_ids += [o.slot.slot_id for o in payload.options if _overlaps(o.slot, mine)]
                        if not obs.first_offered_slot_ids:
                            obs.first_offered_slot_ids = ids
                            obs.first_offers = [
                                {
                                    "slot_id": o.slot.slot_id,
                                    "start_at": o.slot.start_at.isoformat(),
                                    "doctor_id": o.slot.doctor_id,
                                    "department_code": o.slot.department_code,
                                    "hint": o.hint,
                                }
                                for o in payload.options
                            ]
                if not turn_replies:
                    break
                if scored and payload.kind == "confirm_booking" and not stolen:
                    if barrier is not None:
                        await barrier.wait()  # mọi phiên cùng chốt một lúc -> tranh chấp thật
                    state = await container.chat().inspect_state(conversation_id)
                    selected = state.selected_slot_id if state else None
                    if case.faults.slot_taken_after_choice and selected:
                        await container.toolbox().create_appointment(
                            "intruder-patient", selected, None, f"intruder-{conversation_id}", case.now
                        )
                    if case.faults.own_clash_after_choice and selected:
                        await _book_own_clash(container, repos, patient, selected, f"clash-{conversation_id}")
                        existing.update(await db.booked_slot_ids(patient))
                    stolen = True
                reply = turn_replies.pop(0)
                result = await container.chat().resume(conversation_id, patient, UserDecision(**reply.model_dump()))
            if not scored:
                existing.update(await db.booked_slot_ids(patient))
            return result

        try:
            existing.update(await db.booked_slot_ids(patient))
            for turn in case.history:
                await converse(turn.message, list(turn.replies), scored=False)
            reports_before = len(await db.reports(patient))
            result = await converse(case.message, replies, scored=True)
            obs.status = "done" if result.status == "done" else "waiting"
            if result.response is not None:
                obs.outcome = result.response.kind.value
                obs.response_text = result.response.text
            obs.replies_left = len(replies)
            state = await container.chat().inspect_state(conversation_id)
            if state is not None:
                obs.intent = state.intent.value if state.intent else None
                obs.outcome_code = state.outcome.code if state.outcome else None
                obs.final_fields = _fields(state)
                obs.attempts = [
                    AttemptView(
                        attempt_no=a.attempt_no,
                        issue_codes=a.issue_codes,
                        llm_fixable_codes=a.llm_fixable_codes,
                        passed=a.passed,
                        output=a.output,
                    )
                    for a in state.attempts
                ]
                obs.tool_calls = [t.tool for t in state.tool_calls]
        except Exception as exc:  # noqa: BLE001 - một case lỗi không được làm hỏng cả lượt eval
            obs.status = "crashed"
            obs.error = f"{type(exc).__name__}: {exc}"
            reports_before = 0
        obs.latency_ms = int((time.perf_counter() - started) * 1000)

        # Chỉ tính lịch hẹn do tin nhắn chính tạo ra.
        obs.booked_slot_ids = [s for s in await db.booked_slot_ids(patient) if s not in existing]
        obs.offered_slot_ids = sorted(set(offered))
        known = await db.slot_ids()
        obs.hallucinated_slot_ids = sorted(
            {s for s in offered if s not in known} | {s for s in obs.booked_slot_ids if s not in offered}
        )
        obs.own_overlap_slot_ids = sorted(set(obs.own_overlap_slot_ids))
        reports = await db.reports(patient)
        obs.reports_written = len(reports) - reports_before
        obs.saved_report = (
            reports[-1].model_dump(mode="json", exclude={"evidence"}) if obs.reports_written > 0 else None
        )
        return obs


async def run_cases(
    cases: list[EvalCase], llm_factory: LLMFactory, settings: Settings, concurrency: int = 4
) -> list[CaseObservation]:
    runner = CaseRunner(llm_factory, settings)
    limit = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> list[CaseObservation]:
        async with limit:
            return await runner.run(case)

    nested = await asyncio.gather(*[one(c) for c in cases])
    return [obs for group in nested for obs in group]
