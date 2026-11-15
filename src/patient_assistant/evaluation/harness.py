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
from dataclasses import replace
from datetime import datetime
from datetime import time as dtime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from patient_assistant.application.agent.state import AgentState, UserDecision
from patient_assistant.application.ports.llm import (
    AgentMessage,
    AgentReply,
    BookingPrompt,
    GroundedAnswer,
    LLMError,
    LLMPort,
    ReportPrompt,
    ToolSpecification,
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
    cancelled_count: int = 0  # số lịch bị hủy do tin nhắn chính (hủy hoặc dời lịch)
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
    # Chế độ agent thực tế của case (workflow | booking | full) — metric chấm theo chế độ này.
    agent_mode: str = "workflow"
    # Chế độ agent: lượt kết thúc bằng câu hỏi dạng chat (không phải interrupt) mà không còn câu trả lời kịch bản.
    ended_with_question: bool = False
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
        self, question: str, tools: list[ToolSpecification], max_tool_calls: int, memory: str | None = None
    ) -> GroundedAnswer:
        return await self._inner.answer_question(question, tools, max_tool_calls, memory)

    async def converse(self, text: str, context: str | None = None, memory: str | None = None) -> str:
        return await self._inner.converse(text, context, memory)

    async def run_agent(
        self,
        task: Literal["booking", "assistant"],
        context: str,
        history: list[AgentMessage],
        tools: list[ToolSpecification],
        max_tool_calls: int,
    ) -> AgentReply:
        return await self._inner.run_agent(
            task, context, history, [self._inject(tool) for tool in tools], max_tool_calls
        )

    def _inject(self, specification: ToolSpecification) -> ToolSpecification:
        """Chế độ agent không gọi `extract_report`: lỗi tiêm sẵn thay vào tham số LLM gửi cho `propose_report`.
        Tham số sai kiểu (vd. pain_level="năm") làm handler ném ValidationError -> agent nhận lỗi tool như thật."""
        if specification.name != "propose_report":
            return specification
        original = specification.handler

        async def handler(input_data: BaseModel) -> dict[str, Any]:
            if self._reports:
                input_data = specification.input_model.model_validate(self._reports.pop(0))
            return await original(input_data)

        return replace(specification, handler=handler)


class UnavailableLLM:
    async def _fail(self, *_: Any, **__: Any) -> Any:
        raise LLMError("unavailable", "injected fault", retryable=True)

    classify_intent = extract_report = extract_booking = choose_slot = answer_question = converse = run_agent = _fail


async def _raise_unavailable(*_: Any, **__: Any) -> Any:
    raise ConnectionError("injected fault: backend unavailable")


# ---------------------------------------------------------------- chạy case


async def _prebook(repositories: dict[str, Any], specification: dict[str, str], now: datetime) -> None:
    """Chiếm trước mọi slot của một khoa trong một ngày (tới giờ `until`) để buộc agent nới điều kiện."""
    day = datetime.fromisoformat(specification["date"]).date()
    until = dtime.fromisoformat(specification.get("until", "23:59"))
    start = datetime.combine(day, dtime(0, 0), now.tzinfo)
    end = datetime.combine(day, until, now.tzinfo)
    for i, slot in enumerate(
        await repositories["slots"].find_available(specification["department"], start, end, None, 1000)
    ):
        await repositories["appointments"].book("prebook-patient", slot, None, f"prebook-{i}")


async def _setup_appointments(repositories: dict[str, Any], case: EvalCase) -> None:
    for i, slot_id in enumerate(case.setup.appointments):
        slot = await repositories["slots"].get(slot_id)
        if slot is None:
            raise ValueError(f"{case.id}: setup.appointments có slot không tồn tại: {slot_id}")
        await repositories["appointments"].book(case.patient_id, slot, None, f"setup-{case.id}-{i}")


async def _book_own_clash(
    container: Container, repositories: dict[str, Any], patient: str, slot_id: str, key: str
) -> None:
    """Bệnh nhân tự đặt một lịch khoa khác cùng giờ với slot đang chờ xác nhận (giả lập tab khác)."""
    chosen = await repositories["slots"].get(slot_id)
    if chosen is None:
        return
    departments, _ = await container.toolbox().catalog()
    for department in departments:
        if department.code == chosen.department_code:
            continue
        found = await repositories["slots"].find_available(department.code, chosen.start_at, chosen.start_at, None, 1)
        if found:
            await repositories["appointments"].book(patient, found[0], None, key)
            return


def _overlaps(slot: Slot, others: list[Slot]) -> bool:
    return any(other.start_at < slot.end_at and slot.start_at < other.end_at for other in others)


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

    async def cancelled_count(self, patient_id: str) -> int:
        query = select(func.count()).where(
            AppointmentRow.status == AppointmentStatus.CANCELLED.value, AppointmentRow.patient_id == patient_id
        )
        async with self._sessions() as session:
            return int(await session.scalar(query) or 0)

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
        # Chế độ agent không có khoa/khoảng ngày trích xuất: lấy từ slot đã xác nhận (metric so slot_date với
        # date_from/date_to của case).
        chosen = next(
            (option.slot for option in state.offered_slots if option.slot.slot_id == state.selected_slot_id), None
        )
        if chosen is not None and state.resolved_window is None:
            out["department_code"] = out["department_code"] or chosen.department_code
            out["slot_date"] = chosen.start_at.astimezone(state.now.tzinfo).date().isoformat()
        return out
    return {}


def _reply_text(reply: Reply) -> str:
    """Câu trả lời kịch bản viết cho interrupt, đổi thành tin nhắn khi agent hỏi bằng chat thay vì interrupt."""
    if reply.text:
        return reply.text
    if reply.action == "choose" and reply.option_no is not None:
        return f"chọn số {reply.option_no}"
    if reply.action == "edit" and reply.fields:
        return "sửa " + ", ".join(f"{k} = {v}" for k, v in reply.fields.items())
    return {"confirm": "xác nhận", "cancel": "hủy", "other_time": "giờ khác"}.get(reply.action, "xác nhận")


def _chat_reply(result: TurnResult) -> str | None:
    """Câu trả lời dạng chat của agent (lượt đã xong, không chờ interrupt), hoặc None."""
    if result.status != "done" or result.response is None or result.response.kind.value != "chat":
        return None
    return result.response.text


class CaseRunner:
    def __init__(self, llm_factory: LLMFactory, settings: Settings) -> None:
        self._llm_factory = llm_factory
        self._settings = settings

    @asynccontextmanager
    async def _container(self, case: EvalCase) -> AsyncIterator[tuple[Container, dict[str, Any], _Db]]:
        clock = FixedClock(case.now)
        schema = f"eval_{uuid.uuid4().hex[:12]}"
        settings = (
            self._settings.model_copy(update={"agent_mode": case.agent_mode}) if case.agent_mode else self._settings
        )
        async with AsyncExitStack() as stack:
            storage = await open_storage(stack, settings, clock, schema=schema, with_history=False)
            await seed_demo_users(storage.sessions, settings.seed_password, case.now)
            repositories = storage.repositories
            if case.faults.save_unavailable:
                repositories["reports"].add = _raise_unavailable
            if case.faults.booking_unavailable:
                repositories["appointments"].book = _raise_unavailable
            llm: LLMPort = (
                UnavailableLLM() if case.faults.llm_unavailable else ScriptedLLM(self._llm_factory(), case.llm_script)
            )
            container = assemble(settings, repositories, llm, UnavailableSTT(), clock, storage.checkpointer)
            yield container, repositories, _Db(storage.sessions)

    async def run(self, case: EvalCase) -> list[CaseObservation]:
        async with self._container(case) as (container, repositories, db):
            if case.faults.prebook_window:
                await _prebook(repositories, case.faults.prebook_window, case.now)
            await _setup_appointments(repositories, case)
            sessions = case.concurrency.sessions if case.concurrency else 1
            barrier = asyncio.Barrier(sessions) if sessions > 1 else None
            observations = await asyncio.gather(
                *[self._drive(case, container, repositories, db, i, barrier) for i in range(sessions)]
            )
            # Kiểm tra toàn cục sau khi mọi phiên kết thúc.
            doubled = await db.double_booked_slot_ids()
            for observation in observations:
                observation.double_booked_slot_ids = doubled
            return list(observations)

    async def _drive(
        self,
        case: EvalCase,
        container: Container,
        repositories: dict[str, Any],
        db: _Db,
        session: int,
        barrier: asyncio.Barrier | None,
    ) -> CaseObservation:
        conversation_id = f"{case.id}-{session}"
        patient = case.patient_id if barrier is None else f"{case.patient_id}-{session}"
        mode = case.agent_mode or self._settings.effective_agent_mode
        observation = CaseObservation(
            case_id=case.id,
            suite=case.suite,
            category=case.category,
            session=session,
            status="waiting",
            agent_mode=mode,
        )
        started = time.perf_counter()
        offered: list[str] = []
        # Lịch hẹn có từ trước tin nhắn chính (setup, lượt history, lịch tự đặt ở tab khác): không tính là
        # kết quả của case, nhưng mọi slot đề xuất sau đó không được trùng giờ với chúng.
        existing: set[str] = set()
        stolen = False
        replies: list[Reply] = list(case.replies)

        async def converse(message: UserTurn, turn_replies: list[Reply], scored: bool) -> TurnResult:
            result: TurnResult = await container.chat().send_message(
                conversation_id, patient, message.text, message.input_source, message.stt_flags
            )
            while True:
                result = await answer_interrupts(result, turn_replies, scored)
                # Chế độ agent hỏi lại bằng tin nhắn chat (lượt kết thúc, không có interrupt): câu trả lời kịch bản
                # kế tiếp được gửi như một tin nhắn mới — giống người dùng thật nhắn lại.
                question = _chat_reply(result) if mode != "workflow" else None
                if question is None:
                    break
                if not turn_replies:
                    if scored and "?" in question:
                        observation.interrupts.append("chat_question_open")
                        observation.interrupt_messages.append(question)
                        observation.ended_with_question = True
                    break
                reply = turn_replies.pop(0)
                if scored:
                    # Kịch bản trả lời bằng "choose" = agent liệt kê slot cho chọn (tương đương thẻ choose_slot của
                    # workflow), không phải hỏi lại.
                    observation.interrupts.append("chat_choose_slot" if reply.action == "choose" else "chat_question")
                    observation.interrupt_messages.append(question)
                result = await container.chat().send_message(conversation_id, patient, _reply_text(reply), "text", [])
            if not scored:
                existing.update(await db.booked_slot_ids(patient))
            return result

        async def answer_interrupts(result: TurnResult, turn_replies: list[Reply], scored: bool) -> TurnResult:
            nonlocal stolen
            while result.status == "waiting_user" and result.interrupt is not None:
                payload = result.interrupt
                if scored:
                    observation.interrupts.append(payload.kind)
                    observation.interrupt_messages.append(payload.message)
                if payload.kind == "choose_slot":
                    ids = [option.slot.slot_id for option in payload.options]
                    offered.extend(ids)
                    if scored:
                        mine = [
                            slot
                            for slot in [await repositories["slots"].get(slot_id) for slot_id in sorted(existing)]
                            if slot
                        ]
                        observation.own_overlap_slot_ids += [
                            option.slot.slot_id for option in payload.options if _overlaps(option.slot, mine)
                        ]
                        if not observation.first_offered_slot_ids:
                            observation.first_offered_slot_ids = ids
                            observation.first_offers = [
                                {
                                    "slot_id": option.slot.slot_id,
                                    "start_at": option.slot.start_at.isoformat(),
                                    "doctor_id": option.slot.doctor_id,
                                    "department_code": option.slot.department_code,
                                    "hint": option.hint,
                                }
                                for option in payload.options
                            ]
                if payload.kind == "confirm_booking":
                    # Chế độ agent không có thẻ chọn slot: slot hiện trên thẻ xác nhận là slot đã đưa cho người dùng.
                    pending = await container.chat().inspect_state(conversation_id)
                    proposal = pending.agent_proposal if pending else None
                    if proposal is not None and proposal.slot is not None:
                        slot = proposal.slot
                        offered.append(slot.slot_id)
                        if scored:
                            mine = [
                                slot
                                for slot in [await repositories["slots"].get(slot_id) for slot_id in sorted(existing)]
                                if slot
                            ]
                            if _overlaps(slot, mine):
                                observation.own_overlap_slot_ids.append(slot.slot_id)
                            if not observation.first_offered_slot_ids:  # chế độ agent: đề xuất đầu tiên = slot trên thẻ
                                observation.first_offered_slot_ids = [slot.slot_id]
                                observation.first_offers = [
                                    {
                                        "slot_id": slot.slot_id,
                                        "start_at": slot.start_at.isoformat(),
                                        "doctor_id": slot.doctor_id,
                                        "department_code": slot.department_code,
                                        "hint": None,
                                    }
                                ]
                if not turn_replies:
                    break
                if scored and payload.kind == "confirm_booking" and not stolen:
                    if barrier is not None:
                        await barrier.wait()  # mọi phiên cùng chốt một lúc -> tranh chấp thật
                    state = await container.chat().inspect_state(conversation_id)
                    selected = state.selected_slot_id if state else None
                    if selected is None and state and state.agent_proposal and state.agent_proposal.slot:
                        selected = state.agent_proposal.slot.slot_id
                    if case.faults.slot_taken_after_choice and selected:
                        await container.toolbox().create_appointment(
                            "intruder-patient", selected, None, f"intruder-{conversation_id}", case.now
                        )
                    if case.faults.own_clash_after_choice and selected:
                        await _book_own_clash(container, repositories, patient, selected, f"clash-{conversation_id}")
                        existing.update(await db.booked_slot_ids(patient))
                    stolen = True
                reply = turn_replies.pop(0)
                result = await container.chat().resume(conversation_id, patient, UserDecision(**reply.model_dump()))
            return result

        try:
            existing.update(await db.booked_slot_ids(patient))
            for turn in case.history:
                await converse(turn.message, list(turn.replies), scored=False)
            reports_before = len(await db.reports(patient))
            cancelled_before = await db.cancelled_count(patient)
            result = await converse(case.message, replies, scored=True)
            observation.status = "done" if result.status == "done" else "waiting"
            if result.response is not None:
                observation.outcome = result.response.kind.value
                observation.response_text = result.response.text
            observation.replies_left = len(replies)
            state = await container.chat().inspect_state(conversation_id)
            if state is not None:
                observation.intent = state.intent.value if state.intent else None
                observation.outcome_code = state.outcome.code if state.outcome else None
                observation.final_fields = _fields(state)
                observation.attempts = [
                    AttemptView(
                        attempt_no=attempt.attempt_no,
                        issue_codes=attempt.issue_codes,
                        llm_fixable_codes=attempt.llm_fixable_codes,
                        passed=attempt.passed,
                        output=attempt.output,
                    )
                    for attempt in state.attempts
                ]
                observation.tool_calls = [tool_call.tool for tool_call in state.tool_calls]
        except Exception as exception:  # noqa: BLE001 - một case lỗi không được làm hỏng cả lượt eval
            observation.status = "crashed"
            observation.error = f"{type(exception).__name__}: {exception}"
            reports_before = 0
            cancelled_before = 0
        observation.latency_ms = int((time.perf_counter() - started) * 1000)

        # Chỉ tính lịch hẹn do tin nhắn chính tạo ra.
        observation.booked_slot_ids = [
            slot_id for slot_id in await db.booked_slot_ids(patient) if slot_id not in existing
        ]
        observation.cancelled_count = await db.cancelled_count(patient) - cancelled_before
        observation.offered_slot_ids = sorted(set(offered))
        known = await db.slot_ids()
        observation.hallucinated_slot_ids = sorted(
            {slot_id for slot_id in offered if slot_id not in known}
            | {slot_id for slot_id in observation.booked_slot_ids if slot_id not in offered}
        )
        observation.own_overlap_slot_ids = sorted(set(observation.own_overlap_slot_ids))
        reports = await db.reports(patient)
        observation.reports_written = len(reports) - reports_before
        observation.saved_report = (
            reports[-1].model_dump(mode="json", exclude={"evidence"}) if observation.reports_written > 0 else None
        )
        return observation


async def run_cases(
    cases: list[EvalCase], llm_factory: LLMFactory, settings: Settings, concurrency: int = 4
) -> list[CaseObservation]:
    runner = CaseRunner(llm_factory, settings)
    limit = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> list[CaseObservation]:
        async with limit:
            return await runner.run(case)

    nested = await asyncio.gather(*[one(case) for case in cases])
    return [observation for group in nested for observation in group]
