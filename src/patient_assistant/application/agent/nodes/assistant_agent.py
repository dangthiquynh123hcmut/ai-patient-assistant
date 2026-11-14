"""Agent LLM + tool (PA_AGENT_MODE=booking | full): LLM tự hiểu ý, tự tra dữ liệu, chỉ hỏi cái còn thiếu.

Phân vai:
- LLM quyết định hỏi gì, gọi tool đọc nào, tự tính ngày ("cách lịch hiện tại 7 ngày"), rồi gọi một tool
  `propose_*` khi muốn ghi. `booking` chỉ có tool lịch hẹn; `full` thêm báo cáo theo dõi, thông tin bệnh nhân,
  lịch sử báo cáo, tài liệu đã duyệt — một agent cho mọi việc, kể cả nhiều việc trong một câu.
- Code giữ các hàng rào: `patient_id` gắn sẵn trong tool; khoảng ngày theo chính sách đặt lịch; `slot_id` đề xuất
  phải là slot backend đã trả cho agent; lịch đổi/hủy phải là lịch sắp tới của chính bệnh nhân; báo cáo qua đủ
  validator (trích dẫn nguyên văn, phạm vi giá trị); chỉ ghi sau khi người dùng xác nhận; ghi có idempotency và
  đọc lại (dùng chung node ghi/đọc lại với luồng workflow).
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from patient_assistant.application.agent.deps import AgentDeps
from patient_assistant.application.agent.nodes.shared import (
    ask_user,
    idempotency_key,
    is_cancel,
    next_ask,
    tool_error_outcome,
    tool_record,
)
from patient_assistant.application.agent.presenters import CHAT_FALLBACK, final_text, report_explanations
from patient_assistant.application.agent.routing import confirmation_of
from patient_assistant.application.agent.state import (
    AgentProposal,
    AgentState,
    FieldExplanation,
    InterruptPayload,
    Outcome,
    OutcomeKind,
)
from patient_assistant.application.ports.llm import AgentMessage, LLMError, ToolSpec
from patient_assistant.application.tools.result import ToolCallRecord, ToolErrorCode, ToolResult
from patient_assistant.domain.booking.models import (
    Appointment,
    AppointmentStatus,
    BookingRequest,
    Department,
    Doctor,
    Slot,
    SlotOption,
)
from patient_assistant.domain.booking.policy import window_issues
from patient_assistant.domain.common import Origin, Resolution
from patient_assistant.domain.conversation import (
    ConfirmIntent,
    SafetyKind,
    check_input_safety,
    is_filler,
    output_is_safe,
)
from patient_assistant.domain.report.models import REPORT_FIELDS, ReportExtraction
from patient_assistant.domain.report.validators import ReportValidationContext, parse_report, validate_report
from patient_assistant.domain.text.normalize import normalize
from patient_assistant.domain.text.vi_dates import WEEKDAY_NAMES, DateWindow

EN_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MAX_HISTORY = 30  # số tin nhắn giữ lại qua các lượt
MAX_KNOWN_SLOTS = 30
MAX_WRITES_PER_TURN = 3  # trần số việc ghi trong một lượt (nhiều việc trong một câu)
SYSTEM_NOTE = "(System notice, not the user's words)"
WRITE_KINDS = frozenset(
    {
        OutcomeKind.APPOINTMENT_BOOKED,
        OutcomeKind.APPOINTMENT_RESCHEDULED,
        OutcomeKind.APPOINTMENT_CANCELLED,
        OutcomeKind.REPORT_SAVED,
    }
)


# ---------------------------------------------------------------- input của tool (LLM nhìn thấy)


class FindAppointmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    when: Literal["upcoming", "past", "all"] = Field(
        description="upcoming: upcoming (current) appointments, nearest first; past: attended appointments, most "
        "recent first; all: everything, newest first"
    )
    department_code: str | None = Field(description="Filter by department code; null for no filter")


class SearchSlotsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    department_code: str = Field(description="Department code from the catalog")
    doctor_id: str | None = Field(description="Doctor id to require that doctor; null if any doctor is fine")
    date_from: date = Field(description="Start date, YYYY-MM-DD")
    date_to: date = Field(description="End date (may equal date_from), YYYY-MM-DD")
    time_from: time | None = Field(description="Earliest time HH:MM on each day; null if any time is fine")
    time_to: time | None = Field(description="Latest time HH:MM on each day; null if any time is fine")


class ProposeBookingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_id: str = Field(description="slot_id taken from search_slots results")
    reason: str | None = Field(description="Reason for the visit as the user said it (verbatim, short); null if none")
    basis: str = Field(
        description="One short sentence, in the user's language, explaining what this slot is based on, e.g. "
        "'Giống lịch Nội tổng quát 9:00 ngày 05/11, cách 7 ngày'"
    )


class ProposeRescheduleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appointment_id: str = Field(description="id of the appointment to move, taken from find_my_appointments")
    slot_id: str = Field(description="New slot_id, taken from search_slots results")
    basis: str = Field(description="One short sentence, in the user's language: which appointment moves where, and why")


class ProposeCancelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    appointment_id: str = Field(description="id of the appointment to cancel, taken from find_my_appointments")
    basis: str = Field(
        description="One short sentence, in the user's language: which appointment to cancel, per the user"
    )


class SearchKnowledgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(description="Search query for the approved guidance documents (in the user's language)")


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- helper


def _local(dt: datetime, ref: datetime) -> datetime:
    return dt.astimezone(ref.tzinfo)


def _slot_view(slot: Slot) -> dict[str, Any]:
    local = slot.start_at
    return {
        "slot_id": slot.slot_id,
        "department_code": slot.department_code,
        "doctor_id": slot.doctor_id,
        "doctor_name": slot.doctor_name,
        "weekday": WEEKDAY_NAMES[local.weekday()],
        "date": local.date().isoformat(),
        "start": f"{local:%H:%M}",
        "end": f"{slot.end_at:%H:%M}",
    }


def _context(state: AgentState, departments: list[Department], doctors: list[Doctor]) -> str:
    """Bối cảnh do code soạn: hôm nay, danh mục (để LLM dùng đúng mã), slot đã tra ở lượt trước."""
    today = state.now
    lines = [
        f"Today: {EN_WEEKDAYS[today.weekday()]}, {today:%Y-%m-%d} {today:%H:%M} (time zone {today.tzinfo}).",
        f"Current ISO week: {today:%G-W%V}.",
        "",
        "Department catalog (code: name — other names):",
        *[f"- {d.code}: {d.name}" + (f" — {', '.join(d.aliases)}" if d.aliases else "") for d in departments],
        "",
        "Doctors (id: name — department):",
        *[f"- {d.id}: {d.name} — {d.department_code}" for d in doctors],
    ]
    if state.stt_flags:
        # Cờ STT chỉ được validator kiểm khi agent gọi propose_report; báo trước để agent hỏi lại đúng chỗ nghi ngờ.
        lines += [
            "",
            "The user's message is a voice transcript. These words were recognised with LOW confidence "
            "and may be wrong:",
            *[f'- "{f.text}" (confidence {f.probability:.2f})' for f in state.stt_flags],
            "Before using a value that comes from one of these words, ask the user to confirm that value.",
        ]
    if state.agent_slots:
        lines += ["", "Slots looked up in earlier turns (may have been taken since; search again if needed):"]
        lines += [
            f"- {v['slot_id']}: {v['weekday']} {v['date']} {v['start']} — {v['doctor_name']} ({v['department_code']})"
            for v in map(_slot_view, state.agent_slots[-10:])
        ]
    return "\n".join(lines)


def _trim(history: list[AgentMessage]) -> list[AgentMessage]:
    history = history[-MAX_HISTORY:]
    while history and history[0].role != "user":
        history = history[1:]
    return history


def _failure(
    tool: str, message: str, details: list[str] | None = None, code: ToolErrorCode = ToolErrorCode.INVALID_ARGS
) -> dict[str, Any]:
    return ToolResult.failure(tool, code, message, details=details).model_dump(mode="json")


def _ok(tool: str, data: Any, ids: list[str]) -> dict[str, Any]:
    return ToolResult.success(tool, data, ids).model_dump(mode="json")


def _record(sink: list[ToolCallRecord], result: dict[str, Any]) -> dict[str, Any]:
    error = result.get("error") or {}
    sink.append(
        ToolCallRecord(
            tool=result.get("tool", "?"),
            called_by="llm",
            status=result.get("status", "error"),
            error_code=error.get("code"),
            latency_ms=result.get("latency_ms", 0),
            source_ids=result.get("source_ids", []),
        )
    )
    return result


def _user_texts(history: list[AgentMessage]) -> list[str]:
    return [m.text for m in history if m.role == "user" and not m.text.startswith(SYSTEM_NOTE)]


class _Run:
    """Tool cho một lần chạy agent; ghi lại slot đã thấy, đề xuất, nguồn tài liệu và mọi lần gọi tool."""

    def __init__(
        self,
        deps: AgentDeps,
        state: AgentState,
        departments: list[Department],
        doctors: list[Doctor],
        history: list[AgentMessage],
    ):
        self.deps = deps
        self.state = state
        self.departments = {d.code for d in departments}
        self.doctors = {d.id: d for d in doctors}
        self.user_texts = _user_texts(history)
        self.slots: dict[str, Slot] = {s.slot_id: s for s in state.agent_slots}
        self.new_slots: list[Slot] = []
        self.proposal: AgentProposal | None = None
        self.proposal_tool: str | None = None
        self.knowledge_ids: set[str] = set()
        self.calls: list[ToolCallRecord] = []
        # Số lần `propose_report` bị validator từ chối vì lỗi LLM tự sửa được (không tính lỗi phải hỏi người dùng).
        self.rejections = state.correction_count
        self.handoff = False

    # ------------------------------------------------------------ đọc

    async def find_appointments(self, inp: BaseModel) -> dict[str, Any]:
        assert isinstance(inp, FindAppointmentsInput)
        result = await self.deps.tools.find_my_appointments(
            self.state.patient_id, self.state.now, inp.when, inp.department_code
        )
        data = result.model_dump(mode="json")
        for item in data.get("data") or []:
            item["weekday"] = WEEKDAY_NAMES[_local(datetime.fromisoformat(item["start_at"]), self.state.now).weekday()]
        return _record(self.calls, data)

    async def search_slots(self, inp: BaseModel) -> dict[str, Any]:
        assert isinstance(inp, SearchSlotsInput)
        tool = "get_available_slots"
        if inp.department_code not in self.departments:
            return _record(self.calls, _failure(tool, f"Department code {inp.department_code} is not in the catalog."))
        doctor = self.doctors.get(inp.doctor_id) if inp.doctor_id else None
        if inp.doctor_id and (doctor is None or doctor.department_code != inp.department_code):
            return _record(
                self.calls,
                _failure(tool, f"Doctor {inp.doctor_id} does not belong to department {inp.department_code}."),
            )
        if inp.date_to < inp.date_from:
            return _record(self.calls, _failure(tool, "date_to must be on or after date_from."))
        window = DateWindow(date_from=inp.date_from, date_to=inp.date_to, time_from=inp.time_from, time_to=inp.time_to)
        issues = window_issues(window, self.state.now, self.deps.tools.booking_policy)
        if issues:
            return _record(self.calls, _failure(tool, "This date range cannot be booked.", [i.message for i in issues]))
        result = await self.deps.tools.get_available_slots(
            self.state.patient_id, inp.department_code, window, inp.doctor_id, self.state.now
        )
        data = result.model_dump(mode="json")
        if result.ok:
            found = [Slot.model_validate(s) for s in result.data]
            for slot in found:
                if slot.slot_id not in self.slots:
                    self.new_slots.append(slot)
                self.slots[slot.slot_id] = slot
            data["data"] = [
                {
                    **_slot_view(s),
                    **(
                        {"same_day_appointment": raw["same_day_appointment"]} if raw.get("same_day_appointment") else {}
                    ),
                }
                for s, raw in zip(found, result.data, strict=True)
            ]
        return _record(self.calls, data)

    async def patient_plan(self, _: BaseModel) -> dict[str, Any]:
        return _record(
            self.calls, (await self.deps.tools.get_patient_plan(self.state.patient_id)).model_dump(mode="json")
        )

    async def patient_summary(self, _: BaseModel) -> dict[str, Any]:
        result = (await self.deps.tools.get_patient_summary(self.state.patient_id, self.state.now)).model_dump(
            mode="json"
        )
        # LLM không cần ngày sinh / số điện thoại để trả lời -> không gửi dữ liệu cá nhân này cho LLM.
        for key in ("date_of_birth", "phone"):
            ((result.get("data") or {}).get("profile") or {}).pop(key, None)
        return _record(self.calls, result)

    async def report_history(self, _: BaseModel) -> dict[str, Any]:
        return _record(
            self.calls, (await self.deps.tools.get_report_history(self.state.patient_id)).model_dump(mode="json")
        )

    async def search_knowledge(self, inp: BaseModel) -> dict[str, Any]:
        assert isinstance(inp, SearchKnowledgeInput)
        result = await self.deps.tools.search_approved_knowledge(inp.query)
        self.knowledge_ids |= set(result.source_ids)
        return _record(self.calls, result.model_dump(mode="json"))

    # ------------------------------------------------------------ đề xuất ghi (terminal)

    def _proposed(self, tool: str, proposal: AgentProposal, ids: list[str]) -> dict[str, Any]:
        # Hàng rào cho "nhiều việc trong một câu": việc đã ghi trong lượt này thì không đề xuất lại lần nữa.
        done = set(_written_keys(proposal)) & set(self.state.agent_written)
        if done:
            return _record(
                self.calls,
                _failure(
                    tool,
                    "This task was already completed in this turn; do not propose it again. If no other task is left, "
                    "reply with an empty string.",
                    sorted(done),
                ),
            )
        self.proposal, self.proposal_tool = proposal, tool
        return _record(self.calls, _ok(tool, {"shown_to_user": True}, ids))

    async def _upcoming_appointment(self, tool: str, appointment_id: str) -> Appointment | dict[str, Any]:
        back = await self.deps.tools.read_back_appointment(self.state.patient_id, appointment_id)
        if not back.ok:
            return _record(
                self.calls,
                _failure(
                    tool,
                    "The patient has no such appointment. Take the id from find_my_appointments.",
                    code=ToolErrorCode.NOT_FOUND,
                ),
            )
        appt = Appointment.model_validate(back.data)
        if appt.status != AppointmentStatus.BOOKED or appt.slot.start_at < self.state.now:
            return _record(
                self.calls,
                _failure(
                    tool, "This appointment is already cancelled or in the past; it cannot be changed or cancelled."
                ),
            )
        return appt

    def _known_slot(self, tool: str, slot_id: str) -> Slot | dict[str, Any]:
        slot = self.slots.get(slot_id)
        if slot is None:
            return _record(
                self.calls, _failure(tool, "This slot_id is not in the search_slots results. Search for slots first.")
            )
        return slot

    async def propose_booking(self, inp: BaseModel) -> dict[str, Any]:
        assert isinstance(inp, ProposeBookingInput)
        slot = self._known_slot("propose_booking", inp.slot_id)
        if isinstance(slot, dict):
            return slot
        proposal = AgentProposal(kind="booking", slot=slot, reason=inp.reason, basis=inp.basis)
        return self._proposed("propose_booking", proposal, [slot.slot_id])

    async def propose_reschedule(self, inp: BaseModel) -> dict[str, Any]:
        assert isinstance(inp, ProposeRescheduleInput)
        tool = "propose_reschedule"
        appt = await self._upcoming_appointment(tool, inp.appointment_id)
        if isinstance(appt, dict):
            return appt
        slot = self._known_slot(tool, inp.slot_id)
        if isinstance(slot, dict):
            return slot
        if slot.slot_id == appt.slot.slot_id:
            return _record(self.calls, _failure(tool, "The new slot is the same as the current appointment."))
        proposal = AgentProposal(kind="reschedule", slot=slot, appointment=appt, reason=appt.reason, basis=inp.basis)
        return self._proposed(tool, proposal, [appt.id, slot.slot_id])

    async def propose_cancel(self, inp: BaseModel) -> dict[str, Any]:
        assert isinstance(inp, ProposeCancelInput)
        appt = await self._upcoming_appointment("propose_cancel", inp.appointment_id)
        if isinstance(appt, dict):
            return appt
        return self._proposed(
            "propose_cancel", AgentProposal(kind="cancel", appointment=appt, basis=inp.basis), [appt.id]
        )

    async def propose_report(self, inp: BaseModel) -> dict[str, Any]:
        assert isinstance(inp, ReportExtraction)
        tool = "propose_report"
        if inp.reporting_period == f"{self.state.now:%G-W%V}":
            # "Tuần này" -> LLM hay tự điền tuần hiện tại mà không có trích dẫn, bị từ chối rồi phải thử lại (lần thử
            # lại dễ sinh trích dẫn hỏng ký tự). Tuần hiện tại cũng là giá trị mặc định khi lưu -> bỏ đi.
            evidence = [e for e in inp.evidence if e.field != "reporting_period"]
            inp = inp.model_copy(update={"reporting_period": None, "evidence": evidence})
        issues = validate_report(
            inp,
            ReportValidationContext(sources=self.user_texts, now=self.state.now, stt_flags=self.state.stt_flags),
        )
        if issues:
            details = [f"{i.field}: {i.message}" for i in issues]
            if any(i.resolution == Resolution.ASK_USER for i in issues):
                hint = "Ask the user for what is missing/unclear; do not fill it in yourself."
                return _record(self.calls, _failure(tool, f"The report is not valid yet. {hint}", details))
            self.rejections += 1
            if self.rejections > self.deps.limits.max_corrections:
                # Cùng trần với workflow (FR-02.5): sửa quá số lần cho phép -> dừng, chuyển nhân viên, không lưu.
                self.handoff = True
                failure = _failure(tool, "Too many invalid attempts; the request is handed to staff.", details)
                return _record(self.calls, failure | {"stop": True})
            hint = "Fix it to match the user's words (verbatim quotes, values exactly as they said)."
            return _record(self.calls, _failure(tool, f"The report is not valid yet. {hint}", details))
        return self._proposed(tool, AgentProposal(kind="report", report=inp), [])

    def specs(self) -> list[ToolSpec]:
        full = self.deps.agent_mode == "full"
        specs = [
            ToolSpec(
                "find_my_appointments",
                "The logged-in patient's booked appointments (id, department, doctor, time, reason). Use when the "
                "user asks about appointments, refers to a previous/current appointment or visit, or wants to change "
                "or cancel one.",
                FindAppointmentsInput,
                self.find_appointments,
            ),
            ToolSpec(
                "search_slots",
                "Find free appointment slots by department, doctor (optional), date range and time window. Slots that "
                "clash with the patient's own appointments are already excluded.",
                SearchSlotsInput,
                self.search_slots,
            ),
            ToolSpec(
                "get_patient_plan",
                "The patient's current follow-up plan (includes the department following them).",
                EmptyInput,
                self.patient_plan,
            ),
            ToolSpec(
                "propose_booking",
                "Propose booking ONE new slot. The system shows a card for the user to confirm; nothing is written "
                "until they confirm.",
                ProposeBookingInput,
                self.propose_booking,
                terminal=True,
            ),
            ToolSpec(
                "propose_reschedule",
                "Propose moving ONE upcoming appointment to a new slot (the old one is cancelled at the same time). "
                "The system shows a confirmation card.",
                ProposeRescheduleInput,
                self.propose_reschedule,
                terminal=True,
            ),
            ToolSpec(
                "propose_cancel",
                "Propose cancelling ONE upcoming appointment. The system shows a confirmation card.",
                ProposeCancelInput,
                self.propose_cancel,
                terminal=True,
            ),
        ]
        if full:
            specs += [
                ToolSpec(
                    "get_patient_summary",
                    "The logged-in patient's record, follow-up plan, recent reports and upcoming appointments.",
                    EmptyInput,
                    self.patient_summary,
                ),
                ToolSpec(
                    "get_report_history",
                    "The patient's recent follow-up reports (period, pain level, exercise days, symptoms).",
                    EmptyInput,
                    self.report_history,
                ),
                ToolSpec(
                    "search_approved_knowledge",
                    "Search the approved guidance documents. Knowledge questions may be answered only from these "
                    "results, citing the doc_id.",
                    SearchKnowledgeInput,
                    self.search_knowledge,
                ),
                ToolSpec(
                    "propose_report",
                    "Propose saving the weekly follow-up report. Every field with a value needs evidence: a VERBATIM "
                    "quote of the user's words. The system validates it, then shows a card for the user to confirm.",
                    ReportExtraction,
                    self.propose_report,
                    terminal=True,
                ),
            ]
        return specs


# ---------------------------------------------------------------- node


class AgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    async def agent(self, state: AgentState) -> dict[str, Any]:
        if state.agent_note:  # việc trước đã ghi xong -> agent xem còn việc nào người dùng yêu cầu mà chưa làm
            said = f"{SYSTEM_NOTE}: {state.agent_note}"
        elif state.notice:  # slot vừa bị đặt mất khi ghi -> tìm lại, không hỏi lại từ đầu
            said = f"{SYSTEM_NOTE}: {state.notice} Find another slot as close as possible to the request."
        elif state.agent_input:
            said = state.agent_input
            verdict = check_input_safety(said)  # câu trả lời ở thẻ xác nhận cũng qua chặn an toàn
            if verdict.kind != SafetyKind.OK:
                return {
                    "safety": verdict.kind,
                    "agent_input": None,
                    "outcome": Outcome(kind=OutcomeKind.SAFETY, code=verdict.kind.value),
                }
        else:
            said = "\n".join([*state.context_sources, state.raw_input, *state.user_replies])
        history = _trim([*state.agent_history, AgentMessage(role="user", text=said)])
        departments, doctors = await self.deps.tools.catalog()
        run = _Run(self.deps, state, departments, doctors, history)
        updates: dict[str, Any] = {"notice": None, "agent_input": None, "agent_note": None, "agent_proposal": None}
        try:
            reply = await self.deps.llm.run_agent(
                "assistant" if self.deps.agent_mode == "full" else "booking",
                _context(state, departments, doctors),
                history,
                run.specs(),
                self.deps.limits.max_agent_tool_calls,
            )
        except LLMError as exc:
            return updates | {
                "tool_calls": [*state.tool_calls, *run.calls],
                "agent_run_tools": [c.tool for c in run.calls],
                "agent_history": history,
                "agent_active": False,
                "outcome": state.agent_last_write or Outcome(kind=OutcomeKind.ERROR, code=f"llm_{exc.code}"),
            }
        updates |= {
            "tool_calls": [*state.tool_calls, *run.calls],
            "agent_run_tools": [c.tool for c in run.calls],
            "agent_slots": [*state.agent_slots, *run.new_slots][-MAX_KNOWN_SLOTS:],
            "correction_count": run.rejections,
        }
        if run.handoff:
            return updates | {
                "agent_history": history,
                "agent_active": False,
                "outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_corrections"),
            }
        if run.proposal is not None and reply.stopped_by == run.proposal_tool:
            note = reply.text or f"(Proposed: {_proposal_title(run.proposal)})"
            return updates | {
                "agent_proposal": run.proposal,
                "edited_fields": [],
                "agent_history": [*history, AgentMessage(role="assistant", text=note)],
            }
        text = reply.text.strip()
        if text and not output_is_safe(text):
            text = CHAT_FALLBACK
        if state.agent_last_write is not None:
            # Đã ghi xong việc trước trong lượt này: kết quả đã xác minh là câu trả lời chính, lời agent nói thêm.
            return updates | {
                "agent_history": [*history, AgentMessage(role="assistant", text=text)] if text else history,
                "agent_followup": text or None,
                "agent_active": bool(text),
                "outcome": state.agent_last_write,
            }
        text = text or CHAT_FALLBACK
        cited = sorted(d for d in run.knowledge_ids if d in text)
        return updates | {
            "agent_history": [*history, AgentMessage(role="assistant", text=text)],
            "agent_active": True,
            "outcome": Outcome(kind=OutcomeKind.CHAT, data=text, source_ids=cited),
        }

    # ------------------------------------------------------------ thẻ xác nhận

    async def confirm(self, state: AgentState) -> dict[str, Any]:
        proposal = state.agent_proposal
        assert proposal is not None
        key = proposal.model_dump_json(include={"kind", "slot", "appointment", "report"})
        counters = next_ask(state, f"agent_confirm:{key}", self.deps)
        if counters is None:
            return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds")}
        departments, _ = await self.deps.tools.catalog()
        names = {d.code: d.name for d in departments}
        lead = "\n".join([*state.agent_done, *([proposal.basis] if proposal.basis else [])])
        question, explanations, kind = _card(proposal, names, state)
        actions = (
            ["confirm", "edit", "cancel", "reply"] if proposal.kind == "report" else ["confirm", "cancel", "reply"]
        )
        decision = ask_user(
            InterruptPayload(
                kind=kind,
                message=f"{lead}\n{question}" if lead else question,
                explanations=explanations,
                allowed_actions=actions,
            )
        )
        updates: dict[str, Any] = {**counters, "decision": decision}
        if is_cancel(decision):
            return updates | {"outcome": Outcome(kind=OutcomeKind.CANCELLED)}
        if decision.action == "edit" and proposal.kind == "report" and proposal.report is not None:
            return updates | self._edit_report(state, proposal, decision.fields)
        if confirmation_of(state.model_copy(update={"decision": decision})) == ConfirmIntent.CONFIRM:
            updates["decision"] = decision.model_copy(update={"action": "confirm"})
            if proposal.kind == "booking":
                # Đưa về đúng dạng `create_booking`/`verify_booking` của workflow dùng: một slot đã chọn, đã xác nhận.
                assert proposal.slot is not None
                return updates | {
                    "booking_request": BookingRequest(reason=proposal.reason),
                    "offered_slots": [SlotOption(option_no=1, slot=proposal.slot)],
                    "selected_slot_id": proposal.slot.slot_id,
                    "issues": [],
                }
            if proposal.kind == "report":
                return updates | {"report_draft": proposal.report, "issues": []}
            return updates
        text = decision.text or " ".join(f"{k}: {v}" for k, v in decision.fields.items())
        if decision.action == "reply" and is_filler(text):
            # "ờ", "ừm": chưa xác nhận cũng chưa đổi gì -> hiện lại đúng thẻ này (bộ đếm hỏi lại vẫn chạy).
            return updates
        # Nhắn điều muốn đổi ("9:30 được không", "đau mức 4 chứ") -> quay lại agent, đề xuất cũ bỏ.
        return updates | {"agent_input": text or "(The user did not confirm.)", "agent_proposal": None}

    def _edit_report(self, state: AgentState, proposal: AgentProposal, fields: dict[str, Any]) -> dict[str, Any]:
        """Sửa trực tiếp trên thẻ (form): code áp giá trị, chạy lại validator, hiện lại thẻ — không qua LLM."""
        assert proposal.report is not None
        changes = {k: v for k, v in fields.items() if k in REPORT_FIELDS}
        draft, schema_issues = parse_report({**proposal.report.model_dump(mode="json"), **changes})
        edited = sorted(set(state.edited_fields) | set(changes))
        issues = schema_issues or (
            validate_report(
                draft,
                ReportValidationContext(
                    sources=_user_texts(state.agent_history),
                    now=state.now,
                    stt_flags=state.stt_flags,
                    edited_fields=frozenset(edited),
                ),
            )
            if draft
            else []
        )
        if draft is None or issues:
            text = "I want to edit the report: " + ", ".join(f"{k} = {v}" for k, v in changes.items())
            return {"agent_input": text, "agent_proposal": None}
        return {"agent_proposal": proposal.model_copy(update={"report": draft}), "edited_fields": edited}

    # ------------------------------------------------------------ ghi đổi/hủy lịch (code gọi, sau xác nhận)

    async def change_appointment(self, state: AgentState) -> dict[str, Any]:
        proposal = state.agent_proposal
        if proposal is None or proposal.appointment is None or confirmation_of(state) != ConfirmIntent.CONFIRM:
            return {"outcome": Outcome(kind=OutcomeKind.ERROR, code="guard_violation")}
        appt = proposal.appointment
        if proposal.kind == "cancel":
            result = await self.deps.tools.cancel_appointment(state.patient_id, appt.id, state.now)
        else:
            assert proposal.slot is not None
            key = idempotency_key("reschedule", state.conversation_id, state.patient_id, appt.id, proposal.slot.slot_id)
            result = await self.deps.tools.reschedule_appointment(
                state.patient_id, appt.id, proposal.slot.slot_id, None, key, state.now
            )
        return {"tool_calls": tool_record(state, result), "write_result": result.model_dump(mode="json")}

    async def verify_change(self, state: AgentState) -> dict[str, Any]:
        proposal = state.agent_proposal
        assert proposal is not None and proposal.appointment is not None
        written = state.write_result or {}
        error = written.get("error") or {}
        if written.get("status") != "ok":
            if error.get("code") == ToolErrorCode.CONFLICT.value and proposal.kind == "reschedule":
                if state.conflict_retries >= self.deps.limits.max_conflict_retries:
                    return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="conflict")}
                reason = error.get("message") or "The chosen slot can no longer be booked."
                return {
                    "conflict_retries": state.conflict_retries + 1,
                    "notice": f"{reason} The old appointment is unchanged.",
                }
            result = ToolResult.model_validate(written)
            return {"outcome": tool_error_outcome(result, "change_failed")}
        old = await self.deps.tools.read_back_appointment(state.patient_id, proposal.appointment.id)
        tool_calls = tool_record(state, old)
        if not old.ok or old.data["status"] != AppointmentStatus.CANCELLED.value:
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ERROR, code="verify_failed")}
        if proposal.kind == "cancel":
            return {
                "tool_calls": tool_calls,
                "outcome": Outcome(kind=OutcomeKind.APPOINTMENT_CANCELLED, data=old.data, source_ids=[old.data["id"]]),
            }
        assert proposal.slot is not None
        new = await self.deps.tools.read_back_appointment(state.patient_id, (written.get("data") or {}).get("id", ""))
        tool_calls = tool_record(state.model_copy(update={"tool_calls": tool_calls}), new)  # nối sau lần đọc lịch cũ
        if not new.ok or new.data["slot"]["slot_id"] != proposal.slot.slot_id or new.data["status"] != "booked":
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ERROR, code="verify_failed")}
        return {
            "tool_calls": tool_calls,
            "outcome": Outcome(
                kind=OutcomeKind.APPOINTMENT_RESCHEDULED,
                data={"old": old.data, "new": new.data},
                source_ids=[new.data["id"], old.data["id"]],
            ),
        }

    # ------------------------------------------------------------ sau khi ghi + kết thúc

    async def after_write(self, state: AgentState) -> dict[str, Any]:
        """Ghi + xác minh xong: lưu vào lịch sử; ở chế độ full, cho agent làm tiếp việc còn lại trong câu."""
        outcome = state.outcome
        assert outcome is not None and outcome.kind in WRITE_KINDS
        done = final_text(outcome, SafetyKind.OK, [])
        all_done = [*state.agent_done, done]
        written = [*state.agent_written, *(_written_keys(state.agent_proposal) if state.agent_proposal else [])]
        updates: dict[str, Any] = {
            "agent_written": written,
            "agent_history": _trim([*state.agent_history, AgentMessage(role="assistant", text=done)]),
            "agent_done": all_done,
            "agent_last_write": outcome,
            "agent_active": False,
            "agent_proposal": None,
            "edited_fields": [],
            "conflict_retries": 0,
        }
        if self.deps.agent_mode == "full" and len(state.agent_done) + 1 < MAX_WRITES_PER_TURN:
            updates |= {
                "outcome": None,
                "agent_note": "Already completed in this turn (do NOT redo these):\n"
                + "\n".join(f"{i}. {d}" for i, d in enumerate(all_done, start=1))
                + "\nIf the user asked for any other task that is NOT in the list above, do it now; if nothing is "
                "left, reply with an empty string.",
            }
        return updates

    async def wrapup(self, state: AgentState) -> dict[str, Any]:
        """Ghi kết cục không phải ghi dữ liệu (hủy, lỗi...) vào lịch sử agent để lượt sau hiểu ngữ cảnh."""
        outcome = state.outcome
        if outcome is None or outcome.kind in (OutcomeKind.CHAT, *WRITE_KINDS):
            return {"agent_proposal": None}
        if outcome.kind == OutcomeKind.CANCELLED:
            note = "The user pressed cancel on the confirmation card; nothing was written."
        else:
            note = f"Could not be completed ({outcome.code or outcome.kind.value})."
        return {
            "agent_history": _trim([*state.agent_history, AgentMessage(role="assistant", text=note)]),
            "agent_active": False,
            "agent_proposal": None,
        }


# ---------------------------------------------------------------- thẻ xác nhận theo loại việc


def _written_keys(p: AgentProposal) -> list[str]:
    """Khóa nhận diện việc ghi, để chặn đề xuất lại đúng việc đã làm trong lượt."""
    keys = ["report"] if p.kind == "report" else []
    if p.slot is not None:
        keys.append(f"slot:{p.slot.slot_id}")
    if p.appointment is not None:
        keys.append(f"appt:{p.appointment.id}")
    return keys


def _proposal_title(p: AgentProposal) -> str:
    if p.kind == "booking" and p.slot:
        return f"book {p.slot.label()}"
    if p.kind == "reschedule" and p.slot and p.appointment:
        return f"move {p.appointment.slot.label()} to {p.slot.label()}"
    if p.kind == "cancel" and p.appointment:
        return f"cancel {p.appointment.slot.label()}"
    return "save the follow-up report"


def _card(
    p: AgentProposal, names: dict[str, str], state: AgentState
) -> tuple[str, list[FieldExplanation], Literal["confirm_booking", "confirm_report"]]:
    tail = "Bấm “xác nhận” để làm, “hủy” để dừng, hoặc nhắn điều bạn muốn đổi."
    if p.kind == "report":
        assert p.report is not None
        report_rows = report_explanations(p.report, [], state.edited_fields, state.now)
        return f"Bạn xác nhận lưu báo cáo theo dõi này? {tail}", report_rows, "confirm_report"
    rows: list[FieldExplanation] = []
    if p.appointment:
        old = p.appointment.slot
        rows.append(
            _row(
                "appointment",
                "Lịch hiện tại",
                old.label(names.get(old.department_code)),
                Origin.BACKEND,
                "passed",
                "Lịch hẹn sắp tới của bạn, đọc từ hệ thống.",
            )
        )
    if p.slot:
        rows.append(
            _row(
                "slot",
                "Lịch mới" if p.kind == "reschedule" else "Lịch khám",
                p.slot.label(names.get(p.slot.department_code)),
                Origin.BACKEND,
                "passed",
                "Slot còn trống do hệ thống trả về; sẽ được kiểm tra lại ngay trước khi ghi.",
            )
        )
    if p.basis:
        rows.append(
            _row(
                "basis",
                "Căn cứ",
                p.basis,
                Origin.AI_SUMMARY,
                "uncertain",
                "Trợ lý suy ra từ lời bạn và dữ liệu lịch hẹn — bạn kiểm tra lại trước khi xác nhận.",
            )
        )
    if p.kind == "booking" and p.reason:
        said = normalize(p.reason) in normalize(" ".join(_user_texts(state.agent_history)))
        rows.append(
            _row(
                "reason",
                "Lý do khám",
                p.reason,
                Origin.USER_SAID if said else Origin.BACKEND,
                "passed" if said else "uncertain",
                "Lý do bạn nêu."
                if said
                else "Bạn không nêu lý do; trợ lý lấy theo lịch hẹn trước — sửa nếu không đúng.",
                evidence=p.reason if said else None,
            )
        )
    if p.kind == "cancel" and p.appointment:
        question = f"Bạn xác nhận HỦY lịch: {p.appointment.slot.label(names.get(p.appointment.slot.department_code))}?"
    elif p.kind == "reschedule" and p.slot and p.appointment:
        question = (
            f"Bạn xác nhận dời lịch {p.appointment.slot.label()} sang "
            f"{p.slot.label(names.get(p.slot.department_code))}?"
        )
    else:
        assert p.slot is not None
        question = f"Bạn xác nhận đặt lịch: {p.slot.label(names.get(p.slot.department_code))}?"
    return f"{question} {tail}", rows, "confirm_booking"


def _row(
    field: str,
    label: str,
    value: Any,
    origin: Origin,
    validation: Literal["passed", "failed", "missing", "uncertain"],
    why: str,
    evidence: str | None = None,
) -> FieldExplanation:
    return FieldExplanation(
        field=field, label=label, value=value, evidence=evidence, origin=origin, validation=validation, explanation=why
    )
