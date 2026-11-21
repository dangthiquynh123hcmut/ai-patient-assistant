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

from patient_assistant.application.agent.dependencies import AgentDependencies
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
    AttemptRecord,
    FieldExplanation,
    InterruptPayload,
    Outcome,
    OutcomeKind,
)
from patient_assistant.application.ports.llm import AgentMessage, LLMError, ToolSpecification
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
from patient_assistant.domain.common import Origin, Resolution, ValidationIssue
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


def _local(date_time: datetime, ref: datetime) -> datetime:
    return date_time.astimezone(ref.tzinfo)


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
        *[
            f"- {department.code}: {department.name}"
            + (f" — {', '.join(department.aliases)}" if department.aliases else "")
            for department in departments
        ],
        "",
        "Doctors (id: name — department):",
        *[f"- {doctor.id}: {doctor.name} — {doctor.department_code}" for doctor in doctors],
    ]
    if state.stt_flags:
        # Cờ STT chỉ được validator kiểm khi agent gọi propose_report; báo trước để agent hỏi lại đúng chỗ nghi ngờ.
        lines += [
            "",
            "The user's message is a voice transcript. These words were recognised with LOW confidence "
            "and may be wrong:",
            *[f'- "{stt_flag.text}" (confidence {stt_flag.probability:.2f})' for stt_flag in state.stt_flags],
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
    tool: str, message: str, details: list[str] | None = None, code: ToolErrorCode = ToolErrorCode.INVALID_ARGUMENTS
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
    return [message.text for message in history if message.role == "user" and not message.text.startswith(SYSTEM_NOTE)]


class _Run:
    """Tool cho một lần chạy agent; ghi lại slot đã thấy, đề xuất, nguồn tài liệu và mọi lần gọi tool."""

    def __init__(
        self,
        dependencies: AgentDependencies,
        state: AgentState,
        departments: list[Department],
        doctors: list[Doctor],
        history: list[AgentMessage],
    ):
        self.dependencies = dependencies
        self.state = state
        self.departments = {department.code for department in departments}
        self.doctors = {doctor.id: doctor for doctor in doctors}
        self.user_texts = _user_texts(history)
        self.slots: dict[str, Slot] = {slot.slot_id: slot for slot in state.agent_slots}
        self.new_slots: list[Slot] = []
        self.proposal: AgentProposal | None = None
        self.proposal_tool: str | None = None
        self.knowledge_ids: set[str] = set()
        self.calls: list[ToolCallRecord] = []
        # Số lần `propose_report` bị validator từ chối vì lỗi LLM tự sửa được (không tính lỗi phải hỏi người dùng).
        self.rejections = state.correction_count
        self.handoff = False
        # Mỗi lần validator chạy trên đề xuất của LLM -> hiện ở tab "Kiểm tra & sửa lỗi" như ở luồng workflow.
        self.attempts: list[AttemptRecord] = []
        self.issues: list[ValidationIssue] = []

    async def _record_attempt(
        self, flow: Literal["report"], output: dict[str, Any], issues: list[ValidationIssue]
    ) -> None:
        attempt = AttemptRecord(
            flow=flow,
            attempt_no=sum(a.flow == flow for a in [*self.state.attempts, *self.attempts]),
            output=output,
            issue_codes=[i.code for i in issues],
            llm_fixable_codes=[i.code for i in issues if i.resolution == Resolution.LLM_FIXABLE],
            passed=not issues,
        )
        self.attempts.append(attempt)
        self.issues = issues
        await self.dependencies.audit.record("attempt", self.state.conversation_id, attempt.model_dump(mode="json"))

    # ------------------------------------------------------------ đọc

    async def find_appointments(self, input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, FindAppointmentsInput)
        result = await self.dependencies.tools.find_my_appointments(
            self.state.patient_id, self.state.now, input_data.when, input_data.department_code
        )
        data = result.model_dump(mode="json")
        for item in data.get("data") or []:
            item["weekday"] = WEEKDAY_NAMES[_local(datetime.fromisoformat(item["start_at"]), self.state.now).weekday()]
        return _record(self.calls, data)

    async def search_slots(self, input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, SearchSlotsInput)
        tool = "get_available_slots"
        if input_data.department_code not in self.departments:
            return _record(
                self.calls, _failure(tool, f"Department code {input_data.department_code} is not in the catalog.")
            )
        doctor = self.doctors.get(input_data.doctor_id) if input_data.doctor_id else None
        if input_data.doctor_id and (doctor is None or doctor.department_code != input_data.department_code):
            return _record(
                self.calls,
                _failure(
                    tool, f"Doctor {input_data.doctor_id} does not belong to department {input_data.department_code}."
                ),
            )
        if input_data.date_to < input_data.date_from:
            return _record(self.calls, _failure(tool, "date_to must be on or after date_from."))
        window = DateWindow(
            date_from=input_data.date_from,
            date_to=input_data.date_to,
            time_from=input_data.time_from,
            time_to=input_data.time_to,
        )
        issues = window_issues(window, self.state.now, self.dependencies.tools.booking_policy)
        if issues:
            return _record(self.calls, _failure(tool, "This date range cannot be booked.", [i.message for i in issues]))
        result = await self.dependencies.tools.get_available_slots(
            self.state.patient_id, input_data.department_code, window, input_data.doctor_id, self.state.now
        )
        data = result.model_dump(mode="json")
        if result.ok:
            found = [Slot.model_validate(slot_data) for slot_data in result.data]
            for slot in found:
                if slot.slot_id not in self.slots:
                    self.new_slots.append(slot)
                self.slots[slot.slot_id] = slot
            data["data"] = [
                {
                    **_slot_view(found_slot),
                    **(
                        {"same_day_appointment": raw["same_day_appointment"]} if raw.get("same_day_appointment") else {}
                    ),
                }
                for found_slot, raw in zip(found, result.data, strict=True)
            ]
        return _record(self.calls, data)

    async def patient_plan(self, _: BaseModel) -> dict[str, Any]:
        return _record(
            self.calls, (await self.dependencies.tools.get_patient_plan(self.state.patient_id)).model_dump(mode="json")
        )

    async def patient_summary(self, _: BaseModel) -> dict[str, Any]:
        result = (await self.dependencies.tools.get_patient_summary(self.state.patient_id, self.state.now)).model_dump(
            mode="json"
        )
        # LLM không cần ngày sinh / số điện thoại để trả lời -> không gửi dữ liệu cá nhân này cho LLM.
        for key in ("date_of_birth", "phone"):
            ((result.get("data") or {}).get("profile") or {}).pop(key, None)
        return _record(self.calls, result)

    async def report_history(self, _: BaseModel) -> dict[str, Any]:
        return _record(
            self.calls,
            (await self.dependencies.tools.get_report_history(self.state.patient_id)).model_dump(mode="json"),
        )

    async def search_knowledge(self, input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, SearchKnowledgeInput)
        result = await self.dependencies.tools.search_approved_knowledge(input_data.query)
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
        back = await self.dependencies.tools.read_back_appointment(self.state.patient_id, appointment_id)
        if not back.ok:
            return _record(
                self.calls,
                _failure(
                    tool,
                    "The patient has no such appointment. Take the id from find_my_appointments.",
                    code=ToolErrorCode.NOT_FOUND,
                ),
            )
        appointment = Appointment.model_validate(back.data)
        if appointment.status != AppointmentStatus.BOOKED or appointment.slot.start_at < self.state.now:
            return _record(
                self.calls,
                _failure(
                    tool, "This appointment is already cancelled or in the past; it cannot be changed or cancelled."
                ),
            )
        return appointment

    def _known_slot(self, tool: str, slot_id: str) -> Slot | dict[str, Any]:
        slot = self.slots.get(slot_id)
        if slot is None:
            return _record(
                self.calls, _failure(tool, "This slot_id is not in the search_slots results. Search for slots first.")
            )
        return slot

    async def propose_booking(self, input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, ProposeBookingInput)
        slot = self._known_slot("propose_booking", input_data.slot_id)
        if isinstance(slot, dict):
            return slot
        proposal = AgentProposal(kind="booking", slot=slot, reason=input_data.reason, basis=input_data.basis)
        return self._proposed("propose_booking", proposal, [slot.slot_id])

    async def propose_reschedule(self, input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, ProposeRescheduleInput)
        tool = "propose_reschedule"
        appointment = await self._upcoming_appointment(tool, input_data.appointment_id)
        if isinstance(appointment, dict):
            return appointment
        slot = self._known_slot(tool, input_data.slot_id)
        if isinstance(slot, dict):
            return slot
        if slot.slot_id == appointment.slot.slot_id:
            return _record(self.calls, _failure(tool, "The new slot is the same as the current appointment."))
        proposal = AgentProposal(
            kind="reschedule", slot=slot, appointment=appointment, reason=appointment.reason, basis=input_data.basis
        )
        return self._proposed(tool, proposal, [appointment.id, slot.slot_id])

    async def propose_cancel(self, input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, ProposeCancelInput)
        appointment = await self._upcoming_appointment("propose_cancel", input_data.appointment_id)
        if isinstance(appointment, dict):
            return appointment
        return self._proposed(
            "propose_cancel",
            AgentProposal(kind="cancel", appointment=appointment, basis=input_data.basis),
            [appointment.id],
        )

    async def propose_report(self, input_data: BaseModel) -> dict[str, Any]:
        assert isinstance(input_data, ReportExtraction)
        tool = "propose_report"
        if input_data.reporting_period == f"{self.state.now:%G-W%V}":
            # "Tuần này" -> LLM hay tự điền tuần hiện tại mà không có trích dẫn, bị từ chối rồi phải thử lại (lần thử
            # lại dễ sinh trích dẫn hỏng ký tự). Tuần hiện tại cũng là giá trị mặc định khi lưu -> bỏ đi.
            evidence = [
                evidence_item for evidence_item in input_data.evidence if evidence_item.field != "reporting_period"
            ]
            input_data = input_data.model_copy(update={"reporting_period": None, "evidence": evidence})
        issues = validate_report(
            input_data,
            ReportValidationContext(sources=self.user_texts, now=self.state.now, stt_flags=self.state.stt_flags),
        )
        await self._record_attempt("report", input_data.model_dump(mode="json"), issues)
        if issues:
            details = [f"{i.field}: {i.message}" for i in issues]
            if any(i.resolution == Resolution.ASK_USER for i in issues):
                hint = "Ask the user for what is missing/unclear; do not fill it in yourself."
                return _record(self.calls, _failure(tool, f"The report is not valid yet. {hint}", details))
            self.rejections += 1
            if self.rejections > self.dependencies.limits.max_corrections:
                # Cùng trần với workflow (FR-02.5): sửa quá số lần cho phép -> dừng, chuyển nhân viên, không lưu.
                self.handoff = True
                failure = _failure(tool, "Too many invalid attempts; the request is handed to staff.", details)
                return _record(self.calls, failure | {"stop": True})
            hint = "Fix it to match the user's words (verbatim quotes, values exactly as they said)."
            return _record(self.calls, _failure(tool, f"The report is not valid yet. {hint}", details))
        return self._proposed(tool, AgentProposal(kind="report", report=input_data), [])

    def specifications(self) -> list[ToolSpecification]:
        full = self.dependencies.agent_mode == "full"
        specifications = [
            ToolSpecification(
                "find_my_appointments",
                "The logged-in patient's booked appointments (id, department, doctor, time, reason). Use when the "
                "user asks about appointments, refers to a previous/current appointment or visit, or wants to change "
                "or cancel one.",
                FindAppointmentsInput,
                self.find_appointments,
            ),
            ToolSpecification(
                "search_slots",
                "Find free appointment slots by department, doctor (optional), date range and time window. Slots that "
                "clash with the patient's own appointments are already excluded.",
                SearchSlotsInput,
                self.search_slots,
            ),
            ToolSpecification(
                "get_patient_plan",
                "The patient's current follow-up plan (includes the department following them).",
                EmptyInput,
                self.patient_plan,
            ),
            ToolSpecification(
                "propose_booking",
                "Propose booking ONE new slot. The system shows a card for the user to confirm; nothing is written "
                "until they confirm.",
                ProposeBookingInput,
                self.propose_booking,
                terminal=True,
            ),
            ToolSpecification(
                "propose_reschedule",
                "Propose moving ONE upcoming appointment to a new slot (the old one is cancelled at the same time). "
                "The system shows a confirmation card.",
                ProposeRescheduleInput,
                self.propose_reschedule,
                terminal=True,
            ),
            ToolSpecification(
                "propose_cancel",
                "Propose cancelling ONE upcoming appointment. The system shows a confirmation card.",
                ProposeCancelInput,
                self.propose_cancel,
                terminal=True,
            ),
        ]
        if full:
            specifications += [
                ToolSpecification(
                    "get_patient_summary",
                    "The logged-in patient's record, follow-up plan, recent reports and upcoming appointments.",
                    EmptyInput,
                    self.patient_summary,
                ),
                ToolSpecification(
                    "get_report_history",
                    "The patient's recent follow-up reports (period, pain level, exercise days, symptoms).",
                    EmptyInput,
                    self.report_history,
                ),
                ToolSpecification(
                    "search_approved_knowledge",
                    "Search the approved guidance documents. Knowledge questions may be answered only from these "
                    "results, citing the doc_id.",
                    SearchKnowledgeInput,
                    self.search_knowledge,
                ),
                ToolSpecification(
                    "propose_report",
                    "Propose saving the weekly follow-up report. Every field with a value needs evidence: a VERBATIM "
                    "quote of the user's words. The system validates it, then shows a card for the user to confirm.",
                    ReportExtraction,
                    self.propose_report,
                    terminal=True,
                ),
            ]
        return specifications


# ---------------------------------------------------------------- node


class AgentNodes:
    def __init__(self, dependencies: AgentDependencies) -> None:
        self.dependencies = dependencies

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
        departments, doctors = await self.dependencies.tools.catalog()
        run = _Run(self.dependencies, state, departments, doctors, history)
        updates: dict[str, Any] = {"notice": None, "agent_input": None, "agent_note": None, "agent_proposal": None}
        try:
            reply = await self.dependencies.llm.run_agent(
                "assistant" if self.dependencies.agent_mode == "full" else "booking",
                _context(state, departments, doctors),
                history,
                run.specifications(),
                self.dependencies.limits.max_agent_tool_calls,
            )
        except LLMError as exception:
            return updates | {
                "tool_calls": [*state.tool_calls, *run.calls],
                "agent_run_tools": [call.tool for call in run.calls],
                "agent_history": history,
                "agent_active": False,
                "outcome": state.agent_last_write
                or Outcome(kind=OutcomeKind.ERROR, code=f"llm_{exception.code}", detail=str(exception)),
            }
        updates |= {
            "tool_calls": [*state.tool_calls, *run.calls],
            "agent_run_tools": [call.tool for call in run.calls],
            "agent_slots": [*state.agent_slots, *run.new_slots][-MAX_KNOWN_SLOTS:],
            "correction_count": run.rejections,
        }
        if run.attempts:
            updates |= {"attempts": [*state.attempts, *run.attempts], "issues": run.issues}
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
        cited = sorted(knowledge_id for knowledge_id in run.knowledge_ids if knowledge_id in text)
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
        counters = next_ask(state, f"agent_confirm:{key}", self.dependencies)
        if counters is None:
            return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds")}
        departments, _ = await self.dependencies.tools.catalog()
        names = {department.code: department.name for department in departments}
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
        appointment = proposal.appointment
        if proposal.kind == "cancel":
            result = await self.dependencies.tools.cancel_appointment(state.patient_id, appointment.id, state.now)
        else:
            assert proposal.slot is not None
            key = idempotency_key(
                "reschedule", state.conversation_id, state.patient_id, appointment.id, proposal.slot.slot_id
            )
            result = await self.dependencies.tools.reschedule_appointment(
                state.patient_id, appointment.id, proposal.slot.slot_id, None, key, state.now
            )
        return {"tool_calls": tool_record(state, result), "write_result": result.model_dump(mode="json")}

    async def verify_change(self, state: AgentState) -> dict[str, Any]:
        proposal = state.agent_proposal
        assert proposal is not None and proposal.appointment is not None
        written = state.write_result or {}
        error = written.get("error") or {}
        if written.get("status") != "ok":
            if error.get("code") == ToolErrorCode.CONFLICT.value and proposal.kind == "reschedule":
                if state.conflict_retries >= self.dependencies.limits.max_conflict_retries:
                    return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="conflict")}
                reason = error.get("message") or "The chosen slot can no longer be booked."
                return {
                    "conflict_retries": state.conflict_retries + 1,
                    "notice": f"{reason} The old appointment is unchanged.",
                }
            result = ToolResult.model_validate(written)
            return {"outcome": tool_error_outcome(result, "change_failed")}
        old = await self.dependencies.tools.read_back_appointment(state.patient_id, proposal.appointment.id)
        tool_calls = tool_record(state, old)
        if not old.ok or old.data["status"] != AppointmentStatus.CANCELLED.value:
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ERROR, code="verify_failed")}
        if proposal.kind == "cancel":
            return {
                "tool_calls": tool_calls,
                "outcome": Outcome(kind=OutcomeKind.APPOINTMENT_CANCELLED, data=old.data, source_ids=[old.data["id"]]),
            }
        assert proposal.slot is not None
        new = await self.dependencies.tools.read_back_appointment(
            state.patient_id, (written.get("data") or {}).get("id", "")
        )
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
        if self.dependencies.agent_mode == "full" and len(state.agent_done) + 1 < MAX_WRITES_PER_TURN:
            updates |= {
                "outcome": None,
                "agent_note": "Already completed in this turn (do NOT redo these):\n"
                + "\n".join(f"{i}. {done_item}" for i, done_item in enumerate(all_done, start=1))
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


def _written_keys(proposal: AgentProposal) -> list[str]:
    """Khóa nhận diện việc ghi, để chặn đề xuất lại đúng việc đã làm trong lượt."""
    keys = ["report"] if proposal.kind == "report" else []
    if proposal.slot is not None:
        keys.append(f"slot:{proposal.slot.slot_id}")
    if proposal.appointment is not None:
        keys.append(f"appt:{proposal.appointment.id}")
    return keys


def _proposal_title(proposal: AgentProposal) -> str:
    if proposal.kind == "booking" and proposal.slot:
        return f"book {proposal.slot.label()}"
    if proposal.kind == "reschedule" and proposal.slot and proposal.appointment:
        return f"move {proposal.appointment.slot.label()} to {proposal.slot.label()}"
    if proposal.kind == "cancel" and proposal.appointment:
        return f"cancel {proposal.appointment.slot.label()}"
    return "save the follow-up report"


def _card(
    proposal: AgentProposal, names: dict[str, str], state: AgentState
) -> tuple[str, list[FieldExplanation], Literal["confirm_booking", "confirm_report"]]:
    tail = "Bấm “xác nhận” để làm, “hủy” để dừng, hoặc nhắn điều bạn muốn đổi."
    if proposal.kind == "report":
        assert proposal.report is not None
        report_rows = report_explanations(proposal.report, [], state.edited_fields, state.now)
        return f"Bạn xác nhận lưu báo cáo theo dõi này? {tail}", report_rows, "confirm_report"
    rows: list[FieldExplanation] = []
    if proposal.appointment:
        old = proposal.appointment.slot
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
    if proposal.slot:
        rows.append(
            _row(
                "slot",
                "Lịch mới" if proposal.kind == "reschedule" else "Lịch khám",
                proposal.slot.label(names.get(proposal.slot.department_code)),
                Origin.BACKEND,
                "passed",
                "Slot còn trống do hệ thống trả về; sẽ được kiểm tra lại ngay trước khi ghi.",
            )
        )
    if proposal.basis:
        rows.append(
            _row(
                "basis",
                "Căn cứ",
                proposal.basis,
                Origin.AI_SUMMARY,
                "uncertain",
                "Trợ lý suy ra từ lời bạn và dữ liệu lịch hẹn — bạn kiểm tra lại trước khi xác nhận.",
            )
        )
    if proposal.kind == "booking" and proposal.reason:
        said = normalize(proposal.reason) in normalize(" ".join(_user_texts(state.agent_history)))
        rows.append(
            _row(
                "reason",
                "Lý do khám",
                proposal.reason,
                Origin.USER_SAID if said else Origin.BACKEND,
                "passed" if said else "uncertain",
                "Lý do bạn nêu."
                if said
                else "Bạn không nêu lý do; trợ lý lấy theo lịch hẹn trước — sửa nếu không đúng.",
                evidence=proposal.reason if said else None,
            )
        )
    if proposal.kind == "cancel" and proposal.appointment:
        slot = proposal.appointment.slot
        question = f"Bạn xác nhận HỦY lịch: {slot.label(names.get(slot.department_code))}?"
    elif proposal.kind == "reschedule" and proposal.slot and proposal.appointment:
        question = (
            f"Bạn xác nhận dời lịch {proposal.appointment.slot.label()} sang "
            f"{proposal.slot.label(names.get(proposal.slot.department_code))}?"
        )
    else:
        assert proposal.slot is not None
        question = f"Bạn xác nhận đặt lịch: {proposal.slot.label(names.get(proposal.slot.department_code))}?"
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
