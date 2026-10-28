"""Node của booking_flow: trích xuất -> quy đổi ngày giờ -> validate -> tìm slot -> chọn -> xác nhận -> đặt."""

from __future__ import annotations

from typing import Any

from patient_assistant.application.agent.deps import AgentDeps
from patient_assistant.application.agent.nodes.shared import (
    ask_user,
    idempotency_key,
    is_cancel,
    record_attempt,
    tool_error_outcome,
    tool_record,
)
from patient_assistant.application.agent.presenters import (
    booking_confirm_message,
    booking_explanations,
    slot_options_message,
)
from patient_assistant.application.agent.routing import confirmation_of
from patient_assistant.application.agent.state import AgentState, InterruptPayload, Outcome, OutcomeKind
from patient_assistant.application.ports.llm import BookingPrompt, CorrectionFeedback, LLMError
from patient_assistant.application.tools.result import ToolErrorCode
from patient_assistant.domain.booking.models import Appointment, Slot, SlotOption
from patient_assistant.domain.booking.policy import BookingPolicy
from patient_assistant.domain.booking.selection import match_selection
from patient_assistant.domain.booking.validators import (
    BookingValidationContext,
    find_department,
    find_doctor,
    parse_booking,
    validate_booking,
)
from patient_assistant.domain.common import Resolution
from patient_assistant.domain.conversation import ConfirmIntent
from patient_assistant.domain.text.normalize import contains_phrase
from patient_assistant.domain.text.vi_dates import DateResolution, resolve

_EDIT = ("đổi", "sửa", "chuyển sang", "thay")
_OTHER_TIME = ("giờ khác", "ngày khác", "lịch khác", "đổi ngày", "đổi giờ", "không được", "không hợp")


class BookingNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    @property
    def _policy(self) -> BookingPolicy:
        return self.deps.tools.booking_policy

    async def _department_name(self, code: str | None) -> str | None:
        departments, _ = await self.deps.tools.catalog()
        dept = find_department(code, departments)
        return dept.name if dept else None

    # ------------------------------------------------------------------ trích xuất + validate

    async def _generate(self, state: AgentState, feedback: CorrectionFeedback | None) -> dict[str, Any]:
        departments, _ = await self.deps.tools.catalog()
        try:
            raw = await self.deps.llm.extract_booking(BookingPrompt(state.sources(), state.now, departments, feedback))
        except LLMError as exc:
            return {"outcome": Outcome(kind=OutcomeKind.ERROR, code=f"llm_{exc.code}")}
        request, schema_issues = parse_booking(raw)
        from_plan = False
        tool_calls = state.tool_calls
        if request is not None and request.department_code is None:
            # Khoa mặc định chỉ lấy từ kế hoạch theo dõi ở backend — code điền, không phải LLM (FR-09.11).
            plan = await self.deps.tools.get_patient_plan(state.patient_id)
            tool_calls = tool_record(state, plan)
            if plan.ok:
                request = request.model_copy(update={"department_code": plan.data["department_code"]})
                from_plan = True
        return {
            "last_output": raw,
            "booking_request": request,
            "issues": schema_issues,
            "department_from_plan": from_plan,
            "tool_calls": tool_calls,
            "offered_slots": [],
            "selected_slot_id": None,
            "relaxed_search": False,
        }

    async def extract(self, state: AgentState) -> dict[str, Any]:
        return await self._generate(state, None)

    async def correct(self, state: AgentState) -> dict[str, Any]:
        fixable = [i for i in state.issues if i.resolution == Resolution.LLM_FIXABLE]
        updates = await self._generate(state, CorrectionFeedback(state.last_output or {}, fixable))
        updates["correction_count"] = state.correction_count + 1
        return updates

    async def resolve_datetime(self, state: AgentState) -> dict[str, Any]:
        req = state.booking_request
        if req is None:
            return {"resolved_window": None, "date_issues": []}
        result = resolve(req.date_expression, req.time_expression, state.now, self._policy.day_parts)
        return {"resolved_window": result.window, "date_issues": result.issues}

    async def validate(self, state: AgentState) -> dict[str, Any]:
        req = state.booking_request
        doctor_id = None
        if req is None:
            issues = state.issues
        else:
            departments, doctors = await self.deps.tools.catalog()
            issues = validate_booking(
                req,
                DateResolution(state.resolved_window, state.date_issues),
                BookingValidationContext(
                    sources=state.sources(),
                    now=state.now,
                    policy=self._policy,
                    departments=departments,
                    doctors=doctors,
                    stt_flags=state.stt_flags,
                    department_from_plan=state.department_from_plan,
                ),
            )
            if req.doctor_name and not issues:
                doctor = find_doctor(req.doctor_name, req.department_code, doctors)
                doctor_id = doctor.id if doctor else None
        return {
            "issues": issues,
            "doctor_id": doctor_id,
            "attempts": await record_attempt(self.deps, state, "booking", state.last_output, issues),
        }

    # ------------------------------------------------------------------ slot

    async def search_slots(self, state: AgentState) -> dict[str, Any]:
        req, window = state.booking_request, state.resolved_window
        assert req is not None and req.department_code and window is not None
        if state.search_rounds >= self.deps.limits.max_slot_search_rounds:
            return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds")}
        result = await self.deps.tools.get_available_slots(req.department_code, window, state.doctor_id, state.now)
        updates: dict[str, Any] = {
            "tool_calls": tool_record(state, result),
            "search_rounds": state.search_rounds + 1,
            "selected_slot_id": None,
            "relaxed_search": False,
        }
        if not result.ok:
            updates["outcome"] = tool_error_outcome(result, "tool_unavailable")
            return updates
        updates["offered_slots"] = _numbered(result.data)
        return updates

    async def alternatives(self, state: AgentState) -> dict[str, Any]:
        req, window = state.booking_request, state.resolved_window
        assert req is not None and req.department_code and window is not None
        result = await self.deps.tools.get_alternative_slots(req.department_code, window, state.now)
        updates: dict[str, Any] = {"tool_calls": tool_record(state, result), "relaxed_search": True}
        if not result.ok:
            updates["outcome"] = tool_error_outcome(result, "tool_unavailable")
        elif not result.data:
            updates["outcome"] = Outcome(kind=OutcomeKind.HANDOFF, code="no_slots")
        else:
            updates["offered_slots"] = _numbered(result.data)
        return updates

    async def present_slots(self, state: AgentState) -> dict[str, Any]:
        if state.clarify_rounds >= self.deps.limits.max_clarify_rounds:
            return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds")}
        dept_name = await self._department_name(
            state.booking_request.department_code if state.booking_request else None
        )
        message = slot_options_message(state.offered_slots, state.relaxed_search, dept_name)
        decision = ask_user(
            InterruptPayload(
                kind="choose_slot",
                message=f"{state.notice}\n{message}" if state.notice else message,
                options=state.offered_slots,
                allowed_actions=["choose", "reply", "other_time", "cancel"],
            )
        )
        updates: dict[str, Any] = {"decision": decision, "notice": None}
        if is_cancel(decision):
            updates["outcome"] = Outcome(kind=OutcomeKind.CANCELLED)
        elif decision.action == "other_time" or (
            decision.action == "reply" and contains_phrase(decision.text or "", _OTHER_TIME)
        ):
            if state.search_rounds >= self.deps.limits.max_slot_search_rounds:
                updates["outcome"] = Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds")
            else:
                updates["decision"] = decision.model_copy(update={"action": "other_time"})
                updates["user_replies"] = [*state.user_replies, decision.text or ""]
                updates["correction_count"] = 0
        return updates

    async def match(self, state: AgentState) -> dict[str, Any]:
        """Ghép lựa chọn với danh sách đã đưa ra. Kết quả luôn phải thuộc `offered_slots` (FR-09.4, FR-09.6)."""
        decision, options = state.decision, state.offered_slots
        assert decision is not None
        offered = {o.option_no: o for o in options}
        chosen: int | None = decision.option_no if decision.option_no in offered else None
        notice = None
        if chosen is None and decision.text:
            result = match_selection(decision.text, options)
            chosen = result.option_no
            if chosen is None and result.ambiguous:
                notice = f"Có nhiều lịch khớp ({', '.join(map(str, result.candidates))}), bạn chọn giúp mình số cụ thể."
            elif chosen is None:
                try:
                    guess = await self.deps.llm.choose_slot(decision.text, options)
                except LLMError:
                    guess = None
                chosen = guess if guess in offered else None
        if chosen is None:
            return {
                "selected_slot_id": None,
                "clarify_rounds": state.clarify_rounds + 1,
                "notice": notice or "Mình chưa xác định được bạn chọn lịch nào.",
            }
        return {"selected_slot_id": offered[chosen].slot.slot_id}

    # ------------------------------------------------------------------ xác nhận + ghi

    def _selected(self, state: AgentState) -> Slot | None:
        return next((o.slot for o in state.offered_slots if o.slot.slot_id == state.selected_slot_id), None)

    async def present_booking(self, state: AgentState) -> dict[str, Any]:
        slot, req = self._selected(state), state.booking_request
        assert slot is not None and req is not None
        if state.clarify_rounds >= self.deps.limits.max_clarify_rounds:
            return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="too_many_rounds")}
        departments, _ = await self.deps.tools.catalog()
        dept = find_department(req.department_code, departments)
        message = booking_confirm_message(slot, dept.name if dept else None)
        decision = ask_user(
            InterruptPayload(
                kind="confirm_booking",
                message=f"{state.notice} {message}" if state.notice else message,
                explanations=booking_explanations(req, state.resolved_window, dept, state.department_from_plan, slot),
                allowed_actions=["confirm", "edit", "cancel", "reply"],
            )
        )
        updates: dict[str, Any] = {"decision": decision, "notice": None}
        intent = confirmation_of(state.model_copy(update={"decision": decision}))
        if intent == ConfirmIntent.UNCLEAR and contains_phrase(decision.text or "", _EDIT):
            intent = "edit"
            updates["decision"] = decision.model_copy(update={"action": "edit"})
        if intent == ConfirmIntent.CANCEL:
            updates["outcome"] = Outcome(kind=OutcomeKind.CANCELLED)
        elif intent == "edit":
            change = decision.text or " ".join(f"{v}" for v in decision.fields.values())
            updates |= {"user_replies": [*state.user_replies, change], "correction_count": 0}
        elif intent == ConfirmIntent.UNCLEAR:
            updates["clarify_rounds"] = state.clarify_rounds + 1
            updates["notice"] = "Mình chưa rõ bạn đã xác nhận hay chưa."
        return updates

    async def create_booking(self, state: AgentState) -> dict[str, Any]:
        slot = self._selected(state)
        req = state.booking_request
        # Chốt chặn: validate đạt, slot thuộc danh sách đã đưa ra, người dùng xác nhận rõ.
        if slot is None or req is None or state.issues or confirmation_of(state) != ConfirmIntent.CONFIRM:
            return {"outcome": Outcome(kind=OutcomeKind.ERROR, code="guard_violation")}
        key = idempotency_key("booking", state.thread_id, state.patient_id, slot.slot_id)
        result = await self.deps.tools.create_appointment(state.patient_id, slot.slot_id, req.reason, key, state.now)
        return {"tool_calls": tool_record(state, result), "write_result": result.model_dump(mode="json")}

    async def verify_booking(self, state: AgentState) -> dict[str, Any]:
        written = state.write_result or {}
        error = written.get("error") or {}
        if written.get("status") != "ok":
            if error.get("code") == ToolErrorCode.CONFLICT.value:
                if state.conflict_retries >= self.deps.limits.max_conflict_retries:
                    return {"outcome": Outcome(kind=OutcomeKind.HANDOFF, code="conflict")}
                return {
                    "conflict_retries": state.conflict_retries + 1,
                    "offered_slots": [],
                    "selected_slot_id": None,
                    "notice": "Slot bạn chọn vừa được người khác đặt. Đây là các lịch còn trống mới nhất.",
                }
            if error.get("code") == ToolErrorCode.UNAVAILABLE.value:
                return {"outcome": Outcome(kind=OutcomeKind.ERROR, code="tool_unavailable")}
            return {
                "outcome": Outcome(
                    kind=OutcomeKind.ERROR,
                    code="booking_failed",
                    data=error.get("details") or [error.get("message", "")],
                )
            }
        appointment_id = (written.get("data") or {}).get("id")
        back = await self.deps.tools.read_back_appointment(state.patient_id, appointment_id or "")
        tool_calls = tool_record(state, back)
        expected = self._selected(state)
        if not back.ok or expected is None:
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ERROR, code="verify_failed")}
        appt = Appointment.model_validate(back.data)
        if appt.slot.slot_id != expected.slot_id or appt.slot.start_at != expected.start_at:
            return {"tool_calls": tool_calls, "outcome": Outcome(kind=OutcomeKind.ERROR, code="verify_failed")}
        return {
            "tool_calls": tool_calls,
            "outcome": Outcome(kind=OutcomeKind.APPOINTMENT_BOOKED, data=back.data, source_ids=back.source_ids),
        }


def _numbered(slots: list[dict[str, Any]]) -> list[SlotOption]:
    return [SlotOption(option_no=i, slot=Slot.model_validate(s)) for i, s in enumerate(slots, start=1)]
