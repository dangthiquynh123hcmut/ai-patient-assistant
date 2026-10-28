"""Hàm routing thuần: chỉ đọc state, không side effect — unit test trực tiếp được."""

from __future__ import annotations

from typing import Literal

from patient_assistant.application.agent.deps import AgentLimits
from patient_assistant.application.agent.state import AgentState
from patient_assistant.domain.common import Resolution
from patient_assistant.domain.conversation import ConfirmIntent, Intent, SafetyKind, interpret_confirmation

END: Literal["end"] = "end"


def after_guard(state: AgentState) -> Literal["safety_response", "classify"]:
    return "safety_response" if state.safety != SafetyKind.OK else "classify"


def after_classify(state: AgentState) -> str:
    return {
        Intent.CREATE_REPORT: "report_flow",
        Intent.BOOK_APPOINTMENT: "booking_flow",
        Intent.LIST_APPOINTMENTS: "fetch_appointments",
        Intent.GET_REPORT_HISTORY: "fetch_history",
        Intent.ANSWER_FAQ: "rag_answer",
    }.get(state.intent or Intent.UNKNOWN, "clarify_intent")


def after_clarify_intent(state: AgentState) -> Literal["classify", "respond"]:
    return "respond" if state.outcome else "classify"


def after_extract(state: AgentState) -> Literal["validate", "end"]:
    return END if state.outcome else "validate"


def after_write(state: AgentState) -> Literal["verify", "end"]:
    """Tool ghi xong luôn đi qua bước đọc lại, trừ khi đã có lỗi chặn trước đó."""
    return END if state.outcome else "verify"


def after_validate(state: AgentState, limits: AgentLimits) -> Literal["accept", "correct", "clarify", "handoff"]:
    if not state.issues:
        return "accept"
    if any(i.resolution == Resolution.BLOCK for i in state.issues):
        return "handoff"
    if any(i.resolution == Resolution.ASK_USER for i in state.issues):
        return "clarify"
    if state.correction_count < limits.max_corrections:
        return "correct"
    return "handoff"


def after_clarify(state: AgentState) -> Literal["extract", "end"]:
    return END if state.outcome else "extract"


def confirmation_of(state: AgentState) -> ConfirmIntent | Literal["edit"]:
    """Quy quyết định của người dùng về confirm/cancel/edit/unclear. Lời nói mơ hồ = chưa xác nhận."""
    d = state.decision
    if d is None:
        return ConfirmIntent.UNCLEAR
    if d.action == "confirm":
        return ConfirmIntent.CONFIRM
    if d.action == "cancel":
        return ConfirmIntent.CANCEL
    if d.action == "edit":
        return "edit"
    if d.action == "reply" and d.text:
        return interpret_confirmation(d.text)
    return ConfirmIntent.UNCLEAR


def after_present_report(state: AgentState) -> Literal["save", "apply_edits", "present_draft", "end"]:
    if state.outcome:
        return END
    decision = confirmation_of(state)
    if decision == ConfirmIntent.CONFIRM:
        return "save"
    if decision == "edit":
        return "apply_edits"
    if decision == ConfirmIntent.CANCEL:
        return END
    return "present_draft"


# ---------------------------------------------------------------- booking


def after_search(state: AgentState) -> Literal["present_slots", "alternatives", "end"]:
    if state.outcome:
        return END
    return "present_slots" if state.offered_slots else "alternatives"


def after_alternatives(state: AgentState) -> Literal["present_slots", "end"]:
    return END if state.outcome else "present_slots"


def after_present_slots(state: AgentState) -> Literal["match_selection", "extract", "end"]:
    if state.outcome:
        return END
    if state.decision and state.decision.action == "other_time":
        return "extract"
    return "match_selection"


def after_match(state: AgentState) -> Literal["present_booking", "present_slots", "end"]:
    if state.outcome:
        return END
    return "present_booking" if state.selected_slot_id else "present_slots"


def after_present_booking(state: AgentState) -> Literal["create_booking", "extract", "present_booking", "end"]:
    if state.outcome:
        return END
    decision = confirmation_of(state)
    if decision == ConfirmIntent.CONFIRM:
        return "create_booking"
    if decision == "edit":
        return "extract"
    if decision == ConfirmIntent.CANCEL:
        return END
    return "present_booking"


def after_verify_booking(state: AgentState) -> Literal["search_slots", "end"]:
    return END if state.outcome else "search_slots"
