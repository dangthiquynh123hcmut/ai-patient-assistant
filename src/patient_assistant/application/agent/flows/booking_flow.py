from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from patient_assistant.application.agent import routing
from patient_assistant.application.agent.deps import AgentDeps
from patient_assistant.application.agent.nodes.booking import BookingNodes
from patient_assistant.application.agent.nodes.shared import SharedNodes
from patient_assistant.application.agent.state import AgentGraph, AgentState


def build_booking_flow(deps: AgentDeps) -> AgentGraph:
    """Subgraph booking_flow — xem docs/diagrams.html §4."""
    nodes, shared = BookingNodes(deps), SharedNodes(deps)
    g = StateGraph(AgentState)
    g.add_node("extract_booking", nodes.extract)
    g.add_node("resolve_datetime", nodes.resolve_datetime)
    g.add_node("validate_booking", nodes.validate)
    g.add_node("correct", nodes.correct)
    g.add_node("clarify", shared.clarify)
    g.add_node("handoff", shared.handoff)
    g.add_node("search_slots", nodes.search_slots)
    g.add_node("offer_alternatives", nodes.alternatives)
    g.add_node("present_slot_options", nodes.present_slots)
    g.add_node("match_selection", nodes.match)
    g.add_node("present_booking_draft", nodes.present_booking)
    g.add_node("create_booking", nodes.create_booking)
    g.add_node("verify_booking", nodes.verify_booking)

    g.add_edge(START, "extract_booking")
    g.add_conditional_edges("extract_booking", routing.after_extract, {"validate": "resolve_datetime", "end": END})
    g.add_conditional_edges("correct", routing.after_extract, {"validate": "resolve_datetime", "end": END})
    g.add_edge("resolve_datetime", "validate_booking")
    g.add_conditional_edges(
        "validate_booking",
        partial(routing.after_validate, limits=deps.limits),
        {"accept": "search_slots", "correct": "correct", "clarify": "clarify", "handoff": "handoff"},
    )
    g.add_conditional_edges("clarify", routing.after_clarify, {"extract": "extract_booking", "end": END})
    g.add_conditional_edges(
        "search_slots",
        routing.after_search,
        {"present_slots": "present_slot_options", "alternatives": "offer_alternatives", "end": END},
    )
    g.add_conditional_edges(
        "offer_alternatives", routing.after_alternatives, {"present_slots": "present_slot_options", "end": END}
    )
    g.add_conditional_edges(
        "present_slot_options",
        routing.after_present_slots,
        {"match_selection": "match_selection", "extract": "extract_booking", "end": END},
    )
    g.add_conditional_edges(
        "match_selection",
        routing.after_match,
        {"present_booking": "present_booking_draft", "present_slots": "present_slot_options", "end": END},
    )
    g.add_conditional_edges(
        "present_booking_draft",
        routing.after_present_booking,
        {
            "create_booking": "create_booking",
            "extract": "extract_booking",
            "present_booking": "present_booking_draft",
            "end": END,
        },
    )
    g.add_conditional_edges("create_booking", routing.after_write, {"verify": "verify_booking", "end": END})
    g.add_conditional_edges(
        "verify_booking", routing.after_verify_booking, {"search_slots": "search_slots", "end": END}
    )
    g.add_edge("handoff", END)
    return g.compile()
