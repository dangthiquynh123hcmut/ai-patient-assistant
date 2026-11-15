from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from patient_assistant.application.agent import routing
from patient_assistant.application.agent.dependencies import AgentDependencies
from patient_assistant.application.agent.nodes.booking import BookingNodes
from patient_assistant.application.agent.nodes.shared import SharedNodes
from patient_assistant.application.agent.state import AgentGraph, AgentState


def build_booking_flow(dependencies: AgentDependencies) -> AgentGraph:
    """Subgraph booking_flow — xem docs/diagrams.html §4."""
    nodes, shared = BookingNodes(dependencies), SharedNodes(dependencies)
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("extract_booking", nodes.extract)
    graph_builder.add_node("resolve_datetime", nodes.resolve_datetime)
    graph_builder.add_node("validate_booking", nodes.validate)
    graph_builder.add_node("correct", nodes.correct)
    graph_builder.add_node("clarify", shared.clarify)
    graph_builder.add_node("handoff", shared.handoff)
    graph_builder.add_node("search_slots", nodes.search_slots)
    graph_builder.add_node("offer_alternatives", nodes.alternatives)
    graph_builder.add_node("present_slot_options", nodes.present_slots)
    graph_builder.add_node("match_selection", nodes.match)
    graph_builder.add_node("present_booking_draft", nodes.present_booking)
    graph_builder.add_node("create_booking", nodes.create_booking)
    graph_builder.add_node("verify_booking", nodes.verify_booking)

    graph_builder.add_edge(START, "extract_booking")
    graph_builder.add_conditional_edges(
        "extract_booking", routing.after_extract, {"validate": "resolve_datetime", "end": END}
    )
    graph_builder.add_conditional_edges("correct", routing.after_extract, {"validate": "resolve_datetime", "end": END})
    graph_builder.add_edge("resolve_datetime", "validate_booking")
    graph_builder.add_conditional_edges(
        "validate_booking",
        partial(routing.after_validate, limits=dependencies.limits),
        {"accept": "search_slots", "correct": "correct", "clarify": "clarify", "handoff": "handoff"},
    )
    graph_builder.add_conditional_edges("clarify", routing.after_clarify, {"extract": "extract_booking", "end": END})
    graph_builder.add_conditional_edges(
        "search_slots",
        routing.after_search,
        {"present_slots": "present_slot_options", "alternatives": "offer_alternatives", "end": END},
    )
    graph_builder.add_conditional_edges(
        "offer_alternatives", routing.after_alternatives, {"present_slots": "present_slot_options", "end": END}
    )
    graph_builder.add_conditional_edges(
        "present_slot_options",
        routing.after_present_slots,
        {"match_selection": "match_selection", "extract": "extract_booking", "end": END},
    )
    graph_builder.add_conditional_edges(
        "match_selection",
        routing.after_match,
        {"present_booking": "present_booking_draft", "present_slots": "present_slot_options", "end": END},
    )
    graph_builder.add_conditional_edges(
        "present_booking_draft",
        routing.after_present_booking,
        {
            "create_booking": "create_booking",
            "extract": "extract_booking",
            "present_booking": "present_booking_draft",
            "end": END,
        },
    )
    graph_builder.add_conditional_edges("create_booking", routing.after_write, {"verify": "verify_booking", "end": END})
    graph_builder.add_conditional_edges(
        "verify_booking", routing.after_verify_booking, {"search_slots": "search_slots", "end": END}
    )
    graph_builder.add_edge("handoff", END)
    return graph_builder.compile()
