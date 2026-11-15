from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from patient_assistant.application.agent import routing
from patient_assistant.application.agent.dependencies import AgentDependencies
from patient_assistant.application.agent.nodes.report import ReportNodes
from patient_assistant.application.agent.nodes.shared import SharedNodes
from patient_assistant.application.agent.state import AgentGraph, AgentState


def build_report_flow(dependencies: AgentDependencies) -> AgentGraph:
    """Subgraph report_flow — xem docs/diagrams.html §3."""
    nodes, shared = ReportNodes(dependencies), SharedNodes(dependencies)
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("extract", nodes.extract)
    graph_builder.add_node("validate", nodes.validate)
    graph_builder.add_node("correct", nodes.correct)
    graph_builder.add_node("clarify", shared.clarify)
    graph_builder.add_node("handoff", shared.handoff)
    graph_builder.add_node("present_draft", nodes.present_draft)
    graph_builder.add_node("apply_edits", nodes.apply_edits)
    graph_builder.add_node("save", nodes.save)
    graph_builder.add_node("verify_save", nodes.verify_save)

    graph_builder.add_edge(START, "extract")
    graph_builder.add_conditional_edges("extract", routing.after_extract, {"validate": "validate", "end": END})
    graph_builder.add_conditional_edges("correct", routing.after_extract, {"validate": "validate", "end": END})
    graph_builder.add_conditional_edges(
        "validate",
        partial(routing.after_validate, limits=dependencies.limits),
        {"accept": "present_draft", "correct": "correct", "clarify": "clarify", "handoff": "handoff"},
    )
    graph_builder.add_conditional_edges("clarify", routing.after_clarify, {"extract": "extract", "end": END})
    graph_builder.add_conditional_edges(
        "present_draft",
        routing.after_present_report,
        {"save": "save", "apply_edits": "apply_edits", "present_draft": "present_draft", "end": END},
    )
    graph_builder.add_edge("apply_edits", "validate")
    graph_builder.add_conditional_edges("save", routing.after_write, {"verify": "verify_save", "end": END})
    graph_builder.add_edge("verify_save", END)
    graph_builder.add_edge("handoff", END)
    return graph_builder.compile()
