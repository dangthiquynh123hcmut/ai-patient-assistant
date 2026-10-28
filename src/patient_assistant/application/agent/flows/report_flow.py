from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from patient_assistant.application.agent import routing
from patient_assistant.application.agent.deps import AgentDeps
from patient_assistant.application.agent.nodes.report import ReportNodes
from patient_assistant.application.agent.nodes.shared import SharedNodes
from patient_assistant.application.agent.state import AgentGraph, AgentState


def build_report_flow(deps: AgentDeps) -> AgentGraph:
    """Subgraph report_flow — xem docs/diagrams.html §3."""
    nodes, shared = ReportNodes(deps), SharedNodes(deps)
    g = StateGraph(AgentState)
    g.add_node("extract", nodes.extract)
    g.add_node("validate", nodes.validate)
    g.add_node("correct", nodes.correct)
    g.add_node("clarify", shared.clarify)
    g.add_node("handoff", shared.handoff)
    g.add_node("present_draft", nodes.present_draft)
    g.add_node("apply_edits", nodes.apply_edits)
    g.add_node("save", nodes.save)
    g.add_node("verify_save", nodes.verify_save)

    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", routing.after_extract, {"validate": "validate", "end": END})
    g.add_conditional_edges("correct", routing.after_extract, {"validate": "validate", "end": END})
    g.add_conditional_edges(
        "validate",
        partial(routing.after_validate, limits=deps.limits),
        {"accept": "present_draft", "correct": "correct", "clarify": "clarify", "handoff": "handoff"},
    )
    g.add_conditional_edges("clarify", routing.after_clarify, {"extract": "extract", "end": END})
    g.add_conditional_edges(
        "present_draft",
        routing.after_present_report,
        {"save": "save", "apply_edits": "apply_edits", "present_draft": "present_draft", "end": END},
    )
    g.add_edge("apply_edits", "validate")
    g.add_conditional_edges("save", routing.after_write, {"verify": "verify_save", "end": END})
    g.add_edge("verify_save", END)
    g.add_edge("handoff", END)
    return g.compile()
