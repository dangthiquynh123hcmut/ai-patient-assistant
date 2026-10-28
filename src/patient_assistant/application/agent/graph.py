from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from patient_assistant.application.agent import routing
from patient_assistant.application.agent.deps import AgentDeps
from patient_assistant.application.agent.flows.booking_flow import build_booking_flow
from patient_assistant.application.agent.flows.report_flow import build_report_flow
from patient_assistant.application.agent.nodes.common import CommonNodes
from patient_assistant.application.agent.state import AgentGraph, AgentState


def build_agent_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any]) -> AgentGraph:
    """Graph chính — xem docs/diagrams.html §2. Checkpointer bắt buộc vì graph dùng interrupt()."""
    common = CommonNodes(deps)
    g = StateGraph(AgentState)
    g.add_node("ingest", common.ingest)
    g.add_node("guard_input", common.guard_input)
    g.add_node("safety_response", common.safety_response)
    g.add_node("classify", common.classify)
    g.add_node("clarify_intent", common.clarify_intent)
    g.add_node("report_flow", build_report_flow(deps))
    g.add_node("booking_flow", build_booking_flow(deps))
    g.add_node("fetch_history", common.fetch_history)
    g.add_node("fetch_appointments", common.fetch_appointments)
    g.add_node("rag_answer", common.rag_answer)
    g.add_node("respond", common.respond)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "guard_input")
    g.add_conditional_edges("guard_input", routing.after_guard, ["safety_response", "classify"])
    g.add_edge("safety_response", "respond")

    def _after_classify(state: AgentState) -> str:
        return "respond" if state.outcome else routing.after_classify(state)

    g.add_conditional_edges(
        "classify",
        _after_classify,
        [
            "report_flow",
            "booking_flow",
            "fetch_appointments",
            "fetch_history",
            "rag_answer",
            "clarify_intent",
            "respond",
        ],
    )
    g.add_conditional_edges("clarify_intent", routing.after_clarify_intent, ["classify", "respond"])
    for node in ("report_flow", "booking_flow", "fetch_history", "fetch_appointments", "rag_answer"):
        g.add_edge(node, "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)
