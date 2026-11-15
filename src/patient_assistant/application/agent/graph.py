from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from patient_assistant.application.agent import routing
from patient_assistant.application.agent.dependencies import AgentDependencies
from patient_assistant.application.agent.flows.agent_flow import build_agent_flow
from patient_assistant.application.agent.flows.booking_flow import build_booking_flow
from patient_assistant.application.agent.flows.report_flow import build_report_flow
from patient_assistant.application.agent.nodes.common import CommonNodes
from patient_assistant.application.agent.state import AgentGraph, AgentState
from patient_assistant.domain.conversation import Intent


def build_agent_graph(dependencies: AgentDependencies, checkpointer: BaseCheckpointSaver[Any]) -> AgentGraph:
    """Graph chính — xem docs/diagrams.html §2. Checkpointer bắt buộc vì graph dùng interrupt()."""
    common = CommonNodes(dependencies)
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("ingest", common.ingest)
    graph_builder.add_node("guard_input", common.guard_input)
    graph_builder.add_node("safety_response", common.safety_response)
    graph_builder.add_node("classify", common.classify)
    graph_builder.add_node("clarify_intent", common.clarify_intent)
    graph_builder.add_node("report_flow", build_report_flow(dependencies))
    graph_builder.add_node("booking_flow", build_booking_flow(dependencies))
    graph_builder.add_node("agent_flow", build_agent_flow(dependencies))
    graph_builder.add_node("fetch_history", common.fetch_history)
    graph_builder.add_node("fetch_appointments", common.fetch_appointments)
    graph_builder.add_node("fetch_patient_info", common.fetch_patient_info)
    graph_builder.add_node("fetch_doctors", common.fetch_doctors)
    graph_builder.add_node("rag_answer", common.rag_answer)
    graph_builder.add_node("converse", common.converse)
    graph_builder.add_node("respond", common.respond)

    graph_builder.add_edge(START, "ingest")
    graph_builder.add_edge("ingest", "guard_input")
    # full: mọi câu (đã qua chặn an toàn) đi thẳng vào agent chung — không phân loại, không luồng cứng.
    first = "agent_flow" if dependencies.agent_mode == "full" else "classify"

    def _after_guard(state: AgentState) -> str:
        return "safety_response" if routing.after_guard(state) == "safety_response" else first

    graph_builder.add_conditional_edges("guard_input", _after_guard, ["safety_response", first])
    graph_builder.add_edge("safety_response", "respond")

    def _after_classify(state: AgentState) -> str:
        if state.outcome:
            return "respond"
        if dependencies.agent_mode == "booking" and state.intent == Intent.BOOK_APPOINTMENT:
            return "agent_flow"
        return routing.after_classify(state)

    graph_builder.add_conditional_edges(
        "classify",
        _after_classify,
        [
            "report_flow",
            "booking_flow",
            "agent_flow",
            "fetch_appointments",
            "fetch_patient_info",
            "fetch_doctors",
            "fetch_history",
            "rag_answer",
            "converse",
            "clarify_intent",
            "respond",
        ],
    )
    graph_builder.add_conditional_edges("clarify_intent", routing.after_clarify_intent, ["classify", "respond"])
    for node in (
        "report_flow",
        "booking_flow",
        "agent_flow",
        "fetch_history",
        "fetch_appointments",
        "fetch_patient_info",
        "fetch_doctors",
        "rag_answer",
        "converse",
    ):
        graph_builder.add_edge(node, "respond")
    graph_builder.add_edge("respond", END)
    return graph_builder.compile(checkpointer=checkpointer)
