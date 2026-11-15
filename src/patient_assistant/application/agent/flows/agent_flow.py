from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from patient_assistant.application.agent.dependencies import AgentDependencies
from patient_assistant.application.agent.nodes.assistant_agent import WRITE_KINDS, AgentNodes
from patient_assistant.application.agent.nodes.booking import BookingNodes
from patient_assistant.application.agent.nodes.report import ReportNodes
from patient_assistant.application.agent.state import AgentGraph, AgentState


def after_agent(state: AgentState) -> Literal["confirm", "wrapup"]:
    return "confirm" if state.agent_proposal is not None and not state.outcome else "wrapup"


def after_confirm(
    state: AgentState,
) -> Literal["create_booking", "save_report", "change_appointment", "confirm", "agent", "wrapup"]:
    if state.outcome:
        return "wrapup"
    proposal = state.agent_proposal
    if proposal is None:  # người dùng nhắn điều muốn đổi -> agent xử lý
        return "agent"
    if state.decision is None or state.decision.action != "confirm":
        # Sửa báo cáo trên form -> hiện lại thẻ đã sửa; câu trả lời chưa rõ ("ờ") -> hiện lại thẻ. Chỉ ghi khi
        # node xác nhận đã quy về "confirm".
        return "confirm"
    if proposal.kind == "booking":
        return "create_booking"
    if proposal.kind == "report":
        return "save_report"
    return "change_appointment"


def after_write(state: AgentState) -> Literal["verify", "wrapup"]:
    return "wrapup" if state.outcome else "verify"


def after_verify(state: AgentState) -> Literal["after_write", "agent", "wrapup"]:
    if state.outcome is None:  # slot vừa bị đặt mất: agent tìm lại theo `notice`, giữ nguyên yêu cầu
        return "agent"
    return "after_write" if state.outcome.kind in WRITE_KINDS else "wrapup"


def after_done(state: AgentState) -> Literal["agent", "wrapup"]:
    return "agent" if state.agent_note else "wrapup"


def build_agent_flow(dependencies: AgentDependencies) -> AgentGraph:
    """agent (LLM + tool đọc) -> thẻ xác nhận (code) -> ghi + đọc lại (code) -> [full: agent làm tiếp việc còn lại]."""
    nodes, booking, report = AgentNodes(dependencies), BookingNodes(dependencies), ReportNodes(dependencies)
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", nodes.agent)
    graph_builder.add_node("agent_confirm", nodes.confirm)
    graph_builder.add_node("create_booking", booking.create_booking)
    graph_builder.add_node("verify_booking", booking.verify_booking)
    graph_builder.add_node("save", report.save)
    graph_builder.add_node("verify_save", report.verify_save)
    graph_builder.add_node("change_appointment", nodes.change_appointment)
    graph_builder.add_node("verify_change", nodes.verify_change)
    graph_builder.add_node("agent_after_write", nodes.after_write)
    graph_builder.add_node("agent_wrapup", nodes.wrapup)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", after_agent, {"confirm": "agent_confirm", "wrapup": "agent_wrapup"})
    graph_builder.add_conditional_edges(
        "agent_confirm",
        after_confirm,
        {
            "create_booking": "create_booking",
            "save_report": "save",
            "change_appointment": "change_appointment",
            "confirm": "agent_confirm",
            "agent": "agent",
            "wrapup": "agent_wrapup",
        },
    )
    for write, verify in (
        ("create_booking", "verify_booking"),
        ("save", "verify_save"),
        ("change_appointment", "verify_change"),
    ):
        graph_builder.add_conditional_edges(write, after_write, {"verify": verify, "wrapup": "agent_wrapup"})
        graph_builder.add_conditional_edges(
            verify, after_verify, {"after_write": "agent_after_write", "agent": "agent", "wrapup": "agent_wrapup"}
        )
    graph_builder.add_conditional_edges("agent_after_write", after_done, {"agent": "agent", "wrapup": "agent_wrapup"})
    graph_builder.add_edge("agent_wrapup", END)
    return graph_builder.compile()
