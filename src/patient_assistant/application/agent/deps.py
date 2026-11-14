from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from patient_assistant.application.ports.llm import LLMPort
from patient_assistant.application.ports.repositories import AuditLog
from patient_assistant.application.tools.toolbox import Toolbox


@dataclass(frozen=True)
class AgentLimits:
    """Mọi vòng lặp đều có trần (FR-02.5, FR-06.3)."""

    max_corrections: int = 2
    # Số lần hỏi lại CÙNG một câu mà người dùng chưa trả lời được. Trả lời được -> đếm lại từ đầu.
    max_clarify_rounds: int = 3
    # Trần tổng số câu hỏi cho một yêu cầu, kể cả khi mỗi lượt đều có tiến triển.
    max_questions: int = 10
    max_slot_search_rounds: int = 3
    max_conflict_retries: int = 1
    max_tool_calls: int = 4
    # Trần số lần gọi tool trong MỘT lượt của agent đặt lịch (tra lịch cũ, tìm slot vài lần, đề xuất).
    max_agent_tool_calls: int = 10
    recursion_limit: int = 60


@dataclass(frozen=True)
class AgentDeps:
    """Mọi thứ node cần, tiêm từ composition root. Node không tự tạo client/repository."""

    llm: LLMPort
    tools: Toolbox
    audit: AuditLog
    limits: AgentLimits = field(default_factory=AgentLimits)
    # "workflow": các luồng cứng (phân loại -> trích xuất -> hỏi thiếu). "booking": riêng đặt/đổi/hủy lịch do agent
    # LLM + tool đảm nhận. "full": mọi việc do một agent chung đảm nhận (xem nodes/assistant_agent.py).
    agent_mode: Literal["workflow", "booking", "full"] = "workflow"
