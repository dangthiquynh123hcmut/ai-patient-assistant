from __future__ import annotations

from dataclasses import dataclass, field

from patient_assistant.application.ports.llm import LLMPort
from patient_assistant.application.ports.repositories import AuditLog
from patient_assistant.application.tools.toolbox import Toolbox


@dataclass(frozen=True)
class AgentLimits:
    """Mọi vòng lặp đều có trần (FR-02.5, FR-06.3)."""

    max_corrections: int = 2
    max_clarify_rounds: int = 3
    max_slot_search_rounds: int = 3
    max_conflict_retries: int = 1
    max_tool_calls: int = 4
    recursion_limit: int = 60


@dataclass(frozen=True)
class AgentDeps:
    """Mọi thứ node cần, tiêm từ composition root. Node không tự tạo client/repository."""

    llm: LLMPort
    tools: Toolbox
    audit: AuditLog
    limits: AgentLimits = field(default_factory=AgentLimits)
