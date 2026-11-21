from __future__ import annotations

from patient_assistant.application.agent.trace import describe_step


def test_step_details_keep_attempts_for_the_validation_tab() -> None:
    attempt = {
        "flow": "report",
        "attempt_no": 0,
        "output": None,
        "issue_codes": ["evidence_not_found"],
        "llm_fixable_codes": ["evidence_not_found"],
        "passed": False,
    }
    step = describe_step(2, ("agent_flow:abc",), "agent", {"attempts": [attempt], "tool_calls": [{"tool": "x"}]})
    assert step.details["attempts"] == [attempt]
    assert "tool_calls" not in step.details
