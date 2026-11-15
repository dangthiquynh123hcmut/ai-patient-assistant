from __future__ import annotations

from typing import Any

from patient_assistant.domain.booking.models import BookingRequest
from patient_assistant.domain.report.models import ReportExtraction
from patient_assistant.infrastructure.llm.schema import strict_schema


def _walk(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        return [node, *[child for v in node.values() for child in _walk(v)]]
    if isinstance(node, list):
        return [child for v in node for child in _walk(v)]
    return []


def test_strict_schema_is_self_contained_and_closed() -> None:
    for model in (ReportExtraction, BookingRequest):
        schema = strict_schema(model)
        nodes = _walk(schema)
        assert not any("$ref" in node or "$defs" in node for node in nodes)
        for value in (node for node in nodes if node.get("type") == "object" and "properties" in node):
            assert value["additionalProperties"] is False
            assert set(value["required"]) == set(value["properties"])


def test_booking_schema_restricts_department_to_catalog() -> None:
    schema = strict_schema(
        BookingRequest,
        overrides={"department_code": {"anyOf": [{"type": "string", "enum": ["PHCN"]}, {"type": "null"}]}},
    )
    assert schema["properties"]["department_code"]["anyOf"][0]["enum"] == ["PHCN"]
    assert "validation_status" not in schema["properties"]  # LLM không được tự gán trạng thái validation
