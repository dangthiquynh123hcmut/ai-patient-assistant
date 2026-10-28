"""Luật phụ thuộc của clean architecture: lớp trong không được import lớp ngoài."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "patient_assistant"

# lớp -> các package/thư viện cấm import
FORBIDDEN = {
    "domain": (
        "patient_assistant.application",
        "patient_assistant.infrastructure",
        "patient_assistant.interface",
        "patient_assistant.container",
        "patient_assistant.evaluation",
        "langgraph",
        "fastapi",
        "sqlalchemy",
        "anthropic",
        "faster_whisper",
    ),
    "application": (
        "patient_assistant.infrastructure",
        "patient_assistant.interface",
        "patient_assistant.container",
        "patient_assistant.evaluation",
        "fastapi",
        "sqlalchemy",
        "anthropic",
        "faster_whisper",
    ),
    "infrastructure": ("patient_assistant.interface", "patient_assistant.container", "patient_assistant.evaluation"),
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text("utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_dependencies(layer: str) -> None:
    violations = [
        f"{path.relative_to(ROOT)} -> {name}"
        for path in (ROOT / layer).rglob("*.py")
        for name in _imports(path)
        if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN[layer])
    ]
    assert violations == []
