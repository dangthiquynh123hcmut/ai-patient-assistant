"""Chuyển Pydantic model thành JSON Schema dạng strict cho structured outputs / strict tool use."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

_DROP_KEYS = {"title", "default", "examples"}


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline(copy.deepcopy(defs[node["$ref"].split("/")[-1]]), defs)
        out = {k: _inline(v, defs) for k, v in node.items() if k not in _DROP_KEYS and k != "$defs"}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def strict_schema(model: type[BaseModel], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = model.model_json_schema()
    schema: dict[str, Any] = _inline(raw, raw.get("$defs", {}))
    for name, sub in (overrides or {}).items():
        schema["properties"][name] = sub
    return schema
