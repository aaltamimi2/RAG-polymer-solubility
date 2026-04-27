"""Schema helpers for optional Claude SDK structured final results."""

from __future__ import annotations

from typing import Any


FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "markdown": {"type": "string"},
        "artifact_ids": {"type": "array", "items": {"type": "string"}},
        "paths": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "unresolved_checks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["markdown"],
    "additionalProperties": True,
}


def final_answer_output_format() -> dict[str, Any]:
    return {"type": "json_schema", "schema": FINAL_ANSWER_SCHEMA}
