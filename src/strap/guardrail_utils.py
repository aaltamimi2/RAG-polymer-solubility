"""Pure helper functions shared by subagent guardrails."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .services.advanced_separation_service import parse_polymer_list
from .solubility import get_available_polymers, resolve_polymer
from .solvent_registry import resolve_to_biosteam

_STRUCTURED_RESULT_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(.*?)\s*</STRUCTURED_RESULT>",
    re.DOTALL,
)
_TEMPERATURE_LIMIT_PATTERNS = (
    re.compile(
        r"\b(?:up to|max(?:imum)?|at most|no more than|under|below)\s*(\d+(?:\.\d+)?)\s*°?\s*C\b",
        re.IGNORECASE,
    ),
)
_POLYMER_SEGMENT_PATTERNS = (
    re.compile(r"\bmixture of\s+(.+?)(?:\bat\b|\bthen\b|[.;\n]|$)", re.IGNORECASE),
    re.compile(r"\bsequence for\s+(.+?)(?:\bat\b|\bthen\b|[.;\n]|$)", re.IGNORECASE),
    re.compile(r"\bfor\s+(.+?)(?:\bat\b|\bthen\b|[.;\n]|$)", re.IGNORECASE),
)


def extract_text_content(message: AIMessage) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def parse_structured_result_payload(message: AIMessage) -> tuple[dict | None, list[str]]:
    text = extract_text_content(message)
    match = _STRUCTURED_RESULT_RE.search(text)
    if not match:
        return None, ["missing <STRUCTURED_RESULT> block"]

    json_text = match.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if fenced:
        json_text = fenced.group(1).strip()

    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, ["invalid JSON inside <STRUCTURED_RESULT>"]

    if not isinstance(payload, dict):
        return None, ["<STRUCTURED_RESULT> must decode to a JSON object"]
    return payload, []


def coerce_message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def extract_user_temperature_limit_c(messages: list) -> float | None:
    limits: list[float] = []
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        content = coerce_message_text(message.content)
        if not content:
            continue
        for pattern in _TEMPERATURE_LIMIT_PATTERNS:
            for match in pattern.finditer(content):
                try:
                    limits.append(float(match.group(1)))
                except (TypeError, ValueError):
                    continue
    return max(limits) if limits else None


def temperature_pattern(value_c: float) -> str:
    formatted = f"{value_c:.1f}".rstrip("0").rstrip(".")
    integer = str(int(round(value_c)))
    variants = {formatted, integer}
    escaped = "|".join(re.escape(item) for item in sorted(variants))
    return rf"(?:{escaped})\s*°?\s*c"


def mentions_temperature(text: str, value_c: float) -> bool:
    return bool(re.search(temperature_pattern(value_c), text.lower()))


def parse_tool_envelope(content: object) -> dict | None:
    text = content if isinstance(content, str) else str(content)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_required_visualization_tool(messages: list) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        match = re.search(r"Required tool:\s*([a-zA-Z_][a-zA-Z0-9_]*)", content)
        if match:
            return match.group(1)
        match = re.search(r"Use `([a-zA-Z_][a-zA-Z0-9_]*)` to create", content)
        if match:
            return match.group(1)
    return None


def tool_name(tool) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)


def canonicalize_solvent(solvent: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(solvent).strip())
    if not cleaned:
        return ""
    return resolve_to_biosteam(cleaned) or cleaned


def extract_requested_biosteam_batch(args: dict) -> dict[str, object] | None:
    raw = args.get("polymers_json")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        polymers = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(polymers, list):
        return None
    solvents = {
        canonicalize_solvent(spec.get("solvent", ""))
        for spec in polymers
        if isinstance(spec, dict)
    }
    solvents.discard("")
    if not solvents:
        return None
    return {
        "solvents": solvents,
        "energy_case": str(args.get("energy_case", "C1")).strip().upper() or "C1",
        "allocation_method": str(args.get("allocation_method", "value")).strip().lower() or "value",
    }


def extract_prior_successful_biosteam_batches(messages: list) -> list[dict[str, object]]:
    tool_calls: dict[str, dict] = {}
    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                tc_id = tool_call.get("id")
                if tc_id:
                    tool_calls[tc_id] = tool_call

    successful_batches: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = getattr(message, "tool_call_id", None)
        if not tool_call_id:
            continue
        tool_call = tool_calls.get(tool_call_id, {})
        if tool_call.get("name") != "run_biosteam_multi_polymer":
            continue

        envelope = parse_tool_envelope(message.content)
        data = envelope.get("data") if envelope else None
        if not isinstance(data, dict) or data.get("success") is not True:
            continue

        successful_solvents = {
            canonicalize_solvent(item.get("solvent", ""))
            for item in data.get("per_polymer", [])
            if isinstance(item, dict) and item.get("success") is True
        }
        successful_solvents.discard("")
        if not successful_solvents:
            continue

        args = tool_call.get("args", {}) if isinstance(tool_call.get("args"), dict) else {}
        successful_batches.append({
            "solvents": successful_solvents,
            "energy_case": str(data.get("energy_case") or args.get("energy_case", "C1")).strip().upper(),
            "allocation_method": str(data.get("allocation_method") or args.get("allocation_method", "value")).strip().lower(),
        })
    return successful_batches


def extract_completed_tool_names(messages: list) -> list[str]:
    tool_calls: dict[str, str] = {}
    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                tool_call_id = tool_call.get("id")
                tool_name = tool_call.get("name")
                if tool_call_id and tool_name:
                    tool_calls[tool_call_id] = tool_name

    completed: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = getattr(message, "tool_call_id", None)
        tool_name = tool_calls.get(tool_call_id) if tool_call_id else None
        if not tool_name:
            tool_name = getattr(message, "name", None)
        if tool_name:
            completed.append(tool_name)
    return completed


def extract_supported_polymers(messages: list, payload_polymers: list[str] | None = None) -> set[str]:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        payload = parse_tool_envelope(message.content)
        if not payload:
            continue
        data = payload.get("data", {})
        if data.get("tool_name") != "get_supported_polymers_and_solvents":
            continue
        polymers = data.get("polymers")
        if not isinstance(polymers, list):
            return set()
        return {str(polymer).strip().upper() for polymer in polymers if str(polymer).strip()}

    if not payload_polymers:
        return set()
    known_polymers = get_available_polymers()
    resolved_supported: set[str] = set()
    for polymer in payload_polymers:
        resolved = resolve_polymer(polymer, known_polymers)
        if resolved:
            resolved_supported.add(resolved)
    return resolved_supported


def infer_requested_polymer_support(messages: list) -> tuple[list[str], list[str], list[str]]:
    known_polymers = get_available_polymers()
    requested: list[str] = []
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        content = coerce_message_text(message.content)
        if not content:
            continue
        requested.extend(extract_polymer_candidates_from_text(content))

    deduped_requested = list(dict.fromkeys(requested))
    if not deduped_requested:
        return [], [], []

    supported: list[str] = []
    unsupported: list[str] = []
    for polymer in deduped_requested:
        resolved = resolve_polymer(polymer, known_polymers)
        if resolved:
            if resolved not in supported:
                supported.append(resolved)
        elif polymer not in unsupported:
            unsupported.append(polymer)
    return deduped_requested, supported, unsupported


def extract_polymer_candidates_from_text(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in _POLYMER_SEGMENT_PATTERNS:
        for match in pattern.finditer(text):
            segment = match.group(1).strip()
            if not segment:
                continue
            normalized = re.sub(r"\band\b", ",", segment, flags=re.IGNORECASE)
            normalized = normalized.replace("/", ",")
            normalized = normalized.replace(";", ",")
            normalized = normalized.replace("(", ",").replace(")", ",")
            for polymer in parse_polymer_list(normalized):
                cleaned = re.sub(r"[^A-Z0-9_-]", "", polymer.upper())
                if cleaned:
                    candidates.append(cleaned)
    return list(dict.fromkeys(candidates))
