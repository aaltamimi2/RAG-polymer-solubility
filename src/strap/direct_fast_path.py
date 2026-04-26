"""Deterministic direct-tool fast paths for simple DISSOLVE requests.

These handlers bypass chat-model planning/synthesis when a user request maps
cleanly to a structured core tool whose display output is already user-facing.
Complex requests still fall through to the normal agent architecture.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage

from strap.routing_classifier import (
    is_direct_answer_query,
    is_direct_solubility_lookup_query,
    is_direct_solubility_plot_query,
    is_separation_visualization_request,
)
from strap.orchestrator_runtime import make_artifact, make_route_decision, make_run_ledger
from strap.solubility import _get_known_names, _load_coefficients, resolve_polymer, resolve_solvent
from strap.solubility import get_solubility_pair_exclusion_reason
from strap.solvent_registry import resolve_to_interp_key


_POLYMER_RE = re.compile(
    r"\b(LDPE|HDPE|LLDPE|PE|EVOH|PETG?|PP|PS|PVC|PC|PES|PMMA|ABS|PVDF|NYLON[- ]?6|NYLON[- ]?66)\b",
    re.IGNORECASE,
)
_TEMP_RE = re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?:deg\s*)?(?:°\s*)?C\b", re.IGNORECASE)
_PLOT_RE = re.compile(r"\b(plot|chart|graph|visualiz(?:e|ation)?)\b", re.IGNORECASE)
_SOLVENT_LOOKUP_RE = re.compile(r"\b(?:solvents?|dissolv(?:e|es|ing)|solubil)\b", re.IGNORECASE)
_SAFETY_CARD_RE = re.compile(
    r"\b(?:safety\s+(?:cards?|profiles?|suites?|dossiers?)|flash\s+point|vapou?r\s+pressure|"
    r"boiling\s+point|auto[- ]?(?:ignition|emission)|peroxide(?:\s+formation)?|ld50|toxicity|toxic(?:ity)?|"
    r"(?:safely\s+)?(?:heat|heating|handle|handling))\b",
    re.IGNORECASE,
)
_HSP_DOMAIN_RE = re.compile(
    r"\b(?:HSP|RED|Hansen(?:\s+(?:model|solubility))?|compatib(?:le|ility))\b",
    re.IGNORECASE,
)
_FAST_PATH_DOMAIN_CONFLICTS = {
    "safety_lookup": (_HSP_DOMAIN_RE,),
}
_POLYMER_LIST_RE = re.compile(r"\b(?:list|show|what|which).{0,80}\bpolymers?\b", re.IGNORECASE | re.DOTALL)
_SOLVENT_LIST_RE = re.compile(
    r"\b(?:list|show|what|which).{0,80}\bsolvents?\b(?![^.?!]{0,80}\bdissolv)",
    re.IGNORECASE | re.DOTALL,
)
_SINGLE_SOLUBILITY_RE = re.compile(
    r"\bsolubility\s+of\b(?=[^.?!]{0,160}\bat\s+\d+(?:\.\d+)?\s*(?:c|°c)\b)",
    re.IGNORECASE,
)
_QUOTED_PATH_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')
_TOP_N_RE = re.compile(r"\btop\s+(?P<n>\d{1,2})\b", re.IGNORECASE)
_VALUES_RE = re.compile(
    r"\b(?:exact\s+)?(?:values?|data|table|numbers?|points?|csv)\b",
    re.IGNORECASE,
)
_ARTIFACT_UPDATE_RE = re.compile(
    r"\b(?:save|export|write|output|range|from|up\s+to|through|until|only|just)\b",
    re.IGNORECASE,
)
_CONTEXT_REF_RE = re.compile(r"\b(?:these|those|each|same|previous|above|curves?|solvents?|top\s+\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DirectFastPathResult:
    """Result returned by a deterministic direct-tool handler."""

    display: str
    tool_name: str
    data: dict[str, Any]
    raw: str
    route_decision: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    run_ledger: dict[str, Any] = field(default_factory=dict)


def _has_fast_path_domain_conflict(user_request: str, intent: str) -> bool:
    """Return whether explicit domain markers should block a direct fast path."""
    return any(pattern.search(user_request) for pattern in _FAST_PATH_DOMAIN_CONFLICTS.get(intent, ()))


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def _get_last_human_message(messages: list) -> str:
    for msg in reversed(messages):
        msg_type = getattr(msg, "type", None)
        if msg_type == "human" or msg.__class__.__name__ == "HumanMessage":
            return _extract_text(getattr(msg, "content", ""))
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _extract_text(msg.get("content", ""))
    return ""


def _current_user_request(text: str) -> str:
    marker = "User request:"
    if marker not in text:
        return text
    return text.rsplit(marker, 1)[-1].strip()


def _parse_tool_envelope(raw: str, *, fallback_tool_name: str) -> DirectFastPathResult:
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return DirectFastPathResult(str(raw), fallback_tool_name, {}, str(raw))
    if isinstance(parsed, dict) and "display" in parsed:
        data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}
        tool_name = str(data.get("tool_name") or fallback_tool_name)
        return DirectFastPathResult(str(parsed.get("display") or ""), tool_name, data, raw)
    return DirectFastPathResult(str(raw), fallback_tool_name, {}, raw)


def _run_awaitable_sync(awaitable: Awaitable[str]) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, str] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - defensive bridge
            error["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error["exc"]
    return result["value"]


def _call_tool_sync(func, *args, **kwargs) -> str:
    value = func(*args, **kwargs)
    if inspect.isawaitable(value):
        return _run_awaitable_sync(value)
    return value


async def _call_tool_async(func, *args, **kwargs) -> str:
    value = func(*args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


def _known_names() -> tuple[set[str], set[str]]:
    _, lookup = _load_coefficients()
    return _get_known_names(lookup)


def _resolve_polymer_name(value: str | None) -> str | None:
    if not value:
        return None
    known_polymers, _ = _known_names()
    return resolve_polymer(value, known_polymers)


def _resolve_solvent_name(value: str | None) -> str | None:
    if not value:
        return None
    _, known_solvents = _known_names()
    return resolve_solvent(value, known_solvents)


def _resolve_solvent_name_strict(value: str | None) -> str | None:
    if not value:
        return None
    _, known_solvents = _known_names()
    norm = value.strip().lower()
    if norm in known_solvents:
        return norm
    alias = resolve_to_interp_key(norm)
    if alias and alias in known_solvents:
        return alias
    return None


def _extract_target_polymer(user_request: str) -> str | None:
    priority_patterns = (
        r"\bdissolv(?:e|es|ing)\s+(?P<polymer>[A-Za-z0-9-]+)\b",
        r"\bsolubility\s+of\s+(?P<polymer>[A-Za-z0-9-]+)\b",
        r"\bfor\s+(?P<polymer>[A-Za-z0-9-]+)\b",
    )
    for pattern in priority_patterns:
        match = re.search(pattern, user_request, re.IGNORECASE)
        if match:
            polymer = _resolve_polymer_name(match.group("polymer"))
            if polymer:
                return polymer
    for match in _POLYMER_RE.finditer(user_request):
        polymer = _resolve_polymer_name(match.group(1))
        if polymer:
            return polymer
    return None


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        deduped.append(value)
        seen.add(key)
    return deduped


def _split_polymer_text(text: str) -> list[str]:
    cleaned = re.sub(r'"[^"]*"|\'[^\']*\'', "", str(text or ""))
    cleaned = re.sub(r"\s+(?:and|vs\.?|versus)\s+", ",", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("/", ",")
    values: list[str] = []
    for part in [item.strip(" .,:;") for item in cleaned.split(",") if item.strip(" .,:;")]:
        polymer = _resolve_polymer_name(part)
        if polymer:
            values.append(polymer)
            continue
        for match in _POLYMER_RE.finditer(part):
            polymer = _resolve_polymer_name(match.group(1))
            if polymer:
                values.append(polymer)
    return _dedupe_preserving_order(values)


def _extract_explicit_polymers(user_request: str) -> list[str]:
    patterns = (
        r"\bsolubility\s+of\s+(?P<polymers>.+?)(?=\s+in\b|\s+(?:from|between|up\s+to|at|over|to|save|output|write)\b|[?.!]|$)",
        r"\bpolymers?\s+(?P<polymers>.+?)(?=\s+in\b|\s+(?:from|between|up\s+to|at|over|to|save|output|write)\b|[?.!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, user_request, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        polymers = _split_polymer_text(match.group("polymers"))
        if polymers:
            return polymers

    values = [
        polymer
        for match in _POLYMER_RE.finditer(user_request)
        if (polymer := _resolve_polymer_name(match.group(1)))
    ]
    return _dedupe_preserving_order(values)


def _split_requested_solvent_text(text: str) -> list[str]:
    cleaned = re.sub(r'"[^"]*"|\'[^\']*\'', "", str(text or ""))
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\s+(?:and|vs\.?|versus)\s+", ",", cleaned, flags=re.IGNORECASE)
    return _split_solvent_text(cleaned)


def _extract_explicit_solvents(user_request: str) -> list[str]:
    patterns = (
        r"\bsolubility\s+of\s+.+?\s+in\s+(?P<solvents>.+?)(?=\s+(?:from|between|up\s+to|at|over|to|save|output|write)\b|[?.!]|$)",
        r"\bin\s+(?P<solvents>.+?)(?=\s+(?:from|between|up\s+to|at|over|to|save|output|write)\b|[?.!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, user_request, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        candidate = match.group("solvents").strip(" .,:;")
        if re.search(r"\b(each|these|those|solvents?)\b", candidate, re.IGNORECASE):
            continue
        solvents = [
            solvent
            for value in _split_requested_solvent_text(candidate)
            if (solvent := _resolve_solvent_name(value))
        ]
        solvents = _dedupe_preserving_order(solvents)
        if solvents:
            return solvents
    return []


def _extract_explicit_solvent(user_request: str) -> str | None:
    solvents = _extract_explicit_solvents(user_request)
    return solvents[0] if solvents else None


def _extract_context_list(text: str, key: str) -> list[str]:
    match = re.search(rf"\b{re.escape(key)}=([^\n;]+(?:,\s*[^\n;]+)*)", text, re.IGNORECASE)
    if not match:
        return []
    return _split_solvent_text(match.group(1))


def _split_solvent_text(text: str) -> list[str]:
    """Split solvent lists while preserving names like N,N-Dimethylformamide."""
    cleaned = text.strip(" .,:;")
    if not cleaned:
        return []
    if "," not in cleaned and _resolve_solvent_name_strict(cleaned):
        return [cleaned]

    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    values: list[str] = []
    index = 0
    while index < len(parts):
        matched = None
        matched_size = 0
        for size in range(min(3, len(parts) - index), 0, -1):
            candidate = ",".join(parts[index:index + size]).strip()
            if _resolve_solvent_name_strict(candidate):
                matched = candidate
                matched_size = size
                break
        if matched is not None:
            values.append(matched)
            index += matched_size
        else:
            values.append(parts[index])
            index += 1
    return values


def _extract_last_solubility_lookup(text: str) -> dict[str, Any] | None:
    match = re.search(r"Last solubility lookup:\s*(?P<body>[^\n]+)", text, re.IGNORECASE)
    if not match:
        return None
    body = match.group("body")
    polymer_match = re.search(r"\bpolymer=([^;]+)", body, re.IGNORECASE)
    solvent_match = re.search(r"\bsolvents=([^;]+)", body, re.IGNORECASE)
    range_match = re.search(
        r"\btemperature_range=(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\s*C",
        body,
        re.IGNORECASE,
    )
    polymer = _resolve_polymer_name(polymer_match.group(1).strip()) if polymer_match else None
    solvents = []
    if solvent_match:
        for value in _split_solvent_text(solvent_match.group(1)):
            solvent = _resolve_solvent_name(value.strip())
            if solvent:
                solvents.append(solvent)
    if not polymer or not solvents:
        return None
    lookup: dict[str, Any] = {"polymer": polymer, "solvents": solvents}
    if range_match:
        lookup["temperature_min_c"] = float(range_match.group("start"))
        lookup["temperature_max_c"] = float(range_match.group("end"))
    return lookup


def _extract_last_solvent_candidate_table(text: str) -> dict[str, Any] | None:
    match = re.search(r"Last solvent candidates:\s*(?P<body>[^\n]+)", text, re.IGNORECASE)
    body = match.group("body") if match else ""
    polymer_match = re.search(r"\bpolymer=([^;]+)", body, re.IGNORECASE) if body else None
    solvent_match = re.search(r"\bsolvents=([^;]+)", body, re.IGNORECASE) if body else None
    polymer = _resolve_polymer_name(polymer_match.group(1).strip()) if polymer_match else None
    solvents: list[str] = []
    if solvent_match:
        for value in _split_solvent_text(solvent_match.group(1)):
            solvent = _resolve_solvent_name(value.strip())
            if solvent:
                solvents.append(solvent)
    if not solvents:
        solvents = [
            solvent
            for value in _extract_context_list(text, "solvent_candidates")
            if (solvent := _resolve_solvent_name(value))
        ]
    if not solvents:
        return None
    return {"polymer": polymer, "solvents": solvents}


def _extract_last_plot_artifact(text: str) -> dict[str, Any] | None:
    match = re.search(r"Last plot artifact:\s*(?P<body>[^\n]+)", text, re.IGNORECASE)
    if not match:
        return None
    body = match.group("body")
    plot_type_match = re.search(r"\bplot_type=([^;]+)", body, re.IGNORECASE)
    polymers_match = re.search(r"\bpolymers=([^;]+)", body, re.IGNORECASE)
    polymer_match = re.search(r"\bpolymer=([^;]+)", body, re.IGNORECASE)
    solvent_match = re.search(r"\bsolvents=([^;]+)", body, re.IGNORECASE)
    range_match = re.search(
        r"\btemperature_range=(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\s*C",
        body,
        re.IGNORECASE,
    )
    path_match = re.search(r"\bpath=([^;]+)", body, re.IGNORECASE)
    output_dir_match = re.search(r"\boutput_dir=([^;]+)", body, re.IGNORECASE)
    polymers: list[str] = []
    if polymers_match:
        polymers = _split_polymer_text(polymers_match.group(1))
    elif polymer_match:
        polymer = _resolve_polymer_name(polymer_match.group(1).strip())
        if polymer:
            polymers = [polymer]
    solvents: list[str] = []
    if solvent_match:
        for value in _split_solvent_text(solvent_match.group(1)):
            solvent = _resolve_solvent_name(value.strip())
            if solvent:
                solvents.append(solvent)
    polymers = _dedupe_preserving_order(polymers)
    solvents = _dedupe_preserving_order(solvents)
    if not polymers or not solvents:
        return None
    artifact: dict[str, Any] = {
        "plot_type": (plot_type_match.group(1).strip() if plot_type_match else "solubility_vs_temperature"),
        "polymer": polymers[0],
        "polymers": polymers,
        "solvents": solvents,
    }
    if range_match:
        artifact["temperature_min_c"] = float(range_match.group("start"))
        artifact["temperature_max_c"] = float(range_match.group("end"))
    if path_match:
        artifact["path"] = path_match.group(1).strip()
    if output_dir_match:
        artifact["output_dir"] = output_dir_match.group(1).strip()
    return artifact


def _extract_requested_top_n(user_request: str) -> int | None:
    match = _TOP_N_RE.search(user_request)
    if not match:
        return None
    try:
        value = int(match.group("n"))
    except ValueError:
        return None
    if value <= 0:
        return None
    return min(value, 24)


def _limit_requested_solvents(solvents: list[str], user_request: str, *, default_limit: int = 12) -> list[str]:
    limit = _extract_requested_top_n(user_request) or default_limit
    return solvents[:limit]


def _filter_and_limit_solvents(
    polymer: str | None,
    solvents: list[str],
    user_request: str,
    *,
    default_limit: int = 12,
) -> list[str]:
    if polymer:
        solvents = [
            solvent for solvent in solvents
            if not get_solubility_pair_exclusion_reason(polymer, solvent)
        ]
    return _limit_requested_solvents(solvents, user_request, default_limit=default_limit)


def _filter_and_limit_solvents_for_polymers(
    polymers: list[str],
    solvents: list[str],
    user_request: str,
    *,
    default_limit: int = 12,
) -> list[str]:
    if polymers:
        solvents = [
            solvent
            for solvent in solvents
            if any(not get_solubility_pair_exclusion_reason(polymer, solvent) for polymer in polymers)
        ]
    return _limit_requested_solvents(solvents, user_request, default_limit=default_limit)


def _context_default_range(*contexts: dict[str, Any] | None) -> tuple[float, float]:
    for context in contexts:
        if not context:
            continue
        t_min = context.get("temperature_min_c")
        t_max = context.get("temperature_max_c")
        if t_min is not None and t_max is not None:
            return float(t_min), float(t_max)
    return 25.0, 160.0


def _context_default_range_for_polymer(
    polymer: str | None,
    *contexts: dict[str, Any] | None,
) -> tuple[float, float]:
    if not polymer:
        return _context_default_range(*contexts)
    for context in contexts:
        if not context:
            continue
        context_polymer = str(context.get("polymer") or "")
        if context_polymer and context_polymer.upper() != polymer.upper():
            continue
        t_min = context.get("temperature_min_c")
        t_max = context.get("temperature_max_c")
        if t_min is not None and t_max is not None:
            return float(t_min), float(t_max)
    return 25.0, 160.0


def _resolve_context_polymer(user_request: str, *contexts: dict[str, Any] | None) -> str | None:
    polymers = _extract_explicit_polymers(user_request)
    if polymers:
        return polymers[0]
    polymer = _extract_target_polymer(user_request)
    if polymer:
        return polymer
    for context in contexts:
        if context and context.get("polymer"):
            return str(context["polymer"])
    return None


def _resolve_context_polymers(user_request: str, *contexts: dict[str, Any] | None) -> list[str]:
    polymers = _extract_explicit_polymers(user_request)
    if polymers:
        return polymers
    for context in contexts:
        if not context:
            continue
        context_polymers = context.get("polymers")
        if isinstance(context_polymers, list) and context_polymers:
            return [str(polymer) for polymer in context_polymers]
        if context.get("polymer"):
            return [str(context["polymer"])]
    return []


def _resolve_context_solvents(user_request: str, *contexts: dict[str, Any] | None) -> list[str]:
    explicit_solvents = _extract_explicit_solvents(user_request)
    if explicit_solvents:
        return explicit_solvents
    for context in contexts:
        solvents = list((context or {}).get("solvents") or [])
        if solvents:
            return [str(solvent) for solvent in solvents]
    return []


def _infer_output_dir_from_path(path: str | None) -> str | None:
    if not path:
        return None
    from strap.tools._helpers import normalize_wsl_path

    normalized = normalize_wsl_path(path)
    if not normalized:
        return None
    candidate = Path(normalized)
    return str(candidate.parent if candidate.suffix else candidate)


def _resolve_output_path_args(user_request: str, *contexts: dict[str, Any] | None) -> dict[str, str]:
    explicit = _extract_output_path_args(user_request)
    if explicit:
        return explicit
    for context in contexts:
        if not context:
            continue
        if context.get("output_dir"):
            output_dir = _infer_output_dir_from_path(str(context["output_dir"]))
            if output_dir:
                return {"output_dir": output_dir}
        output_dir = _infer_output_dir_from_path(str(context.get("path") or ""))
        if output_dir:
            return {"output_dir": output_dir}
    return {}


def _ordered_followup_contexts(
    user_request: str,
    *,
    lookup: dict[str, Any] | None,
    last_plot: dict[str, Any] | None,
    candidates: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, ...]:
    if last_plot and re.search(r"\bcurves?\b", user_request, re.IGNORECASE):
        return last_plot, lookup, candidates
    if candidates and re.search(r"\b(top\s+\d+|these|those|each)\b", user_request, re.IGNORECASE):
        return candidates, lookup, last_plot
    if last_plot and _ARTIFACT_UPDATE_RE.search(user_request):
        return last_plot, lookup, candidates
    return lookup, last_plot, candidates


def _build_plot_call(query_text: str):
    user_request = _current_user_request(query_text)
    if is_separation_visualization_request(user_request):
        return None
    wants_plot = bool(_PLOT_RE.search(user_request))
    wants_plot_update = bool(_ARTIFACT_UPDATE_RE.search(user_request) and _extract_last_plot_artifact(query_text))
    if not wants_plot and is_direct_solubility_lookup_query(user_request):
        return None
    if not wants_plot and not wants_plot_update:
        return None

    lookup = _extract_last_solubility_lookup(query_text)
    last_plot = _extract_last_plot_artifact(query_text)
    candidates = _extract_last_solvent_candidate_table(query_text)
    context_refs = bool(_CONTEXT_REF_RE.search(user_request))
    if not (lookup or last_plot or candidates or _extract_explicit_solvent(user_request)):
        return None
    if candidates and not context_refs and not _extract_target_polymer(user_request):
        return None

    contexts = _ordered_followup_contexts(
        user_request,
        lookup=lookup,
        last_plot=last_plot,
        candidates=candidates,
    )
    polymers = _resolve_context_polymers(user_request, *contexts)
    solvents = _resolve_context_solvents(user_request, *contexts)
    solvents = _filter_and_limit_solvents_for_polymers(polymers, solvents, user_request, default_limit=12)
    if not polymers or not solvents:
        return None

    from strap.tools.visualization import plot_solubility_vs_temperature

    t_start, t_end = _extract_temperature_range(
        user_request,
        _context_default_range_for_polymer(polymers[0], *contexts),
    )
    kwargs = {
        "table_name": "solubility_data",
        "polymer_column": "polymer",
        "solvent_column": "solvent",
        "temperature_column": "temperature_c",
        "solubility_column": "solubility_percentage",
        "polymers": ", ".join(polymers),
        "solvents": ", ".join(str(item) for item in solvents),
        "temperature_min": t_start,
        "temperature_max": t_end,
        **_resolve_output_path_args(user_request, *contexts),
    }
    return plot_solubility_vs_temperature, kwargs, "plot_solubility_vs_temperature"


def _build_values_followup_calls(query_text: str):
    user_request = _current_user_request(query_text)
    if not _VALUES_RE.search(user_request):
        return None
    lookup = _extract_last_solubility_lookup(query_text)
    last_plot = _extract_last_plot_artifact(query_text)
    candidates = _extract_last_solvent_candidate_table(query_text)
    if not (lookup or last_plot or candidates):
        return None
    contexts = _ordered_followup_contexts(
        user_request,
        lookup=lookup,
        last_plot=last_plot,
        candidates=candidates,
    )
    polymer = _resolve_context_polymer(user_request, *contexts)
    solvents = _resolve_context_solvents(user_request, *contexts)
    solvents = _filter_and_limit_solvents(polymer, solvents, user_request, default_limit=12)
    if not polymer or not solvents:
        return None
    default_range = _context_default_range_for_polymer(polymer, *contexts)
    t_start, t_end = _extract_temperature_range(user_request, default_range)
    from strap.tools.interpolation import predict_solubility_range

    return [
        (
            predict_solubility_range,
            {"polymer_name": polymer, "solvent_name": solvent, "t_start_c": t_start, "t_end_c": t_end, "t_step_c": 5.0},
            "predict_solubility_range",
        )
        for solvent in solvents
    ]


def _extract_temperature_range(user_request: str, default: tuple[float, float] = (25.0, 160.0)) -> tuple[float, float]:
    temps = [float(match.group("value")) for match in _TEMP_RE.finditer(user_request)]
    if re.search(r"\broom\s+temp(?:erature)?\b", user_request, re.IGNORECASE):
        if temps:
            return 25.0, temps[-1]
        return 25.0, default[1]
    if len(temps) >= 2:
        return temps[0], temps[-1]
    if len(temps) == 1:
        if re.search(r"\b(at)\s+\d", user_request, re.IGNORECASE) and not re.search(
            r"\b(up\s+to|to|through|until|from)\b", user_request, re.IGNORECASE
        ):
            return temps[0], temps[0]
        return default[0], temps[0]
    return default


def _extract_output_path_args(user_request: str) -> dict[str, str]:
    for match in _QUOTED_PATH_RE.finditer(user_request):
        path = (match.group(1) or match.group(2) or "").strip()
        if not path:
            continue
        from strap.tools._helpers import normalize_wsl_path

        path = normalize_wsl_path(path)
        if re.search(r"\.(png|jpg|jpeg|svg|pdf|html?)$", path, re.IGNORECASE):
            return {"output_path": path}
        return {"output_dir": path}
    return {}


def _extract_operating_temperature(user_request: str) -> float | None:
    matches = [float(match.group("value")) for match in _TEMP_RE.finditer(user_request)]
    return matches[-1] if matches else None


def _resolve_safety_solvent_name(value: str) -> str:
    from strap.solvent_registry import ABBREVIATION_MAP, resolve_to_property_db

    cleaned = re.sub(r"\b(?:solvent|chemical|compound)\b", "", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")
    if not cleaned:
        return cleaned
    resolved = resolve_to_property_db(cleaned)
    if resolved:
        return resolved
    expanded = ABBREVIATION_MAP.get(cleaned.lower())
    if expanded:
        return resolve_to_property_db(expanded) or expanded
    return cleaned


def _split_safety_solvents(raw: str) -> list[str]:
    cleaned = re.sub(r"\b(?:please|show|give|render|create|make|me|the)\b", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")
    if not cleaned:
        return []
    if cleaned.lower().startswith("n,n-") and "," in cleaned and " and " not in cleaned.lower():
        return [_resolve_safety_solvent_name(cleaned)]
    list_text = re.sub(r"\s+(?:and|vs\.?|versus)\s+", ",", cleaned, flags=re.IGNORECASE)
    values = _split_solvent_text(list_text)
    if not values:
        values = [part.strip() for part in list_text.split(",") if part.strip()]
    return [_resolve_safety_solvent_name(value) for value in values if value.strip()]


def _extract_safety_solvents(user_request: str) -> list[str]:
    stop = r"(?=\s+(?:at|to|above|below|as|with|under|while|when|during)\b|\s+and\s+(?:is|are|would|could|can|does)\b|[?.!]|$)"
    patterns = (
        rf"\b(?:compare\s+)?(?:safety\s+(?:cards?|profiles?|suites?|dossiers?))\s+(?:for|of|on)\s+(?P<solvents>.+?){stop}",
        rf"\b(?:flash\s+point|boiling\s+point|vapou?r\s+pressure|auto[- ]?(?:ignition|emission)(?:\s+temperature)?|ld50|toxicity|peroxide(?:\s+formation)?)\s+(?:of|for|in)\s+(?P<solvents>.+?){stop}",
        rf"\b(?:safely\s+)?(?:heat|heating|handle|handling)\s+(?P<solvents>.+?){stop}",
        rf"\b(?P<solvents>[A-Za-z0-9,.\- ]+?)\s+safety(?:\s+(?:card|profile))?{stop}",
    )
    for pattern in patterns:
        match = re.search(pattern, user_request, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        solvents = _split_safety_solvents(match.group("solvents"))
        if solvents:
            return solvents
    return []


def _combine_results(results: list[DirectFastPathResult], *, tool_name: str) -> DirectFastPathResult:
    if len(results) == 1:
        return results[0]
    display_parts = [result.display.strip() for result in results if result.display.strip()]
    data = {
        "tool_name": tool_name,
        "success": all(result.data.get("success", True) for result in results),
        "results": [result.data for result in results],
    }
    raw = json.dumps({"display": "\n\n".join(display_parts), "data": data}, ensure_ascii=False, indent=2)
    return DirectFastPathResult("\n\n".join(display_parts), tool_name, data, raw)


def _artifact_solvents_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    solvents: list[str] = []
    seen: set[str] = set()
    for row in rows:
        solvent = str(row.get("solvent") or row.get("solvent_name") or "").strip()
        if solvent and solvent.lower() not in seen:
            solvents.append(solvent)
            seen.add(solvent.lower())
    return solvents


def _build_artifacts_for_result(result: DirectFastPathResult) -> list[dict[str, Any]]:
    data = result.data or {}
    tool_name = str(data.get("tool_name") or result.tool_name)

    if tool_name == "list_available_solvents":
        rows = data.get("solvents") if isinstance(data.get("solvents"), list) else []
        polymer = data.get("polymer")
        solvents = _artifact_solvents_from_rows([row for row in rows if isinstance(row, dict)])
        if polymer and solvents:
            return [
                make_artifact(
                    artifact_type="solvent_candidate_table",
                    producer=tool_name,
                    entities={"polymer": polymer, "solvents": solvents},
                    data={"rows": rows, "limit": data.get("limit")},
                    row_order=solvents,
                    display_title=f"Solvents with solubility data for {polymer}",
                )
            ]

    if tool_name in {"predict_solubility", "predict_solubility_range"}:
        rows = data.get("results") if isinstance(data.get("results"), list) else [data]
        rows = [row for row in rows if isinstance(row, dict)]
        if not rows:
            return []
        polymer = rows[0].get("polymer_name") or rows[0].get("polymer")
        solvents = _artifact_solvents_from_rows(rows)
        t_min_values = [
            row.get("t_start_c", row.get("temperature_c"))
            for row in rows
            if row.get("t_start_c", row.get("temperature_c")) is not None
        ]
        t_max_values = [
            row.get("t_end_c", row.get("temperature_c"))
            for row in rows
            if row.get("t_end_c", row.get("temperature_c")) is not None
        ]
        artifact_data: dict[str, Any] = {"results": rows}
        if t_min_values:
            artifact_data["temperature_min_c"] = min(float(value) for value in t_min_values)
        if t_max_values:
            artifact_data["temperature_max_c"] = max(float(value) for value in t_max_values)
        if polymer and solvents:
            return [
                make_artifact(
                    artifact_type="solubility_table",
                    producer=tool_name,
                    entities={"polymer": polymer, "solvents": solvents},
                    data=artifact_data,
                    row_order=solvents,
                    display_title=f"{polymer} solubility lookup",
                )
            ]

    if tool_name == "plot_solubility_vs_temperature":
        polymers = data.get("polymers") if isinstance(data.get("polymers"), list) else []
        solvents = data.get("solvents") if isinstance(data.get("solvents"), list) else []
        polymer = str(polymers[0]) if polymers else ""
        polymer_values = [str(polymer_item) for polymer_item in polymers if str(polymer_item)]
        solvent_values = [str(solvent) for solvent in solvents if str(solvent)]
        plot_path = data.get("plot_filepath")
        output_dir = data.get("output_dir") or (os.path.dirname(str(plot_path)) if plot_path else None)
        if polymer and polymer_values and solvent_values:
            return [
                make_artifact(
                    artifact_type="plot_artifact",
                    producer=tool_name,
                    entities={
                        "plot_type": "solubility_vs_temperature",
                        "polymer": polymer,
                        "polymers": polymer_values,
                        "solvents": solvent_values,
                    },
                    data={
                        "path": plot_path,
                        "output_dir": output_dir,
                        "url": data.get("plot_url"),
                        "temperature_min_c": data.get("temperature_min_c"),
                        "temperature_max_c": data.get("temperature_max_c"),
                        "requested_temperature_max_c": data.get("requested_temperature_max_c"),
                    },
                    row_order=solvent_values,
                    display_title=f"{', '.join(polymer_values[:4])} solubility vs temperature",
                )
            ]

    return []


def _route_intent_for_tool(tool_name: str) -> tuple[str, str]:
    if tool_name == "plot_solubility_vs_temperature":
        return "artifact_transform", "solubility_plot"
    if tool_name in {"predict_solubility", "predict_solubility_range"}:
        return "direct_tool", "solubility_lookup"
    if tool_name == "list_available_solvents":
        return "direct_tool", "solvent_candidate_lookup"
    if tool_name in {"get_solvent_safety_card", "compare_solvent_safety_cards"}:
        return "direct_tool", "safety_lookup"
    return "direct_tool", "catalog_lookup"


def _attach_orchestration_metadata(result: DirectFastPathResult, *, tool_names: list[str]) -> DirectFastPathResult:
    artifacts = _build_artifacts_for_result(result)
    mode, intent = _route_intent_for_tool(result.tool_name)
    route_decision = make_route_decision(
        mode=mode,
        intent=intent,
        allowed_tools=sorted(set(tool_names or [result.tool_name])),
        tool_call_budget=len(tool_names or [result.tool_name]),
        reason="request matched deterministic direct-tool route",
    )
    run_ledger = make_run_ledger(route_decision=route_decision, tools=tool_names or [result.tool_name])
    return DirectFastPathResult(
        display=result.display,
        tool_name=result.tool_name,
        data=result.data,
        raw=result.raw,
        route_decision=route_decision,
        artifacts=artifacts,
        run_ledger=run_ledger,
    )


def _build_solubility_calls(query_text: str) -> tuple[str, list[str], float, float, bool] | None:
    user_request = _current_user_request(query_text)
    lookup = _extract_last_solubility_lookup(query_text)
    last_plot = _extract_last_plot_artifact(query_text)
    candidates = _extract_last_solvent_candidate_table(query_text)
    contexts = _ordered_followup_contexts(
        user_request,
        lookup=lookup,
        last_plot=last_plot,
        candidates=candidates,
    )
    polymer = _resolve_context_polymer(user_request, *contexts)
    if not polymer:
        return None

    explicit_solvent = _extract_explicit_solvent(user_request)
    solvents: list[str] = []
    if explicit_solvent:
        solvents = [explicit_solvent]
    elif _CONTEXT_REF_RE.search(user_request) or _VALUES_RE.search(user_request):
        solvents = _resolve_context_solvents(user_request, *contexts)
    elif lookup:
        solvents = list(lookup.get("solvents") or [])
    elif last_plot:
        solvents = list(last_plot.get("solvents") or [])
    if not solvents:
        return None
    solvents = _filter_and_limit_solvents(polymer, solvents, user_request, default_limit=8)

    default_range = _context_default_range_for_polymer(polymer, *contexts)
    t_start, t_end = _extract_temperature_range(user_request, default_range)
    single_point = bool(
        len(list(_TEMP_RE.finditer(user_request))) == 1
        and re.search(r"\bat\s+\d", user_request, re.IGNORECASE)
        and not re.search(r"\b(from|to|up\s+to|between|range|over)\b", user_request, re.IGNORECASE)
    )
    return polymer, solvents[:8], t_start, t_end, single_point


def _try_direct_fast_path_impl(query_text: str, *, async_mode: bool = False):
    user_request = _current_user_request(query_text)
    if is_separation_visualization_request(user_request):
        return None

    if _SAFETY_CARD_RE.search(user_request) and not _has_fast_path_domain_conflict(
        user_request, "safety_lookup"
    ):
        solvents = _extract_safety_solvents(user_request)
        if solvents:
            operating_temp = _extract_operating_temperature(user_request)
            if len(solvents) > 1:
                from strap.tools.safety_card import compare_solvent_safety_cards

                return (
                    compare_solvent_safety_cards,
                    {"solvent_names": ", ".join(solvents[:6]), "operating_temp_c": operating_temp},
                    "compare_solvent_safety_cards",
                )
            from strap.tools.safety_card import get_solvent_safety_card

            return (
                get_solvent_safety_card,
                {"solvent_name": solvents[0], "operating_temp_c": operating_temp},
                "get_solvent_safety_card",
            )

    if values_spec := _build_values_followup_calls(query_text):
        return values_spec

    if plot_spec := _build_plot_call(query_text):
        return plot_spec

    if is_direct_solubility_plot_query(user_request):
        lookup = _extract_last_solubility_lookup(query_text)
        if not lookup:
            return None
        last_plot = _extract_last_plot_artifact(query_text)
        from strap.tools.visualization import plot_solubility_vs_temperature

        t_start, t_end = _extract_temperature_range(
            user_request,
            (
                float(lookup.get("temperature_min_c", 25.0)),
                float(lookup.get("temperature_max_c", 160.0)),
            ),
        )
        solvents = _filter_and_limit_solvents(str(lookup["polymer"]), list(lookup["solvents"]), user_request, default_limit=12)
        if not solvents:
            return None
        kwargs = {
            "table_name": "solubility_data",
            "polymer_column": "polymer",
            "solvent_column": "solvent",
            "temperature_column": "temperature_c",
            "solubility_column": "solubility_percentage",
            "polymers": str(lookup["polymer"]),
            "solvents": ", ".join(str(item) for item in solvents),
            "temperature_min": t_start,
            "temperature_max": t_end,
            **_resolve_output_path_args(user_request, last_plot, lookup),
        }
        return plot_solubility_vs_temperature, kwargs, "plot_solubility_vs_temperature"

    if _PLOT_RE.search(user_request) and re.search(r"\bsolubility\b", user_request, re.IGNORECASE):
        lookup = _extract_last_solubility_lookup(query_text)
        last_plot = _extract_last_plot_artifact(query_text)
        candidates = _extract_last_solvent_candidate_table(query_text)
        polymer = _extract_target_polymer(user_request)
        solvents = [
            solvent
            for value in _extract_context_list(query_text, "solvent_candidates")
            if (solvent := _resolve_solvent_name(value))
        ]
        if polymer and solvents and re.search(r"\b(these|those|each)\b", user_request, re.IGNORECASE):
            from strap.tools.visualization import plot_solubility_vs_temperature

            solvents = _filter_and_limit_solvents(polymer, solvents, user_request, default_limit=12)
            if not solvents:
                return None
            t_start, t_end = _extract_temperature_range(user_request, (25.0, 160.0))
            kwargs = {
                "table_name": "solubility_data",
                "polymer_column": "polymer",
                "solvent_column": "solvent",
                "temperature_column": "temperature_c",
                "solubility_column": "solubility_percentage",
                "polymers": polymer,
                "solvents": ", ".join(solvents),
                "temperature_min": t_start,
                "temperature_max": t_end,
                **_resolve_output_path_args(user_request, candidates, lookup, last_plot),
            }
            return plot_solubility_vs_temperature, kwargs, "plot_solubility_vs_temperature"

    solubility_query = (
        is_direct_solubility_lookup_query(user_request)
        or bool(_SINGLE_SOLUBILITY_RE.search(user_request))
        or bool(re.search(r"\bsolubility\b", user_request, re.IGNORECASE) and _CONTEXT_REF_RE.search(user_request))
    )
    if solubility_query:
        call_spec = _build_solubility_calls(query_text)
        if not call_spec:
            return None
        polymer, solvents, t_start, t_end, single_point = call_spec
        if single_point:
            from strap.tools.interpolation import predict_solubility

            return [
                (
                    predict_solubility,
                    {"polymer_name": polymer, "solvent_name": solvent, "temperature_c": t_end},
                    "predict_solubility",
                )
                for solvent in solvents
            ]
        from strap.tools.interpolation import predict_solubility_range

        return [
            (
                predict_solubility_range,
                {"polymer_name": polymer, "solvent_name": solvent, "t_start_c": t_start, "t_end_c": t_end, "t_step_c": 5.0},
                "predict_solubility_range",
            )
            for solvent in solvents
        ]

    if is_direct_answer_query(user_request) and _SOLVENT_LOOKUP_RE.search(user_request):
        polymer = _extract_target_polymer(user_request)
        if polymer:
            from strap.tools.listing import list_available_solvents

            return list_available_solvents, {"polymer": polymer, "limit": 12}, "list_available_solvents"

    if _POLYMER_LIST_RE.search(user_request):
        from strap.tools.listing import list_available_polymers

        return list_available_polymers, {}, "list_available_polymers"

    if _SOLVENT_LIST_RE.search(user_request):
        from strap.tools.listing import list_available_solvents

        return list_available_solvents, {}, "list_available_solvents"

    return None


def try_direct_tool_fast_path(query_text: str) -> DirectFastPathResult | None:
    """Return a direct tool result for simple requests, or None to use the agent."""
    spec = _try_direct_fast_path_impl(query_text)
    if spec is None:
        return None
    specs = spec if isinstance(spec, list) else [spec]
    results: list[DirectFastPathResult] = []
    for func, kwargs, fallback_tool_name in specs:
        raw = _call_tool_sync(func, **kwargs)
        results.append(_parse_tool_envelope(raw, fallback_tool_name=fallback_tool_name))
    combined = _combine_results(results, tool_name=results[0].tool_name if results else "direct_tool_fast_path")
    return _attach_orchestration_metadata(combined, tool_names=[result.tool_name for result in results])


async def atry_direct_tool_fast_path(query_text: str) -> DirectFastPathResult | None:
    """Async variant of try_direct_tool_fast_path."""
    spec = _try_direct_fast_path_impl(query_text, async_mode=True)
    if spec is None:
        return None
    specs = spec if isinstance(spec, list) else [spec]
    results: list[DirectFastPathResult] = []
    for func, kwargs, fallback_tool_name in specs:
        raw = await _call_tool_async(func, **kwargs)
        results.append(_parse_tool_envelope(raw, fallback_tool_name=fallback_tool_name))
    combined = _combine_results(results, tool_name=results[0].tool_name if results else "direct_tool_fast_path")
    return _attach_orchestration_metadata(combined, tool_names=[result.tool_name for result in results])


class DirectToolFastPathMiddleware(AgentMiddleware):
    """Short-circuit simple structured-tool requests before any chat model call."""

    def wrap_model_call(self, request, handler):
        query_text = _get_last_human_message(request.messages)
        result = try_direct_tool_fast_path(query_text)
        if result is None:
            return handler(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content=result.display,
                    additional_kwargs={
                        "strap_origin": "direct_tool_fast_path",
                        "strap_tool_name": result.tool_name,
                        "strap_fast_path": True,
                        "strap_route_decision": result.route_decision,
                        "strap_artifacts": result.artifacts,
                        "strap_run_ledger": result.run_ledger,
                    },
                )
            ]
        )

    async def awrap_model_call(self, request, handler):
        query_text = _get_last_human_message(request.messages)
        result = await atry_direct_tool_fast_path(query_text)
        if result is None:
            return await handler(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content=result.display,
                    additional_kwargs={
                        "strap_origin": "direct_tool_fast_path",
                        "strap_tool_name": result.tool_name,
                        "strap_fast_path": True,
                        "strap_route_decision": result.route_decision,
                        "strap_artifacts": result.artifacts,
                        "strap_run_ledger": result.run_ledger,
                    },
                )
            ]
        )
