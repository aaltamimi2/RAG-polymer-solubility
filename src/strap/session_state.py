"""Durable CLI session transcript and compact domain context."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .solvent_registry import SOLVENT_REGISTRY, resolve_to_interp_key
from .user_input_parsing import (
    extract_output_destination,
    extract_temperatures_c,
    has_temperature_ceiling,
    last_temperature_c,
)

_SCHEMA_VERSION = "1.1"
_MAX_CONTEXT_BLOCK_CHARS = 2400

_CAPACITY_RE = re.compile(
    r"\b(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>metric\s*)?(?:tonnes?|tons?|mt|tpy)\s*"
    r"(?:/|per)?\s*(?:year|yr|annum)?\b",
    re.IGNORECASE,
)
_KNOWN_POLYMER_PATTERN = r"LDPE|HDPE|LLDPE|PE|EVOH|PET|PP|PS|PVC|PC|PLA|PA|Nylon"
_COMPOSITION_RE = re.compile(
    rf"\b(?P<pct>\d+(?:\.\d+)?)\s*(?:wt\s*)?%\s*(?P<polymer>{_KNOWN_POLYMER_PATTERN})\b",
    re.IGNORECASE,
)
_SCENARIO_RE = re.compile(r"\bscenario\s+(?P<scenario>[A-Z0-9_-]+)\b", re.IGNORECASE)
_ENERGY_CASE_RE = re.compile(r"\b(?:energy\s+case\s*)?(?P<case>C[123])\b", re.IGNORECASE)
_TEMP_RE = re.compile(r"\b(?P<value>\d+(?:\.\d+)?)\s*(?:deg\s*)?(?:°\s*)?C\b", re.IGNORECASE)
_PRECIP_TEMP_CONTEXT_RE = re.compile(
    r"\b(?:precipitation|precip|cooling)\s+(?:temperature\s*)?(?:of|=|at|to)?(?:(?:\d+\.\d+)|[^.?!\n]){0,80}",
    re.IGNORECASE,
)
_DISSOLUTION_TEMP_CONTEXT_RE = re.compile(
    r"\b(?:use|using|dissolv(?:e|es|ing|ution)?|operat(?:e|ing)?|process(?:ing)?|stage\s+\d+)"
    r"(?:(?:\d+\.\d+)|[^.?!\n]){0,100}?"
    r"\b(?:at|temperature\s*(?:of|=)?|temp(?:erature)?\s*(?:of|=)?)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:deg\s*)?(?:°\s*)?C\b",
    re.IGNORECASE,
)
_TEMPERATURE_RANGE_CONTEXT_RE = re.compile(
    r"\b(?:from|between|range|up\s+to|through|until|below|under|max(?:imum)?)\b"
    r"(?:(?:\d+\.\d+)|[^.?!\n]){0,80}\b\d+(?:\.\d+)?\s*(?:deg\s*)?(?:°\s*)?C\b",
    re.IGNORECASE,
)
_TARGET_FRACTION_RE = re.compile(
    r"\b(?:target_plastic_percent\s*=\s*)?(?P<pct>\d+(?:\.\d+)?)\s*"
    r"(?:wt\s*)?%\s+target\s+plastic\s+in\s+feed\b|"
    r"\btarget_plastic_percent\s*=\s*(?P<pct_alt>\d+(?:\.\d+)?)\s*(?:wt\s*)?%",
    re.IGNORECASE,
)
_SOLVENT_RE = re.compile(
    r"\b(?:using|with|recovered\s+with|dissolution\s+in)\s+"
    r"(?P<solvent>[A-Z][A-Za-z0-9 ,/-]+?)"
    r"(?=\s+(?:at|under|for|and|with|in|to|from)|[,.$])",
)
_OUTPUT_PATH_RE = re.compile(
    r"\b(?:save|saved|write|written|export|output)\b[^.?!\n]{0,80}?"
    r"\b(?:to|under|in)\s+"
    r"(?P<path>\"[^\"]+\"|'[^']+'|\\\\[^\s]+|/[^\s,;]+|~[^\s,;]+)",
    re.IGNORECASE,
)
_METRIC_PATTERNS = (
    ("MSP", re.compile(r"\bmsp\b|minimum\s+selling\s+price", re.IGNORECASE)),
    ("TCI/CAPEX", re.compile(r"\btci\b|\bcapex\b|capital\s+cost", re.IGNORECASE)),
    ("AOC/OPEX", re.compile(r"\baoc\b|\bopex\b|operating\s+cost", re.IGNORECASE)),
    ("GWP", re.compile(r"\bgwp\b|carbon\s+footprint|co2e?", re.IGNORECASE)),
)
_KNOWN_POLYMER_RE = re.compile(rf"\b({_KNOWN_POLYMER_PATTERN})\b", re.IGNORECASE)
_SOLUBILITY_SUBJECT_RE = re.compile(
    rf"\bsolubility\s+of\s+(?P<polymer>{_KNOWN_POLYMER_PATTERN})\b",
    re.IGNORECASE,
)
_FOLLOWUP_RE = re.compile(
    r"\b(this|that|it|same|previous|above|those|these|there|now|again|instead|under\s+C[123])\b",
    re.IGNORECASE,
)
_SOLUBILITY_PLOT_FOLLOWUP_RE = re.compile(
    r"\b(plot|chart|graph|visualiz(?:e|ation)?)\b",
    re.IGNORECASE,
)
_ARTIFACT_FOLLOWUP_RE = re.compile(
    r"\b(plot|chart|graph|visualiz(?:e|ation)?|save|export|write|values?|data|table|"
    r"numbers?|csv|top\s+\d+|these|those|each|same|previous|above|curves?|only|just)\b",
    re.IGNORECASE,
)
_DOMAIN_QUERY_RE = re.compile(
    r"\b(feedstock|composition|scenario|energy\s+case|msp|capex|opex|gwp|"
    r"biosteam|tea|lca|solvent|solubility|polymer|separation|optimization|pareto|"
    r"plot|chart|graph|visualiz(?:e|ation)?)\b",
    re.IGNORECASE,
)

_SESSION_SOLVENT_ALIAS_DENYLIST = {
    # "deg" is a common abbreviation for degrees in "100 deg C"; treating it
    # as diethylene glycol polluted follow-up solvent context.
    "deg",
}
_SOLVENT_ALIAS_PATTERNS: list[tuple[re.Pattern[str], str]] = []
_seen_solvent_aliases: set[str] = set()
for _key, _info in SOLVENT_REGISTRY.items():
    canonical = str(_info.get("property_db") or _info.get("interp_key") or _key)
    for _alias in {_key, *[str(alias) for alias in _info.get("aliases", [])]}:
        alias = _alias.strip()
        alias_lower = alias.lower()
        if not alias or alias_lower in _seen_solvent_aliases or alias_lower in _SESSION_SOLVENT_ALIAS_DENYLIST:
            continue
        _seen_solvent_aliases.add(alias_lower)
        left = r"(?<![A-Za-z0-9])"
        right = r"(?![A-Za-z0-9])"
        _SOLVENT_ALIAS_PATTERNS.append((re.compile(left + re.escape(alias) + right, re.IGNORECASE), canonical))
_SOLVENT_ALIAS_PATTERNS.sort(key=lambda item: len(item[0].pattern), reverse=True)


def _is_runtime_excluded_solvent(solvent: str) -> bool:
    interp_key = resolve_to_interp_key(solvent) or solvent
    return interp_key.strip().lower() == "triethylamine"


def _filter_runtime_solvents(solvents: list[str]) -> list[str]:
    return [solvent for solvent in solvents if not _is_runtime_excluded_solvent(solvent)]


def _solvent_mentions(text: str) -> list[str]:
    hits: list[tuple[int, int, str]] = []
    for pattern, canonical in _SOLVENT_ALIAS_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append((match.start(), match.end(), canonical))
    accepted: list[tuple[int, int, str]] = []
    for start, end, canonical in sorted(hits, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(not (end <= acc_start or start >= acc_end) for acc_start, acc_end, _ in accepted):
            continue
        accepted.append((start, end, canonical))
    solvents = [canonical for _start, _end, canonical in sorted(accepted)]
    return _filter_runtime_solvents(_merge_unique([], solvents))


def _extract_displayed_solvent_recommendations(text: str) -> dict[str, Any] | None:
    """Extract the small user-visible recommendation table from assistant prose."""
    if not re.search(r"\b(Recommended\s+Solvent|Target\s+Polymer|Polymer)\b", text, re.IGNORECASE):
        return None

    rows: list[dict[str, str]] = []
    row_re = re.compile(
        rf"^\s*(?P<polymer>{_KNOWN_POLYMER_PATTERN})\s+(?P<body>.+)$",
        re.IGNORECASE,
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("━", "-", "|")):
            continue
        match = row_re.match(line)
        if not match:
            continue
        body = match.group("body")
        solvents = _solvent_mentions(body)
        if not solvents:
            continue
        rows.append(
            {
                "polymer": match.group("polymer").upper(),
                "solvent": solvents[0],
            }
        )

    if not rows:
        return None
    return {
        "source": "assistant_displayed_recommendations",
        "polymers": _merge_unique([], [row["polymer"] for row in rows]),
        "solvents": _merge_unique([], [row["solvent"] for row in rows]),
        "rows": rows,
    }


def get_session_root() -> Path:
    """Return the directory where DISSOLVE CLI sessions are stored."""
    root = os.getenv("DISSOLVE_SESSION_DIR")
    if root:
        return Path(root).expanduser().resolve()
    return (Path.home() / ".dissolve" / "sessions").resolve()


def get_session_dir(thread_id: str) -> Path:
    """Return the durable directory for a thread id."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", thread_id).strip("._") or "default"
    return get_session_root() / safe


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_context(thread_id: str) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "thread_id": thread_id,
        "updated_at": _utc_now(),
        "feedstock": {},
        "process": {},
        "analysis": {},
        "artifacts": [],
        "route_decisions": [],
        "run_ledgers": [],
        "last_user_query": "",
    }


def load_session_context(thread_id: str) -> dict[str, Any]:
    """Load compact structured context for a thread."""
    path = get_session_dir(thread_id) / "context.json"
    if not path.exists():
        return _empty_context(thread_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_context(thread_id)
    context = _empty_context(thread_id)
    if isinstance(raw, dict):
        context.update(raw)
        for key in ("feedstock", "process", "analysis"):
            if not isinstance(context.get(key), dict):
                context[key] = {}
        for key in ("artifacts", "route_decisions", "run_ledgers"):
            if not isinstance(context.get(key), list):
                context[key] = []
    return context


def save_session_context(thread_id: str, context: dict[str, Any]) -> None:
    """Persist compact structured context for a thread."""
    session_dir = get_session_dir(thread_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    context = dict(context)
    context["schema_version"] = _SCHEMA_VERSION
    context["thread_id"] = thread_id
    context["updated_at"] = _utc_now()
    (session_dir / "context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def append_transcript_event(thread_id: str, role: str, content: str, **metadata: Any) -> None:
    """Append one durable JSONL transcript event."""
    session_dir = get_session_dir(thread_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": _utc_now(),
        "role": role,
        "content": content,
        "metadata": metadata,
    }
    with (session_dir / "transcript.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def session_paths(thread_id: str) -> dict[str, Path]:
    """Return standard durable paths for a thread."""
    session_dir = get_session_dir(thread_id)
    return {
        "dir": session_dir,
        "context": session_dir / "context.json",
        "transcript": session_dir / "transcript.jsonl",
        "checkpoint": session_dir / "checkpoints.sqlite",
    }


def _merge_unique(existing: list[str], new_values: list[str]) -> list[str]:
    seen = {value.lower() for value in existing}
    merged = list(existing)
    for value in new_values:
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in seen:
            merged.append(cleaned)
            seen.add(cleaned.lower())
    return merged


def _extract_query_facts(text: str) -> dict[str, Any]:
    facts: dict[str, Any] = {"feedstock": {}, "process": {}, "analysis": {}}

    if capacity := _CAPACITY_RE.search(text):
        facts["feedstock"]["capacity_mt_yr"] = float(capacity.group("value").replace(",", ""))

    composition = {
        match.group("polymer").upper(): float(match.group("pct"))
        for match in _COMPOSITION_RE.finditer(text)
    }
    if composition:
        facts["feedstock"]["composition_wt_pct"] = composition

    polymers = [match.group(1).upper() for match in _KNOWN_POLYMER_RE.finditer(text)]
    if polymers:
        facts["feedstock"]["polymers"] = _merge_unique([], polymers)

    if scenario := _SCENARIO_RE.search(text):
        facts["process"]["scenario"] = scenario.group("scenario").upper()

    if energy_case := _ENERGY_CASE_RE.search(text):
        facts["process"]["energy_case"] = energy_case.group("case").upper()

    if solvent := _SOLVENT_RE.search(text):
        solvent_name = solvent.group("solvent").strip(" ,.")
        if solvent_name:
            facts["process"]["solvent"] = solvent_name

    if output_path := _extract_output_dir(text):
        facts["process"]["output_dir"] = output_path

    solvent_mentions: list[str] = []
    if re.search(r"\b(solvents?|solubility|dissolv(?:e|es|ing)?)\b", text, re.IGNORECASE):
        solvent_mentions = _solvent_mentions(text)
        if solvent_mentions:
            facts["process"]["solvent_candidates"] = solvent_mentions[:12]

    if re.search(r"\bsolubility\b", text, re.IGNORECASE):
        subject = _SOLUBILITY_SUBJECT_RE.search(text)
        solvents = solvent_mentions
        if subject and solvents:
            temps = extract_temperatures_c(text)
            start_temp = 25.0 if re.search(r"\broom\s+temp(?:erature)?\b", text, re.IGNORECASE) else (temps[0] if len(temps) >= 2 else 25.0)
            end_temp = temps[-1] if temps else None
            lookup: dict[str, Any] = {
                "polymer": subject.group("polymer").upper(),
                "solvents": _merge_unique([], solvents[:6]),
            }
            if end_temp is not None:
                lookup["temperature_min_c"] = start_temp
                lookup["temperature_max_c"] = end_temp
            facts["analysis"]["last_solubility_lookup"] = lookup

    precipitation_temp = _extract_precipitation_temp(text)
    if precipitation_temp is not None:
        facts["process"]["precipitation_temp_c"] = precipitation_temp

    if "precipitation_temp_c" not in facts["process"]:
        dissolution_temp = _extract_dissolution_temp(text)
        if dissolution_temp is not None:
            facts["process"]["dissolution_temp_c"] = dissolution_temp

    if target_fraction := _TARGET_FRACTION_RE.search(text):
        pct = target_fraction.group("pct") or target_fraction.group("pct_alt")
        facts["process"]["target_plastic_percent"] = float(pct)

    metrics = [label for label, pattern in _METRIC_PATTERNS if pattern.search(text)]
    if metrics:
        facts["analysis"]["metrics"] = metrics

    return facts


def _extract_output_dir(text: str) -> str | None:
    """Extract a user-requested output directory without touching the filesystem."""
    destination = extract_output_destination(text)
    return destination.output_dir if destination is not None else None


def _extract_precipitation_temp(text: str) -> float | None:
    for match in _PRECIP_TEMP_CONTEXT_RE.finditer(text):
        temp = last_temperature_c(match.group(0))
        if temp is not None:
            return temp
    return None


def _extract_dissolution_temp(text: str) -> float | None:
    """Extract an actual process temperature, not a plot/query temperature range."""
    if match := _DISSOLUTION_TEMP_CONTEXT_RE.search(text):
        return float(match.group("value"))
    context_match = re.search(
        r"\b(?:use|using|dissolv(?:e|es|ing|ution)?|operat(?:e|ing)?|process(?:ing)?|stage\s+\d+)"
        r"(?:(?:\d+\.\d+)|[^.?!\n]){0,140}",
        text,
        re.IGNORECASE,
    )
    if context_match:
        temp = last_temperature_c(context_match.group(0))
        if temp is not None:
            return temp
    if (
        _TEMPERATURE_RANGE_CONTEXT_RE.search(text)
        or has_temperature_ceiling(text)
        or (
            re.search(r"\b(?:from|between|range|through|until)\b", text, re.IGNORECASE)
            and extract_temperatures_c(text)
        )
    ):
        return None
    if re.search(r"\b(plot|chart|graph|visualiz(?:e|ation)?|curve|solubility\s+(?:plot|curve|range))\b", text, re.IGNORECASE):
        return None
    temps = extract_temperatures_c(text)
    return temps[0] if temps else None


def update_session_context_from_text(
    context: dict[str, Any],
    text: str,
    *,
    role: str = "user",
) -> dict[str, Any]:
    """Update structured context from a user query or CLI clarification text."""
    updated = dict(context)
    updated.setdefault("feedstock", {})
    updated.setdefault("process", {})
    updated.setdefault("analysis", {})
    facts = _extract_query_facts(text)

    for section in ("feedstock", "process", "analysis"):
        for key, value in facts[section].items():
            if key == "polymers":
                existing = updated[section].get(key) or []
                updated[section][key] = _merge_unique(list(existing), list(value))
            elif key == "solvent_candidates":
                existing = _filter_runtime_solvents(list(updated[section].get(key) or []))
                new_values = _filter_runtime_solvents(list(value))
                updated[section][key] = _merge_unique(existing, new_values)
            elif key == "composition_wt_pct":
                existing = dict(updated[section].get(key) or {})
                existing.update(value)
                updated[section][key] = existing
            else:
                updated[section][key] = value

    if role == "assistant":
        displayed_recommendations = _extract_displayed_solvent_recommendations(text)
        if displayed_recommendations:
            updated["analysis"]["last_solvent_candidate_table"] = displayed_recommendations
            updated["process"]["solvent_candidates"] = list(displayed_recommendations["solvents"])

    if role == "user":
        updated["last_user_query"] = text
    updated["updated_at"] = _utc_now()
    return updated


def _append_limited(existing: list[Any], new_values: list[Any], *, limit: int = 50) -> list[Any]:
    merged = list(existing)
    merged.extend(new_values)
    if len(merged) > limit:
        return merged[-limit:]
    return merged


def _merge_artifacts(existing: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for artifact in [*existing, *artifacts]:
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id") or "")
        if not artifact_id:
            continue
        if artifact_id not in by_id:
            ordered.append(artifact_id)
        by_id[artifact_id] = artifact
    ordered = ordered[-50:]
    return [by_id[artifact_id] for artifact_id in ordered if artifact_id in by_id]


def _artifact_entities(artifact: dict[str, Any]) -> dict[str, Any]:
    entities = artifact.get("entities")
    return entities if isinstance(entities, dict) else {}


def _artifact_data(artifact: dict[str, Any]) -> dict[str, Any]:
    data = artifact.get("data")
    return data if isinstance(data, dict) else {}


def _compact_optimization_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed for follow-up summaries and plotting."""
    analysis_type = str(payload.get("analysis_type") or "").strip()
    allowed = {
        "analysis_type",
        "schema_version",
        "objective",
        "scenario",
        "feed_composition",
        "profit",
        "total_cost",
        "emissions",
        "circularity_score",
        "ce_score",
        "raw_circularity_score",
        "capital_cost",
        "operational_cost",
        "transport_cost",
        "sales",
        "stage1_tech",
        "stage2_tech",
        "stage3_tech",
        "wash1_selection",
        "wash2_selection",
        "optimal_washes",
        "x_metric",
        "y_metric",
        "pareto_payload_path",
        "pareto_slices_payload_path",
    }
    compact = {key: payload[key] for key in allowed if key in payload}
    if analysis_type == "pareto_front":
        points = payload.get("points")
        if isinstance(points, list) and len(points) <= 25:
            compact["points"] = points
        elif payload.get("pareto_payload_path"):
            compact["points"] = []
    return compact


def update_session_context_from_direct_metadata(
    context: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist route decisions and artifact frames from direct-tool turns."""
    if not isinstance(metadata, dict):
        return context

    updated = dict(context)
    updated.setdefault("feedstock", {})
    updated.setdefault("process", {})
    updated.setdefault("analysis", {})
    updated.setdefault("artifacts", [])
    updated.setdefault("route_decisions", [])
    updated.setdefault("run_ledgers", [])

    route_decision = metadata.get("strap_route_decision")
    if isinstance(route_decision, dict) and route_decision:
        updated["route_decisions"] = _append_limited(
            list(updated.get("route_decisions") or []),
            [route_decision],
        )

    run_ledger = metadata.get("strap_run_ledger")
    if isinstance(run_ledger, dict) and run_ledger:
        updated["run_ledgers"] = _append_limited(
            list(updated.get("run_ledgers") or []),
            [run_ledger],
        )

    artifacts = metadata.get("strap_artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
    if artifacts:
        updated["artifacts"] = _merge_artifacts(list(updated.get("artifacts") or []), artifacts)

    for artifact in artifacts:
        artifact_type = str(artifact.get("type") or "")
        entities = _artifact_entities(artifact)
        data = _artifact_data(artifact)

        if artifact_type == "solvent_candidate_table":
            polymer = entities.get("polymer")
            solvents = list(entities.get("solvents") or artifact.get("row_order") or [])
            if solvents:
                updated["process"]["solvent_candidates"] = _merge_unique(
                    list(updated["process"].get("solvent_candidates") or []),
                    [str(solvent) for solvent in solvents],
                )
            if polymer and solvents:
                updated["analysis"]["last_solvent_candidate_table"] = {
                    "artifact_id": artifact.get("artifact_id"),
                    "polymer": str(polymer),
                    "solvents": [str(solvent) for solvent in solvents],
                }

        elif artifact_type == "solubility_table":
            polymer = entities.get("polymer")
            solvents = list(entities.get("solvents") or artifact.get("row_order") or [])
            if polymer and solvents:
                lookup: dict[str, Any] = {
                    "artifact_id": artifact.get("artifact_id"),
                    "polymer": str(polymer),
                    "solvents": [str(solvent) for solvent in solvents],
                }
                if data.get("temperature_min_c") is not None:
                    lookup["temperature_min_c"] = float(data["temperature_min_c"])
                if data.get("temperature_max_c") is not None:
                    lookup["temperature_max_c"] = float(data["temperature_max_c"])
                updated["analysis"]["last_solubility_lookup"] = lookup

        elif artifact_type == "plot_artifact":
            polymer = entities.get("polymer")
            polymers = list(entities.get("polymers") or ([polymer] if polymer else []))
            solvents = list(entities.get("solvents") or artifact.get("row_order") or [])
            plot_type = str(entities.get("plot_type") or "")
            if plot_type.startswith("optimization_"):
                plot: dict[str, Any] = {
                    "artifact_id": artifact.get("artifact_id"),
                    "plot_type": plot_type,
                }
                if data.get("path"):
                    plot["path"] = str(data["path"])
                if data.get("paths"):
                    plot["paths"] = [str(path) for path in data["paths"] if str(path)]
                if data.get("output_dir"):
                    plot["output_dir"] = str(data["output_dir"])
                updated["analysis"]["last_plot_artifact"] = plot
                continue
            if polymers and solvents:
                plot: dict[str, Any] = {
                    "artifact_id": artifact.get("artifact_id"),
                    "plot_type": str(entities.get("plot_type") or "solubility_vs_temperature"),
                    "polymer": str(polymers[0]),
                    "polymers": [str(polymer_item) for polymer_item in polymers],
                    "solvents": [str(solvent) for solvent in solvents],
                }
                if data.get("temperature_min_c") is not None:
                    plot["temperature_min_c"] = float(data["temperature_min_c"])
                if data.get("temperature_max_c") is not None:
                    plot["temperature_max_c"] = float(data["temperature_max_c"])
                if data.get("path"):
                    plot["path"] = str(data["path"])
                if data.get("output_dir"):
                    plot["output_dir"] = str(data["output_dir"])
                updated["analysis"]["last_plot_artifact"] = plot

        elif artifact_type in {"optimization_point_result", "optimization_pareto_front"}:
            payload = data.get("payload") if isinstance(data.get("payload"), dict) else None
            if payload:
                updated["analysis"]["last_optimization_result"] = {
                    "artifact_id": artifact.get("artifact_id"),
                    "artifact_type": artifact_type,
                    "payload": _compact_optimization_payload(payload),
                }

    updated["updated_at"] = _utc_now()
    return updated


def build_session_context_block(context: dict[str, Any]) -> str:
    """Build a compact context block for injection into the next turn."""
    lines = [
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):"
    ]
    feedstock = context.get("feedstock") or {}
    process = context.get("process") or {}
    analysis = context.get("analysis") or {}

    feed_parts = []
    if capacity := feedstock.get("capacity_mt_yr"):
        feed_parts.append(f"capacity={float(capacity):,.0f} MT/yr")
    if composition := feedstock.get("composition_wt_pct"):
        comp = ", ".join(f"{polymer}={pct:g}%" for polymer, pct in sorted(composition.items()))
        feed_parts.append(f"composition={comp}")
    elif polymers := feedstock.get("polymers"):
        feed_parts.append("polymers=" + ", ".join(polymers))
    if feed_parts:
        lines.append("- Feedstock: " + "; ".join(feed_parts))

    process_parts = []
    for key, label in (
        ("scenario", "scenario"),
        ("energy_case", "energy_case"),
        ("solvent", "solvent"),
        ("solvent_candidates", "solvent_candidates"),
        ("output_dir", "output_dir"),
        ("dissolution_temp_c", "dissolution_temp_c"),
        ("precipitation_temp_c", "precipitation_temp_c"),
        ("target_plastic_percent", "target_plastic_percent"),
    ):
        if key in process:
            value = process[key]
            if isinstance(value, float):
                value = f"{value:g}"
            elif isinstance(value, list):
                if key == "solvent_candidates":
                    value = _filter_runtime_solvents([str(item) for item in value])
                    if not value:
                        continue
                value = ", ".join(str(item) for item in value[:12])
            process_parts.append(f"{label}={value}")
    if process_parts:
        lines.append("- Process: " + "; ".join(process_parts))

    if metrics := analysis.get("metrics"):
        lines.append("- Analysis metrics: " + ", ".join(metrics))

    if opt_result := analysis.get("last_optimization_result"):
        if isinstance(opt_result, dict) and isinstance(opt_result.get("payload"), dict):
            payload = dict(opt_result["payload"])
            payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            if len(payload_json) <= 1800:
                parts = []
                if artifact_id := opt_result.get("artifact_id"):
                    parts.append(f"artifact_id={artifact_id}")
                if analysis_type := payload.get("analysis_type"):
                    parts.append(f"analysis_type={analysis_type}")
                parts.append(f"payload_json={payload_json}")
                lines.append("- Last optimization result: " + "; ".join(parts))

    if candidate_table := analysis.get("last_solvent_candidate_table"):
        if isinstance(candidate_table, dict):
            parts = []
            polymers = candidate_table.get("polymers")
            if isinstance(polymers, list) and polymers:
                parts.append("polymers=" + ", ".join(str(item) for item in polymers[:8]))
            elif polymer := candidate_table.get("polymer"):
                parts.append(f"polymer={polymer}")
            if solvents := candidate_table.get("solvents"):
                solvents = _filter_runtime_solvents([str(item) for item in solvents])
                if solvents:
                    parts.append("solvents=" + ", ".join(str(item) for item in solvents[:12]))
            if artifact_id := candidate_table.get("artifact_id"):
                parts.append(f"artifact_id={artifact_id}")
            if parts:
                lines.append("- Last solvent candidates: " + "; ".join(parts))

    if lookup := analysis.get("last_solubility_lookup"):
        if isinstance(lookup, dict):
            parts = []
            if polymer := lookup.get("polymer"):
                parts.append(f"polymer={polymer}")
            if solvents := lookup.get("solvents"):
                parts.append("solvents=" + ", ".join(str(item) for item in solvents[:6]))
            t_min = lookup.get("temperature_min_c")
            t_max = lookup.get("temperature_max_c")
            if t_min is not None and t_max is not None:
                parts.append(f"temperature_range={float(t_min):g}-{float(t_max):g} C")
            if parts:
                lines.append("- Last solubility lookup: " + "; ".join(parts))

    if plot := analysis.get("last_plot_artifact"):
        if isinstance(plot, dict):
            parts = []
            if plot_type := plot.get("plot_type"):
                parts.append(f"plot_type={plot_type}")
            polymers = plot.get("polymers")
            if isinstance(polymers, list) and len(polymers) > 1:
                parts.append("polymers=" + ", ".join(str(item) for item in polymers[:8]))
            elif polymer := plot.get("polymer"):
                parts.append(f"polymer={polymer}")
            if solvents := plot.get("solvents"):
                parts.append("solvents=" + ", ".join(str(item) for item in solvents[:12]))
            t_min = plot.get("temperature_min_c")
            t_max = plot.get("temperature_max_c")
            if t_min is not None and t_max is not None:
                parts.append(f"temperature_range={float(t_min):g}-{float(t_max):g} C")
            if path := plot.get("path"):
                parts.append(f"path={path}")
            if output_dir := plot.get("output_dir"):
                parts.append(f"output_dir={output_dir}")
            if parts:
                lines.append("- Last plot artifact: " + "; ".join(parts))

    if len(lines) == 1:
        return ""
    block = "\n".join(lines)
    if len(block) > _MAX_CONTEXT_BLOCK_CHARS:
        return block[: _MAX_CONTEXT_BLOCK_CHARS - 3].rstrip() + "..."
    return block


def inject_session_context(user_input: str, context: dict[str, Any]) -> str:
    """Prepend compact session context to a user input when context exists."""
    block = build_session_context_block(context)
    if not block:
        return user_input
    return f"{block}\n\nUser request:\n{user_input}"


def should_inject_session_context(user_input: str, context: dict[str, Any]) -> bool:
    """Return whether compact context is useful enough to inject this turn."""
    if not build_session_context_block(context):
        return False
    analysis = context.get("analysis") or {}
    has_last_solubility_lookup = bool(analysis.get("last_solubility_lookup"))
    has_artifact_context = bool(
        analysis.get("last_solvent_candidate_table")
        or analysis.get("last_plot_artifact")
        or context.get("artifacts")
    )
    if has_artifact_context and _ARTIFACT_FOLLOWUP_RE.search(user_input):
        return True
    if has_last_solubility_lookup and _SOLUBILITY_PLOT_FOLLOWUP_RE.search(user_input):
        return True
    if _FOLLOWUP_RE.search(user_input):
        return True
    if not _DOMAIN_QUERY_RE.search(user_input):
        return False
    has_feedstock = bool((context.get("feedstock") or {}).get("composition_wt_pct"))
    has_capacity = bool((context.get("feedstock") or {}).get("capacity_mt_yr"))
    has_solvents = bool((context.get("process") or {}).get("solvent_candidates"))
    query_has_composition = bool(_COMPOSITION_RE.search(user_input))
    query_has_capacity = bool(_CAPACITY_RE.search(user_input))
    if has_solvents and re.search(r"\b(these|those|each|solvents?|solubility)\b", user_input, re.IGNORECASE):
        return True
    return (has_feedstock and not query_has_composition) or (has_capacity and not query_has_capacity)
