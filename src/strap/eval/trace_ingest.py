"""Turn logged agent traces into scorable RL Episodes — offline, no inference.

Last night's live multistage runs are logged trajectories from a fixed policy:
the classic offline / off-policy RL setting. The expensive part (the model
deciding routes and shortlisting solvents) is already paid for and sits in the
LangSmith traces; the deterministic v10 engines that the reward model and
best-of-N run on are free and unlimited. So we can score those real
trajectories, evaluate the harness changes off-policy, warm-start a bandit, and
run trajectory-rooted counterfactual exploration — all with **zero new model
calls**.

Two stages, deliberately split so everything downstream is reproducible offline:

1. ``harvest_trace_cache`` — the ONLY network step. Reads our *own* logged
   LangSmith traces (no model inference) and materializes per-run structured
   results, token ledgers, tool-call counts, config, and final answers into a
   local JSON cache. Mirrors the case-study "replay from cache" rule.
2. ``load_trace_cache`` / ``episodes_from_trace_cache`` — pure offline. Turn the
   cache into canonical :class:`~strap.eval.reward.Episode` objects in the exact
   shape the scorers expect (same as ``strap.eval.query_suite``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strap.eval.reward import Episode

_STRUCTURED_RE = re.compile(r"<STRUCTURED_RESULT>\s*(\{.*?\})\s*</STRUCTURED_RESULT>", re.DOTALL)


# ---------------------------------------------------------------------------
# Harvest (network: reads our own logs only, no model inference)
# ---------------------------------------------------------------------------

@dataclass
class TraceRunSpec:
    """One logged run to harvest, plus offline-known metadata."""

    run_id: str
    label: str                       # short human tag, e.g. "run3_clean_staged"
    config_note: str = ""            # which harness fixes were live (for the fix-evaluation axis)
    order: int = 0                   # chronological order across the fix sequence


def _coerce_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(value or "")


def _structured_results_from_runs(runs: list) -> dict[str, dict]:
    """Latest STRUCTURED_RESULT per agent, parsed from task/tool run outputs."""
    found: dict[str, dict] = {}
    for run in sorted(runs, key=lambda r: r.start_time or 0):
        if run.run_type != "tool":
            continue
        blob = json.dumps(run.outputs or {}, default=str)
        for match in _STRUCTURED_RE.finditer(blob.replace("\\n", "\n").replace('\\"', '"')):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            agent = payload.get("agent")
            if agent:
                found[agent] = payload  # keep the latest
    return found


def _ledger_by_agent(runs: list) -> dict[str, dict[str, int]]:
    """Per-subagent token + llm-call ledger, grouped by the Subagent chain
    that is each llm run's ancestor."""
    sub_ids = {r.id: r.name.split(":", 1)[1].strip()
               for r in runs if (r.name or "").startswith("Subagent:")}
    ledger: dict[str, dict[str, int]] = {}
    for run in runs:
        if run.run_type != "llm":
            continue
        agent = None
        for parent in (run.parent_run_ids or []):
            if parent in sub_ids:
                agent = sub_ids[parent]
                break
        if agent is None:
            agent = "orchestrator"
        entry = ledger.setdefault(agent, {"prompt_tokens": 0, "output_tokens": 0, "llm_calls": 0})
        entry["prompt_tokens"] += int(run.prompt_tokens or 0)
        entry["output_tokens"] += int(run.completion_tokens or 0)
        entry["llm_calls"] += 1
    return ledger


def _tool_calls_by_agent(runs: list) -> dict[str, int]:
    sub_ids = {r.id: r.name.split(":", 1)[1].strip()
               for r in runs if (r.name or "").startswith("Subagent:")}
    counts: dict[str, int] = {}
    for run in runs:
        if run.run_type != "tool":
            continue
        agent = "orchestrator"
        for parent in (run.parent_run_ids or []):
            if parent in sub_ids:
                agent = sub_ids[parent]
                break
        counts[agent] = counts.get(agent, 0) + 1
    return counts


def harvest_trace_cache(specs: list[TraceRunSpec], out_path: Path) -> dict:
    """Read our own logged LangSmith traces into a local offline cache.

    This is the only step that touches the network, and it reads *logged runs*
    (our data) — never the model. Requires ``LANGSMITH_API_KEY``. Raises with a
    clear message if the client or a trace is unavailable so a caller can fall
    back to a previously committed cache.
    """
    try:
        from langsmith import Client
    except ImportError as exc:  # pragma: no cover - env guard
        raise RuntimeError("langsmith not installed; cannot harvest (use the committed cache).") from exc

    client = Client()
    cache: dict[str, Any] = {"schema": "dissolve.trace_rl.v1", "runs": []}
    for spec in sorted(specs, key=lambda s: s.order):
        runs = list(client.list_runs(trace_id=spec.run_id))
        if not runs:
            raise RuntimeError(f"no runs found for trace {spec.run_id} ({spec.label})")
        root = next((r for r in runs if str(r.id) == spec.run_id), None)
        query = ""
        final_answer = ""
        if root is not None:
            query = _coerce_content((root.inputs or {}).get("message", ""))
            final_answer = _coerce_content((root.outputs or {}).get("output", ""))
        # fall back to the human/last-ai messages if the root chain omitted them
        if not query:
            humans = [r for r in runs if r.run_type == "llm"]
            query = spec.label
        structured = _structured_results_from_runs(runs)
        cache["runs"].append({
            "run_id": spec.run_id,
            "label": spec.label,
            "config_note": spec.config_note,
            "order": spec.order,
            "query": query,
            "structured_results": structured,
            "ledger_by_agent": _ledger_by_agent(runs),
            "tool_calls_by_agent": _tool_calls_by_agent(runs),
            "final_answer": final_answer,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=2))
    return cache


# ---------------------------------------------------------------------------
# Load + normalize (pure offline)
# ---------------------------------------------------------------------------

def load_trace_cache(path: Path) -> dict:
    return json.loads(Path(path).read_text())


@dataclass
class TraceEpisode:
    """An Episode plus the provenance needed for the fix-evaluation axis."""

    episode: Episode
    run_label: str
    agent: str
    order: int
    config_note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_separation_result(payload: dict) -> dict[str, Any]:
    """Map a separation-engineer STRUCTURED_RESULT to the canonical scorer shape."""
    steps = []
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        steps.append({
            "polymer": step.get("polymer") or step.get("target"),
            "solvent": step.get("solvent"),
            "temperature_c": step.get("temperature_c"),
            "selectivity_pct": step.get("selectivity_pct", step.get("selectivity")),
        })
    result = {
        "polymers": payload.get("polymers"),
        "best_sequence": payload.get("best_sequence"),
        "steps": steps,
        "polymer_solvent_candidates": payload.get("polymer_solvent_candidates"),
        "top_k_sequences": payload.get("top_k_sequences"),
        "min_selectivity": _num(payload.get("min_selectivity_pct")),
    }
    return result


def _normalize_optimization_result(payload: dict) -> dict[str, Any]:
    """Map an optimization-engineer STRUCTURED_RESULT to the canonical shape."""
    analysis = str(payload.get("analysis_type") or "").strip().lower()
    points = payload.get("points") or []
    frontier = points if isinstance(points, list) else []
    result: dict[str, Any] = {
        "polymers": payload.get("polymers") or ["PE", "EVOH"],
        "analysis_type": analysis,
        "frontier": frontier,
        "points": frontier,
        "n_points_feasible": payload.get("n_points_feasible"),
    }
    if analysis == "infeasible":
        result["infeasible"] = True
        result["no_data"] = True
        result["failure_reason"] = payload.get("failure_reason")
    return result


def episodes_from_trace_cache(cache: dict) -> list[TraceEpisode]:
    """Canonical Episodes from the harvested cache (offline)."""
    episodes: list[TraceEpisode] = []
    for run in cache.get("runs", []):
        query = run.get("query") or run.get("label", "")
        structured = run.get("structured_results") or {}
        ledger = run.get("ledger_by_agent") or {}
        tool_counts = run.get("tool_calls_by_agent") or {}
        for agent, payload in structured.items():
            if agent == "separation-engineer":
                result = _normalize_separation_result(payload)
                budget = 12
            elif agent == "optimization-engineer":
                result = _normalize_optimization_result(payload)
                budget = 10
            else:
                result = dict(payload)
                budget = 12
            agent_ledger = ledger.get(agent, {})
            episode = Episode(
                query=query,
                result=result,
                context={
                    "polymers": result.get("polymers") or ["PE", "EVOH"],
                    "tool_call_budget": budget,
                },
                ledger={
                    "tool_calls": tool_counts.get(agent, 0),
                    "prompt_tokens": agent_ledger.get("prompt_tokens"),
                    "output_tokens": agent_ledger.get("output_tokens"),
                },
                label=f"{run.get('label')}::{agent}",
            )
            episodes.append(TraceEpisode(
                episode=episode,
                run_label=run.get("label", ""),
                agent=agent,
                order=int(run.get("order", 0)),
                config_note=run.get("config_note", ""),
                extras={"raw_structured": payload},
            ))
    return episodes
