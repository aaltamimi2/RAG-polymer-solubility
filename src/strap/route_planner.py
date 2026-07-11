"""Planner-first routing core for the DISSOLVE orchestrator.

One planner call produces a validated :class:`RoutePlan` per user query.
Every routing consumer — advisory hints, tool guards, the direct-tool fast
path, the typed runtime, and the output verifier — reads that same plan, so
intent is decided exactly once.

Decision order:

1. ``RoutePlanner.plan()`` asks the configured backend (an LLM prompted with
   the capability catalog) for a plan payload.
2. The payload is validated structurally against the capability graph:
   unknown specialists are dropped, dependency edges are checked and
   cycle-broken, explicit exclusions are enforced.
3. If the backend is missing, errors, or returns an unusable payload, the
   deterministic keyword fallback (``fallback_route_plan``) is used instead.

Keyword/regex matching never overrides a valid planner decision — it exists
only as the offline fallback path.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

RouteMode = str  # "direct" | "orchestrator" | "specialists"
_VALID_MODES = ("direct", "orchestrator", "specialists")
_VALID_CONFIDENCE = ("high", "medium", "low")
_MAX_PLAN_STEPS = 8
_ACTIVE_PLAN_LIMIT = 64

# Specialists whose job is retrieving/summarizing external or indexed
# knowledge. Domain tokens (TEA, HSP, Pareto...) inside such requests must
# not divert them to numeric pipelines — see typed-runtime gating.
RESEARCH_SUBAGENTS = frozenset({
    "scholar-researcher",
    "patent-researcher",
    "rag-analyst",
})


# ---------------------------------------------------------------------------
# Plan model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteStep:
    """One specialist dispatch in a route plan."""

    subagent: str
    objective: str = ""
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutePlan:
    """Validated routing decision for a single user query."""

    query: str
    mode: RouteMode
    steps: tuple[RouteStep, ...] = ()
    excluded_subagents: tuple[str, ...] = ()
    confidence: str = "medium"
    rationale: str = ""
    source: str = "planner"  # "planner" | "fallback"
    validation_notes: tuple[str, ...] = ()
    # Typed artifact names the planner identified as the user's deliverables
    # (e.g. "biosteam_tea_lca_result"). Consumed by the typed runtime as
    # authoritative intent so keyword detection is only a fallback. Kept as
    # opaque strings here; consumers filter against their artifact catalog.
    deliverables: tuple[str, ...] = ()

    @property
    def is_direct(self) -> bool:
        return self.mode == "direct"

    @property
    def is_specialists(self) -> bool:
        return self.mode == "specialists" and bool(self.steps)

    def subagent_names(self) -> list[str]:
        return [step.subagent for step in self.steps]

    def dependency_map(self) -> dict[str, set[str]]:
        return {step.subagent: set(step.depends_on) for step in self.steps}

    def to_rules(self) -> list[dict]:
        """Project plan steps onto the classifier rule-dict contract.

        Downstream workflow machinery (progress tracking, guards, handoffs)
        consumes ``{"subagent", "description", ...}`` dicts; enrich them with
        the plan's objective and dependencies.
        """
        from .routing_classifier import ROUTING_RULES

        rules_by_name = {rule["subagent"]: rule for rule in ROUTING_RULES}
        projected: list[dict] = []
        for step in self.steps:
            rule = rules_by_name.get(step.subagent)
            if rule is None:
                continue
            projected.append({
                **rule,
                "objective": step.objective,
                "depends_on": tuple(step.depends_on),
            })
        return projected

    def explain(self) -> dict:
        """Compact, log/JSON-friendly description of the decision."""
        return {
            "query": self.query,
            "mode": self.mode,
            "steps": [
                {
                    "subagent": step.subagent,
                    "objective": step.objective,
                    "depends_on": list(step.depends_on),
                }
                for step in self.steps
            ],
            "excluded_subagents": list(self.excluded_subagents),
            "confidence": self.confidence,
            "source": self.source,
            "rationale": self.rationale,
            "validation_notes": list(self.validation_notes),
            "deliverables": list(self.deliverables),
        }


# ---------------------------------------------------------------------------
# Capability catalog + planner prompt
# ---------------------------------------------------------------------------

def build_route_catalog() -> str:
    """Render the specialist catalog from capability metadata for the prompt."""
    from .routing_classifier import PLANNING_GRAPH, ROUTING_RULES

    lines: list[str] = []
    for rule in ROUTING_RULES:
        name = rule["subagent"]
        parts = [f"- {name}: {rule['description']}"]
        node = PLANNING_GRAPH.nodes.get(name) if PLANNING_GRAPH else None
        if node is not None:
            if node.goals:
                parts.append(f"  goals: {', '.join(node.goals)}")
            if node.produces:
                parts.append(f"  produces: {', '.join(node.produces)}")
            consumes = [c for c in node.consumes if c != "generic.context.v1"]
            if consumes:
                parts.append(f"  consumes: {', '.join(consumes)}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


_PLANNER_PROMPT_TEMPLATE = """\
You are the routing planner for a polymer dissolution / plastics recycling \
analysis system. Read the user's request and produce a routing plan as JSON.

The orchestrator's own core tools already handle simple lookups directly: \
listing polymers/solvents, solvent candidates for one or more polymers \
(including with a simple temperature ceiling like "below 100 C"), point or \
range solubility values for a named polymer-solvent pair, solvent properties \
(boiling point, LogP), solvent safety cards for named solvents (flash point, \
GHS hazards, exposure data), plotting solubility vs temperature for named \
polymer-solvent pairs, and re-plotting or tabulating the previous lookup. \
Core tools do NOT include Hansen/HSP parameter data: any HSP-based listing, \
lookup, or screening — even a simple one — belongs to statistics-ml.

Specialists available for delegation:
{catalog}

Respond with ONLY a JSON object:
{{
  "mode": "direct" | "orchestrator" | "specialists",
  "steps": [{{"subagent": "<name>", "objective": "<short imperative>", "depends_on": ["<name>", ...]}}],
  "excluded_subagents": ["<name>", ...],
  "deliverables": ["<typed artifact name>", ...],
  "confidence": "high" | "medium" | "low",
  "rationale": "<one short sentence>"
}}

"deliverables" names the typed artifacts the user is asking for, chosen ONLY
from this catalog (leave the list empty when none clearly applies):
biosteam_tea_lca_result, biosteam_tea_lca_plot, optimization_point_result,
optimization_pareto_front, optimization_pareto_landscape,
optimization_pareto_slices, optimization_pareto_slices_plot,
separation_dp_state_map, separation_tree_plot, separation_selectivity_heatmap,
solvent_safety_card, solvent_safety_comparison, hsp_single_pair_summary,
hsp_red_heatmap

Modes:
- "direct": the request is one of the simple core-tool lookups listed above. No steps.
- "orchestrator": conversational, meta, or ambiguous requests the orchestrator \
should answer or clarify itself. No steps.
- "specialists": everything needing specialist analysis. Give ordered steps; \
use depends_on when a step needs another step's results (e.g. economics after \
a separation route exists). Independent steps may run in parallel.

Routing principles, in priority order:
1. Route by the user's PRIMARY DELIVERABLE, never by keyword. A request to \
find/summarize papers, patents, or indexed documents is research even when it \
mentions TEA, LCA, HSP, BioSTEAM, Pareto, or other domain vocabulary.
2. Requests about the system's indexed/local document corpus (RAG, "our \
documents", retrieved chunks, citations from the corpus) go to rag-analyst; \
published-literature searches go to scholar-researcher; patents to \
patent-researcher.
3. Choose the smallest sufficient specialist set. Do not add safety, \
economics, optimization, or visualization steps the user did not ask for.
4. Honor explicit negations: if the user says to skip/omit an analysis \
("no TEA", "without safety scoring"), list that specialist in \
excluded_subagents and do not include it in steps.
5. Temperature-dependent solubility questions, selectivity, separation \
sequences, and process design belong to separation-engineer. Hansen/HSP/RED \
screening and statistical/ML/thermal predictions belong to statistics-ml. \
Cost/TEA/LCA/MSP/CAPEX/OPEX/GWP simulation belongs to biosteam-analyst. \
Superstructure/pathway/Pareto optimization belongs to optimization-engineer.
6. If the user asks for a figure/plot of specialist results, add \
visualization-specialist as a dependent final step — except when the \
producing specialist already renders that artifact itself.
7. When genuinely unsure between "direct" and a single specialist, prefer \
the specialist with confidence "low".

A [SESSION CONTEXT] block, when present, lists what this conversation has
already produced. Use it to route follow-ups:
8. If the requested information can be answered from analyses that already
completed (reading values off a prior result, comparing/ranking existing
options, "which one was best", explaining a produced figure), use mode
"orchestrator" — the orchestrator answers from stored session results. Do
NOT re-dispatch specialists whose results already exist unless the user
changes parameters or explicitly asks to redo the analysis.
9. If the follow-up asks for a NEW analysis stage on top of existing results
("now run TEA on that route", "add safety scoring for those solvents"), plan
ONLY the new specialists; upstream results are passed to them automatically.
Do not re-plan the completed upstream stages.

Examples:

[SESSION CONTEXT]
Previous user requests:
1. "Generate a separation state map for all LDPE/EVOH/PET sequences under 100 C."
Already produced this session:
- separation-engineer: completed (has: best_sequence, steps, top_k_sequences, polymers)
[CURRENT REQUEST]
From that state map, which separation sequence maximizes predicted separation efficiency? Use the structured sequence results rather than re-describing the feedstock.
A: {{"mode": "orchestrator", "steps": [], "excluded_subagents": [], "deliverables": [], "confidence": "high", "rationale": "answerable from the completed separation results already in session"}}

[SESSION CONTEXT]
Previous user requests:
1. "Design a separation sequence for LDPE/PP below 120 C."
Already produced this session:
- separation-engineer: completed (has: best_sequence, steps, solvent_mapping)
[CURRENT REQUEST]
Now estimate MSP and GWP for that route.
A: {{"mode": "specialists", "steps": [{{"subagent": "biosteam-analyst", "objective": "estimate MSP and GWP for the already-designed route", "depends_on": []}}], "excluded_subagents": [], "deliverables": ["biosteam_tea_lca_result"], "confidence": "high", "rationale": "new economics stage on the existing route; separation already completed"}}

Q: "What solvents dissolve LDPE?"
A: {{"mode": "direct", "steps": [], "excluded_subagents": [], "confidence": "high", "rationale": "simple solvent-candidate lookup"}}

Q: "Find recent journal articles on techno-economic analysis of solvent-based polyolefin recycling."
A: {{"mode": "specialists", "steps": [{{"subagent": "scholar-researcher", "objective": "find recent TEA-focused articles on solvent-based polyolefin recycling", "depends_on": []}}], "excluded_subagents": [], "confidence": "high", "rationale": "literature search is the deliverable; TEA is just the topic"}}

Q: "What do our indexed PDFs say about EVOH barrier layers? Cite the retrieved chunks."
A: {{"mode": "specialists", "steps": [{{"subagent": "rag-analyst", "objective": "retrieve and cite indexed-corpus passages on EVOH barrier layers", "depends_on": []}}], "excluded_subagents": [], "confidence": "high", "rationale": "local corpus retrieval with citations"}}

Q: "Design a separation sequence for an LDPE/PP/EVOH stream below 120 C, then estimate MSP and GWP for the route."
A: {{"mode": "specialists", "steps": [{{"subagent": "separation-engineer", "objective": "design the LDPE/PP/EVOH separation sequence below 120 C", "depends_on": []}}, {{"subagent": "biosteam-analyst", "objective": "estimate MSP and GWP for the designed route", "depends_on": ["separation-engineer"]}}], "excluded_subagents": [], "deliverables": ["biosteam_tea_lca_result"], "confidence": "high", "rationale": "route design feeds economics"}}

Q: "Optimize the processing pathway for maximum profit. Do not run TEA or LCA."
A: {{"mode": "specialists", "steps": [{{"subagent": "optimization-engineer", "objective": "optimize the processing pathway for maximum profit", "depends_on": []}}], "excluded_subagents": ["biosteam-analyst"], "deliverables": ["optimization_point_result"], "confidence": "high", "rationale": "optimization only; TEA/LCA explicitly excluded"}}

Q: "Screen PS vs PVC solvent selectivity with Hansen parameters and show the RED matrix."
A: {{"mode": "specialists", "steps": [{{"subagent": "statistics-ml", "objective": "HSP/RED screening of PS vs PVC with matrix output", "depends_on": []}}], "excluded_subagents": [], "deliverables": ["hsp_red_heatmap"], "confidence": "high", "rationale": "explicit Hansen/RED screening"}}

Q: "Plot the solubility of LDPE, EVOH, and PET in dodecane and o-xylene from 25 to 100 C; save the figures and structured data."
A: {{"mode": "specialists", "steps": [{{"subagent": "visualization-specialist", "objective": "multi-polymer solubility-vs-temperature plots for dodecane and o-xylene", "depends_on": []}}], "excluded_subagents": [], "confidence": "high", "rationale": "plotting is the deliverable; visualization tools pull solubility data directly, no separation analysis requested"}}

Q: "thanks, that looks right"
A: {{"mode": "orchestrator", "steps": [], "excluded_subagents": [], "confidence": "high", "rationale": "conversational"}}

A [PLAN REVISION REQUEST] block, when present, means a step of the active plan
ended in a state (failure, step-budget exhaustion, infeasible result) that may
invalidate the remaining steps. Rules for revisions:
10. Produce a corrected FULL plan for what still needs to happen to answer the
user's request from here. Do not include steps that already completed
successfully — their results are retained and passed downstream automatically.
11. Re-dispatching the failed specialist is allowed only when the outcome
suggests different instructions would succeed (e.g. narrow the scope). If the
outcome is a physical/data infeasibility, do NOT retry the same work: either
pivot to a specialist that can still add value, or return mode "orchestrator"
so the orchestrator synthesizes an honest final answer (including the
infeasibility and its suggested relaxation) from the results already produced.

[PLAN REVISION REQUEST]
Prior plan for this request:
1. separation-engineer — shortlist solvents [completed]
2. optimization-engineer — Pareto optimization on the shortlist [FAILED: infeasible — no candidate pair could be evaluated]
[CURRENT REQUEST]
For the PE/EVOH feed, shortlist solvents then run the cost-emissions Pareto and report the knee point.
A: {{"mode": "orchestrator", "steps": [], "excluded_subagents": [], "deliverables": [], "confidence": "high", "rationale": "optimization is infeasible for the produced shortlist; synthesize the honest infeasibility answer with the suggested relaxation instead of retrying"}}
"""

_prompt_cache: str | None = None


def build_planner_system_prompt() -> str:
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = _PLANNER_PROMPT_TEMPLATE.format(catalog=build_route_catalog())
    return _prompt_cache


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class RoutePlannerBackend(Protocol):
    """Produce a raw plan payload for a query, or None on failure."""

    def __call__(self, query_text: str) -> dict | str | None: ...


def extract_json_payload(text: str) -> dict | None:
    """Tolerantly extract one JSON object from model output."""
    if not text:
        return None
    candidates: list[str] = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        for index in range(brace_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[brace_start:index + 1])
                    break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


class LLMRoutePlannerBackend:
    """Plan payloads from a chat model, with one malformed-output retry."""

    def __init__(self, model: "BaseChatModel") -> None:
        self._model = model

    def __call__(self, query_text: str, session_digest: str | None = None) -> dict | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = SystemMessage(content=build_planner_system_prompt())
        if session_digest:
            human_content = f"{session_digest}\n[CURRENT REQUEST]\n{query_text}"
        else:
            human_content = query_text
        messages = [system, HumanMessage(content=human_content)]
        for attempt in range(2):
            try:
                result = self._model.invoke(messages)
            except Exception:
                logger.warning("route_planner: backend model call failed", exc_info=True)
                return None
            text = _message_text(result.content)
            payload = extract_json_payload(text)
            if payload is not None:
                return payload
            logger.warning(
                "route_planner: unparseable plan payload (attempt %d): %s",
                attempt + 1,
                text[:200],
            )
            messages = [
                system,
                HumanMessage(content=human_content),
                result,
                HumanMessage(
                    content="That was not parseable JSON. Respond again with ONLY the JSON object."
                ),
            ]
        return None


# ---------------------------------------------------------------------------
# Session digest — what this conversation already produced
# ---------------------------------------------------------------------------

_DIGEST_MAX_PRIOR_QUERIES = 3
_DIGEST_MAX_RESULT_FIELDS = 5
_DIGEST_MAX_CHARS = 1400
_STRUCTURED_RESULT_DIGEST_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(\{.*?\})\s*</STRUCTURED_RESULT>", re.DOTALL
)


def build_session_digest(messages: list | None) -> str | None:
    """Deterministic one-block summary of prior turns for the planner.

    Only history BEFORE the latest user message is summarized, so the digest
    is stable for every planner consumer across the whole current turn.
    Includes: prior user requests, completed/failed specialist runs (with
    salient structured-result fields), typed-runtime executions, and direct
    fast-path lookups. Returns None when there is no prior history.
    """
    if not messages:
        return None

    last_human_index = None
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "type", None) == "human":
            last_human_index = index
            break
    if not last_human_index:  # None or 0 — no history before the current turn
        return None
    history = messages[:last_human_index]

    prior_queries: list[str] = []
    specialist_lines: list[str] = []
    runtime_lines: list[str] = []
    task_subagents: dict[str, str] = {}  # tool_call_id -> subagent

    for message in history:
        message_type = getattr(message, "type", None)
        if message_type == "human":
            content = message.content if isinstance(message.content, str) else str(message.content)
            content = " ".join(content.split())
            if content:
                prior_queries.append(content[:160])
            continue

        if message_type == "ai":
            kwargs = getattr(message, "additional_kwargs", {}) or {}
            origin = kwargs.get("strap_origin")
            if origin == "direct_tool_fast_path":
                runtime_lines.append(
                    f"- direct lookup answered via {kwargs.get('strap_tool_name') or 'core tools'}"
                )
            elif origin == "typed_runtime":
                status = kwargs.get("strap_typed_runtime_status") or "executed"
                workflow = kwargs.get("strap_workflow_id") or kwargs.get("strap_plan_id") or "typed workflow"
                runtime_lines.append(f"- typed runtime {status}: {workflow}")
            for tool_call in getattr(message, "tool_calls", None) or []:
                if tool_call.get("name") == "task":
                    subagent = str((tool_call.get("args") or {}).get("subagent_type") or "")
                    if subagent:
                        task_subagents[tool_call.get("id")] = subagent
            continue

        if message_type == "tool":
            call_id = getattr(message, "tool_call_id", None)
            subagent = task_subagents.get(call_id)
            if not subagent:
                continue
            content = message.content if isinstance(message.content, str) else str(message.content)
            if getattr(message, "status", None) == "error":
                specialist_lines.append(f"- {subagent}: FAILED")
                continue
            fields = ""
            match = _STRUCTURED_RESULT_DIGEST_RE.search(content)
            if match:
                try:
                    payload = json.loads(match.group(1))
                    keys = [
                        key for key in payload
                        if key not in {"agent", "schema_version"} and payload[key] not in (None, [], {})
                    ][:_DIGEST_MAX_RESULT_FIELDS]
                    if payload.get("no_data") is True:
                        fields = " (no data)"
                    elif keys:
                        fields = f" (has: {', '.join(keys)})"
                except (json.JSONDecodeError, TypeError):
                    pass
            specialist_lines.append(f"- {subagent}: completed{fields}")

    if not prior_queries and not specialist_lines and not runtime_lines:
        return None

    lines = ["[SESSION CONTEXT]"]
    if prior_queries:
        lines.append("Previous user requests:")
        for index, text in enumerate(prior_queries[-_DIGEST_MAX_PRIOR_QUERIES:], start=1):
            lines.append(f'{index}. "{text}"')
    if specialist_lines or runtime_lines:
        lines.append("Already produced this session:")
        lines.extend(dict.fromkeys(specialist_lines))  # dedupe, keep order
        lines.extend(dict.fromkeys(runtime_lines))
    digest = "\n".join(lines)
    return digest[:_DIGEST_MAX_CHARS]


def _render_revision_request(prior_plan: RoutePlan, step_statuses: dict[str, str]) -> str:
    """Render the [PLAN REVISION REQUEST] block the planner prompt understands."""
    lines = [
        "[PLAN REVISION REQUEST]",
        "A step of the active plan ended in a state that may invalidate the remaining steps.",
        "Prior plan for this request:",
    ]
    for index, step in enumerate(prior_plan.steps, start=1):
        status = step_statuses.get(step.subagent, "not started")
        objective = step.objective or step.subagent
        lines.append(f"{index}. {step.subagent} — {objective} [{status}]")
    if not prior_plan.steps:
        lines.append(f"(mode {prior_plan.mode} with no specialist steps)")
    lines.append(
        "Produce the corrected plan for what still needs to happen (see revision rules)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _known_subagent_names() -> set[str]:
    from .routing_classifier import ROUTING_RULES

    return {rule["subagent"] for rule in ROUTING_RULES}


def _toposort(names: list[str], deps: dict[str, set[str]]) -> list[str] | None:
    """Kahn topological order preserving input order among ready nodes."""
    order_index = {name: index for index, name in enumerate(names)}
    remaining = {name: set(dep for dep in deps.get(name, set()) if dep in order_index) for name in names}
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            (name for name, pending in remaining.items() if not pending),
            key=lambda name: order_index[name],
        )
        if not ready:
            return None  # cycle
        for name in ready:
            ordered.append(name)
            del remaining[name]
        for pending in remaining.values():
            pending.difference_update(ready)
    return ordered


def validate_route_payload(query_text: str, payload: dict) -> RoutePlan | None:
    """Structurally validate a raw payload into a RoutePlan.

    Returns None when the payload is unusable and the caller should fall
    back. Validation may repair a plan (drop unknown names, break cycles);
    every repair is recorded in ``validation_notes``.
    """
    if not isinstance(payload, dict):
        return None

    notes: list[str] = []
    known = _known_subagent_names()

    mode = str(payload.get("mode") or "").strip().lower()
    if mode not in _VALID_MODES:
        notes.append(f"invalid mode {mode!r}")
        mode = ""

    raw_steps = payload.get("steps")
    steps: list[RouteStep] = []
    seen: set[str] = set()
    dropped_unknown = 0
    if isinstance(raw_steps, list):
        for raw in raw_steps[: _MAX_PLAN_STEPS * 2]:
            if isinstance(raw, str):
                raw = {"subagent": raw}
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("subagent") or "").strip()
            if not name:
                continue
            if name not in known:
                dropped_unknown += 1
                notes.append(f"dropped unknown subagent {name!r}")
                continue
            if name in seen:
                continue
            seen.add(name)
            raw_deps = raw.get("depends_on")
            deps = tuple(
                str(dep).strip()
                for dep in (raw_deps if isinstance(raw_deps, list) else [])
                if str(dep).strip() and str(dep).strip() != name
            )
            steps.append(RouteStep(
                subagent=name,
                objective=str(raw.get("objective") or "").strip(),
                depends_on=deps,
            ))
            if len(steps) >= _MAX_PLAN_STEPS:
                notes.append("step list truncated")
                break
    elif raw_steps not in (None, ()):
        notes.append("steps was not a list")

    excluded = tuple(
        name for name in (
            str(item).strip()
            for item in (payload.get("excluded_subagents") or [])
            if str(item).strip()
        )
        if name in known
    )
    if excluded:
        before = len(steps)
        steps = [step for step in steps if step.subagent not in excluded]
        if len(steps) != before:
            notes.append("removed steps contradicting excluded_subagents")

    step_names = {step.subagent for step in steps}
    # Dependencies must reference selected steps.
    cleaned: list[RouteStep] = []
    for step in steps:
        deps = tuple(dep for dep in step.depends_on if dep in step_names)
        if deps != step.depends_on:
            notes.append(f"pruned unknown depends_on for {step.subagent}")
        cleaned.append(RouteStep(step.subagent, step.objective, deps))
    steps = cleaned

    # A payload with no dependency edges at all usually means the planner
    # omitted them, not that it wants full parallelism — the handoff
    # machinery needs producer→consumer edges to pass structured results.
    # Enrich from the capability graph; explicit planner edges always win.
    if len(steps) > 1 and not any(step.depends_on for step in steps):
        from .routing_classifier import graph_workflow_dependencies

        graph_deps = graph_workflow_dependencies(query_text, {step.subagent for step in steps})
        if any(graph_deps.values()):
            step_names_local = {step.subagent for step in steps}
            steps = [
                RouteStep(
                    step.subagent,
                    step.objective,
                    tuple(sorted(dep for dep in graph_deps.get(step.subagent, set()) if dep in step_names_local)),
                )
                for step in steps
            ]
            notes.append("dependencies enriched from capability graph")

    ordered_names = _toposort(
        [step.subagent for step in steps],
        {step.subagent: set(step.depends_on) for step in steps},
    )
    if ordered_names is None:
        notes.append("dependency cycle; dependencies reset")
        steps = [RouteStep(step.subagent, step.objective, ()) for step in steps]
    else:
        by_name = {step.subagent: step for step in steps}
        steps = [by_name[name] for name in ordered_names]

    # Mode coherence.
    if steps:
        if mode != "specialists":
            if mode:
                notes.append(f"mode {mode!r} coerced to specialists (steps present)")
            mode = "specialists"
    elif mode == "specialists":
        if dropped_unknown:
            # The planner wanted specialists but named none we know — unusable.
            return None
        notes.append("specialists mode with no steps coerced to orchestrator")
        mode = "orchestrator"
    elif not mode:
        return None

    confidence = str(payload.get("confidence") or "medium").strip().lower()
    if confidence not in _VALID_CONFIDENCE:
        confidence = "medium"

    raw_deliverables = payload.get("deliverables")
    deliverables: list[str] = []
    if isinstance(raw_deliverables, list):
        seen_deliverables: set[str] = set()
        for item in raw_deliverables[:12]:
            name = str(item or "").strip()
            if name and name not in seen_deliverables:
                seen_deliverables.add(name)
                deliverables.append(name)

    return RoutePlan(
        query=query_text,
        mode=mode,
        steps=tuple(steps),
        excluded_subagents=excluded,
        confidence=confidence,
        rationale=str(payload.get("rationale") or "").strip(),
        source="planner",
        validation_notes=tuple(notes),
        deliverables=tuple(deliverables),
    )


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def fallback_route_plan(query_text: str) -> RoutePlan:
    """Keyword-classifier fallback used when no planner decision is available.

    This is the only remaining consumer of the legacy keyword/regex routing
    chain; it preserves offline behavior (no API key, backend failure).
    """
    from langchain_core.messages import HumanMessage

    from .routing_classifier import (
        classify_query_keywords,
        is_direct_answer_query,
        plan_workflow_rules,
        select_workflow_rules,
    )

    if is_direct_answer_query(query_text):
        return RoutePlan(
            query=query_text,
            mode="direct",
            confidence="medium",
            rationale="keyword fallback: direct core-tool lookup",
            source="fallback",
        )

    keyword_matched = classify_query_keywords([HumanMessage(content=query_text)])
    matched = select_workflow_rules(query_text, keyword_matched=keyword_matched)
    planned = plan_workflow_rules(query_text, matched)
    if not planned:
        return RoutePlan(
            query=query_text,
            mode="orchestrator",
            confidence="low",
            rationale="keyword fallback: no specialist match",
            source="fallback",
        )

    from .routing_classifier import graph_workflow_dependencies

    names = [rule["subagent"] for rule in planned]
    dependency_sets = graph_workflow_dependencies(query_text, set(names))
    steps = tuple(
        RouteStep(
            subagent=name,
            objective="",
            depends_on=tuple(dep for dep in names if dep in dependency_sets.get(name, set())),
        )
        for name in names
    )
    return RoutePlan(
        query=query_text,
        mode="specialists",
        steps=steps,
        confidence="medium",
        rationale="keyword fallback: stem/phrase match",
        source="fallback",
    )


# ---------------------------------------------------------------------------
# Active-plan registry
# ---------------------------------------------------------------------------

_active_plans: "OrderedDict[str, RoutePlan]" = OrderedDict()


def normalize_query_key(query_text: str) -> str:
    return " ".join((query_text or "").split()).casefold()


def activate_route_plan(plan: RoutePlan) -> None:
    key = normalize_query_key(plan.query)
    if not key:
        return
    _active_plans[key] = plan
    _active_plans.move_to_end(key)
    while len(_active_plans) > _ACTIVE_PLAN_LIMIT:
        _active_plans.popitem(last=False)


def get_active_route_plan(query_text: str) -> RoutePlan | None:
    return _active_plans.get(normalize_query_key(query_text))


def clear_active_route_plans() -> None:
    _active_plans.clear()


def active_plan_dependency_map(query_text: str, subagent_names: set[str]) -> dict[str, set[str]] | None:
    """Dependency map from the active plan, when it covers the requested names."""
    plan = get_active_route_plan(query_text)
    if plan is None or not plan.steps:
        return None
    plan_names = set(plan.subagent_names())
    if not subagent_names.issubset(plan_names):
        return None
    full = plan.dependency_map()
    return {
        name: {dep for dep in full.get(name, set()) if dep in subagent_names}
        for name in subagent_names
    }


def is_direct_route(query_text: str) -> bool:
    """Plan-aware direct-mode check with legacy regex fallback."""
    plan = get_active_route_plan(query_text)
    if plan is not None:
        return plan.is_direct
    from .routing_classifier import is_direct_answer_query

    return is_direct_answer_query(query_text)


# ---------------------------------------------------------------------------
# Planner façade
# ---------------------------------------------------------------------------

class RoutePlanner:
    """Compute, cache, and activate route plans for user queries.

    Caching is planner-sourced only: fallback plans are recomputed on every
    call so a transient backend failure self-heals on the next model call
    within the same turn instead of pinning the query to keyword routing.
    """

    def __init__(
        self,
        backend: RoutePlannerBackend | Callable[[str], dict | str | None] | None = None,
    ) -> None:
        self._backend = backend
        self._cache: dict[str, RoutePlan] = {}

    @property
    def has_backend(self) -> bool:
        """Whether an LLM planner backend is configured.

        Distinguishes a deliberate keyword-mode deployment (no backend:
        legacy behavior stays authoritative) from a degraded state (backend
        configured but this query fell back: keyword output is advisory only).
        """
        return self._backend is not None

    @staticmethod
    def _cache_key(query_text: str, session_digest: str | None) -> str:
        key = normalize_query_key(query_text)
        if session_digest:
            import hashlib

            key = f"{key}|{hashlib.sha1(session_digest.encode()).hexdigest()[:10]}"
        return key

    def plan(self, query_text: str, *, session_digest: str | None = None) -> RoutePlan:
        query_text = (query_text or "").strip()
        if not query_text:
            return RoutePlan(query="", mode="orchestrator", confidence="low",
                             rationale="empty query", source="fallback")
        # Same query in a different session state must replan: the digest is
        # part of the cache identity, so follow-ups never reuse a plan made
        # under different context.
        key = self._cache_key(query_text, session_digest)
        cached = self._cache.get(key)
        if cached is not None:
            activate_route_plan(cached)
            return cached

        plan: RoutePlan | None = None
        if self._backend is not None:
            payload = None
            try:
                payload = self._call_backend(query_text, session_digest)
            except Exception:
                logger.warning("route_planner: backend raised", exc_info=True)
            if isinstance(payload, str):
                payload = extract_json_payload(payload)
            if payload is not None:
                plan = validate_route_payload(query_text, payload)
                if plan is None:
                    logger.warning(
                        "route_planner: backend payload failed validation for query=%s",
                        query_text[:80],
                    )

        if plan is None:
            plan = fallback_route_plan(query_text)
            if self._backend is not None:
                logger.warning(
                    "route_planner: DEGRADED — planner backend unavailable/unusable; "
                    "keyword fallback is advisory only for query=%s",
                    query_text[:80],
                )

        logger.info(
            "route_planner: mode=%s steps=%s source=%s confidence=%s notes=%s query=%s",
            plan.mode,
            plan.subagent_names(),
            plan.source,
            plan.confidence,
            list(plan.validation_notes),
            query_text[:80],
        )
        if plan.source == "planner":
            self._cache[key] = plan
        activate_route_plan(plan)
        return plan

    def _call_backend(self, query_text: str, session_digest: str | None):
        """Invoke the backend, passing the digest only if it accepts one."""
        import inspect

        try:
            parameters = inspect.signature(self._backend).parameters
            accepts_digest = "session_digest" in parameters or any(
                param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
            )
        except (TypeError, ValueError):
            accepts_digest = False
        if accepts_digest:
            return self._backend(query_text, session_digest=session_digest)
        return self._backend(query_text)

    def is_authoritative(self, plan: RoutePlan) -> bool:
        """Whether hard (execution-affecting) decisions may rely on this plan.

        Planner-sourced plans (including mid-turn revisions) are always
        authoritative. Fallback plans are authoritative only in deliberate
        keyword-mode deployments (no backend); with a backend configured they
        are advisory-only.
        """
        return plan.source in ("planner", "planner_revision") or not self.has_backend

    def revise(
        self,
        query_text: str,
        *,
        session_digest: str | None = None,
        prior_plan: RoutePlan,
        step_statuses: dict[str, str],
        outcome_key: str,
    ) -> RoutePlan | None:
        """Re-plan mid-turn after a step outcome that may invalidate the plan.

        ``step_statuses`` maps each prior-plan subagent to a short status
        annotation ("completed", "FAILED: <reason>", "not started").
        ``outcome_key`` identifies the triggering outcome (the failed task's
        tool_call_id); it is stamped into the revised plan's validation notes
        so the same outcome never triggers a second revision.

        On success the revised plan **overwrites the cached plan** for
        (query, session_digest) — every routing consumer sees the revision for
        the rest of the turn. Returns None (original plan stays active) when
        the backend is missing, errors, or emits an unusable payload.
        """
        if self._backend is None:
            return None
        query_text = (query_text or "").strip()
        if not query_text:
            return None

        revision_block = _render_revision_request(prior_plan, step_statuses)
        augmented_digest = (
            f"{session_digest}\n\n{revision_block}" if session_digest else revision_block
        )
        payload = None
        try:
            payload = self._call_backend(query_text, augmented_digest)
        except Exception:
            logger.warning("route_planner: revision backend raised", exc_info=True)
            return None
        if isinstance(payload, str):
            payload = extract_json_payload(payload)
        if payload is None:
            return None
        revised = validate_route_payload(query_text, payload)
        if revised is None:
            logger.warning(
                "route_planner: revision payload failed validation for query=%s",
                query_text[:80],
            )
            return None

        # Carry prior revision markers forward so a per-turn revision cap can
        # count every revision, not just the latest one.
        prior_markers = tuple(
            note for note in prior_plan.validation_notes
            if str(note).startswith("revised_after:")
        )
        revised = replace(
            revised,
            source="planner_revision",
            validation_notes=revised.validation_notes + prior_markers + (f"revised_after:{outcome_key}",),
        )
        key = self._cache_key(query_text, session_digest)
        self._cache[key] = revised
        activate_route_plan(revised)
        logger.info(
            "route_planner: REVISED plan after %s — mode=%s steps=%s (was steps=%s)",
            outcome_key,
            revised.mode,
            revised.subagent_names(),
            prior_plan.subagent_names(),
        )
        return revised


def plan_query(
    query_text: str,
    *,
    backend: RoutePlannerBackend | None = None,
) -> RoutePlan:
    """One-shot convenience wrapper around :class:`RoutePlanner`."""
    return RoutePlanner(backend=backend).plan(query_text)
