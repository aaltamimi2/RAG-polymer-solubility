"""Routing middleware for the DISSOLVE orchestrator agent.

Provides LLM-based semantic routing with keyword fallback. Advisory hints
are appended to the system prompt before each LLM call. The hints are
advisory — the LLM remains in control and can override them.

Single source of truth: ROUTING_RULES defines both the prompt routing table
(via generate_routing_table()), the runtime keyword matcher (via
classify_query_keywords()), and the LLM classifier prompt (via
_build_subagent_list()).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

logger = logging.getLogger(__name__)

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel
    from collections.abc import Callable

# ------------------------------------------------------------------
# Routing rules — loaded from subagents.yaml at import time
# ------------------------------------------------------------------

def _load_routing_rules() -> tuple[list[dict], set[frozenset], set[frozenset], dict[tuple, None]]:
    """Load routing rules from subagents.yaml.

    Builds ROUTING_RULES from subagent specs that have a ``routing:`` key,
    and PARALLEL_PAIRS / SEQUENTIAL_PAIRS from the ``execution_pairs:`` section.
    """
    try:
        yaml_path = Path(__file__).parent / "subagents.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Support both flat-list format and new mapping format
        if isinstance(data, dict):
            specs = data.get("subagents", [])
            exec_pairs = data.get("execution_pairs", {})
        else:
            specs = data
            exec_pairs = {}

        # Build ROUTING_RULES from subagent specs that have a routing: section.
        # The description field comes from the subagent's top-level description.
        routing_rules: list[dict] = []
        for spec in specs:
            routing = spec.get("routing")
            if not routing:
                continue
            rule = {
                "subagent": spec["name"],
                "priority": routing.get("priority", 999),
                "description": spec.get("description", "").strip(),
                "phrases": list(routing.get("phrases", [])),
                "high_stems": list(routing.get("high_stems", [])),
                "low_stems": list(routing.get("low_stems", [])),
                "negatives": list(routing.get("negatives", [])),
            }
            routing_rules.append(rule)

        # Sort by priority so order matches the original hardcoded list
        routing_rules.sort(key=lambda r: r["priority"])

        # Build PARALLEL_PAIRS
        parallel_pairs: set[frozenset] = set()
        for pair in exec_pairs.get("parallel", []):
            parallel_pairs.add(frozenset(pair))

        # Build PARALLEL_3WAY
        parallel_3way: set[frozenset] = set()
        for group in exec_pairs.get("parallel_3way", []):
            parallel_3way.add(frozenset(group))

        # Build SEQUENTIAL_PAIRS
        sequential_pairs: dict[tuple, None] = {}
        for pair in exec_pairs.get("sequential", []):
            sequential_pairs[(pair[0], pair[1])] = None

        return routing_rules, parallel_pairs, parallel_3way, sequential_pairs

    except Exception as e:
        logger.warning(
            "Failed to load routing rules from YAML: %s — using empty defaults", e
        )
        return {}, [], set(), []


ROUTING_RULES, PARALLEL_PAIRS, PARALLEL_3WAY, SEQUENTIAL_PAIRS = _load_routing_rules()


# ------------------------------------------------------------------
# LLM-based semantic classifier
# ------------------------------------------------------------------

def _build_subagent_list() -> str:
    """Build the specialist list for the classifier prompt from ROUTING_RULES."""
    lines = []
    for rule in ROUTING_RULES:
        lines.append(f'- {rule["subagent"]}: {rule["description"]}')
    return "\n".join(lines)


_CLASSIFIER_SYSTEM_PROMPT = """\
You are a query router for a polymer dissolution analysis system.
Given a user query, identify which specialist(s) should handle it.

Available specialists:
{subagent_list}

Respond with JSON only:
{{"subagents": ["name1"], "confidence": "HIGH"|"MEDIUM"|"LOW"}}

Rules:
- Return 1-3 subagent names ordered by relevance
- Return {{"subagents": []}} if the orchestrator can handle it directly \
(e.g. listing polymers, simple lookups)
- HIGH = clear specialist match, LOW = ambiguous
- "separation-engineer" handles dissolution, purification, separation \
sequences, selective solvents, mixed-stream processing
- "safety-analyst" handles safety, toxicity, GSK scores, hazard data
- When a query involves BOTH separation AND safety (e.g. "safest sequence"), \
return both specialists""".format(subagent_list=_build_subagent_list())


def classify_query_llm(
    query: str,
    classifier_model: BaseChatModel,
) -> list[dict] | None:
    """Classify a query using an LLM and return matched routing rules.

    Returns a list of matched ROUTING_RULES dicts ordered by relevance,
    or None on failure (caller should fall back to keywords).
    """
    try:
        result = classifier_model.invoke([
            SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])
    except Exception:
        logger.warning("classify_query_llm: model call failed", exc_info=True)
        return None

    # Parse JSON response — handle Gemini's list-of-dicts content format
    content = result.content
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        # Gemini returns [{"type": "text", "text": "...", ...}]
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        text = "\n".join(parts).strip()
    else:
        text = str(content).strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Try extracting from markdown code block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except (json.JSONDecodeError, TypeError):
                pass

    if not parsed or not isinstance(parsed.get("subagents"), list):
        logger.warning("classify_query_llm: could not parse response: %s", text[:200])
        return None

    subagent_names = parsed["subagents"]
    if not subagent_names:
        return []  # Orchestrator handles directly

    # Map names back to ROUTING_RULES entries
    rules_by_name = {r["subagent"]: r for r in ROUTING_RULES}
    matched = []
    for name in subagent_names:
        rule = rules_by_name.get(name)
        if rule:
            matched.append(rule)
        else:
            logger.warning("classify_query_llm: unknown subagent name: %s", name)

    confidence = parsed.get("confidence", "MEDIUM")
    logger.info(
        "classify_query_llm: subagents=%s confidence=%s",
        [r["subagent"] for r in matched],
        confidence,
    )
    return matched if matched else None


# ------------------------------------------------------------------
# Keyword classifier
# ------------------------------------------------------------------

def _match_rule(rule: dict, query_lower: str) -> int:
    """Score a single routing rule against a query.

    Returns:
        Score: 3 (phrase match), 2 (high-stem), 1 (2+ low-stems), 0 (no match).
        Returns -1 if a negative keyword cancels the match.
    """
    # Check negatives first
    for neg in rule["negatives"]:
        if neg in query_lower:
            return -1

    # Phrase match (any phrase → strong match)
    for phrase in rule["phrases"]:
        if re.search(phrase, query_lower):
            return 3

    # High-stem match (any hit → strong match)
    for stem in rule["high_stems"]:
        if re.search(stem, query_lower):
            return 2

    # Low-stem match (2+ hits → moderate match)
    low_hits = sum(1 for stem in rule["low_stems"] if re.search(stem, query_lower))
    if low_hits >= 2:
        return 1

    return 0


def classify_query_keywords(messages: list) -> list[dict]:
    """Classify the latest human message using keyword matching.

    Returns a list of matched ROUTING_RULES dicts sorted by
    (score desc, priority asc), or an empty list if no match.
    """
    # Extract last human message
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not query:
        return []

    query_lower = query.lower()

    # Score all rules
    matches: list[tuple[int, int, dict]] = []  # (score, priority, rule)
    for rule in ROUTING_RULES:
        score = _match_rule(rule, query_lower)
        if score > 0:
            matches.append((score, rule["priority"], rule))

    if not matches:
        return []

    # Sort by score descending, then priority ascending (lower = higher priority)
    matches.sort(key=lambda x: (-x[0], x[1]))
    return [m[2] for m in matches]


# ------------------------------------------------------------------
# Shared hint builder
# ------------------------------------------------------------------

def _build_hint_from_matches(matched_rules: list[dict]) -> str | None:
    """Build an advisory routing hint string from matched routing rules.

    Used by both the LLM classifier and keyword fallback paths.
    Returns None if no rules matched.
    """
    if not matched_rules:
        return None

    if len(matched_rules) == 1:
        rule = matched_rules[0]
        return (
            f'\n\n[ADVISORY: This query is well-suited for the "{rule["subagent"]}" '
            f'specialist ({rule["description"]}). '
            f'Consider delegating via task(subagent_type="{rule["subagent"]}"). '
            f'For simple queries you can also answer directly using your own tools.]'
        )

    # 3-way parallel check
    if len(matched_rules) == 3:
        trio_set = frozenset(r["subagent"] for r in matched_rules)
        if trio_set in PARALLEL_3WAY:
            agent_descs = ", ".join(
                f'"{r["subagent"]}" ({r["description"]})'
                for r in matched_rules
            )
            return (
                f'\n\n[ADVISORY: This query may benefit from three specialists in parallel: '
                f'{agent_descs}. '
                f'You may delegate to all three via concurrent task() calls.]'
            )

    # Multi-agent routing
    if len(matched_rules) == 2:
        primary = matched_rules[0]
        secondary = matched_rules[1]
        pair_set = frozenset({primary["subagent"], secondary["subagent"]})
        pair_tuple = (primary["subagent"], secondary["subagent"])
        pair_tuple_rev = (secondary["subagent"], primary["subagent"])

        # Check if parallel
        if pair_set in PARALLEL_PAIRS:
            return (
                f'\n\n[ADVISORY: This query may benefit from multiple specialists: '
                f'"{primary["subagent"]}" ({primary["description"]}) and '
                f'"{secondary["subagent"]}" ({secondary["description"]}). '
                f'You may delegate to them in parallel or sequentially as appropriate. '
                f'For simple selectivity queries, you can answer directly using '
                f'rank_solvents_selectivity.]'
            )

        # Check if sequential (order matters — check both directions)
        if pair_tuple in SEQUENTIAL_PAIRS:
            first, second = primary, secondary
        elif pair_tuple_rev in SEQUENTIAL_PAIRS:
            first, second = secondary, primary
        else:
            first, second = primary, secondary

        return (
            f'\n\n[ADVISORY: This query may benefit from two specialists in sequence: '
            f'first "{first["subagent"]}" ({first["description"]}), '
            f'then "{second["subagent"]}" ({second["description"]}). '
            f'You may delegate to them sequentially, or handle simpler aspects '
            f'directly with your own tools.]'
        )

    # 3+ agents: advisory list
    agent_list = ", ".join(
        f'"{r["subagent"]}" ({r["description"]})'
        for r in matched_rules
    )
    return (
        f'\n\n[ADVISORY: This query may benefit from multiple specialists: '
        f'{agent_list}. '
        f'You may delegate to them sequentially or in parallel as appropriate. '
        f'For simpler aspects, you can answer directly using your own tools.]'
    )


def classify_query(messages: list) -> str | None:
    """Classify the latest human message and return an advisory routing hint.

    Backward-compatible wrapper: keyword-only classification.
    Returns None if no routing hint should be injected.
    """
    matched = classify_query_keywords(messages)
    return _build_hint_from_matches(matched)


# ------------------------------------------------------------------
# Prompt table generator (used in SYSTEM_PROMPT)
# ------------------------------------------------------------------

def generate_routing_table() -> str:
    """Generate the routing rules section for the system prompt.

    Built from ROUTING_RULES so the prompt and middleware stay in sync.
    """
    lines = [
        "## Routing rules",
        "Your core tools handle: listing polymers/solvents, solvent properties "
        "(BP, LogP), SQL queries, and data exploration. You also clarify ambiguous "
        "requests, summarize subagent results, and format final responses.",
        "",
        "For specialist work, delegate to the appropriate subagent:",
        "",
        "| Query involves... | Delegate to |",
        "|---|---|",
    ]

    for rule in ROUTING_RULES:
        lines.append(f'| {rule["description"]} | {rule["subagent"]} |')

    lines.extend([
        "",
        "Subagent contracts:",
        "- separation-engineer owns feasibility/sequence/selectivity",
        "- safety-analyst owns hazard/safety scores",
        "- biosteam-analyst owns all TEA/LCA: process simulation, MSP, CAPEX, OPEX, GWP, cost, emissions",
        "- For cross-domain queries, delegate to the primary domain first, "
        "then pass results to the secondary specialist.",
        "",
        "When in doubt, delegate rather than attempting specialist work yourself.",
    ])

    return "\n".join(lines)


# ------------------------------------------------------------------
# Middleware helpers
# ------------------------------------------------------------------

def _extract_completed_subagents(messages: list) -> list[str]:
    """Extract subagent names from completed task() calls in message history.

    A task() call is "completed" only when its corresponding ToolMessage
    exists in the history.  This prevents counting in-flight parallel task()
    dispatches as completed before the subagent has actually returned.
    """
    # Collect all task() dispatches: tool_call_id -> subagent_type
    pending: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "task":
                    sa = tc.get("args", {}).get("subagent_type", "")
                    tc_id = tc.get("id", "")
                    if sa and tc_id:
                        pending[tc_id] = sa

    if not pending:
        return []

    # Find tool_call_ids that have a matching ToolMessage (actually completed)
    completed_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id and tc_id in pending:
                completed_ids.add(tc_id)

    # Return completed subagent names in ToolMessage order, deduplicated
    completed: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", None)
            if tc_id and tc_id in completed_ids:
                sa = pending[tc_id]
                if sa not in seen:
                    seen.add(sa)
                    completed.append(sa)

    return completed


def _extract_all_task_subagents(messages: list) -> list[str]:
    """Extract all subagent names dispatched via task(), completed or in-flight.

    Returns deduplicated names in the order they were first dispatched.
    Used to reconstruct the actual execution plan the orchestrator chose.
    """
    seen: set[str] = set()
    dispatched: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "task":
                    sa = tc.get("args", {}).get("subagent_type", "")
                    if sa and sa not in seen:
                        seen.add(sa)
                        dispatched.append(sa)
    return dispatched


def _build_progress_directive(
    completed: list[str], ordered_plan: list[dict]
) -> str | None:
    """Build a progress-tracking directive for multi-agent sequential plans.

    Returns a system prompt injection telling the orchestrator which steps
    are done and what to do next, or None if the plan is complete.
    """
    done_set = set(completed)
    remaining = [r for r in ordered_plan if r["subagent"] not in done_set]

    done_names = ", ".join(dict.fromkeys(completed)) if completed else "(none)"

    if not remaining:
        return (
            "\n\n[PROGRESS: All subagent steps are complete. "
            "Consider synthesizing findings from all subagents into a final answer. "
            "Prefer not to call additional task() functions unless needed.]"
        )

    next_agent = remaining[0]
    return (
        f"\n\n[PROGRESS: Completed subagents: {done_names}. "
        f'Suggested next: task(subagent_type="{next_agent["subagent"]}") '
        f"for {next_agent['description']}. "
        f"Prefer not to re-delegate to an already-completed subagent. "
        f"Remaining steps: {', '.join(r['subagent'] for r in remaining)}.]"
    )


def _get_ordered_plan(messages: list) -> list[dict]:
    """Build the ordered execution plan from actual orchestrator history.

    Instead of re-classifying the original query with keyword matching (which
    may disagree with the LLM's routing decision), the plan is derived from
    the actual task() dispatches in the message history.

    1. Start with subagents actually dispatched (in dispatch order) — ground truth.
    2. Append keyword-recommended agents not yet dispatched — advisory hints.
    """
    rules_by_name = {r["subagent"]: r for r in ROUTING_RULES}

    # Ground truth: what the orchestrator actually dispatched
    dispatched_names = _extract_all_task_subagents(messages)

    plan: list[dict] = []
    seen: set[str] = set()
    for name in dispatched_names:
        rule = rules_by_name.get(name)
        if rule and name not in seen:
            plan.append(rule)
            seen.add(name)

    # Advisory supplement: keyword-recommended agents not yet dispatched
    keyword_matches = classify_query_keywords(messages)
    for rule in keyword_matches:
        name = rule["subagent"]
        if name not in seen:
            plan.append(rule)
            seen.add(name)

    return plan


def _get_last_human_message(messages: list) -> str | None:
    """Extract text from the last HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None


# ------------------------------------------------------------------
# Middleware (class-based)
# ------------------------------------------------------------------

class RoutingMiddleware(AgentMiddleware):
    """LLM-based semantic routing with keyword fallback.

    - First call (no completed subagents): try LLM classifier → fall back
      to keywords → build advisory hint.
    - Subsequent calls: progress tracking only (keyword-based, no LLM needed).
    """

    def __init__(self, classifier_model: BaseChatModel | None = None) -> None:
        self._classifier_model = classifier_model

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        request = self._inject_hint(request)
        response = handler(request)
        self._log_decision(response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        request = self._inject_hint(request)
        response = await handler(request)
        self._log_decision(response)
        return response

    def _inject_hint(self, request: ModelRequest) -> ModelRequest:
        """Inject routing or progress hint into the system message."""
        completed = _extract_completed_subagents(request.messages)
        query_text = _get_last_human_message(request.messages)

        if not completed:
            # First call — try LLM, fall back to keywords
            matched = None
            if self._classifier_model and query_text:
                matched = classify_query_llm(query_text, self._classifier_model)

            if matched is None:
                # LLM unavailable or failed — keyword fallback
                matched = classify_query_keywords(request.messages)

            hint = _build_hint_from_matches(matched)

            if hint:
                logger.info(
                    "routing_middleware: advisory hint injected for query=%s",
                    (query_text or "")[:80],
                )
            else:
                logger.info(
                    "routing_middleware: no routing hint for query=%s",
                    (query_text or "")[:80],
                )

            if hint and request.system_message is not None:
                new_system = append_to_system_message(request.system_message, hint)
                return request.override(system_message=new_system)

            return request

        # After task() calls: inject progress directive based on actual history.
        # _get_ordered_plan derives the plan from what the orchestrator actually
        # dispatched, not from re-predicting the original query.
        ordered_plan = _get_ordered_plan(request.messages)

        if len(ordered_plan) > 1:
            progress = _build_progress_directive(completed, ordered_plan)
            if progress and request.system_message is not None:
                new_system = append_to_system_message(
                    request.system_message, progress
                )
                logger.info(
                    "routing_middleware: progress directive injected, "
                    "completed=%s remaining=%s",
                    completed,
                    [r["subagent"] for r in ordered_plan
                     if r["subagent"] not in set(completed)],
                )
                return request.override(system_message=new_system)

        return request

    def _log_decision(self, response) -> None:
        """Log what the LLM decided to do after routing."""
        result = getattr(response, "result", None)
        if result:
            ai_msg = result[0]
            tool_calls = getattr(ai_msg, "tool_calls", None)
            if tool_calls:
                tool_names = [tc.get("name", "?") for tc in tool_calls]
                logger.info(
                    "routing_middleware: LLM decided tool_calls=%s", tool_names,
                )


# Module-level instance for backward compat (keyword-only, no LLM)
routing_middleware = RoutingMiddleware(classifier_model=None)
