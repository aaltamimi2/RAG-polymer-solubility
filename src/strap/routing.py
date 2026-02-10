"""Routing middleware for the DISSOLVE orchestrator agent.

Provides keyword-based routing hints that are appended to the system prompt
before each LLM call. The hints are advisory — the LLM remains in control
and can override them.

Single source of truth: ROUTING_RULES defines both the prompt routing table
(via generate_routing_table()) and the runtime keyword matcher (via
classify_query()).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import wrap_model_call
from langchain_core.messages import HumanMessage, ToolMessage

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
    from collections.abc import Callable

# ------------------------------------------------------------------
# Routing rules — single source of truth for prompt table + matcher
# ------------------------------------------------------------------

ROUTING_RULES: list[dict] = [
    {
        "subagent": "separation-engineer",
        "priority": 1,
        "description": "Separation sequences, selectivity, dissolution, precipitation",
        "phrases": [
            "separation sequence", "optimal separation", "selective solvent",
            "polymer separation", "separation order", "separation plan",
            "sequential separation", "selective dissolution",
            "separation cascade", r"dissolve.*but not",
        ],
        "high_stems": [
            "precipitat", "antisolvent", "antisolvents",
            "selectiv", "greedy", r"branch.and.bound",
        ],
        "low_stems": [
            "separat", "dissolution", "dissolve", "mixed stream",
            "solvent",
        ],
        "negatives": [
            "sql", "database", "table schema", "list polymers",
            "list solvents", "available solvents", "available polymers",
            "describe table",
        ],
    },
    {
        "subagent": "safety-analyst",
        "priority": 2,
        "description": "GSK G-scores, PubChem hazard/GHS, toxicity data",
        "phrases": [
            r"g.score", "ghs hazard", "pubchem safety",
            "safety comparison", "solvent safety",
        ],
        "high_stems": [
            "pubchem", "gscore", "ld50", "lc50", "biodegradation",
            "safe",
        ],
        "low_stems": ["hazard", "toxic"],
        "negatives": [],
    },
    {
        "subagent": "tea-lca-analyst",
        "priority": 3,
        "description": "TEA, LCA, STRAP process, costs, emissions, MSP",
        "phrases": [
            r"techno.economic", "life cycle", "operating cost",
            "capital cost", "minimum selling price",
            "scale economics", "strap process",
        ],
        "high_stems": ["msp", "ghg", "payback"],
        "low_stems": ["tea", "lca", "emission", "cost"],
        "negatives": [],
    },
    {
        "subagent": "scholar-researcher",
        "priority": 4,
        "description": "Google Scholar, Web of Science, research papers",
        "phrases": [
            "google scholar", "web of science",
            "literature search", "research article",
        ],
        "high_stems": ["scholar"],
        "low_stems": ["publication", "journal", "paper"],
        "negatives": [],
    },
    {
        "subagent": "patent-researcher",
        "priority": 5,
        "description": "Patent search, patent lookup",
        "phrases": ["patent search", "patent number"],
        "high_stems": ["patent"],
        "low_stems": [],
        "negatives": [],
    },
    {
        "subagent": "rag-analyst",
        "priority": 6,
        "description": "RAG ingestion, literature Q&A, retrieval diagnostics",
        "phrases": [
            r"literature q&a", "rag index", "chunk quality",
            "retrieval diagnostics",
        ],
        "high_stems": ["rag", "ingest"],
        "low_stems": ["indexed", "embedding", "retrieval"],
        "negatives": [],
    },
    {
        "subagent": "visualization-specialist",
        "priority": 7,
        "description": "Plots, charts, heatmaps, dashboards",
        "phrases": [
            "solubility plot", "selectivity heatmap",
            "comparison dashboard", "process flow diagram",
        ],
        "high_stems": ["heatmap", "dashboard"],
        "low_stems": ["plot", "chart", "visualiz", "diagram", "figure"],
        "negatives": [],
    },
    {
        "subagent": "statistics-ml",
        "priority": 8,
        "description": "Statistics, regression, hypothesis testing, ML prediction, thermal properties, new polymer solubility",
        "phrases": [
            "statistical summary", "confidence interval",
            "hansen parameter", "ml predict",
            "solubility prediction",
            "thermal properties", "new polymer", "novel polymer",
            "unknown polymer", "psmiles",
            "melting temperature prediction", "predict tm",
            "predict thermal", "enthalpy of fusion",
            "heat capacity prediction", "generate solubility",
            "polymer not in database",
        ],
        "high_stems": ["anova", "regression", "psmiles", "thermal", "enthalpy"],
        "low_stems": [
            "statistic", "correlation", "hypothesis", "machine learning",
            "novel",
        ],
        "negatives": [],
    },
]

# ------------------------------------------------------------------
# Multi-agent execution patterns
# ------------------------------------------------------------------

PARALLEL_PAIRS: set[frozenset[str]] = {
    frozenset({"separation-engineer", "safety-analyst"}),
}

SEQUENTIAL_PAIRS: dict[tuple[str, str], None] = {
    ("separation-engineer", "tea-lca-analyst"): None,
    ("separation-engineer", "visualization-specialist"): None,
    ("statistics-ml", "visualization-specialist"): None,
    ("scholar-researcher", "rag-analyst"): None,
}


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


def classify_query(messages: list) -> str | None:
    """Classify the latest human message and return a routing hint string.

    Returns None if no routing hint should be injected.
    """
    # Extract last human message
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not query:
        return None

    query_lower = query.lower()

    # Score all rules
    matches: list[tuple[int, int, dict]] = []  # (score, priority, rule)
    for rule in ROUTING_RULES:
        score = _match_rule(rule, query_lower)
        if score > 0:
            matches.append((score, rule["priority"], rule))

    if not matches:
        return None

    # Sort by score descending, then priority ascending (lower = higher priority)
    matches.sort(key=lambda x: (-x[0], x[1]))

    if len(matches) == 1:
        # Single-agent hint
        rule = matches[0][2]
        return (
            f'\n\n[ROUTING: Your NEXT action must be a task() call. '
            f'Delegate to "{rule["subagent"]}" for '
            f'{rule["description"]}. Do NOT run query_database or other tools first — '
            f'the subagent has the right tools. Call: '
            f'task(description="<describe what the user wants>", '
            f'subagent_type="{rule["subagent"]}")]'
        )

    # Multi-agent routing
    if len(matches) == 2:
        primary = matches[0][2]
        secondary = matches[1][2]
        pair_set = frozenset({primary["subagent"], secondary["subagent"]})
        pair_tuple = (primary["subagent"], secondary["subagent"])
        pair_tuple_rev = (secondary["subagent"], primary["subagent"])

        # Check if parallel
        if pair_set in PARALLEL_PAIRS:
            return (
                "\n\n[ROUTING: Your NEXT action must be TWO task() calls in a single response.\n"
                "Launch both specialists in parallel:\n"
                f'- task(description="...", subagent_type="{primary["subagent"]}")\n'
                f'- task(description="...", subagent_type="{secondary["subagent"]}")\n'
                "Do NOT run query_database or other tools first.\n"
                "After both return, synthesize their results into a final answer.]"
            )

        # Check if sequential (order matters — check both directions)
        if pair_tuple in SEQUENTIAL_PAIRS:
            first, second = primary, secondary
        elif pair_tuple_rev in SEQUENTIAL_PAIRS:
            first, second = secondary, primary
        else:
            first, second = primary, secondary

        return (
            "\n\n[ROUTING: Your NEXT action must be a task() call. "
            "This requires two specialists in sequence.\n"
            f'Step 1: Delegate to "{first["subagent"]}" for {first["description"]}. '
            "Do NOT run query_database or other tools first. "
            "In the task description, instruct the subagent to write its findings to "
            f'"/chain_state/step_1_{first["subagent"]}.md" using write_file.\n'
            f'Step 2: Read the file from Step 1, then delegate to "{second["subagent"]}" '
            f"for {second['description']}. In the task description, include a brief "
            "summary of Step 1 results AND the file path "
            f'"/chain_state/step_1_{first["subagent"]}.md" '
            "so the subagent can read_file for full details. "
            "Instruct it to write its findings to "
            f'"/chain_state/step_2_{second["subagent"]}.md".\n'
            "After Step 2 returns, read_file both chain_state files and "
            "synthesize a final answer.]"
        )

    # 3+ agents: sequential chain — delegate one at a time, writing results to files
    ordered = [m[2] for m in matches]
    steps = []
    for i, rule in enumerate(ordered, 1):
        file_path = f"/chain_state/step_{i}_{rule['subagent']}.md"
        if i == 1:
            ctx = (
                " Instruct the subagent to write its findings to "
                f'"{file_path}" using write_file.'
            )
        else:
            prev_paths = ", ".join(
                f'"/chain_state/step_{j}_{ordered[j-1]["subagent"]}.md"'
                for j in range(1, i)
            )
            ctx = (
                f" Before delegating, read_file the prior results ({prev_paths}). "
                "Include a brief summary and the file path(s) in the task description "
                "so the subagent can read_file for full details. "
                f'Instruct the subagent to write its findings to "{file_path}".'
            )
        steps.append(
            f'Step {i}: Delegate to "{rule["subagent"]}" for {rule["description"]}.{ctx}'
        )
    step_text = "\n".join(steps)
    first_rule = ordered[0]
    return (
        f"\n\n[ROUTING: This query requires {len(ordered)} specialists in sequence. "
        "Execute them ONE AT A TIME — call the first task() now, wait for its result, "
        "then call the next. Each subagent writes its findings to a file in /chain_state/.\n"
        f"{step_text}\n"
        "After all steps complete, read_file the chain_state files and synthesize "
        "a final answer.\n"
        f'Your NEXT action must be: task(description="<step 1 task>", '
        f'subagent_type="{first_rule["subagent"]}"). '
        "Do NOT run query_database or other tools first.]"
    )


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
        "- tea-lca-analyst owns cost/LCA numbers",
        "- For cross-domain queries, delegate to the primary domain first, "
        "then pass results to the secondary specialist.",
        "",
        "When in doubt, delegate rather than attempting specialist work yourself.",
    ])

    return "\n".join(lines)


# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------

def _extract_completed_subagents(messages: list) -> list[str]:
    """Extract subagent names from completed task() calls in message history."""
    from langchain_core.messages import AIMessage
    completed = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "task":
                    sa = tc.get("args", {}).get("subagent_type", "")
                    if sa:
                        completed.append(sa)
    return completed


def _build_progress_directive(
    completed: list[str], ordered_plan: list[dict]
) -> str | None:
    """Build a progress-tracking directive for multi-agent sequential plans.

    Returns a system prompt injection telling the orchestrator which steps
    are done and what to do next, or None if the plan is complete.
    """
    done_set = set(completed)
    remaining = [r for r in ordered_plan if r["subagent"] not in done_set]

    if not remaining:
        return (
            "\n\n[PROGRESS: All subagent steps are COMPLETE. "
            "Synthesize findings from all subagents into a final answer NOW. "
            "Do NOT call any more task() functions.]"
        )

    next_agent = remaining[0]
    done_names = ", ".join(completed) if completed else "(none)"
    return (
        f"\n\n[PROGRESS: Completed subagents: {done_names}. "
        f"Your NEXT action MUST be: task(subagent_type=\"{next_agent['subagent']}\") "
        f"for {next_agent['description']}. "
        f"Do NOT re-delegate to an already-completed subagent. "
        f"Remaining steps: {', '.join(r['subagent'] for r in remaining)}.]"
    )


def _get_ordered_plan(messages: list) -> list[dict]:
    """Re-classify the user query and return the ordered plan of subagents."""
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    if not query:
        return []
    query_lower = query.lower()
    matches = []
    for rule in ROUTING_RULES:
        score = _match_rule(rule, query_lower)
        if score > 0:
            matches.append((score, rule["priority"], rule))
    matches.sort(key=lambda x: (-x[0], x[1]))
    return [m[2] for m in matches]


def _rewrite_duplicate_task_calls(
    response: ModelResponse,
    completed: list[str],
    ordered_plan: list[dict],
) -> ModelResponse:
    """Hard enforcement: if the LLM tries to re-call an already-completed
    subagent, rewrite the tool_call to target the correct next subagent."""
    from langchain_core.messages import AIMessage as _AIMessage

    if not response.result:
        return response

    ai_msg = response.result[0]
    tool_calls = getattr(ai_msg, "tool_calls", None)
    if not tool_calls:
        return response

    done_set = set(completed)
    remaining = [r for r in ordered_plan if r["subagent"] not in done_set]

    if not remaining:
        # All steps complete — strip any task() calls, force synthesis
        has_task_call = any(
            tc.get("name") == "task" for tc in tool_calls
        )
        if has_task_call:
            logger.warning(
                "routing_middleware: ALL plan steps complete — stripping task() "
                "calls to force synthesis (completed: %s)", completed,
            )
            from langchain_core.messages import AIMessage as _AIMsg
            return _AIMsg(
                content=(
                    "[ALL STEPS COMPLETE] All specialist subagents have returned "
                    "results. Synthesize the findings from separation-engineer, "
                    "safety-analyst, and tea-lca-analyst into a comprehensive "
                    "final answer NOW. Do NOT call any more task() functions."
                ),
            )
        return response

    next_agent = remaining[0]["subagent"]
    rewritten = False

    new_tool_calls = []
    for tc in tool_calls:
        if tc.get("name") == "task":
            target = tc.get("args", {}).get("subagent_type", "")
            if target in done_set and target != next_agent:
                logger.warning(
                    "routing_middleware: REWRITING task(%s) -> task(%s) "
                    "(already completed: %s)",
                    target, next_agent, completed,
                )
                new_args = dict(tc["args"])
                new_args["subagent_type"] = next_agent
                new_tc = dict(tc)
                new_tc["args"] = new_args
                new_tool_calls.append(new_tc)
                rewritten = True
            else:
                new_tool_calls.append(tc)
        else:
            new_tool_calls.append(tc)

    if rewritten:
        new_msg = ai_msg.model_copy(update={"tool_calls": new_tool_calls})
        return type(response)(result=[new_msg])

    return response


@wrap_model_call
def routing_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelCallResult:
    """Append routing or progress hints to the system prompt.

    - First call (no task results): injects routing hint from keyword classifier.
    - Subsequent calls: injects progress directive AND hard-rewrites duplicate
      task() calls to enforce the sequential plan.
    """
    completed = _extract_completed_subagents(request.messages)

    if not completed:
        # First call — inject full routing hint from keyword classifier
        hint = classify_query(request.messages)
        if hint and request.system_message is not None:
            new_system = append_to_system_message(request.system_message, hint)
            request = request.override(system_message=new_system)
        return handler(request)

    # After task() calls: inject progress directive + hard-rewrite responses
    ordered_plan = _get_ordered_plan(request.messages)

    if len(ordered_plan) > 1:
        progress = _build_progress_directive(completed, ordered_plan)
        if progress and request.system_message is not None:
            new_system = append_to_system_message(
                request.system_message, progress
            )
            request = request.override(system_message=new_system)

    response = handler(request)

    # Hard enforcement: rewrite duplicate task() calls
    if len(ordered_plan) > 1:
        response = _rewrite_duplicate_task_calls(
            response, completed, ordered_plan
        )

    return response
