"""Query-classification helpers for orchestrator routing."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from .subagent_config import load_routing_configuration

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

try:
    ROUTING_RULES, PARALLEL_PAIRS, PARALLEL_3WAY, SEQUENTIAL_PAIRS = load_routing_configuration()
except Exception as e:  # pragma: no cover - import-time fallback
    logger.warning(
        "Failed to load routing rules from subagent config: %s — using empty defaults", e
    )
    ROUTING_RULES, PARALLEL_PAIRS, PARALLEL_3WAY, SEQUENTIAL_PAIRS = [], set(), set(), {}

_HSP_QUERY_RE = re.compile(
    r"\b(hansen|hsp|red\b|relative energy difference|ml prediction|machine learning)\b",
    re.IGNORECASE,
)
_HSP_SCREENING_RE = re.compile(
    r"\b("
    r"screen|screening|matrix|heatmap|show your work|compare red|rank|ranking|"
    r"look up .*hsp|relevant hsp data|evaluate using hansen|group .*separab|"
    r"which of .* would you expect to dissolve|would .* work as a selective solvent"
    r")\b",
    re.IGNORECASE,
)
_PROCESS_DESIGN_RE = re.compile(
    r"\b("
    r"sequence|workflow|process|recover|recovery|recycling facility|facility|waste stream|"
    r"mixed[- ]plastics|stage|staged|route|rt separation|separation sequence|"
    r"tea|lca|biosteam|capex|opex|msp|gwp"
    r")\b",
    re.IGNORECASE,
)
_BIOSTEAM_INTENT_RE = re.compile(
    r"\b("
    r"tea|lca|techno[- ]economic|life cycle|biosteam|process simulation|"
    r"msp|capex|opex|operating cost|capital cost|gwp|emissions?|"
    r"uncertainty|sensitivity|tornado|parameter sweep|monte carlo|payback"
    r")\b",
    re.IGNORECASE,
)
_NEGATED_BIOSTEAM_RE = re.compile(
    r"\b(do not|don't|no)\b[^.]{0,48}\b("
    r"tea|lca|techno[- ]economic|life cycle|biosteam|msp|capex|opex|gwp"
    r")\b",
    re.IGNORECASE,
)
_CONTAMINANT_INTENT_RE = re.compile(
    r"\b("
    r"contamin|decontamin|pfas|phthalate|additive removal|"
    r"leaching mode|contaminant screening|strap contaminant removal"
    r")\b",
    re.IGNORECASE,
)
_NEGATED_SAFETY_RE = re.compile(
    r"\b(do not|don't|no)\b[^.]{0,48}\b("
    r"safety|pubchem|gsk|gscore|g-score|ghs|hazard|toxicity"
    r")\b",
    re.IGNORECASE,
)
_SEPARATION_ROUTE_INTENT_RE = re.compile(
    r"\b("
    r"process design|separation|route|sequence|solvent route|"
    r"isolat|atmospheric pressure|best .* route|identify .* route"
    r")\b",
    re.IGNORECASE,
)
_NEGATED_PROCESS_DESIGN_RE = re.compile(
    r"\b(do not|don't|no)\b[^.]{0,48}\b("
    r"process design|separation|route|sequence"
    r")\b",
    re.IGNORECASE,
)
_SAFETY_INTENT_RE = re.compile(
    r"\b("
    r"gsk|gscore|pubchem|ghs|hazard|toxicity|toxic|health risk|risk profile|"
    r"safety score|safety scores|exposure|flammab|flammability|sds|msds"
    r")\b",
    re.IGNORECASE,
)


def _build_subagent_list() -> str:
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


def classify_query_llm(query: str, classifier_model: BaseChatModel) -> list[dict] | None:
    try:
        result = classifier_model.invoke([
            SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])
    except Exception:
        logger.warning("classify_query_llm: model call failed", exc_info=True)
        return None

    content = result.content
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
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
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                pass

    if not parsed or not isinstance(parsed.get("subagents"), list):
        logger.warning("classify_query_llm: could not parse response: %s", text[:200])
        return None

    rules_by_name = {rule["subagent"]: rule for rule in ROUTING_RULES}
    matched_rules: list[dict] = []
    for name in parsed["subagents"]:
        rule = rules_by_name.get(name)
        if rule:
            matched_rules.append(rule)
        else:
            logger.warning("classify_query_llm: unknown subagent name: %s", name)

    logger.info(
        "classify_query_llm: subagents=%s confidence=%s",
        parsed["subagents"],
        parsed.get("confidence"),
    )
    return matched_rules


def _match_rule(rule: dict, query_lower: str) -> int:
    for neg in rule["negatives"]:
        if neg in query_lower:
            return -1
    for phrase in rule["phrases"]:
        if re.search(phrase, query_lower):
            return 3
    for stem in rule["high_stems"]:
        if re.search(stem, query_lower):
            return 2
    low_hits = sum(1 for stem in rule["low_stems"] if re.search(stem, query_lower))
    if low_hits >= 2:
        return 1
    return 0


def classify_query_keywords(messages: list) -> list[dict]:
    query = _extract_query_text(messages)
    if not query:
        return []

    query_lower = query.lower()
    matches: list[tuple[int, int, dict]] = []
    for rule in ROUTING_RULES:
        score = _match_rule(rule, query_lower)
        if score > 0:
            matches.append((score, rule["priority"], rule))
    if not matches:
        return []
    matches.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in matches]


def _extract_query_text(messages: list) -> str:
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    return query


def _normalize_matched_rules(query_text: str, matched_rules: list[dict] | None) -> list[dict] | None:
    if not matched_rules:
        return matched_rules

    query_lower = query_text.lower()
    names = {rule["subagent"] for rule in matched_rules}
    if {"statistics-ml", "separation-engineer"}.issubset(names):
        is_hsp = bool(_HSP_QUERY_RE.search(query_text))
        is_process_design = bool(_PROCESS_DESIGN_RE.search(query_text))
        needs_explicit_hsp_screening = is_hsp and bool(_HSP_SCREENING_RE.search(query_text))
        if is_hsp and not is_process_design:
            return [rule for rule in matched_rules if rule["subagent"] != "separation-engineer"]
        if is_process_design and not needs_explicit_hsp_screening:
            return [rule for rule in matched_rules if rule["subagent"] != "statistics-ml"]
    if {"separation-engineer", "biosteam-analyst"}.issubset(names):
        has_biosteam_intent = bool(_BIOSTEAM_INTENT_RE.search(query_text)) and not bool(_NEGATED_BIOSTEAM_RE.search(query_text))
        if not has_biosteam_intent:
            matched_rules = [rule for rule in matched_rules if rule["subagent"] != "biosteam-analyst"]
            names = {rule["subagent"] for rule in matched_rules}
    if {"contaminant-removal-analyst", "biosteam-analyst"}.issubset(names):
        has_biosteam_intent = bool(_BIOSTEAM_INTENT_RE.search(query_text)) and not bool(_NEGATED_BIOSTEAM_RE.search(query_text))
        has_contaminant_intent = bool(_CONTAMINANT_INTENT_RE.search(query_text))
        if has_contaminant_intent and not has_biosteam_intent:
            matched_rules = [rule for rule in matched_rules if rule["subagent"] != "biosteam-analyst"]
            names = {rule["subagent"] for rule in matched_rules}
    if {"separation-engineer", "safety-analyst"}.issubset(names):
        has_explicit_safety_intent = bool(_SAFETY_INTENT_RE.search(query_text)) and not bool(_NEGATED_SAFETY_RE.search(query_text))
        process_only = "only do process design" in query_lower
        if process_only and not has_explicit_safety_intent:
            matched_rules = [rule for rule in matched_rules if rule["subagent"] != "safety-analyst"]
            names = {rule["subagent"] for rule in matched_rules}
    if {"contaminant-removal-analyst", "safety-analyst"}.issubset(names):
        has_explicit_safety_intent = bool(_SAFETY_INTENT_RE.search(query_text)) and not bool(_NEGATED_SAFETY_RE.search(query_text))
        contaminant_only = "only do contaminant-removal screening" in query_lower
        if contaminant_only and not has_explicit_safety_intent:
            matched_rules = [rule for rule in matched_rules if rule["subagent"] != "safety-analyst"]
            names = {rule["subagent"] for rule in matched_rules}
    if "contaminant-removal-analyst" in names and "separation-engineer" not in names:
        has_contaminant_intent = bool(_CONTAMINANT_INTENT_RE.search(query_text))
        has_separation_route_intent = bool(_SEPARATION_ROUTE_INTENT_RE.search(query_text)) and not bool(_NEGATED_PROCESS_DESIGN_RE.search(query_text))
        if has_contaminant_intent and has_separation_route_intent:
            separation_rule = next(
                (rule for rule in ROUTING_RULES if rule["subagent"] == "separation-engineer"),
                None,
            )
            if separation_rule is not None:
                matched_rules = [separation_rule, *matched_rules]
    return matched_rules


def _build_hint_from_matches(matched_rules: list[dict]) -> str | None:
    if not matched_rules:
        return None

    if len(matched_rules) == 1:
        rule = matched_rules[0]
        direct_answer_suffix = " For simple queries you can also answer directly using your own tools."
        if rule["subagent"] == "separation-engineer":
            direct_answer_suffix = ""
        return (
            f'\n\n[ADVISORY: This query is well-suited for the "{rule["subagent"]}" '
            f'specialist ({rule["description"]}). '
            f'Consider delegating via task(subagent_type="{rule["subagent"]}"). '
            f"{direct_answer_suffix}]"
        )

    if len(matched_rules) == 3:
        trio_set = frozenset(rule["subagent"] for rule in matched_rules)
        if trio_set in PARALLEL_3WAY:
            agent_descs = ", ".join(
                f'"{rule["subagent"]}" ({rule["description"]})' for rule in matched_rules
            )
            return (
                f'\n\n[ADVISORY: This query may benefit from three specialists in parallel: '
                f'{agent_descs}. '
                f'You may delegate to all three via concurrent task() calls.]'
            )

    if len(matched_rules) == 2:
        primary = matched_rules[0]
        secondary = matched_rules[1]
        pair_set = frozenset({primary["subagent"], secondary["subagent"]})
        pair_tuple = (primary["subagent"], secondary["subagent"])
        pair_tuple_rev = (secondary["subagent"], primary["subagent"])

        if pair_set in PARALLEL_PAIRS:
            direct_answer_suffix = ""
            if "separation-engineer" not in {
                primary["subagent"],
                secondary["subagent"],
            }:
                direct_answer_suffix = (
                    " For simple selectivity queries, you can answer directly using "
                    "rank_solvents_selectivity."
                )
            return (
                f'\n\n[ADVISORY: This query may benefit from multiple specialists: '
                f'"{primary["subagent"]}" ({primary["description"]}) and '
                f'"{secondary["subagent"]}" ({secondary["description"]}). '
                f'You may delegate to them in parallel or sequentially as appropriate. '
                f"{direct_answer_suffix}]"
            )

        if pair_tuple in SEQUENTIAL_PAIRS:
            first, second = primary, secondary
        elif pair_tuple_rev in SEQUENTIAL_PAIRS:
            first, second = secondary, primary
        else:
            first, second = primary, secondary
        return (
            f'\n\n[ADVISORY: This query may benefit from a sequential workflow: '
            f'first "{first["subagent"]}" ({first["description"]}), then '
            f'"{second["subagent"]}" ({second["description"]}). '
            'Consider delegating in that order and passing the structured result forward.]'
        )

    return None


def classify_query(messages: list) -> str | None:
    query_text = _extract_query_text(messages)
    matched = classify_query_keywords(messages)
    matched = _normalize_matched_rules(query_text, matched)
    return _build_hint_from_matches(matched)


def generate_routing_table() -> str:
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
