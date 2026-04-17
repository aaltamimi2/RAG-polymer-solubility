"""Query-classification helpers for orchestrator routing."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from .planning_graph import GENERIC_CONTEXT_ARTIFACT, build_planning_graph
from .query_context import extract_query_context
from .subagent_config import load_routing_configuration

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)
_HINT_ORDER = {"low": 0, "medium": 1, "high": 2}

try:
    ROUTING_RULES, PARALLEL_PAIRS, PARALLEL_3WAY, SEQUENTIAL_PAIRS = load_routing_configuration()
except Exception as e:  # pragma: no cover - import-time fallback
    logger.warning(
        "Failed to load routing rules from subagent config: %s — using empty defaults", e
    )
    ROUTING_RULES, PARALLEL_PAIRS, PARALLEL_3WAY, SEQUENTIAL_PAIRS = [], set(), set(), {}

try:
    PLANNING_GRAPH = build_planning_graph()
except Exception as e:  # pragma: no cover - import-time fallback
    logger.warning(
        "Failed to load planning graph from subagent config: %s — using empty defaults", e
    )
    PLANNING_GRAPH = None

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
_EXPLICIT_BIOSTEAM_ANALYSIS_RE = re.compile(
    r"\b("
    r"tea|lca|techno[- ]economic|life cycle|biosteam|process simulation|"
    r"msp|capex|opex|operating cost|capital cost|"
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
_OPTIMIZATION_INTENT_RE = re.compile(
    r"\b("
    r"optimi[sz](?:e|ation)|max(?:imize)? profit|min(?:imize)? emissions?|"
    r"min(?:imize)? cost|max(?:imize)? circularity|superstructure|pyomo|minlp|"
    r"optimal pathway|waste management"
    r")\b",
    re.IGNORECASE,
)
_NEGATED_SAFETY_RE = re.compile(
    r"\b(do not|don't|no)\b[^.]{0,48}\b("
    r"safety|pubchem|gsk|gscore|g-score|ghs|hazard|toxicity"
    r")\b",
    re.IGNORECASE,
)
_NEGATED_LITERATURE_RE = re.compile(
    r"\b(do not|don't|no)\b[^.]{0,64}\b(literature|scholar|paper|papers|journal|web of science)\b",
    re.IGNORECASE,
)
_NEGATED_PATENT_RE = re.compile(
    r"\b(do not|don't|no)\b[^.]{0,64}\bpatents?\b",
    re.IGNORECASE,
)
_NEGATED_RAG_RE = re.compile(
    r"\b(do not|don't|no)\b[^.]{0,64}\b(rag|retrieval|indexed documents?)\b",
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
_LCA_INTENT_RE = re.compile(
    r"\b(lca|life cycle|gwp|emissions?|environmental)\b",
    re.IGNORECASE,
)
_SEPARATION_FEASIBILITY_RE = re.compile(
    r"\b(feasible|feasibility|atmospheric pressure|boiling point)\b",
    re.IGNORECASE,
)
_SOLVENT_SHORTLIST_RE = re.compile(
    r"\b(shortlist|best solvents?|rank solvents?|solvent screening)\b",
    re.IGNORECASE,
)
_LITERATURE_SEARCH_RE = re.compile(
    r"\b(literature|google scholar|web of science|research articles?|papers?|journal)\b",
    re.IGNORECASE,
)
_PATENT_SEARCH_RE = re.compile(
    r"\bpatents?\b",
    re.IGNORECASE,
)
_RAG_INTENT_RE = re.compile(
    r"\b(rag|retrieval-augmented|indexed documents?|retrieved findings|retrieval diagnostics)\b",
    re.IGNORECASE,
)
_VISUALIZATION_INTENT_RE = re.compile(
    r"\b(plot|chart|graph|visualiz|dashboard|figure|heatmap|diagram)\b",
    re.IGNORECASE,
)
_STATISTICS_INTENT_RE = re.compile(
    r"\b(statistics?|statistical|confidence interval|hypothesis|anova|correlation|regression)\b",
    re.IGNORECASE,
)
_ML_PREDICTION_RE = re.compile(
    r"\b(machine learning|ml prediction|predict(?:ion)?|hansen|hsp|relative energy difference|red)\b",
    re.IGNORECASE,
)
_THERMAL_PREDICTION_RE = re.compile(
    r"\b(glass transition|tg\b|melting|thermal prediction|thermal propert)\b",
    re.IGNORECASE,
)
_SEQUENTIAL_CUE_RE = re.compile(
    r"\b(then|and then|after|followed by|before finally|finally|using the result|based on the result)\b",
    re.IGNORECASE,
)
_QUERY_CONTEXT_LABEL_TO_GOALS: dict[str, tuple[str, ...]] = {
    "separation.route": ("separation.route",),
    "separation.feasibility": ("separation.feasibility",),
    "route.atmospheric_pressure": ("separation.feasibility",),
    "route.wash_step": ("separation.route",),
    "route.solvent_recovery": ("separation.route",),
    "solvent.shortlist": ("solvent.shortlist",),
    "safety.assessment": ("safety.assessment", "hazard.screening"),
    "tea.economics": ("tea.economics",),
    "lca.environmental": ("lca.environmental",),
    "literature.search": ("literature.search",),
    "patent.search": ("patent.search",),
    "literature.answer": ("literature.answer", "rag.retrieval"),
    "visualization.plot": ("visualization.plot",),
    "statistics.analysis": ("statistics.analysis",),
    "ml.prediction": ("ml.prediction",),
    "thermal.prediction": ("thermal.prediction",),
    "contaminant.screening": ("contaminant.screening", "contaminant.removal"),
    "optimization.pathway": ("optimization.pathway",),
}


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
- Return the smallest sufficient set of subagent names ordered by relevance
- You may return more than 3 subagent names when the query explicitly spans multiple stages or deliverables
- Return {{"subagents": []}} if the orchestrator can handle it directly \
(e.g. listing polymers, simple lookups)
- HIGH = clear specialist match, LOW = ambiguous
- "separation-engineer" handles dissolution, purification, separation \
sequences, selective solvents, mixed-stream processing
- "safety-analyst" handles safety, toxicity, GSK scores, hazard data
- "optimization-engineer" handles waste management optimization, profit \
maximization, emission minimization, MINLP superstructure, Pyomo models, \
and optimal processing pathway selection for multilayer plastic feeds
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


def score_query_rules(query_text: str) -> dict[str, int]:
    """Return keyword-match scores for every routing rule."""
    query_lower = query_text.lower()
    return {
        rule["subagent"]: _match_rule(rule, query_lower)
        for rule in ROUTING_RULES
    }


def infer_requested_goals(query_text: str) -> set[str]:
    """Infer requested planning goals directly from the query text."""
    requested: set[str] = set()
    if not query_text:
        return requested

    query_context = extract_query_context(query_text)
    negated_process = bool(_NEGATED_PROCESS_DESIGN_RE.search(query_text))
    negated_biosteam = bool(_NEGATED_BIOSTEAM_RE.search(query_text))
    negated_safety = bool(_NEGATED_SAFETY_RE.search(query_text))
    negated_literature = bool(_NEGATED_LITERATURE_RE.search(query_text))
    negated_patent = bool(_NEGATED_PATENT_RE.search(query_text))
    negated_rag = bool(_NEGATED_RAG_RE.search(query_text))

    for label in (*query_context.route_labels, *query_context.request_labels):
        for goal in _QUERY_CONTEXT_LABEL_TO_GOALS.get(label, ()):
            if (goal.startswith("separation.") or goal == "solvent.shortlist") and negated_process:
                continue
            if goal.startswith(("tea.", "lca.")) and negated_biosteam:
                continue
            if goal.startswith(("safety.", "hazard.")) and negated_safety:
                continue
            if goal == "literature.search" and negated_literature:
                continue
            if goal == "patent.search" and negated_patent:
                continue
            if goal in {"literature.answer", "rag.retrieval"} and negated_rag:
                continue
            requested.add(goal)

    if "optimization.pathway" in requested and not bool(_EXPLICIT_BIOSTEAM_ANALYSIS_RE.search(query_text)):
        requested.discard("tea.economics")
        requested.discard("lca.environmental")

    return requested


def infer_available_query_inputs(query_text: str) -> set[str]:
    """Infer which planning `user.*` requirements are satisfied by the query itself."""
    return set(extract_query_context(query_text).available_inputs)


def _requirements_satisfied(requirements: tuple[str, ...], available_inputs: set[str]) -> bool:
    if not requirements:
        return True
    return all(requirement in available_inputs for requirement in requirements)


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
    if {"optimization-engineer", "biosteam-analyst"}.issubset(names):
        has_explicit_biosteam_intent = bool(_EXPLICIT_BIOSTEAM_ANALYSIS_RE.search(query_text)) and not bool(_NEGATED_BIOSTEAM_RE.search(query_text))
        has_optimization_intent = bool(_OPTIMIZATION_INTENT_RE.search(query_text))
        if has_optimization_intent and not has_explicit_biosteam_intent:
            matched_rules = [rule for rule in matched_rules if rule["subagent"] != "biosteam-analyst"]
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


def _pair_normalization_allows(
    query_text: str,
    producer: str,
    consumer: str,
    rules_by_name: dict[str, dict],
) -> bool:
    producer_rule = rules_by_name.get(producer)
    consumer_rule = rules_by_name.get(consumer)
    if producer_rule is None or consumer_rule is None:
        return False
    normalized = _normalize_matched_rules(query_text, [producer_rule, consumer_rule]) or []
    names = {rule["subagent"] for rule in normalized}
    return producer in names and consumer in names


def _edge_direction_rank(
    producer: str,
    artifacts: set[str],
    *,
    score_map: dict[str, int],
    rules_by_name: dict[str, dict],
) -> tuple[int, int, int, int, int, str]:
    node = PLANNING_GRAPH.nodes[producer]
    return (
        len(artifacts),
        score_map.get(producer, 0),
        -rules_by_name[producer]["priority"],
        -_HINT_ORDER[node.cost_hint],
        -_HINT_ORDER[node.latency_hint],
        producer,
    )


def _build_capability_edge_map(
    query_text: str,
    *,
    candidate_names: set[str],
    protected_names: set[str],
    allowed_optional_names: set[str],
    allow_generic_fallback: bool,
    score_map: dict[str, int],
    rules_by_name: dict[str, dict],
) -> dict[tuple[str, str], set[str]]:
    if PLANNING_GRAPH is None:
        return {}

    edge_map: dict[tuple[str, str], set[str]] = {}
    capability_consumers: set[str] = set()
    capability_producers: set[str] = set()
    for consumer in candidate_names:
        for edge in PLANNING_GRAPH.incoming(consumer, kind="capability"):
            producer = edge.producer
            if producer == consumer or producer not in candidate_names:
                continue
            if producer not in protected_names:
                if producer not in allowed_optional_names:
                    continue
                if not _pair_normalization_allows(query_text, producer, consumer, rules_by_name):
                    continue
            edge_map[(producer, consumer)] = set(edge.artifacts)
            capability_consumers.add(consumer)
            capability_producers.add(producer)

    if allow_generic_fallback:
        generic_consumers = {
            consumer
            for consumer in candidate_names - capability_consumers
            if consumer not in capability_producers
        }
        for consumer in generic_consumers:
            for edge in PLANNING_GRAPH.incoming(consumer, kind="generic"):
                producer = edge.producer
                if producer == consumer or producer not in candidate_names:
                    continue
                if producer not in protected_names:
                    if producer not in allowed_optional_names:
                        continue
                    if not _pair_normalization_allows(query_text, producer, consumer, rules_by_name):
                        continue
                edge_map[(producer, consumer)] = {GENERIC_CONTEXT_ARTIFACT}

    blocked_edges: set[tuple[str, str]] = set()
    for producer, consumer in list(edge_map):
        reverse = (consumer, producer)
        if reverse not in edge_map or reverse in blocked_edges:
            continue
        rank = _edge_direction_rank(
            producer,
            edge_map[(producer, consumer)],
            score_map=score_map,
            rules_by_name=rules_by_name,
        )
        reverse_rank = _edge_direction_rank(
            consumer,
            edge_map[reverse],
            score_map=score_map,
            rules_by_name=rules_by_name,
        )
        if rank >= reverse_rank:
            blocked_edges.add(reverse)
        else:
            blocked_edges.add((producer, consumer))

    return {
        key: artifacts
        for key, artifacts in edge_map.items()
        if key not in blocked_edges
    }


def _producer_selection_rank(
    producer: str,
    coverage: set[str],
    *,
    protected_names: set[str],
    goal_candidate_names: set[str],
    score_map: dict[str, int],
    rules_by_name: dict[str, dict],
) -> tuple[int, int, int, int, int, int, str]:
    node = PLANNING_GRAPH.nodes[producer]
    return (
        len(coverage),
        int(producer in protected_names),
        int(producer in goal_candidate_names),
        score_map.get(producer, 0),
        -rules_by_name[producer]["priority"],
        -_HINT_ORDER[node.cost_hint] - _HINT_ORDER[node.latency_hint],
        producer,
    )


def _resolve_artifact_workflow(
    query_text: str,
    *,
    target_names: set[str],
    candidate_names: set[str],
    protected_names: set[str],
    goal_candidate_names: set[str],
) -> tuple[set[str], dict[str, set[str]]]:
    if PLANNING_GRAPH is None or not target_names:
        return set(target_names), {name: set() for name in target_names}

    rules_by_name = {rule["subagent"]: rule for rule in ROUTING_RULES}
    score_map = score_query_rules(query_text)
    available_inputs = infer_available_query_inputs(query_text)
    allowed_optional_names = {
        name
        for name in candidate_names - protected_names
        if _requirements_satisfied(PLANNING_GRAPH.nodes[name].requires, available_inputs)
        if name in goal_candidate_names or score_map.get(name, 0) > 0
    }
    edge_map = _build_capability_edge_map(
        query_text,
        candidate_names=candidate_names,
        protected_names=protected_names,
        allowed_optional_names=allowed_optional_names,
        allow_generic_fallback=bool(_SEQUENTIAL_CUE_RE.search(query_text)),
        score_map=score_map,
        rules_by_name=rules_by_name,
    )
    incoming_by_consumer: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for (producer, consumer), artifacts in edge_map.items():
        incoming_by_consumer[consumer][producer] = set(artifacts)

    memo: dict[str, tuple[set[str], set[str]]] = {}
    visiting: set[str] = set()

    def resolve(consumer: str) -> tuple[set[str], set[str]]:
        cached = memo.get(consumer)
        if cached is not None:
            return cached

        node = PLANNING_GRAPH.nodes[consumer]
        if consumer in visiting:
            return set(), set(node.produces)

        visiting.add(consumer)
        relevant_artifacts = set().union(*incoming_by_consumer.get(consumer, {}).values()) if incoming_by_consumer.get(consumer) else set()
        chosen_dependencies: set[str] = set()
        covered_artifacts: set[str] = set()

        while relevant_artifacts - covered_artifacts:
            best_producer: str | None = None
            best_coverage: set[str] = set()
            best_rank: tuple[int, int, int, int, int, int, str] | None = None

            for producer, direct_artifacts in incoming_by_consumer.get(consumer, {}).items():
                if producer in chosen_dependencies:
                    continue
                _producer_deps, producer_closure = resolve(producer)
                coverage = (set(direct_artifacts) | (producer_closure & relevant_artifacts)) - covered_artifacts
                if not coverage:
                    continue
                rank = _producer_selection_rank(
                    producer,
                    coverage,
                    protected_names=protected_names,
                    goal_candidate_names=goal_candidate_names,
                    score_map=score_map,
                    rules_by_name=rules_by_name,
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_producer = producer
                    best_coverage = coverage

            if best_producer is None:
                break

            chosen_dependencies.add(best_producer)
            covered_artifacts |= best_coverage

        closure = set(node.produces)
        for producer in chosen_dependencies:
            _producer_deps, producer_closure = resolve(producer)
            closure |= producer_closure

        visiting.remove(consumer)
        memo[consumer] = (chosen_dependencies, closure)
        return memo[consumer]

    selected_names: set[str] = set()

    def collect(consumer: str) -> None:
        if consumer in selected_names:
            return
        selected_names.add(consumer)
        dependencies, _closure = resolve(consumer)
        for producer in dependencies:
            collect(producer)

    for target in target_names:
        collect(target)

    dependency_map = {
        name: set(resolve(name)[0])
        for name in selected_names
    }
    return selected_names, dependency_map


def derive_workflow_dependencies(query_text: str, subagent_names: set[str]) -> dict[str, set[str]]:
    """Infer specialist prerequisites from capability contracts and closure coverage."""
    _selected_names, dependencies = _resolve_artifact_workflow(
        query_text,
        target_names=set(subagent_names),
        candidate_names=set(subagent_names),
        protected_names=set(subagent_names),
        goal_candidate_names=set(subagent_names),
    )
    return {
        name: set(dependencies.get(name, set()))
        for name in subagent_names
    }


def select_workflow_rules(
    query_text: str,
    *,
    llm_matched: list[dict] | None = None,
    keyword_matched: list[dict] | None = None,
) -> list[dict]:
    """Select the active specialist set for a query before workflow ordering."""
    llm_matched = _normalize_matched_rules(query_text, llm_matched) if llm_matched is not None else None
    keyword_matched = _normalize_matched_rules(query_text, keyword_matched)

    if not llm_matched:
        return keyword_matched or []
    if not keyword_matched:
        return llm_matched

    merged = list(llm_matched)
    seen = {rule["subagent"] for rule in merged}
    keyword_names = {rule["subagent"] for rule in keyword_matched}
    llm_names = {rule["subagent"] for rule in llm_matched}

    should_union_keywords = bool(_SEQUENTIAL_CUE_RE.search(query_text)) or len(keyword_matched) > len(llm_matched)
    if (
        {"separation-engineer", "contaminant-removal-analyst"}.issubset(keyword_names)
        and not {"separation-engineer", "contaminant-removal-analyst"}.issubset(llm_names)
    ):
        should_union_keywords = True

    if should_union_keywords:
        for rule in keyword_matched:
            if rule["subagent"] in seen:
                continue
            merged.append(rule)
            seen.add(rule["subagent"])

    return merged


def _collect_goal_candidate_names(
    query_text: str,
    seed_rules: list[dict] | None,
    score_map: dict[str, int],
) -> tuple[list[str], set[str]]:
    seed_rules = _normalize_matched_rules(query_text, seed_rules) if seed_rules is not None else None
    selected_names = list(dict.fromkeys(rule["subagent"] for rule in (seed_rules or [])))
    goal_candidate_names: set[str] = set()
    requested_goals = infer_requested_goals(query_text)
    available_inputs = infer_available_query_inputs(query_text)

    if PLANNING_GRAPH is None or not requested_goals:
        return selected_names, goal_candidate_names

    treat_as_seedless = not selected_names
    covered_goals: set[str] = set()
    if not treat_as_seedless:
        for name in selected_names:
            node = PLANNING_GRAPH.nodes.get(name)
            if node is not None:
                covered_goals.update(set(node.goals) & requested_goals)
    uncovered_goals = requested_goals - covered_goals
    for rule in ROUTING_RULES:
        node = PLANNING_GRAPH.nodes.get(rule["subagent"])
        if node is None:
            continue
        node_goals = set(node.goals) & requested_goals
        if not node_goals:
            continue
        if treat_as_seedless:
            if not _requirements_satisfied(node.requires, available_inputs):
                continue
        elif score_map.get(rule["subagent"], 0) <= 0:
            # When some specialists are already seeded, still admit a node if it
            # is the only goal-covered producer for an uncovered requested goal
            # and the query itself satisfies its declared input contract.
            if not (node_goals & uncovered_goals):
                continue
            if not _requirements_satisfied(node.requires, available_inputs):
                continue
        goal_candidate_names.add(rule["subagent"])
        if rule["subagent"] not in selected_names:
            selected_names.append(rule["subagent"])
            covered_goals.update(node_goals)
            uncovered_goals = requested_goals - covered_goals
        else:
            covered_goals.update(node_goals)
            uncovered_goals = requested_goals - covered_goals

    return selected_names, goal_candidate_names


def infer_planner_seed_rules(
    query_text: str,
    seed_rules: list[dict] | None,
) -> list[dict]:
    """Infer target specialists from planning goals before backchaining prerequisites."""
    score_map = score_query_rules(query_text)
    rules_by_name = {rule["subagent"]: rule for rule in ROUTING_RULES}
    selected_names, _goal_candidate_names = _collect_goal_candidate_names(
        query_text,
        seed_rules,
        score_map,
    )

    if not selected_names:
        selected_names = [
            rule["subagent"]
            for rule in ROUTING_RULES
            if score_map.get(rule["subagent"], 0) > 0
        ]

    selected_rules = [rules_by_name[name] for name in selected_names if name in rules_by_name]
    normalized = _normalize_matched_rules(query_text, selected_rules)
    return normalized or []


def plan_workflow_rules(
    query_text: str,
    seed_rules: list[dict] | None,
) -> list[dict]:
    """Build a workflow plan from query goals, then backchain graph prerequisites."""
    target_rules = infer_planner_seed_rules(query_text, seed_rules)
    if not target_rules:
        return []
    if PLANNING_GRAPH is None:
        return order_workflow_rules(query_text, target_rules)

    rules_by_name = {rule["subagent"]: rule for rule in ROUTING_RULES}
    score_map = score_query_rules(query_text)
    _seed_names, goal_candidate_names = _collect_goal_candidate_names(
        query_text,
        seed_rules,
        score_map,
    )
    positive_names = {name for name, score in score_map.items() if score > 0}
    target_names = {rule["subagent"] for rule in target_rules}
    candidate_names = {
        name
        for name in (target_names | positive_names | goal_candidate_names)
        if name in rules_by_name
    }
    selected_names, _dependencies = _resolve_artifact_workflow(
        query_text,
        target_names=target_names,
        candidate_names=candidate_names,
        protected_names=target_names,
        goal_candidate_names=goal_candidate_names,
    )

    planned_names = [rule["subagent"] for rule in target_rules]
    extra_names = [
        name
        for name in sorted(
            selected_names - set(planned_names),
            key=lambda name: rules_by_name[name]["priority"],
        )
    ]
    planned_rules = [rules_by_name[name] for name in [*planned_names, *extra_names] if name in selected_names]
    return order_workflow_rules(query_text, planned_rules)


def order_workflow_rules(query_text: str, matched_rules: list[dict] | None) -> list[dict]:
    """Return a dependency-respecting specialist order for the query."""
    if not matched_rules:
        return []
    if len(matched_rules) <= 1:
        return list(matched_rules)

    original_order = {
        rule["subagent"]: index
        for index, rule in enumerate(matched_rules)
    }
    rules_by_name = {rule["subagent"]: rule for rule in matched_rules}
    dependencies = derive_workflow_dependencies(query_text, set(rules_by_name))
    if not any(dependencies.values()):
        return list(matched_rules)

    remaining = {name: set(deps) for name, deps in dependencies.items()}
    ordered_names: list[str] = []
    ready = sorted(
        [name for name, deps in remaining.items() if not deps],
        key=lambda name: original_order[name],
    )

    while ready:
        current = ready.pop(0)
        if current in ordered_names:
            continue
        ordered_names.append(current)
        for name, deps in remaining.items():
            if current in deps:
                deps.discard(current)
                if not deps and name not in ordered_names and name not in ready:
                    ready.append(name)
        ready.sort(key=lambda name: original_order[name])

    if len(ordered_names) != len(rules_by_name):
        return list(matched_rules)

    return [rules_by_name[name] for name in ordered_names]


def _build_hint_from_matches(matched_rules: list[dict], query_text: str = "") -> str | None:
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
        dependencies = derive_workflow_dependencies(
            query_text,
            {rule["subagent"] for rule in matched_rules},
        )
        if any(dependencies.values()):
            chain = " -> ".join(
                f'"{rule["subagent"]}"'
                for rule in matched_rules
            )
            return (
                f'\n\n[ADVISORY: This query may benefit from a sequential workflow: '
                f'{chain}. Consider delegating in that order and passing validated '
                f'structured results forward.]'
            )

    if len(matched_rules) > 3:
        dependencies = derive_workflow_dependencies(
            query_text,
            {rule["subagent"] for rule in matched_rules},
        )
        if any(dependencies.values()):
            chain = " -> ".join(
                f'"{rule["subagent"]}"'
                for rule in matched_rules
            )
            return (
                f'\n\n[ADVISORY: This query may benefit from a staged workflow across multiple specialists: '
                f'{chain}. Consider delegating in that order and passing validated '
                f'structured results forward.]'
            )
        agent_descs = ", ".join(
            f'"{rule["subagent"]}" ({rule["description"]})' for rule in matched_rules
        )
        return (
            f'\n\n[ADVISORY: This query may benefit from multiple specialists: '
            f'{agent_descs}. Coordinate the workflow in stages as needed.]'
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
    matched = select_workflow_rules(
        query_text,
        keyword_matched=classify_query_keywords(messages),
    )
    matched = plan_workflow_rules(query_text, matched)
    return _build_hint_from_matches(matched, query_text=query_text)


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
