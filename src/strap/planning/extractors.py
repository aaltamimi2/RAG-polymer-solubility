"""Deterministic fact extraction for typed plan compilation.

The extractor intentionally produces facts only. It does not choose routes,
roles, or tools; the compiler maps requested artifacts to capabilities.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import Field

from strap.hsp_registry import (
    POLYMER_CATEGORY_ALIASES,
    SOLVENT_CATEGORY_ALIASES,
    SOLVENT_POLARITY_ALIASES,
    normalize_hsp_key,
)
from strap.planning.models import PlanningModel
from strap.planning.runtime_paths import normalize_runtime_path
from strap.user_input_parsing import extract_output_destination, extract_temperatures_c


_POLYMER_ALIASES = {
    "LDPE": "PE",
    "HDPE": "PE",
    "PE": "PE",
    "EVOH": "EVOH",
    "PET": "PET",
    "PP": "PP",
    "PS": "PS",
    "PVC": "PVC",
    "PC": "PC",
    "PA6": "PA6",
    "PA66": "PA66",
}

_SOLVENT_ALIASES = {
    "THF": "Tetrahydrofuran",
    "TETRAHYDROFURAN": "Tetrahydrofuran",
    "TETRAHYDROPYRAN": "Tetrahydropyran",
    "THP": "Tetrahydropyran",
    "TOLUENE": "Toluene",
    "CYCLOHEXANE": "Cyclohexane",
    "HEPTANE": "Heptane",
    "N-HEPTANE": "Heptane",
    "DIETHYL ETHER": "Diethyl ether",
    "DICHLOROMETHANE": "Dichloromethane",
    "DCM": "Dichloromethane",
    "N,N-DIMETHYLFORMAMIDE": "N,N-Dimethylformamide",
    "DIMETHYLFORMAMIDE": "N,N-Dimethylformamide",
    "DMF": "N,N-Dimethylformamide",
    "DIMETHYL SULFOXIDE": "Dimethyl sulfoxide",
    "DMSO": "Dimethyl sulfoxide",
    "GVL": "gamma-Valerolactone",
    "GAMMA-VALEROLACTONE": "gamma-Valerolactone",
    "DODECANE": "Dodecane",
    "PYRIDAZINE": "Pyridazine",
    "ETHYLENE GLYCOL": "Ethylene Glycol",
    "GLYCOL": "Ethylene Glycol",
}

_METRIC_ALIASES = {
    "cost": "total_cost",
    "total cost": "total_cost",
    "emissions": "emissions",
    "gwp": "emissions",
    "circularity": "circularity",
    "profit": "profit",
    "revenue": "profit",
}

_OUTPUT_PATH_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".html", ".json", ".csv", ".xlsx", ".md"}

_NEGATED_PARETO_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|no|not|without)\b[^.]{0,80}\b(?:pareto|frontier|trade[- ]?offs?)\b"
    r"|\b(?:pareto|frontier|trade[- ]?offs?)\b[^.]{0,48}\b(?:not\s+wanted|unwanted|not\s+needed)\b",
    re.I,
)
_PARETO_REQUEST_RE = re.compile(r"\b(?:pareto|frontier|trade[- ]?offs?)\b", re.I)
_SINGLE_OBJECTIVE_RE = re.compile(
    r"\b(?:single[- ]objective|single[- ]point|point[- ]optimum|one\s+optimum|just\s+optimi[sz]e|only\s+optimi[sz]e)\b",
    re.I,
)
_OPTIMIZATION_WORD_RE = re.compile(
    r"\b(?:optimi[sz](?:e|ation)|optimizaiotn|optimzation|waste[- ]management|superstructure|pyomo|minlp)\b",
    re.I,
)
_OBJECTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "max_circularity",
        r"\b(?:max(?:imize)?|maximum|highest)\s+(?:circularity|ce\s+score)\b|\bmax_circularity\b",
    ),
    (
        "min_emissions",
        r"\b(?:min(?:imize)?|minimum|lowest)\s+(?:emissions?|gwp|greenhouse\s+gas|co2e?)\b|\bmin_emissions\b",
    ),
    (
        "min_total_cost",
        r"\b(?:min(?:imize)?|minimum|lowest)\s+(?:total\s+)?cost\b|\bmin_total_cost\b",
    ),
    (
        "max_profit",
        r"\b(?:max(?:imize)?|maximum|highest)\s+(?:profit|revenue)\b|\bmax_profit\b|\bprofit\s+objective\b",
    ),
)


class ExtractedFacts(PlanningModel):
    query: str
    polymers: list[str] = Field(default_factory=list)
    polymer_aliases: dict[str, str] = Field(default_factory=dict)
    solvents: list[str] = Field(default_factory=list)
    temperatures_c: list[float] = Field(default_factory=list)
    feed_capacity_tpy: float | None = None
    feed_composition: dict[str, float] = Field(default_factory=dict)
    composition_slices: list[dict[str, float]] = Field(default_factory=list)
    scenario: str | None = None
    energy_case: str | None = None
    polymer_solvent_filters: dict[str, list[str]] = Field(default_factory=dict)
    route_candidates: list[dict[str, Any]] = Field(default_factory=list)
    constraint_mode: str | None = None
    fallback_policy: str | None = None
    route_pool_mode: str | None = None
    hsp_polymer_category: str | None = None
    hsp_solvent_category: str | None = None
    hsp_solvent_polarity: str | None = None
    top_k_per_polymer: int | None = None
    n_points: int | None = None
    min_washes: int | None = None
    max_washes: int | None = None
    objective: str | None = None
    pareto_requested: bool = False
    pareto_negated: bool = False
    single_objective_requested: bool = False
    metrics: list[str] = Field(default_factory=list)
    requested_artifact_types: list[str] = Field(default_factory=list)
    forbidden_artifact_types: list[str] = Field(default_factory=list)
    output_dir: str | None = None
    output_filename_hint: str | None = None
    plot_title: str | None = None
    workflow_markers: list[str] = Field(default_factory=list)
    missing_required_inputs: list[str] = Field(default_factory=list)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _context_list(context: dict[str, Any], key: str) -> list[Any]:
    value = context.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set | frozenset):
        return list(value)
    return []


def _context_dict(context: dict[str, Any], key: str) -> dict[str, Any]:
    value = context.get(key)
    return value if isinstance(value, dict) else {}


def _extract_polymers(query: str) -> tuple[list[str], dict[str, str]]:
    aliases: dict[str, str] = {}
    found: list[str] = []
    for raw, canonical in _POLYMER_ALIASES.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(raw)}(?![A-Za-z0-9])", query, re.I):
            aliases[raw] = canonical
            found.append(canonical)
    return _dedupe(found), aliases


def _extract_solvents(query: str) -> list[str]:
    found: list[str] = []
    for raw, canonical in _SOLVENT_ALIASES.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(raw)}(?![A-Za-z0-9])", query, re.I):
            found.append(canonical)
    return _dedupe(found)


def _extract_temperatures(query: str) -> list[float]:
    return extract_temperatures_c(query)


def _extract_capacity(query: str) -> float | None:
    match = re.search(
        r"(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s*(?:tonnes|tons|tpy)\s*(?:/year|per year|year)?",
        query,
        re.I,
    )
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _extract_feed_composition(query: str) -> dict[str, float]:
    composition: dict[str, float] = {}
    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:wt\s*)?%\s*(LDPE|HDPE|PE|EVOH|PET|PP|PS|PVC|PC|PA6|PA66)\b",
        query,
        re.I,
    ):
        raw_polymer = match.group(2).upper()
        canonical = _POLYMER_ALIASES.get(raw_polymer, raw_polymer)
        composition[canonical] = float(match.group(1)) / 100.0
    return composition


def _extract_composition_slices(query: str, polymers: list[str]) -> list[dict[str, float]]:
    if len(polymers) < 2:
        return []
    slices: list[dict[str, float]] = []
    for match in re.finditer(r"\b(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?){1,5})\b", query):
        values = [float(part.strip()) / 100.0 for part in match.group(1).split("/")]
        if len(values) != len(polymers):
            continue
        slices.append(dict(zip(polymers, values, strict=True)))
    if not slices and len(polymers) == 2:
        # "slices of 20% PE, 50% PE, and 80% PE" — repeated percentages of the
        # SAME polymer; the remainder goes to the other polymer. Feed
        # compositions ("60% PE and 40% EVOH") name different polymers and are
        # excluded by the same-polymer requirement.
        pattern = r"\b(\d+(?:\.\d+)?)\s*%\s*(" + "|".join(re.escape(p) for p in polymers) + r")\b"
        found = [(m.group(2), float(m.group(1)) / 100.0) for m in re.finditer(pattern, query, re.I)]
        if len(found) >= 2 and len({name.upper() for name, _ in found}) == 1:
            primary = next(p for p in polymers if p.upper() == found[0][0].upper())
            other = next(p for p in polymers if p.upper() != primary.upper())
            slices = [{primary: value, other: round(1.0 - value, 6)} for _, value in found]
    return slices


def _extract_scenario(query: str) -> str | None:
    match = re.search(r"\bscenario\s+([A-Z0-9]+)\b", query, re.I)
    return match.group(1).upper() if match else None


def _extract_energy_case(query: str) -> str | None:
    match = re.search(r"\b(?:energy\s+case|case|under)\s+(C[123])\b", query, re.I)
    if match:
        return match.group(1).upper()
    # Named configurations, as advertised in the biosteam-analyst description:
    # C1 = Combined Heat & Power (CHP), C2 = Grid + AMCOR heat, C3 = Grid + boiler.
    lowered = query.lower()
    if re.search(r"\bchp\b|combined\s+heat\s*(?:&|and)?\s*power", lowered):
        return "C1"
    if re.search(r"grid\s*\+\s*boiler|(?:natural\s+)?gas\s+boiler", lowered):
        return "C3"
    if re.search(r"grid\s*\+\s*amcor|\bamcor\b", lowered):
        return "C2"
    if re.search(
        r"\bgrid\b[^.]{0,24}\b(?:electricity|energy|scenario|configuration|config|case)\b"
        r"|\b(?:electricity|energy|scenario|configuration|config|case)\b[^.]{0,24}\bgrid\b",
        lowered,
    ):
        return "C2"
    return None


def _extract_top_k(query: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", query, re.I)
    return int(match.group(1)) if match else None


def _extract_n_points(query: str) -> int | None:
    match = re.search(r"\b(?:with\s+)?(\d+)\s+points?\b", query, re.I)
    return int(match.group(1)) if match else None


def _pareto_negated(query: str) -> bool:
    return bool(_NEGATED_PARETO_RE.search(query))


def _pareto_requested(query: str) -> bool:
    return bool(_PARETO_REQUEST_RE.search(query)) and not _pareto_negated(query)


def _extract_objective(query: str) -> str | None:
    for objective, pattern in _OBJECTIVE_PATTERNS:
        if re.search(pattern, query, re.I):
            return objective
    if re.search(r"\bmax(?:imize)?\b", query, re.I):
        return "max_profit"
    return None


def _single_objective_requested(query: str) -> bool:
    return bool(_SINGLE_OBJECTIVE_RE.search(query)) or _pareto_negated(query) or _extract_objective(query) is not None


def _optimization_requested(query: str) -> bool:
    return bool(_OPTIMIZATION_WORD_RE.search(query)) or _single_objective_requested(query)


def _extract_washes(query: str) -> tuple[int | None, int | None]:
    min_washes = None
    max_washes = None
    min_match = re.search(r"(?:at least|require(?:ing)?)\s+(\d+)\s+STRAP wash", query, re.I)
    max_match = re.search(r"(?:up to|allow(?:ing)? up to|maximum of)\s+(\d+)\s+(?:STRAP\s+)?wash", query, re.I)
    exact_match = re.search(r"(?:exactly|one[- ]wash|single[- ]wash)\s+(\d+)?\s*(?:active\s+)?(?:STRAP\s+)?wash", query, re.I)
    if min_match:
        min_washes = int(min_match.group(1))
    if max_match:
        max_washes = int(max_match.group(1))
    if exact_match:
        exact = int(exact_match.group(1) or 1)
        min_washes = exact
        max_washes = exact
    return min_washes, max_washes


def _extract_metrics(query: str) -> list[str]:
    text = query.lower()
    metrics: list[str] = []
    for phrase, metric in _METRIC_ALIASES.items():
        if phrase in text:
            metrics.append(metric)
    if "cost-vs-circularity" in text:
        metrics.extend(["total_cost", "circularity"])
    if "cost-vs-emissions" in text or "cost vs emissions" in text:
        metrics.extend(["total_cost", "emissions"])
    return _dedupe(metrics)


def _match_hsp_alias(text: str, aliases: set[str]) -> str | None:
    normalized = f" {normalize_hsp_key(text)} "
    for alias in sorted(aliases, key=len, reverse=True):
        if f" {normalize_hsp_key(alias)} " in normalized:
            return alias
    return None


def _extract_hsp_polymer_category(query: str) -> str | None:
    normalized = normalize_hsp_key(query)
    candidate_regions = [normalized]
    against_match = re.search(r"\b(?:screen|compare|evaluate|test)\s+(.+?)\s+against\b", normalized)
    if against_match:
        candidate_regions.insert(0, against_match.group(1))

    family_aliases = {
        key
        for key in POLYMER_CATEGORY_ALIASES
        if key
        not in {
            "nonpolar",
            "non polar",
            "low polarity",
            "weakly polar",
            "polar",
            "hydrogen bonding",
            "h bonding",
            "semicrystalline",
            "amorphous",
        }
    }
    for region in candidate_regions:
        match = _match_hsp_alias(region, family_aliases)
        if match:
            return match

    contextual = re.search(
        r"\b(nonpolar|non polar|low polarity|weakly polar|polar|hydrogen bonding|h bonding|semicrystalline|amorphous)\s+polymers?\b",
        normalized,
    )
    return contextual.group(1) if contextual else None


def _extract_hsp_solvent_selectors(query: str) -> tuple[str | None, str | None]:
    normalized = normalize_hsp_key(query)
    solvent_region = normalized
    against_match = re.search(r"\bagainst\s+(.+?)\s+solvents?\b", normalized)
    if against_match:
        solvent_region = against_match.group(1)
    category = _match_hsp_alias(solvent_region, set(SOLVENT_CATEGORY_ALIASES))
    polarity = _match_hsp_alias(solvent_region, set(SOLVENT_POLARITY_ALIASES))
    return category, polarity


def _normalize_extracted_polymer(raw: str) -> str | None:
    return _POLYMER_ALIASES.get(str(raw or "").strip().upper())


def _extract_polymer_solvent_filters(query: str) -> dict[str, list[str]]:
    """Extract explicit optimizer shortlists such as ``PP: Toluene and Cyclohexane``."""
    filters: dict[str, list[str]] = {}
    polymer_pattern = "|".join(sorted((re.escape(key) for key in _POLYMER_ALIASES), key=len, reverse=True))
    for match in re.finditer(
        rf"\b({polymer_pattern})\s*:\s*([^.;\n]+)",
        query,
        re.I,
    ):
        polymer = _normalize_extracted_polymer(match.group(1))
        if polymer is None:
            continue
        solvents = _extract_solvents(match.group(2))
        if solvents:
            filters[polymer] = solvents
    return filters


def _extract_route_candidates(query: str) -> list[dict[str, Any]]:
    """Extract a small explicit route constraint for selected Pareto plots."""
    lowered = query.lower()
    if not any(word in lowered for word in ("route", "frontier", "anchor", "constrain", "force")):
        return []
    route_region = query
    region_match = re.search(
        r"(?:constrain|force|restrict|anchor)[^.]{0,180}?(?:route|frontier|pareto)[^.]{0,180}",
        query,
        re.I,
    )
    if region_match:
        route_region = region_match.group(0)

    polymers, _aliases = _extract_polymers(route_region)
    solvents = _extract_solvents(route_region)
    if len(polymers) != 1 or len(solvents) != 1:
        return []
    polymer = polymers[0]
    solvent = solvents[0]
    route_id = f"{polymer.lower()}_{re.sub(r'[^a-z0-9]+', '_', solvent.lower()).strip('_')}_route"
    return [
        {
            "route_id": route_id,
            "rank": 1,
            "sequence": [polymer],
            "source": "query_constraint",
            "polymer_solvent_map": {polymer: solvent},
            "step_conditions": [
                {
                    "polymer": polymer,
                    "solvent": solvent,
                    "optimizer_option": solvent,
                }
            ],
        }
    ]


def _extract_optimizer_constraint_modes(query: str) -> tuple[str | None, str | None, str | None]:
    text = query.lower()
    constraint_mode = None
    fallback_policy = None
    route_pool_mode = None
    for value in ("ranked_soft", "ranked soft", "fixed", "hard", "soft"):
        if value in text:
            constraint_mode = value.replace(" ", "_")
            break
    for value in ("fail_closed", "fail closed", "broaden_disclosed", "broaden disclosed"):
        if value in text:
            fallback_policy = value.replace(" ", "_")
            break
    for value in ("slot_independent", "slot independent", "exact"):
        if value in text:
            route_pool_mode = value.replace(" ", "_")
            break
    return constraint_mode, fallback_policy, route_pool_mode


def _extract_plot_title(query: str) -> str | None:
    match = re.search(r"\btitle (?:it|the plot|the figure)?\s*[\"']([^\"']+)[\"']", query, re.I)
    if match:
        return match.group(1).strip()
    return None


def _extract_requested_artifacts(query: str) -> list[str]:
    text = query.lower()
    artifacts: list[str] = []
    pareto_requested = _pareto_requested(query)
    optimization_requested = _optimization_requested(query)
    if (
        "safety card" in text
        or "safety profile" in text
        or "safely heat" in text
        or "peroxide formation" in text
        or ("flammability" in text and "compare" in text)
        or ("volatility" in text and "compare" in text)
    ):
        artifacts.append("solvent_safety_comparison" if "compare" in text else "solvent_safety_card")
    if "hsp" in text or "hansen" in text or "red" in text:
        artifacts.append("hsp_red_heatmap" if "heatmap" in text or "screen" in text else "hsp_single_pair_summary")
    if "dynamic-programming" in text or "dynamic programming" in text:
        artifacts.extend(["separation_topk_sequences", "optimization_stage_candidates"])
    if "state map" in text:
        artifacts.append("separation_dp_state_map")
    if "separation tree" in text:
        artifacts.append("separation_tree_plot")
    hsp_requested = "hsp" in text or "hansen" in text or "red" in text
    if "selectivity heatmap" in text or ("compatibility heatmap" in text and not hsp_requested):
        artifacts.append("separation_selectivity_heatmap")
    if optimization_requested or pareto_requested:
        if pareto_requested:
            if "fixed feed compositions" in text or "composition slices" in text or "one png per composition" in text:
                artifacts.append("optimization_pareto_slices")
            else:
                artifacts.extend(["optimization_pareto_front", "optimization_pareto_landscape"])
        else:
            artifacts.append("optimization_point_result")
    if "plot" in text or "figure" in text or "png" in text or "visual" in text:
        if pareto_requested and "fixed feed compositions" in text:
            artifacts.append("optimization_pareto_slices_plot")
        elif pareto_requested:
            artifacts.append("optimization_pareto_plot")
        elif optimization_requested:
            artifacts.append("optimization_point_plot")
    if pareto_requested and "landscape" in text and re.search(r"\b(?:generate|create|make|save|plot|figure|visual)\b", text):
        artifacts.append("optimization_pareto_plot")
    if (
        "biosteam" in text
        or "capex" in text
        or "opex" in text
        or re.search(r"\btea\b|\blca\b|\bmsp\b", text)
        or "techno-economic" in text
        or "technoeconomic" in text
        or "life cycle" in text
        or "life-cycle" in text
        or "minimum selling price" in text
    ):
        artifacts.append("biosteam_tea_lca_result")
        if "plot" in text or "chart" in text or "visual" in text or "png" in text:
            artifacts.append("biosteam_tea_lca_plot")
    return _dedupe(artifacts)


def _extract_forbidden_artifacts(query: str) -> list[str]:
    text = query.lower()
    artifacts: list[str] = []
    if "state map" in text:
        artifacts.append("solubility_curve")
    return artifacts


def _extract_workflow_markers(query: str) -> list[str]:
    text = query.lower()
    markers: list[str] = []
    if "separation engineer" in text:
        markers.append("separation")
    if "pass" in text and ("optimization" in text or "optimizer" in text):
        markers.append("handoff")
    if "optimization engineer" in text or "optimize" in text or "pareto" in text:
        markers.append("optimization")
    if "visualization specialist" in text or "plot" in text or "figure" in text or "png" in text:
        markers.append("visualization")
    if "five fixed feed compositions" in text or "one png per composition" in text:
        markers.append("multi_slice")
    return _dedupe(markers)


def _split_output_destination(raw_path: str) -> tuple[str, str | None]:
    destination = extract_output_destination(
        f"save to {raw_path}",
        output_extensions=set(_OUTPUT_PATH_EXTENSIONS),
    )
    if destination is not None:
        return destination.output_dir, destination.filename_hint
    normalized = normalize_runtime_path(raw_path)
    path = Path(normalized)
    if path.suffix.lower() in _OUTPUT_PATH_EXTENSIONS:
        return str(path.parent), path.name
    return normalized, None


def _extract_output_destination(query: str) -> tuple[str | None, str | None]:
    destination = extract_output_destination(
        query,
        output_extensions=set(_OUTPUT_PATH_EXTENSIONS),
    )
    if destination is not None:
        return destination.output_dir, destination.filename_hint
    return None, None


def extract_facts(query: str, context: dict[str, Any] | None = None) -> ExtractedFacts:
    """Extract deterministic facts from a user query."""
    context = context or {}
    hydrated = bool(context.get("hydrated_from_typed_runtime"))
    polymers, polymer_aliases = _extract_polymers(query)
    feed_composition = _extract_feed_composition(query)
    if not feed_composition:
        feed_composition = {
            str(key): float(value)
            for key, value in _context_dict(context, "feed_composition").items()
            if value is not None
        }
    if feed_composition:
        polymers = _dedupe(list(feed_composition) + polymers)
    if not polymers:
        polymers = [str(item) for item in _context_list(context, "polymers") if str(item)]
    composition_slices = _extract_composition_slices(query, polymers)
    if not composition_slices:
        composition_slices = [
            {str(key): float(value) for key, value in item.items()}
            for item in _context_list(context, "composition_slices")
            if isinstance(item, dict)
        ]
    min_washes, max_washes = _extract_washes(query)
    if min_washes is None and context.get("min_washes") is not None:
        min_washes = int(context["min_washes"])
    if max_washes is None and context.get("max_washes") is not None:
        max_washes = int(context["max_washes"])
    pareto_negated = _pareto_negated(query)
    pareto_requested = _pareto_requested(query)
    single_objective_requested = _single_objective_requested(query)
    objective = _extract_objective(query)
    if pareto_negated and objective is None:
        objective = "max_profit"
    objective = objective or (str(context["objective"]) if context.get("objective") else None)
    query_output_dir, query_filename_hint = _extract_output_destination(query)
    if hydrated:
        output_dir = query_output_dir or context.get("output_dir")
        output_filename_hint = query_filename_hint or context.get("output_filename_hint")
    else:
        output_dir = context.get("output_dir") or query_output_dir
        output_filename_hint = context.get("output_filename_hint") or query_filename_hint
    hsp_solvent_category, hsp_solvent_polarity = _extract_hsp_solvent_selectors(query)
    polymer_solvent_filters = _extract_polymer_solvent_filters(query)
    route_candidates = _extract_route_candidates(query)
    constraint_mode, fallback_policy, route_pool_mode = _extract_optimizer_constraint_modes(query)
    requested_artifact_types = _extract_requested_artifacts(query)
    context_requested = [str(item) for item in _context_list(context, "requested_artifact_types") if str(item)]
    if context_requested and (hydrated or not requested_artifact_types):
        requested_artifact_types = _dedupe(requested_artifact_types + context_requested)
    # Route-plan deliverables are authoritative intent from the LLM planner:
    # merge them unconditionally so keyword detection is only a fallback.
    plan_requested = [str(item) for item in _context_list(context, "plan_requested_artifact_types") if str(item)]
    if plan_requested:
        requested_artifact_types = _dedupe(requested_artifact_types + plan_requested)
    workflow_markers = _extract_workflow_markers(query)
    context_markers = [str(item) for item in _context_list(context, "workflow_markers") if str(item)]
    if context_markers and hydrated:
        workflow_markers = _dedupe(workflow_markers + context_markers)
    # Markers derived from the route plan's step graph are authoritative.
    plan_markers = [str(item) for item in _context_list(context, "plan_workflow_markers") if str(item)]
    if plan_markers:
        workflow_markers = _dedupe(workflow_markers + plan_markers)
    metrics = _extract_metrics(query)
    if not metrics:
        metrics = [str(item) for item in _context_list(context, "metrics") if str(item)]
    return ExtractedFacts(
        query=query,
        polymers=polymers,
        polymer_aliases=polymer_aliases,
        solvents=_extract_solvents(query),
        temperatures_c=_extract_temperatures(query),
        feed_capacity_tpy=_extract_capacity(query) or context.get("feed_capacity_tpy"),
        feed_composition=feed_composition,
        composition_slices=composition_slices,
        scenario=_extract_scenario(query) or (str(context["scenario"]) if context.get("scenario") else None),
        energy_case=(context.get("energy_case") or _extract_energy_case(query)),
        polymer_solvent_filters=polymer_solvent_filters,
        route_candidates=route_candidates,
        constraint_mode=constraint_mode,
        fallback_policy=fallback_policy,
        route_pool_mode=route_pool_mode,
        hsp_polymer_category=_extract_hsp_polymer_category(query),
        hsp_solvent_category=hsp_solvent_category,
        hsp_solvent_polarity=hsp_solvent_polarity,
        top_k_per_polymer=_extract_top_k(query) or context.get("top_k_per_polymer"),
        n_points=_extract_n_points(query) or context.get("n_points"),
        min_washes=min_washes,
        max_washes=max_washes,
        objective=objective,
        pareto_requested=pareto_requested,
        pareto_negated=pareto_negated,
        single_objective_requested=single_objective_requested,
        metrics=metrics,
        requested_artifact_types=requested_artifact_types,
        forbidden_artifact_types=_extract_forbidden_artifacts(query),
        output_dir=normalize_runtime_path(output_dir) if output_dir else None,
        output_filename_hint=str(output_filename_hint) if output_filename_hint else None,
        plot_title=_extract_plot_title(query),
        workflow_markers=workflow_markers,
    )
