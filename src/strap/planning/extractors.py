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
    "HEPTANE": "Heptane",
    "N-HEPTANE": "Heptane",
    "CYCLOHEXANE": "Cyclohexane",
    "DIETHYL ETHER": "Diethyl ether",
    "DICHLOROMETHANE": "Dichloromethane",
    "DCM": "Dichloromethane",
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
    hsp_polymer_category: str | None = None
    hsp_solvent_category: str | None = None
    hsp_solvent_polarity: str | None = None
    top_k_per_polymer: int | None = None
    n_points: int | None = None
    min_washes: int | None = None
    max_washes: int | None = None
    objective: str | None = None
    metrics: list[str] = Field(default_factory=list)
    requested_artifact_types: list[str] = Field(default_factory=list)
    forbidden_artifact_types: list[str] = Field(default_factory=list)
    output_dir: str | None = None
    output_filename_hint: str | None = None
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
    return [
        float(match.group(1))
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:°\s*)?C\b", query, re.I)
    ]


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
    return slices


def _extract_scenario(query: str) -> str | None:
    match = re.search(r"\bscenario\s+([A-Z0-9]+)\b", query, re.I)
    return match.group(1).upper() if match else None


def _extract_energy_case(query: str) -> str | None:
    match = re.search(r"\b(?:energy\s+case|case|under)\s+(C[123])\b", query, re.I)
    return match.group(1).upper() if match else None


def _extract_top_k(query: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", query, re.I)
    return int(match.group(1)) if match else None


def _extract_n_points(query: str) -> int | None:
    match = re.search(r"\b(?:with\s+)?(\d+)\s+points?\b", query, re.I)
    return int(match.group(1)) if match else None


def _extract_washes(query: str) -> tuple[int | None, int | None]:
    min_washes = None
    max_washes = None
    min_match = re.search(r"(?:at least|require(?:ing)?)\s+(\d+)\s+STRAP wash", query, re.I)
    max_match = re.search(r"(?:up to|allow(?:ing)? up to|maximum of)\s+(\d+)\s+(?:STRAP\s+)?wash", query, re.I)
    if min_match:
        min_washes = int(min_match.group(1))
    if max_match:
        max_washes = int(max_match.group(1))
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


def _extract_requested_artifacts(query: str) -> list[str]:
    text = query.lower()
    artifacts: list[str] = []
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
    if "optimize" in text or "optimization" in text or "pareto" in text:
        if "pareto" in text:
            if "fixed feed compositions" in text or "composition slices" in text or "one png per composition" in text:
                artifacts.append("optimization_pareto_slices")
            else:
                artifacts.extend(["optimization_pareto_front", "optimization_pareto_landscape"])
        else:
            artifacts.append("optimization_point_result")
    if "plot" in text or "figure" in text or "png" in text or "visual" in text:
        if "pareto" in text and "fixed feed compositions" in text:
            artifacts.append("optimization_pareto_slices_plot")
        elif "pareto" in text:
            artifacts.append("optimization_pareto_plot")
        elif "optimization" in text or "optimize" in text:
            artifacts.append("optimization_point_plot")
    if "pareto" in text and "landscape" in text and re.search(r"\b(?:generate|create|make|save|plot|figure|visual)\b", text):
        artifacts.append("optimization_pareto_plot")
    if "biosteam" in text or "tea/lca" in text or "capex" in text or "opex" in text:
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
    normalized = normalize_runtime_path(raw_path)
    path = Path(normalized)
    if path.suffix.lower() in _OUTPUT_PATH_EXTENSIONS:
        return str(path.parent), path.name
    return normalized, None


def _extract_output_destination(query: str) -> tuple[str | None, str | None]:
    quoted = re.search(
        r"\b(?:save|write|store|output|export|put|place|create|generate)\b.{0,120}?\b(?:to|under|in|at)\s+"
        r"(?P<quote>[\"'`])(?P<path>.+?)(?P=quote)",
        query,
        re.I | re.S,
    )
    if quoted:
        return _split_output_destination(quoted.group("path"))

    unquoted = re.search(
        r"\b(?:save|write|store|output|export|put|place|create|generate)\b.{0,120}?\b(?:to|under|in|at)\s+"
        r"(?P<path>(?:\\\\|//|/|[A-Za-z]:[\\/])[^,.;\n]+)",
        query,
        re.I,
    )
    if unquoted:
        return _split_output_destination(unquoted.group("path").strip())
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
    objective = "max_profit" if "profit" in query.lower() or "maximize" in query.lower() else None
    objective = objective or (str(context["objective"]) if context.get("objective") else None)
    query_output_dir, query_filename_hint = _extract_output_destination(query)
    if hydrated:
        output_dir = query_output_dir or context.get("output_dir")
        output_filename_hint = query_filename_hint or context.get("output_filename_hint")
    else:
        output_dir = context.get("output_dir") or query_output_dir
        output_filename_hint = context.get("output_filename_hint") or query_filename_hint
    hsp_solvent_category, hsp_solvent_polarity = _extract_hsp_solvent_selectors(query)
    requested_artifact_types = _extract_requested_artifacts(query)
    context_requested = [str(item) for item in _context_list(context, "requested_artifact_types") if str(item)]
    if context_requested and (hydrated or not requested_artifact_types):
        requested_artifact_types = _dedupe(requested_artifact_types + context_requested)
    workflow_markers = _extract_workflow_markers(query)
    context_markers = [str(item) for item in _context_list(context, "workflow_markers") if str(item)]
    if context_markers and hydrated:
        workflow_markers = _dedupe(workflow_markers + context_markers)
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
        hsp_polymer_category=_extract_hsp_polymer_category(query),
        hsp_solvent_category=hsp_solvent_category,
        hsp_solvent_polarity=hsp_solvent_polarity,
        top_k_per_polymer=_extract_top_k(query) or context.get("top_k_per_polymer"),
        n_points=_extract_n_points(query) or context.get("n_points"),
        min_washes=min_washes,
        max_washes=max_washes,
        objective=objective,
        metrics=metrics,
        requested_artifact_types=requested_artifact_types,
        forbidden_artifact_types=_extract_forbidden_artifacts(query),
        output_dir=normalize_runtime_path(output_dir) if output_dir else None,
        output_filename_hint=str(output_filename_hint) if output_filename_hint else None,
        workflow_markers=workflow_markers,
    )
