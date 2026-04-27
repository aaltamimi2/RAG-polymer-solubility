"""Structured query-context extraction for planner-facing user inputs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from .services.contaminant_data_service import (
    list_supported_contaminant_families,
    list_supported_contaminants,
)
from .solubility import POLYMER_ALIASES, get_available_polymers
from .solvent_registry import SOLVENT_REGISTRY

_SEQUENCE_PATTERNS = (
    re.compile(r"\band then\b", re.IGNORECASE),
    re.compile(r"\bfollowed by\b", re.IGNORECASE),
    re.compile(r"\busing the result\b", re.IGNORECASE),
    re.compile(r"\bbased on the result\b", re.IGNORECASE),
    re.compile(r"\bafter\b", re.IGNORECASE),
    re.compile(r"\bthen\b", re.IGNORECASE),
    re.compile(r"\bfinally\b", re.IGNORECASE),
)
_RESEARCH_TOPIC_LEAD_RE = re.compile(
    r"\b(?:literature(?: search)?|google scholar|web of science|patent(?: search)?|patents?)\b"
    r"[^.?!]{0,120}?\bfor\b",
    re.IGNORECASE,
)
_ROUTE_PATTERNS = (
    ("separation.route", re.compile(r"\boptimal separation sequence\b", re.IGNORECASE)),
    ("separation.route", re.compile(r"\bseparation sequences?\b", re.IGNORECASE)),
    ("separation.route", re.compile(r"\bseparation routes?\b", re.IGNORECASE)),
    ("separation.route", re.compile(r"\bsolvent routes?\b", re.IGNORECASE)),
    ("separation.route", re.compile(r"\btop separation routes?\b", re.IGNORECASE)),
    ("separation.route", re.compile(r"\bselective dissolution\b", re.IGNORECASE)),
    ("separation.route", re.compile(r"\bprocess design\b", re.IGNORECASE)),
    ("separation.feasibility", re.compile(r"\bfeasible\b", re.IGNORECASE)),
    ("separation.feasibility", re.compile(r"\bfeasibility\b", re.IGNORECASE)),
    ("separation.feasibility", re.compile(r"\bboiling point\b", re.IGNORECASE)),
    ("solvent.shortlist", re.compile(r"\bshortlist\b", re.IGNORECASE)),
    ("solvent.shortlist", re.compile(r"\bbest solvents?\b", re.IGNORECASE)),
    ("solvent.shortlist", re.compile(r"\brank solvents?\b", re.IGNORECASE)),
    ("solvent.shortlist", re.compile(r"\bsolvent screening\b", re.IGNORECASE)),
    ("route.wash_step", re.compile(r"\bwash step\b", re.IGNORECASE)),
    ("route.solvent_recovery", re.compile(r"\bsolvent recovery\b", re.IGNORECASE)),
    ("route.atmospheric_pressure", re.compile(r"\batmospheric pressure\b", re.IGNORECASE)),
)
_REQUEST_PATTERNS = (
    ("visualization.plot", re.compile(r"\b(plot|chart|graph|visualiz|dashboard|figure|heatmap|diagram)\b", re.IGNORECASE)),
    ("literature.search", re.compile(r"\b(literature|google scholar|web of science|research articles?|papers?|journal)\b", re.IGNORECASE)),
    ("patent.search", re.compile(r"\bpatents?\b", re.IGNORECASE)),
    ("literature.answer", re.compile(r"\b(rag|retrieval-augmented|indexed documents?|retrieved findings|retrieval diagnostics)\b", re.IGNORECASE)),
    ("statistics.analysis", re.compile(r"\b(statistics?|statistical|confidence interval|hypothesis|anova|correlation|regression)\b", re.IGNORECASE)),
    ("thermal.prediction", re.compile(r"\b(glass transition|tg\b|melting|thermal prediction|thermal propert)\b", re.IGNORECASE)),
    ("ml.prediction", re.compile(r"\b(machine learning|ml prediction|hansen|hsp|relative energy difference|red\b)\b", re.IGNORECASE)),
    ("safety.assessment", re.compile(r"\b(gsk|gscore|pubchem|ghs|hazard|toxicity|toxic|health risk|risk profile|safety score|safety scores|exposure|flammab|flammability|sds|msds)\b", re.IGNORECASE)),
    ("tea.economics", re.compile(r"\b(tea|techno[- ]economic|biosteam|msp|capex|opex|operating cost|capital cost|payback)\b", re.IGNORECASE)),
    ("lca.environmental", re.compile(r"\b(lca|life cycle|gwp|emissions?|environmental)\b", re.IGNORECASE)),
    ("contaminant.screening", re.compile(r"\b(contaminant screening|strap contaminant removal|remove phthalates|remove pfas|leaching mode|pfas|phthalate|contamin|decontamin)\b", re.IGNORECASE)),
    ("optimization.pathway", re.compile(r"\b(optimi[sz](?:e|ation)|max(?:imize)? profit|min(?:imize)? emissions?|min(?:imize)? cost|max(?:imize)? circularity|superstructure|pyomo|minlp|optimal pathway|waste management)\b", re.IGNORECASE)),
)
_POLYMER_CANONICAL_MAP = {
    "POLYETHYLENE": "PE",
    "POLYPROPYLENE": "PP",
    "POLYSTYRENE": "PS",
    "POLYCARBONATE": "PC",
    "POLYETHYLENE TEREPHTHALATE": "PET",
    "POLYVINYL CHLORIDE": "PVC",
    "POLYVINYLCHLORIDE": "PVC",
    "POLYETHERSULFONE": "PES",
    "NYLON 6": "NYLON6",
    "NYLON-6": "NYLON6",
    "NYLON 66": "NYLON66",
    "NYLON-66": "NYLON66",
    "PA6": "NYLON6",
    "PA66": "NYLON66",
    "PE": "PE",
    "HDPE": "HDPE",
    "LDPE": "LDPE",
    "LLDPE": "LLDPE",
    "EVOH": "EVOH",
    "PET": "PET",
    "PETG": "PETG",
    "PS": "PS",
    "PVC": "PVC",
    "PC": "PC",
    "PES": "PES",
    "PP": "PP",
    "PMMA": "PMMA",
    "ABS": "ABS",
    "PVDF": "PVDF",
}
_CONTAMINANT_FAMILY_ALIASES = {
    "pfas": "PFAS",
    "per- and polyfluoroalkyl substances": "PFAS",
    "perfluoroalkyl substances": "PFAS",
    "phthalate": "Phthalates",
    "phthalates": "Phthalates",
}
_AMBIGUOUS_SOLVENT_ALIASES = {
    # DEG is a solvent abbreviation, but "deg C" is far more common in user
    # temperature text. Require the full solvent name in query-context memory.
    "deg",
    "tea",
}
_FEED_CAPACITY_PATTERNS = (
    re.compile(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:metric\s+tons?|tonnes?|tons?|mt)\s*/\s*(?:year|yr)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:metric\s+tons?|tonnes?|tons?|mt)\s+per\s+year\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*t\s*/\s*y\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class QuerySpan:
    kind: str
    normalized: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class QueryContext:
    text: str
    polymer_spans: tuple[QuerySpan, ...]
    solvent_spans: tuple[QuerySpan, ...]
    contaminant_spans: tuple[QuerySpan, ...]
    contaminant_family_spans: tuple[QuerySpan, ...]
    route_spans: tuple[QuerySpan, ...]
    request_spans: tuple[QuerySpan, ...]
    sequence_spans: tuple[QuerySpan, ...]
    feed_composition_items: tuple[tuple[str, float], ...]
    feed_composition_slices_items: tuple[tuple[tuple[str, float], ...], ...]
    feed_capacity_tpy: float | None

    @property
    def polymers(self) -> tuple[str, ...]:
        return _unique_span_values(self.polymer_spans)

    @property
    def solvents(self) -> tuple[str, ...]:
        return _unique_span_values(self.solvent_spans)

    @property
    def contaminants(self) -> tuple[str, ...]:
        return _unique_span_values(self.contaminant_spans)

    @property
    def contaminant_families(self) -> tuple[str, ...]:
        return _unique_span_values(self.contaminant_family_spans)

    @property
    def route_labels(self) -> tuple[str, ...]:
        return _unique_span_values(self.route_spans)

    @property
    def request_labels(self) -> tuple[str, ...]:
        return _unique_span_values(self.request_spans)

    @property
    def feed_composition(self) -> dict[str, float]:
        return dict(self.feed_composition_items)

    @property
    def feed_composition_slices(self) -> tuple[dict[str, float], ...]:
        return tuple(dict(items) for items in self.feed_composition_slices_items)

    @property
    def available_inputs(self) -> frozenset[str]:
        available: set[str] = set()
        if self.polymers:
            available.update({
                "user.polymers",
                "user.target_plastic",
                "user.target_polymer",
            })
        if self.contaminants or self.contaminant_families:
            available.add("user.contaminants")
        if self.solvents or self.route_spans:
            available.add("user.solvents_or_route")
        if "visualization.plot" in self.request_labels:
            available.add("user.visualization_request")
        if {
            "literature.search",
            "patent.search",
            "literature.answer",
        } & set(self.request_labels) or "?" in self.text:
            available.add("user.research_question")
        if {
            "statistics.analysis",
            "ml.prediction",
            "thermal.prediction",
        } & set(self.request_labels):
            available.add("user.data_or_prediction_target")
        if "optimization.pathway" in self.request_labels:
            available.add("user.optimization_request")
        if self.feed_composition_items or self.feed_composition_slices_items:
            available.add("user.feed_composition")
        if self.feed_capacity_tpy is not None:
            available.add("user.feed_capacity")
        return frozenset(available)


def _unique_span_values(spans: tuple[QuerySpan, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for span in spans:
        if span.normalized in seen:
            continue
        seen.add(span.normalized)
        values.append(span.normalized)
    return tuple(values)


def _contains_overlap(accepted: list[QuerySpan], start: int, end: int) -> bool:
    return any(not (end <= span.start or start >= span.end) for span in accepted)


def _range_overlaps(start: int, end: int, excluded_ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(not (end <= excluded_start or start >= excluded_end) for excluded_start, excluded_end in excluded_ranges)


def _compile_term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _collect_term_spans(
    text: str,
    *,
    kind: str,
    term_map: dict[str, str],
) -> tuple[QuerySpan, ...]:
    accepted: list[QuerySpan] = []
    patterns = [
        (term, normalized, _compile_term_pattern(term))
        for term, normalized in sorted(term_map.items(), key=lambda item: (-len(item[0]), item[0]))
    ]
    for term, normalized, pattern in patterns:
        for match in pattern.finditer(text):
            if _contains_overlap(accepted, match.start(), match.end()):
                continue
            accepted.append(
                QuerySpan(
                    kind=kind,
                    normalized=normalized,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
    accepted.sort(key=lambda span: (span.start, span.end, span.normalized))
    return tuple(accepted)


def _collect_pattern_spans(
    text: str,
    *,
    kind: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...] | tuple[re.Pattern[str], ...],
    excluded_ranges: tuple[tuple[int, int], ...] = (),
) -> tuple[QuerySpan, ...]:
    accepted: list[QuerySpan] = []
    if patterns and isinstance(patterns[0], tuple):
        labeled_patterns = patterns  # type: ignore[assignment]
    else:
        labeled_patterns = tuple(("sequence.marker", pattern) for pattern in patterns)  # type: ignore[arg-type]
    for normalized, pattern in labeled_patterns:
        for match in pattern.finditer(text):
            if _range_overlaps(match.start(), match.end(), excluded_ranges):
                continue
            if _contains_overlap(accepted, match.start(), match.end()):
                continue
            accepted.append(
                QuerySpan(
                    kind=kind,
                    normalized=normalized,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
    accepted.sort(key=lambda span: (span.start, span.end, span.normalized))
    return tuple(accepted)


def _collect_feed_fraction_items(text: str) -> tuple[tuple[str, float], ...]:
    accepted: list[QuerySpan] = []
    feed_items: list[tuple[str, float]] = []
    patterns = [
        (term, normalized, _compile_term_pattern(term))
        for term, normalized in sorted(_polymer_term_map().items(), key=lambda item: (-len(item[0]), item[0]))
    ]
    for _term, normalized, pattern in patterns:
        for match in pattern.finditer(text):
            if _contains_overlap(accepted, match.start(), match.end()):
                continue
            prefix = text[max(0, match.start() - 24):match.start()]
            percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*$", prefix)
            if percent_match is None:
                continue
            accepted.append(
                QuerySpan(
                    kind="feed_fraction",
                    normalized=normalized,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
            feed_items.append((normalized, float(percent_match.group(1)) / 100.0))
    deduped: dict[str, float] = {}
    for polymer, fraction in feed_items:
        deduped[polymer] = fraction
    return tuple(deduped.items())


def _collect_feed_composition_slices(
    text: str,
    polymer_spans: tuple[QuerySpan, ...],
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Extract slash-delimited composition slices such as 20/60/20.

    The parser intentionally requires at least three requested polymers so a
    free-standing numeric fraction elsewhere in the prompt is not interpreted
    as a feed composition. Slice order follows the first three polymers as
    mentioned by the user, e.g. LDPE/EVOH/PET.
    """
    ordered_polymers: list[str] = []
    for span in polymer_spans:
        if span.normalized not in ordered_polymers:
            ordered_polymers.append(span.normalized)
        if len(ordered_polymers) >= 3:
            break
    if len(ordered_polymers) < 3:
        return ()

    slice_pattern = re.compile(
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?!\d|\.\d)"
    )
    slices: list[tuple[tuple[str, float], ...]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for match in slice_pattern.finditer(text):
        values = [float(match.group(i)) for i in range(1, 4)]
        total = sum(values)
        if 99.0 <= total <= 101.0:
            fractions = [value / 100.0 for value in values]
        elif 0.99 <= total <= 1.01:
            fractions = values
        else:
            continue
        item = tuple(
            (polymer, round(fraction, 6))
            for polymer, fraction in zip(ordered_polymers, fractions)
        )
        if item in seen:
            continue
        seen.add(item)
        slices.append(item)
    return tuple(slices)


def _extract_feed_capacity_tpy(text: str) -> float | None:
    for pattern in _FEED_CAPACITY_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        raw = match.group(1).replace(",", "")
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _collect_research_topic_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    for match in _RESEARCH_TOPIC_LEAD_RE.finditer(text):
        start = match.end()
        end_candidates = [len(text)]
        punctuation_match = re.search(r"[,.;?!]", text[start:])
        if punctuation_match is not None:
            end_candidates.append(start + punctuation_match.start())
        for pattern in _SEQUENCE_PATTERNS:
            sequence_match = pattern.search(text, start)
            if sequence_match is not None:
                end_candidates.append(sequence_match.start())
        end = min(candidate for candidate in end_candidates if candidate >= start)
        if start < end:
            ranges.append((start, end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


@lru_cache(maxsize=1)
def _polymer_term_map() -> dict[str, str]:
    term_map: dict[str, str] = {}
    for polymer in sorted(get_available_polymers()):
        term_map[polymer.lower()] = polymer.upper()
    for term, normalized in _POLYMER_CANONICAL_MAP.items():
        term_map[term.lower()] = normalized
    for alias, normalized in POLYMER_ALIASES.items():
        term_map.setdefault(alias.lower(), _POLYMER_CANONICAL_MAP.get(normalized, normalized))
    return term_map


@lru_cache(maxsize=1)
def _solvent_term_map() -> dict[str, str]:
    term_map: dict[str, str] = {}
    for key, entry in SOLVENT_REGISTRY.items():
        normalized = str(entry.get("interp_key") or key).strip().lower()
        if normalized:
            term_map[normalized] = normalized
        canonical = str(entry.get("canonical") or entry.get("property_db") or entry.get("biosteam") or "").strip()
        if canonical:
            term_map[canonical.lower()] = normalized or canonical.lower()
        for alias in entry.get("aliases", []):
            cleaned = str(alias).strip().lower()
            if cleaned in _AMBIGUOUS_SOLVENT_ALIASES:
                continue
            if cleaned:
                term_map[cleaned] = normalized or cleaned
    return term_map


@lru_cache(maxsize=1)
def _contaminant_term_map() -> dict[str, str]:
    term_map: dict[str, str] = {}
    for family in list_supported_contaminant_families():
        for name in list_supported_contaminants(family):
            lowered = name.lower()
            term_map[lowered] = name
            abbrev_match = re.search(r"\(([A-Za-z0-9-]+)\)", name)
            if abbrev_match:
                term_map[abbrev_match.group(1).lower()] = name
    return term_map


def _contaminant_family_term_map() -> dict[str, str]:
    return dict(_CONTAMINANT_FAMILY_ALIASES)


@lru_cache(maxsize=512)
def extract_query_context(text: str) -> QueryContext:
    """Extract structured entities and spans from a user query."""
    research_topic_ranges = _collect_research_topic_ranges(text)
    polymer_spans = _collect_term_spans(text, kind="polymer", term_map=_polymer_term_map())
    return QueryContext(
        text=text,
        polymer_spans=polymer_spans,
        solvent_spans=_collect_term_spans(text, kind="solvent", term_map=_solvent_term_map()),
        contaminant_spans=_collect_term_spans(text, kind="contaminant", term_map=_contaminant_term_map()),
        contaminant_family_spans=_collect_term_spans(
            text,
            kind="contaminant_family",
            term_map=_contaminant_family_term_map(),
        ),
        route_spans=_collect_pattern_spans(
            text,
            kind="route",
            patterns=_ROUTE_PATTERNS,
            excluded_ranges=research_topic_ranges,
        ),
        request_spans=_collect_pattern_spans(
            text,
            kind="request",
            patterns=_REQUEST_PATTERNS,
            excluded_ranges=research_topic_ranges,
        ),
        sequence_spans=_collect_pattern_spans(text, kind="sequence", patterns=_SEQUENCE_PATTERNS),
        feed_composition_items=_collect_feed_fraction_items(text),
        feed_composition_slices_items=_collect_feed_composition_slices(text, polymer_spans),
        feed_capacity_tpy=_extract_feed_capacity_tpy(text),
    )
