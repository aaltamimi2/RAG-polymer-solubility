"""Relevancy scoring for literature and patent search results.

Dual-signal approach:
  1. Keyword/heuristic scoring against DISSOLVE domain vocabulary (zero LLM cost)
  2. Vector similarity against existing RAG index (reuses ScientificEmbedder)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Domain vocabulary (extracted from DISSOLVE system prompt + subagents)
# ------------------------------------------------------------------

TIER1_PHRASES: list[str] = [
    "polymer dissolution", "solvent-based separation", "hansen solubility",
    "selective dissolution", "polymer recycling", "solvent recovery",
    "multilayer packaging", "polymer waste", "plastic recycling",
    "dissolution temperature", "polymer solubility", "solubility parameter",
    "antisolvent precipitation", "differential dissolution",
    "solvent-based recycling", "selective solubility", "polymer separation",
    "delamination", "multilayer film", "polymer blend separation",
]

TIER2_STEMS: list[str] = [
    "polymer", "solvent", "dissolution", "solubility", "recycling",
    "separation", "selectivity", "precipitat", "polyethylene",
    "polypropylene", "polystyrene", "nylon", "polyamide",
    "toluene", "xylene", "dmf", "nmp", "thf", "mek", "acetone",
    "dichloromethane", "chloroform", "cyclohexan",
    "green solvent", "bio-based", "circular economy",
    "hdpe", "ldpe", "lldpe", "pet", "pvc", "abs", "hips",
    "waste stream", "mixed plastic", "packaging waste",
]

TIER3_STEMS: list[str] = [
    "chemical engineer", "process design", "thermodynamic",
    "flory-huggins", "chi parameter", "miscibility", "compatibility",
    "melt process", "extrusion", "injection mold", "rheolog",
    "crystallin", "glass transition", "melting point",
    "distillation", "evaporation", "filtration",
    "life cycle", "techno-economic", "sustainability",
]

TIER1_WEIGHT = 3.0
TIER2_WEIGHT = 1.0
TIER3_WEIGHT = 0.3

# Max possible score used for normalisation
_MAX_KEYWORD_SCORE = (
    len(TIER1_PHRASES) * TIER1_WEIGHT
    + len(TIER2_STEMS) * TIER2_WEIGHT
    + len(TIER3_STEMS) * TIER3_WEIGHT
)

# Thresholds for categories
HIGH_THRESHOLD = 0.6
MEDIUM_THRESHOLD = 0.3

# Weights for combining signals
KEYWORD_WEIGHT = 0.6
VECTOR_WEIGHT = 0.4


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class RelevancyResult:
    title: str
    score: float  # combined relevancy score [0.0, 1.0]
    keyword_score: float
    vector_score: float
    category: str  # "HIGH", "MEDIUM", "LOW"
    matched_terms: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Keyword scorer
# ------------------------------------------------------------------

def compute_keyword_score(text: str) -> Tuple[float, list[str]]:
    """Score text against DISSOLVE domain vocabulary.

    Returns (normalised_score, matched_terms).
    """
    if not text:
        return 0.0, []

    text_lower = text.lower()
    raw_score = 0.0
    matched: list[str] = []

    # Tier 1: exact phrase matching (case-insensitive)
    for phrase in TIER1_PHRASES:
        if phrase in text_lower:
            raw_score += TIER1_WEIGHT
            matched.append(phrase)

    # Tier 2: stem/substring matching
    for stem in TIER2_STEMS:
        if stem in text_lower:
            raw_score += TIER2_WEIGHT
            matched.append(stem)

    # Tier 3: stem/substring matching
    for stem in TIER3_STEMS:
        if stem in text_lower:
            raw_score += TIER3_WEIGHT
            matched.append(stem)

    # Apply sqrt scaling to reward partial matches
    if _MAX_KEYWORD_SCORE > 0:
        ratio = raw_score / _MAX_KEYWORD_SCORE
        normalised = min(1.0, ratio ** 0.5)  # sqrt scaling
    else:
        normalised = 0.0

    return normalised, matched


# ------------------------------------------------------------------
# Vector scorer (uses existing RAG embedder)
# ------------------------------------------------------------------

def compute_vector_score(text: str) -> float:
    """Embed *text* and compare against existing RAG index.

    Returns average cosine similarity of top-3 nearest chunks,
    or 0.5 (neutral) if the RAG system is unavailable/empty.
    """
    if not text or len(text.strip()) < 20:
        return 0.5

    try:
        from strap.vendor.rag import get_rag_system

        rag = get_rag_system()

        # Check RAG is ready with indexed docs
        if not rag.is_ready() or rag.vector_db is None:
            return 0.5

        status = rag.get_status()
        points_count = status.get("collection", {}).get("points_count", 0)
        if points_count == 0:
            return 0.5

        # Encode the candidate text
        embedding = rag.embedder.encode_dense(
            [text], is_query=True, normalize=True
        )[0]

        # Query Qdrant for top-3 most similar existing chunks
        from qdrant_client.models import models as qmodels

        hits = rag.vector_db.client.query_points(
            collection_name=rag.vector_db.collection_name,
            query=embedding.tolist(),
            using="dense",
            limit=3,
        ).points

        if not hits:
            return 0.5

        scores = [h.score for h in hits]
        avg_sim = float(np.mean(scores))

        # Cosine similarity is in [-1, 1]; clamp to [0, 1]
        return max(0.0, min(1.0, avg_sim))

    except Exception as e:
        logger.debug(f"Vector scoring unavailable: {e}")
        return 0.5


# ------------------------------------------------------------------
# Combined scorer
# ------------------------------------------------------------------

def score_result(
    title: str,
    abstract_or_snippet: str,
    keywords: Optional[list[str]] = None,
) -> RelevancyResult:
    """Score a single search result for domain relevancy."""

    # Combine available text
    parts = [title or "", abstract_or_snippet or ""]
    if keywords:
        parts.append(" ".join(keywords))
    combined = " ".join(parts)

    kw_score, matched = compute_keyword_score(combined)
    vec_score = compute_vector_score(combined)

    combined_score = KEYWORD_WEIGHT * kw_score + VECTOR_WEIGHT * vec_score

    if combined_score >= HIGH_THRESHOLD:
        category = "HIGH"
    elif combined_score >= MEDIUM_THRESHOLD:
        category = "MEDIUM"
    else:
        category = "LOW"

    return RelevancyResult(
        title=title or "Untitled",
        score=round(combined_score, 3),
        keyword_score=round(kw_score, 3),
        vector_score=round(vec_score, 3),
        category=category,
        matched_terms=matched,
    )


# ------------------------------------------------------------------
# Batch filter
# ------------------------------------------------------------------

def filter_results(
    results: list[dict],
    text_field: str = "snippet",
    title_field: str = "title",
    keywords_field: Optional[str] = None,
    min_category: str = "MEDIUM",
) -> Tuple[list[dict], list[RelevancyResult]]:
    """Filter a list of parsed search results by domain relevancy.

    Args:
        results: list of dicts from a search tool
        text_field: key containing abstract/snippet
        title_field: key containing the paper title
        keywords_field: optional key containing keyword list
        min_category: minimum category to keep ("HIGH", "MEDIUM", "LOW")

    Returns:
        (kept_results, all_scores) — kept results and full score list
    """
    category_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    min_rank = category_rank.get(min_category, 2)

    kept: list[dict] = []
    all_scores: list[RelevancyResult] = []

    for item in results:
        title = item.get(title_field, "")
        text = item.get(text_field, "") or ""
        kws = item.get(keywords_field, []) if keywords_field else None

        rel = score_result(title, text, kws)
        all_scores.append(rel)

        if category_rank.get(rel.category, 0) >= min_rank:
            item["_relevancy"] = rel
            kept.append(item)

    return kept, all_scores


def format_relevancy_summary(
    all_scores: list[RelevancyResult],
    kept_count: int,
    total_count: int,
) -> str:
    """Return a short markdown summary of filtering results."""
    high = sum(1 for s in all_scores if s.category == "HIGH")
    med = sum(1 for s in all_scores if s.category == "MEDIUM")
    low = sum(1 for s in all_scores if s.category == "LOW")
    removed = total_count - kept_count

    lines = [
        "## Relevancy Filtering",
        f"**Kept:** {kept_count} of {total_count} results "
        f"({high} HIGH, {med} MEDIUM, {low} LOW)",
    ]
    if removed:
        lines.append(f"**Filtered out:** {removed} low-relevancy results")
    return "\n".join(lines)
