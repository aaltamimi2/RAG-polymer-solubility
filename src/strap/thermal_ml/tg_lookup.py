"""Lightweight Tg lookup with fuzzy name search.

Pre-computed glass transition temperature predictions for ~7,700 polymers.
No model loading required — instant lookups from a JSON file.

Usage::

    from strap.thermal_ml.tg_lookup import search_tg, get_tg_by_psmiles

    # Search by name (fuzzy)
    results = search_tg("polystyrene")
    results = search_tg("PMMA")
    results = search_tg("nylon")

    # Search by structural tag
    results = search_tg("fluorinated aromatic")

    # Exact PSMILES lookup
    entry = get_tg_by_psmiles("[*]CC([*])c1ccccc1")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Singleton state
_LOOKUP_DATA: dict | None = None
_NAME_INDEX: dict[str, list[int]] | None = None
_PSMILES_INDEX: dict[str, int] | None = None
_ALL_SEARCHABLE: list[tuple[str, int]] | None = None

_LOOKUP_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "thermal_properties" / "tg_lookup.json"


def load_tg_lookup(path: str | Path | None = None) -> dict:
    """Load the Tg lookup JSON (lazy singleton).

    Parameters
    ----------
    path : str or Path, optional
        Override the default lookup JSON path.

    Returns
    -------
    dict
        The full lookup data with ``entries``, ``model_info``, ``stats``.
    """
    global _LOOKUP_DATA, _NAME_INDEX, _PSMILES_INDEX, _ALL_SEARCHABLE

    if _LOOKUP_DATA is not None and path is None:
        return _LOOKUP_DATA

    lookup_path = Path(path) if path else _LOOKUP_PATH
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Tg lookup file not found at {lookup_path}. "
            "Run scripts/build_tg_lookup.py to generate it."
        )

    with open(lookup_path) as f:
        _LOOKUP_DATA = json.load(f)

    entries = _LOOKUP_DATA["entries"]

    # Build indices
    _NAME_INDEX = {}
    _PSMILES_INDEX = {}
    _ALL_SEARCHABLE = []

    for i, entry in enumerate(entries):
        # PSMILES index
        _PSMILES_INDEX[entry["psmiles"]] = i

        # Name index (lowercase)
        for name in entry.get("names", []):
            key = name.lower().strip()
            _NAME_INDEX.setdefault(key, []).append(i)
            _ALL_SEARCHABLE.append((key, i))

        # Tag index (for keyword search)
        for tag in entry.get("tags", []):
            key = tag.lower().strip()
            _NAME_INDEX.setdefault(key, []).append(i)

    logger.info("Loaded Tg lookup: %d entries, %d searchable names",
                len(entries), len(_ALL_SEARCHABLE))

    return _LOOKUP_DATA


def _format_entry(entry: dict) -> dict[str, Any]:
    """Format a lookup entry for return."""
    tg = entry.get("Tg_K") or entry.get("Tg_K_predicted")
    return {
        "psmiles": entry["psmiles"],
        "names": entry.get("names", []),
        "tags": entry.get("tags", []),
        "Tg_K": tg,
        "Tg_K_predicted": entry.get("Tg_K_predicted"),
        "Tg_std": entry.get("Tg_std"),
        "source": entry.get("source", ""),
    }


def get_tg_by_psmiles(psmiles: str) -> dict[str, Any] | None:
    """Exact PSMILES lookup. Returns None if not found."""
    data = load_tg_lookup()
    idx = _PSMILES_INDEX.get(psmiles)
    if idx is None:
        return None
    return _format_entry(data["entries"][idx])


def search_tg(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search the Tg database by polymer name, abbreviation, or structural tag.

    Uses multi-strategy matching:
    1. Exact name/abbreviation match (case-insensitive)
    2. Fuzzy name match via rapidfuzz
    3. Tag/keyword match
    4. Direct PSMILES match if query looks like SMILES

    Parameters
    ----------
    query : str
        Polymer name, abbreviation, structural keyword, or PSMILES.
    top_k : int
        Maximum results to return.

    Returns
    -------
    list[dict]
        Matching entries sorted by relevance, each with keys:
        ``psmiles``, ``names``, ``tags``, ``Tg_K``, ``Tg_K_predicted``,
        ``Tg_std``, ``source``, ``match_score``, ``match_type``.
    """
    data = load_tg_lookup()
    entries = data["entries"]
    query_lower = query.lower().strip()

    results: list[tuple[float, str, int]] = []  # (score, match_type, index)
    seen_indices: set[int] = set()

    # Strategy 1: Exact name match
    if query_lower in _NAME_INDEX:
        for idx in _NAME_INDEX[query_lower]:
            if idx not in seen_indices:
                results.append((100.0, "exact_name", idx))
                seen_indices.add(idx)

    # Strategy 2: Check if query is a PSMILES
    if "[*]" in query or "(*)" in query:
        idx = _PSMILES_INDEX.get(query)
        if idx is not None and idx not in seen_indices:
            results.append((100.0, "exact_psmiles", idx))
            seen_indices.add(idx)

    # Strategy 3: Substring match on names
    if len(results) < top_k:
        for i, entry in enumerate(entries):
            if i in seen_indices:
                continue
            for name in entry.get("names", []):
                if query_lower in name.lower():
                    results.append((85.0, "substring", i))
                    seen_indices.add(i)
                    break

    # Strategy 4: Fuzzy match using rapidfuzz
    if len(results) < top_k and _ALL_SEARCHABLE:
        try:
            from rapidfuzz import process, fuzz
            searchable_names = [s[0] for s in _ALL_SEARCHABLE]
            matches = process.extract(
                query_lower,
                searchable_names,
                scorer=fuzz.WRatio,
                limit=top_k * 2,
                score_cutoff=65,
            )
            for match_str, score, match_idx in matches:
                entry_idx = _ALL_SEARCHABLE[match_idx][1]
                if entry_idx not in seen_indices:
                    results.append((score, "fuzzy", entry_idx))
                    seen_indices.add(entry_idx)
        except ImportError:
            logger.debug("rapidfuzz not available, skipping fuzzy search")

    # Strategy 5: Tag keyword match
    if len(results) < top_k:
        query_words = set(query_lower.split())
        for i, entry in enumerate(entries):
            if i in seen_indices:
                continue
            entry_tags = set(t.lower() for t in entry.get("tags", []))
            overlap = query_words & entry_tags
            if overlap:
                score = 60.0 * len(overlap) / len(query_words)
                results.append((score, "tag", i))
                seen_indices.add(i)
                if len(results) >= top_k * 3:
                    break

    # Sort by score descending, take top_k
    results.sort(key=lambda x: -x[0])
    results = results[:top_k]

    output = []
    for score, match_type, idx in results:
        entry = _format_entry(entries[idx])
        entry["match_score"] = round(score, 1)
        entry["match_type"] = match_type
        output.append(entry)

    return output


def get_lookup_stats() -> dict[str, Any]:
    """Return summary statistics about the lookup database."""
    data = load_tg_lookup()
    return {
        "version": data.get("version"),
        "model_info": data.get("model_info"),
        "stats": data.get("stats"),
    }
