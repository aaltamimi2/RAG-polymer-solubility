"""Unified solubility API — interpolation model with SQL fallback.

This module is the **single source of truth** for solubility lookups across
the entire STRAP engine/analysis/tool stack.  All SQL-based solubility
queries in engines/, analysis.py, and tools/ should call through here.

Primary source:  pre-fitted  ln(S) = A + B/T + C/T²  coefficients
Fallback source: DuckDB SQL on the common_solvents_database table
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Method constants
AUTO = "auto"
INTERPOLATION = "interpolation"
SQL = "sql"

# ==================================================================
# Coefficient data — lazy-loaded singleton
# ==================================================================

_COEFFICIENTS: Optional[dict] = None
_LOOKUP: Optional[dict[tuple[str, str], dict]] = None
_KNOWN_POLYMERS: Optional[set[str]] = None
_KNOWN_SOLVENTS: Optional[set[str]] = None

_DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "solubility_coefficients.json"
)
_GENERATED_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "generated_coefficients.json"
)


def _load_coefficients() -> tuple[dict, dict[tuple[str, str], dict]]:
    """Load JSON coefficients and build normalized lookup dict.

    Implements three-tier loading:
      1. Static (highest priority) — from solubility_coefficients.json
      2. Validated (medium) — generated_coefficients.json entries with tier=validated
      3. Dynamic (lowest) — generated_coefficients.json entries with tier=dynamic

    Each lookup entry gets a ``"source"`` field: ``"static"``, ``"validated"``,
    or ``"dynamic"``.
    """
    global _COEFFICIENTS, _LOOKUP, _KNOWN_POLYMERS, _KNOWN_SOLVENTS
    if _COEFFICIENTS is not None:
        return _COEFFICIENTS, _LOOKUP

    with open(_DATA_PATH) as f:
        _COEFFICIENTS = json.load(f)

    _LOOKUP = {}
    # --- Tier 1: static entries (always win) ---
    for entry in _COEFFICIENTS["entries"]:
        key = (entry["polymer"].strip().upper(), entry["solvent"].strip().lower())
        entry["source"] = "static"
        _LOOKUP[key] = entry

    # --- Tiers 2 & 3: generated entries (validated before dynamic) ---
    if _GENERATED_PATH.exists():
        try:
            with open(_GENERATED_PATH) as f:
                generated = json.load(f)
            # Sort so validated entries are processed after dynamic ones,
            # meaning validated overwrites dynamic for the same key.
            gen_entries = sorted(
                generated.get("entries", []),
                key=lambda e: 0 if e.get("tier") == "dynamic" else 1,
            )
            for entry in gen_entries:
                key = (entry["polymer"].strip().upper(), entry["solvent"].strip().lower())
                if key in _LOOKUP and _LOOKUP[key].get("source") == "static":
                    continue  # static always takes precedence
                entry["source"] = entry.get("tier", "dynamic")
                _LOOKUP[key] = entry
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load generated coefficients: %s", e)

    # Pre-compute known name sets once (avoids 352-iteration rebuild per call)
    _KNOWN_POLYMERS = {p for p, _ in _LOOKUP}
    _KNOWN_SOLVENTS = {s for _, s in _LOOKUP}

    return _COEFFICIENTS, _LOOKUP


# ==================================================================
# Fuzzy name matching — aliases & resolution
# ==================================================================

from strap.solvent_registry import resolve_to_interp_key, resolve_to_bp_db_key

POLYMER_ALIASES: dict[str, str] = {
    "POLYETHYLENE": "HDPE",
    "NYLON 6": "NYLON6",
    "NYLON 66": "NYLON66",
    "NYLON-6": "NYLON6",
    "NYLON-66": "NYLON66",
    "PA6": "NYLON6",
    "PA66": "NYLON66",
    "POLYCARBONATE": "PC",
    "POLYSTYRENE": "PS",
    "POLYETHERSULFONE": "PES",
    "POLYVINYLCHLORIDE": "PVC",
    "POLYVINYL CHLORIDE": "PVC",
    "POLYPROPYLENE": "PP",
}


def resolve_polymer(name: str, known_polymers: set[str]) -> Optional[str]:
    """3-strategy fuzzy match: exact → alias → substring."""
    norm = name.strip().upper()
    if norm in known_polymers:
        return norm
    alias = POLYMER_ALIASES.get(norm)
    if alias and alias in known_polymers:
        return alias
    for kp in known_polymers:
        if norm in kp or kp in norm:
            return kp
    return None


def resolve_solvent(name: str, known_solvents: set[str]) -> Optional[str]:
    """3-strategy fuzzy match: exact → alias → substring."""
    norm = name.strip().lower()
    if norm in known_solvents:
        return norm
    alias = resolve_to_interp_key(norm)
    if alias and alias in known_solvents:
        return alias
    for ks in known_solvents:
        if norm in ks or ks in norm:
            return ks
    return None


def _get_known_names(lookup: dict) -> tuple[set[str], set[str]]:
    """Return cached known polymer/solvent name sets (O(1) after first load)."""
    return _KNOWN_POLYMERS, _KNOWN_SOLVENTS


# ==================================================================
# Prediction helper
# ==================================================================

def predict(entry: dict, temp_c: float) -> dict:
    """Predict solubility for a single temperature using coefficients.

    Returns:
        {"solubility_pct": float, "temperature_c": float, "extrapolation": str,
         "source": str}
    """
    t_k = temp_c + 273.15
    ln_s = entry["A"] + entry["B"] / t_k + entry["C"] / t_k**2
    s_pct = np.clip(np.exp(ln_s), 0.0, 100.0)

    extrapolation = ""
    if temp_c < entry["t_min_c"]:
        extrapolation = "below"
    elif temp_c > entry["t_max_c"]:
        extrapolation = "above"

    return {
        "solubility_pct": round(float(s_pct), 6),
        "temperature_c": temp_c,
        "extrapolation": extrapolation,
        "source": entry.get("source", "static"),
    }


# ==================================================================
# Public API
# ==================================================================

def get_solubility(
    polymer: str,
    solvent: str,
    temperature_c: float,
    method: str = AUTO,
) -> Optional[float]:
    """Get solubility (%) for a polymer-solvent pair at a temperature.

    Args:
        polymer: Polymer name (fuzzy-matched via aliases).
        solvent: Solvent name (fuzzy-matched via aliases).
        temperature_c: Temperature in °C.
        method: "auto" (interpolation first, SQL fallback),
                "interpolation", or "sql".

    Returns:
        Solubility percentage (0–100), or None if no data.
    """
    if method in (AUTO, INTERPOLATION):
        result = _interp_get_solubility(polymer, solvent, temperature_c)
        if result is not None:
            return result
        if method == INTERPOLATION:
            return None

    resolved_polymer = POLYMER_ALIASES.get(polymer.strip().upper(), polymer.strip().upper())
    resolved_solvent = resolve_to_interp_key(solvent) or solvent
    return _sql_get_solubility(resolved_polymer, resolved_solvent, temperature_c)


def get_solubility_batch(
    polymer_solvent_pairs: Sequence[tuple[str, str]],
    temperature_c: float,
    method: str = AUTO,
) -> dict[tuple[str, str], Optional[float]]:
    """Get solubility for multiple polymer-solvent pairs at one temperature."""
    return {
        (p, s): get_solubility(p, s, temperature_c, method=method)
        for p, s in polymer_solvent_pairs
    }


def get_solubility_curve(
    polymer: str,
    solvent: str,
    t_start_c: float = 25.0,
    t_end_c: float = 160.0,
    t_step_c: float = 5.0,
    method: str = AUTO,
) -> list[dict]:
    """Get solubility curve over a temperature range.

    Returns:
        List of {"temperature": float, "solubility": float} dicts,
        sorted by temperature.  Empty list if pair not found.
    """
    if method in (AUTO, INTERPOLATION):
        result = _interp_get_curve(polymer, solvent, t_start_c, t_end_c, t_step_c)
        if result is not None:
            return result
        if method == INTERPOLATION:
            return []

    resolved_polymer = POLYMER_ALIASES.get(polymer.strip().upper(), polymer.strip().upper())
    resolved_solvent = resolve_to_interp_key(solvent) or solvent
    return _sql_get_curve(resolved_polymer, resolved_solvent, t_start_c, t_end_c, t_step_c)


def get_selectivity(
    target: str,
    others: list[str],
    temperature_c: float,
    used_solvents: Optional[set[str]] = None,
    method: str = AUTO,
) -> tuple[str, float, float, float]:
    """Find the best solvent to separate *target* from *others*.

    selectivity = target_solubility - max(other_solubilities)

    Returns:
        (best_solvent, selectivity, target_solubility, max_other_solubility).
        ("none", -999, 0, 0) when no data is found.
    """
    if not others:
        return ("N/A", float("inf"), 100.0, 0.0)

    if method in (AUTO, INTERPOLATION):
        result = _interp_get_selectivity(target, others, temperature_c, used_solvents)
        if result is not None:
            return result
        if method == INTERPOLATION:
            return ("none", -999.0, 0.0, 0.0)

    resolved_target = POLYMER_ALIASES.get(target.strip().upper(), target.strip().upper())
    resolved_others = [POLYMER_ALIASES.get(o.strip().upper(), o.strip().upper()) for o in others]
    return _sql_get_selectivity(resolved_target, resolved_others, temperature_c, used_solvents)


def get_all_solvents_selectivity(
    target: str,
    others: list[str],
    temperature_c: float,
    method: str = AUTO,
) -> list[dict]:
    """Selectivity for every available solvent (for ranking).

    Returns:
        List sorted by selectivity descending::

            [{"solvent", "selectivity", "target_sol", "max_other_sol"}, ...]
    """
    if method in (AUTO, INTERPOLATION):
        result = _interp_all_solvents_selectivity(target, others, temperature_c)
        if result:
            return result
        if method == INTERPOLATION:
            return []

    resolved_target = POLYMER_ALIASES.get(target.strip().upper(), target.strip().upper())
    resolved_others = [POLYMER_ALIASES.get(o.strip().upper(), o.strip().upper()) for o in others]
    return _sql_all_solvents_selectivity(resolved_target, resolved_others, temperature_c)


# ------------------------------------------------------------------
# Listing helpers
# ------------------------------------------------------------------

def get_available_polymers() -> set[str]:
    """All polymer names in the interpolation dataset."""
    _, lookup = _load_coefficients()
    p, _ = _get_known_names(lookup)
    return p


def get_available_solvents() -> set[str]:
    """All solvent names in the interpolation dataset."""
    _, lookup = _load_coefficients()
    _, s = _get_known_names(lookup)
    return s


def get_available_solvents_for_polymer(polymer: str) -> set[str]:
    """Solvents with *fitted* interpolation data for a given polymer."""
    _, lookup = _load_coefficients()
    known_p, _ = _get_known_names(lookup)
    resolved = resolve_polymer(polymer, known_p)
    if resolved is None:
        return set()
    return {
        s for (p, s), entry in lookup.items()
        if p == resolved and entry["category"] == "fitted"
    }


def get_available_pairs() -> set[tuple[str, str]]:
    """All (polymer, solvent) pairs with fitted interpolation data."""
    _, lookup = _load_coefficients()
    return {
        (p, s) for (p, s), entry in lookup.items()
        if entry["category"] == "fitted"
    }


def resolve_names(
    polymer: str, solvent: str,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve polymer/solvent names via alias + fuzzy matching."""
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)
    return resolve_polymer(polymer, known_p), resolve_solvent(solvent, known_s)


def get_entry(polymer: str, solvent: str) -> Optional[dict]:
    """Get the raw coefficient entry for a polymer-solvent pair (or None)."""
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)
    p = resolve_polymer(polymer, known_p)
    s = resolve_solvent(solvent, known_s)
    if p is None or s is None:
        return None
    return lookup.get((p, s))


def get_coefficients_metadata() -> dict:
    """Return the top-level metadata (n_entries, categories, etc.)."""
    coeffs, _ = _load_coefficients()
    return coeffs


def get_entry_source(polymer: str, solvent: str) -> Optional[str]:
    """Return the data source tier for a polymer-solvent pair.

    Returns:
        ``"static"``, ``"validated"``, ``"dynamic"``, or ``None`` if no entry.
    """
    entry = get_entry(polymer, solvent)
    if entry is None:
        return None
    return entry.get("source", "static")


def get_dynamic_entries() -> list[dict]:
    """Return all ML-generated entries (both validated and dynamic tiers)."""
    _, lookup = _load_coefficients()
    return [
        entry for entry in lookup.values()
        if entry.get("source") in ("validated", "dynamic")
    ]


def reload_coefficients() -> None:
    """Clear the singleton cache and force a full reload on next access.

    Useful after new ML-generated coefficients have been written to disk.
    """
    global _COEFFICIENTS, _LOOKUP, _KNOWN_POLYMERS, _KNOWN_SOLVENTS
    _COEFFICIENTS = None
    _LOOKUP = None
    _KNOWN_POLYMERS = None
    _KNOWN_SOLVENTS = None


# ==================================================================
# Interpolation internals
# ==================================================================

def _interp_get_solubility(
    polymer: str, solvent: str, temperature_c: float,
) -> Optional[float]:
    """Returns None for anomalous/missing (caller should fall back to SQL)."""
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)

    p = resolve_polymer(polymer, known_p)
    s = resolve_solvent(solvent, known_s)
    if p is None or s is None:
        return None

    entry = lookup.get((p, s))
    if entry is None:
        return None

    if entry["category"] == "insoluble":
        return 0.0
    if entry["category"] == "anomalous":
        return None  # fall through to SQL

    return predict(entry, temperature_c)["solubility_pct"]


def _interp_get_curve(
    polymer: str, solvent: str,
    t_start: float, t_end: float, t_step: float,
) -> Optional[list[dict]]:
    """Returns None if the pair can't be interpolated (anomalous/missing)."""
    _, lookup = _load_coefficients()
    known_p, known_s = _get_known_names(lookup)

    p = resolve_polymer(polymer, known_p)
    s = resolve_solvent(solvent, known_s)
    if p is None or s is None:
        return None

    entry = lookup.get((p, s))
    if entry is None:
        return None

    step = max(t_step, 1.0)
    temps = np.arange(t_start, t_end + step / 2, step)

    if entry["category"] == "insoluble":
        return [{"temperature": float(t), "solubility": 0.0} for t in temps]
    if entry["category"] == "anomalous":
        return None

    return [
        {
            "temperature": float(t),
            "solubility": predict(entry, float(t))["solubility_pct"],
        }
        for t in temps
    ]


def _interp_get_selectivity(
    target: str,
    others: list[str],
    temperature_c: float,
    used_solvents: Optional[set[str]] = None,
) -> Optional[tuple[str, float, float, float]]:
    """Returns None when interpolation can't fully handle the request."""
    _, lookup = _load_coefficients()
    known_p, _ = _get_known_names(lookup)

    target_r = resolve_polymer(target, known_p)
    if target_r is None:
        return None

    others_r = []
    for o in others:
        resolved = resolve_polymer(o, known_p)
        if resolved is None:
            return None
        others_r.append(resolved)

    target_solvents = {
        s for (p, s), entry in lookup.items()
        if p == target_r and entry["category"] == "fitted"
    }
    if not target_solvents:
        return None

    used_lower = {s.lower() for s in used_solvents} if used_solvents else set()

    best: tuple[str, float, float, float] = ("none", -999.0, 0.0, 0.0)
    any_computed = False

    for solvent in target_solvents:
        if solvent in used_lower:
            continue

        target_sol = _interp_get_solubility(target_r, solvent, temperature_c)
        if target_sol is None or target_sol <= 0:
            continue

        max_other = 0.0
        skip = False
        for other_p in others_r:
            other_sol = _interp_get_solubility(other_p, solvent, temperature_c)
            if other_sol is None:
                skip = True
                break
            max_other = max(max_other, other_sol)

        if skip:
            continue

        any_computed = True
        selectivity = target_sol - max_other
        if selectivity > best[1]:
            best = (solvent, selectivity, target_sol, max_other)

    return best if any_computed else None


def _interp_all_solvents_selectivity(
    target: str, others: list[str], temperature_c: float,
) -> list[dict]:
    _, lookup = _load_coefficients()
    known_p, _ = _get_known_names(lookup)

    target_r = resolve_polymer(target, known_p)
    if target_r is None:
        return []

    others_r = []
    for o in others:
        resolved = resolve_polymer(o, known_p)
        if resolved is None:
            return []
        others_r.append(resolved)

    target_solvents = {
        s for (p, s), entry in lookup.items()
        if p == target_r and entry["category"] == "fitted"
    }

    results: list[dict] = []
    for solvent in target_solvents:
        target_sol = _interp_get_solubility(target_r, solvent, temperature_c)
        if target_sol is None or target_sol <= 0:
            continue

        max_other = 0.0
        skip = False
        for other_p in others_r:
            other_sol = _interp_get_solubility(other_p, solvent, temperature_c)
            if other_sol is None:
                skip = True
                break
            max_other = max(max_other, other_sol)

        if skip:
            continue

        results.append({
            "solvent": solvent,
            "selectivity": target_sol - max_other,
            "target_sol": target_sol,
            "max_other_sol": max_other,
        })

    results.sort(key=lambda r: r["selectivity"], reverse=True)
    return results


# ==================================================================
# Boiling-point lookup (cached)
# ==================================================================

_BP_CACHE: Optional[dict[str, float]] = None


def get_boiling_point(solvent: str) -> Optional[float]:
    """Return boiling point (°C) from Solvent_Data, or None if not found."""
    global _BP_CACHE
    if _BP_CACHE is None:
        conn = _get_db_conn()
        rows = conn.execute(
            "SELECT LOWER(solvent_name), bp__oc_ FROM solvent_data"
        ).fetchall()
        _BP_CACHE = {name: float(bp) for name, bp in rows if bp is not None}
    key = solvent.lower()
    bp = _BP_CACHE.get(key)
    if bp is None:
        alias = resolve_to_bp_db_key(key)
        if alias:
            bp = _BP_CACHE.get(alias)
    return bp


# ==================================================================
# LogP lookup (cached)
# ==================================================================

_LOGP_CACHE: Optional[dict[str, float]] = None


def get_logp(solvent: str) -> Optional[float]:
    """Return LogP from Solvent_Data, or None if not found."""
    global _LOGP_CACHE
    if _LOGP_CACHE is None:
        conn = _get_db_conn()
        rows = conn.execute(
            "SELECT LOWER(solvent_name), logp FROM solvent_data"
        ).fetchall()
        _LOGP_CACHE = {name: float(lp) for name, lp in rows if lp is not None}
    key = solvent.lower()
    lp = _LOGP_CACHE.get(key)
    if lp is None:
        alias = resolve_to_bp_db_key(key)
        if alias:
            lp = _LOGP_CACHE.get(alias)
    return lp


# ==================================================================
# SQL fallback internals
# ==================================================================

def _get_db_conn():
    """Lazy-import the DuckDB connection singleton."""
    from strap.database import get_connection
    return get_connection()


def _sql_get_solubility(
    polymer: str,
    solvent: str,
    temperature_c: float,
    temp_tolerance: float = 10.0,
) -> Optional[float]:
    conn = _get_db_conn()
    query = """
    SELECT AVG(solubility____) as avg_sol
    FROM common_solvents_database
    WHERE UPPER(polymer) = UPPER(?)
      AND LOWER(solvent) = LOWER(?)
      AND temperature___c_ BETWEEN ? AND ?
    """
    try:
        row = conn.execute(query, [polymer, solvent,
                                   temperature_c - temp_tolerance,
                                   temperature_c + temp_tolerance]).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as e:
        logger.debug("SQL fallback failed for %s/%s: %s", polymer, solvent, e)
    return None


def _sql_get_curve(
    polymer: str, solvent: str,
    t_start: float, t_end: float, t_step: float,
) -> list[dict]:
    conn = _get_db_conn()
    query = """
    SELECT temperature___c_ as temperature, AVG(solubility____) as solubility
    FROM common_solvents_database
    WHERE UPPER(polymer) = UPPER(?)
      AND LOWER(solvent) = LOWER(?)
      AND temperature___c_ BETWEEN ? AND ?
    GROUP BY temperature___c_
    ORDER BY temperature
    """
    try:
        rows = conn.execute(query, [polymer, solvent, t_start, t_end]).fetchall()
        return [
            {"temperature": float(r[0]), "solubility": float(r[1])}
            for r in rows
        ]
    except Exception as e:
        logger.debug("SQL curve failed for %s/%s: %s", polymer, solvent, e)
    return []


def _sql_get_selectivity(
    target: str,
    others: list[str],
    temperature_c: float,
    used_solvents: Optional[set[str]] = None,
    temp_tolerance: float = 10.0,
) -> tuple[str, float, float, float]:
    conn = _get_db_conn()
    all_polymers = [target] + others
    # Escape each name for safe IN-list interpolation
    def _esc(s: str) -> str:
        return s.replace("'", "''")
    polymer_filter = "', '".join(_esc(p) for p in all_polymers)
    others_upper = ",".join(f"UPPER('{_esc(p)}')" for p in others)
    target_safe = _esc(target)

    query = f"""
    WITH solubility_data AS (
        SELECT solvent, polymer, AVG(solubility____) as avg_sol
        FROM common_solvents_database
        WHERE polymer IN ('{polymer_filter}')
          AND temperature___c_ BETWEEN {temperature_c - temp_tolerance}
                                    AND {temperature_c + temp_tolerance}
        GROUP BY solvent, polymer
    ),
    target_sol AS (
        SELECT solvent, avg_sol as target_solubility
        FROM solubility_data
        WHERE UPPER(polymer) = UPPER('{target_safe}')
    ),
    others_max AS (
        SELECT solvent, MAX(avg_sol) as max_other
        FROM solubility_data
        WHERE UPPER(polymer) IN ({others_upper})
        GROUP BY solvent
    )
    SELECT t.solvent,
           t.target_solubility,
           COALESCE(o.max_other, t.target_solubility) as max_other,
           (t.target_solubility - COALESCE(o.max_other, t.target_solubility)) as selectivity
    FROM target_sol t
    LEFT JOIN others_max o ON LOWER(t.solvent) = LOWER(o.solvent)
    WHERE t.target_solubility > 0
    ORDER BY selectivity DESC
    """
    try:
        result = conn.execute(query).fetchall()
        if not result:
            return ("none", -999.0, 0.0, 0.0)

        if used_solvents:
            used_lower = {s.lower() for s in used_solvents}
            filtered = [r for r in result if r[0].lower() not in used_lower]
            if filtered:
                result = filtered

        best = result[0]
        return (best[0], float(best[3]), float(best[1]), float(best[2]))
    except Exception as e:
        logger.debug("SQL selectivity failed: %s", e)
        return ("error", -999.0, 0.0, 0.0)


def _sql_all_solvents_selectivity(
    target: str,
    others: list[str],
    temperature_c: float,
    temp_tolerance: float = 10.0,
) -> list[dict]:
    conn = _get_db_conn()
    all_polymers = [target] + others
    # Escape each name for safe IN-list interpolation
    def _esc(s: str) -> str:
        return s.replace("'", "''")
    polymer_filter = "', '".join(_esc(p) for p in all_polymers)
    others_upper = ",".join(f"UPPER('{_esc(p)}')" for p in others)
    target_safe = _esc(target)

    query = f"""
    WITH solubility_data AS (
        SELECT solvent, polymer, AVG(solubility____) as avg_sol
        FROM common_solvents_database
        WHERE polymer IN ('{polymer_filter}')
          AND temperature___c_ BETWEEN {temperature_c - temp_tolerance}
                                    AND {temperature_c + temp_tolerance}
        GROUP BY solvent, polymer
    ),
    target_sol AS (
        SELECT solvent, avg_sol as target_solubility
        FROM solubility_data
        WHERE UPPER(polymer) = UPPER('{target_safe}')
    ),
    others_max AS (
        SELECT solvent, MAX(avg_sol) as max_other
        FROM solubility_data
        WHERE UPPER(polymer) IN ({others_upper})
        GROUP BY solvent
    )
    SELECT t.solvent,
           t.target_solubility,
           COALESCE(o.max_other, t.target_solubility) as max_other,
           (t.target_solubility - COALESCE(o.max_other, t.target_solubility)) as selectivity
    FROM target_sol t
    LEFT JOIN others_max o ON LOWER(t.solvent) = LOWER(o.solvent)
    WHERE t.target_solubility > 0
    ORDER BY selectivity DESC
    """
    try:
        rows = conn.execute(query).fetchall()
        return [
            {
                "solvent": r[0],
                "selectivity": float(r[3]),
                "target_sol": float(r[1]),
                "max_other_sol": float(r[2]),
            }
            for r in rows
        ]
    except Exception as e:
        logger.debug("SQL all-solvents selectivity failed: %s", e)
        return []
