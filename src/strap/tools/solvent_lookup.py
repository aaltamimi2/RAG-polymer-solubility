"""Solvent price and GWP lookup tools.
Two retrieval strategies:
- **Price**: built-in database → web search fallback (chemical market data is open)
- **GWP**: built-in database → web search fallback → solvent-class average estimate
  (ecoinvent/GaBi values are mostly paywalled)
"""
from __future__ import annotations
import csv
import json
import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from strap.paths import get_data_path
from strap.solvent_registry import resolve_to_biosteam
from strap.tools._helpers import safe_tool_wrapper
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Built-in solvent database
# ---------------------------------------------------------------------------
# Sources:
#   - 16 existing solvents from biosteam_runner.py (price + GWP from Branch-TEA)
#   - ~10 common solvents from ChemAnalyst/IMARC/ECHEMI + published LCA studies
_SOLVENT_DB: dict[str, dict[str, Any]] = {
    # ── BioSTEAM solvents (high confidence — Branch-TEA validated) ──────
    "sec-Butyl Acetate": {
        "price_usd_kg": 1.60,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 4.98,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "105-46-4",
        "aliases": ["SBA", "sec-butyl acetate", "2-butyl acetate"],
    },
    "Isobutyl Acetate": {
        "price_usd_kg": 1.60,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 4.81,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "110-19-0",
        "aliases": ["IBA", "isobutyl acetate"],
    },
    "Tetrachloroethylene": {
        "price_usd_kg": 1.38,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 3.85,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "127-18-4",
        "aliases": ["PCE", "perc", "perchloroethylene", "PERC"],
    },
    "o-Chlorotoluene": {
        "price_usd_kg": 2.40,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 2.74,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "95-49-8",
        "aliases": ["2-chlorotoluene", "OCT"],
    },
    "Methylcyclohexane": {
        "price_usd_kg": 1.55,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 2.55,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "108-87-2",
        "aliases": ["MCH", "methylcyclohexane"],
    },
    "Dodecanol": {
        "price_usd_kg": 1.50,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 4.12,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "112-53-8",
        "aliases": ["1-dodecanol", "lauryl alcohol"],
    },
    "Heptane": {
        "price_usd_kg": 1.42,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 0.897,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "142-82-5",
        "aliases": ["n-heptane"],
    },
    "Toluene": {
        "price_usd_kg": 0.82,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 1.61,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "108-88-3",
        "aliases": ["toluol", "methylbenzene"],
    },
    "Xylene": {
        "price_usd_kg": 0.84,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 1.52,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "1330-20-7",
        "aliases": ["xylol", "dimethylbenzene", "mixed xylenes"],
    },
    "Ethylene Glycol": {
        "price_usd_kg": 0.53,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 2.70,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "107-21-1",
        "aliases": ["EG", "MEG", "monoethylene glycol", "1,2-ethanediol"],
    },
    "Pyridazine": {
        "price_usd_kg": 4.95,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 10.7,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "289-80-5",
        "aliases": ["1,2-diazine"],
    },
    "butane-1,4-diol": {
        "price_usd_kg": 1.22,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 5.50,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "110-63-4",
        "aliases": ["1,4-butanediol", "BDO", "1,4-BDO"],
    },
    "Diethanolamine": {
        "price_usd_kg": 1.06,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 3.71,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "111-42-2",
        "aliases": ["DEA", "2,2'-iminodiethanol"],
    },
    "Diethylene glycol": {
        "price_usd_kg": 0.59,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 3.15,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "111-46-6",
        "aliases": ["DEG", "diglycol"],
    },
    "Propylene Glycol": {
        "price_usd_kg": 1.53,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 5.16,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "57-55-6",
        "aliases": ["PG", "MPG", "1,2-propanediol", "monopropylene glycol"],
    },
    "gamma-butyrolactone": {
        "price_usd_kg": 2.58,
        "price_source": "Branch-TEA reference data",
        "price_region": "North America",
        "gwp_kg_co2e": 6.54,
        "gwp_source": "Branch-TEA LCA (ecoinvent-derived)",
        "gwp_confidence": "high",
        "cas": "96-48-0",
        "aliases": ["GBL", "butyrolactone"],
    },
    # ── Common industrial solvents (web research) ──────────────────────
    "Acetone": {
        "price_usd_kg": 1.05,
        "price_source": "ChemAnalyst Q4 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 2.55,
        "gwp_source": "LanzaTech/Nature Biotech 2022, cumene process baseline",
        "gwp_confidence": "high",
        "cas": "67-64-1",
        "aliases": ["propanone", "dimethyl ketone", "2-propanone"],
    },
    "Methyl Ethyl Ketone": {
        "price_usd_kg": 1.39,
        "price_source": "ChemAnalyst 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 3.2,
        "gwp_source": "Estimated from ketone class average + sec-butanol dehydrogenation route",
        "gwp_confidence": "estimated",
        "cas": "78-93-3",
        "aliases": ["MEK", "2-butanone", "methyl ethyl ketone", "butan-2-one"],
    },
    "Tetrahydrofuran": {
        "price_usd_kg": 2.10,
        "price_source": "ECHEMI/IMARC 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 5.5,
        "gwp_source": "Estimated from Rein (1970) process energy + ether class data",
        "gwp_confidence": "estimated",
        "cas": "109-99-9",
        "aliases": ["THF", "oxolane", "tetrahydrofuran"],
    },
    "Ethyl Acetate": {
        "price_usd_kg": 1.13,
        "price_source": "ChemAnalyst 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 2.4,
        "gwp_source": "Tischer-Fischer esterification route, literature estimate",
        "gwp_confidence": "estimated",
        "cas": "141-78-6",
        "aliases": ["EtOAc", "ethyl acetate", "ethyl ethanoate", "EA"],
    },
    "Cyclohexanone": {
        "price_usd_kg": 1.30,
        "price_source": "ECHEMI 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 3.0,
        "gwp_source": "Estimated from cyclohexane oxidation route, ketone class",
        "gwp_confidence": "estimated",
        "cas": "108-94-1",
        "aliases": ["cyclohexanone", "pimelic ketone", "ketohexamethylene"],
    },
    "N-Methyl-2-pyrrolidone": {
        "price_usd_kg": 2.50,
        "price_source": "ECHEMI/ChemAnalyst 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 5.0,
        "gwp_source": "Estimated from GBL + methylamine route, amide class",
        "gwp_confidence": "estimated",
        "cas": "872-50-4",
        "aliases": ["NMP", "N-methylpyrrolidone", "1-methyl-2-pyrrolidone"],
    },
    "Dimethylformamide": {
        "price_usd_kg": 1.20,
        "price_source": "ChemAnalyst 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 3.8,
        "gwp_source": "Estimated from methanol + CO + dimethylamine route",
        "gwp_confidence": "estimated",
        "cas": "68-12-2",
        "aliases": ["DMF", "N,N-dimethylformamide", "dimethyl formamide"],
    },
    "Dimethyl Sulfoxide": {
        "price_usd_kg": 1.50,
        "price_source": "ECHEMI 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 2.8,
        "gwp_source": "Estimated from dimethyl sulfide oxidation route",
        "gwp_confidence": "estimated",
        "cas": "67-68-5",
        "aliases": ["DMSO", "dimethylsulfoxide", "methyl sulfoxide"],
    },
    "Dichloromethane": {
        "price_usd_kg": 0.55,
        "price_source": "ChemAnalyst 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 2.8,
        "gwp_source": "Estimated from methanol chlorination route, chlorinated class",
        "gwp_confidence": "estimated",
        "cas": "75-09-2",
        "aliases": ["DCM", "methylene chloride", "MC"],
    },
    "d-Limonene": {
        "price_usd_kg": 3.50,
        "price_source": "ECHEMI/citrus market data 2024",
        "price_region": "North America",
        "gwp_kg_co2e": 1.0,
        "gwp_source": "Estimated — biogenic (citrus peel byproduct), low fossil input",
        "gwp_confidence": "estimated",
        "cas": "5989-27-5",
        "aliases": ["limonene", "R-limonene", "orange terpene", "(+)-limonene"],
    },
}
# ---------------------------------------------------------------------------
# Alias index (built once at import time)
# ---------------------------------------------------------------------------
_ALIAS_INDEX: dict[str, str] = {}  # lowercase alias → canonical name
_CANONICAL_LOWER: dict[str, str] = {}  # lowercase canonical → canonical name
for _canonical, _info in _SOLVENT_DB.items():
    _CANONICAL_LOWER[_canonical.lower()] = _canonical
    for _alias in _info.get("aliases", []):
        _ALIAS_INDEX[_alias.lower()] = _canonical
# ---------------------------------------------------------------------------
# GWP class averages for fallback estimates
# ---------------------------------------------------------------------------
_GWP_CLASS_AVERAGES: dict[str, tuple[float, float]] = {
    "alkane": (0.8, 1.5),
    "aromatic": (1.5, 2.0),
    "ketone": (2.5, 3.5),
    "ester": (2.0, 3.0),
    "glycol": (2.5, 5.5),
    "amine": (3.5, 5.0),
    "chlorinated": (2.5, 4.0),
    "ether": (3.0, 6.0),
    "amide": (3.0, 5.0),
    "alcohol": (1.5, 3.5),
    "terpene": (0.5, 2.0),
}
_TEA_LCA_SOLVENT_CSV = get_data_path("60_common_solvents-TEA-LCA.csv")
# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------
def _resolve_name(query: str) -> str | None:
    """Resolve a solvent query to its canonical name in _SOLVENT_DB.
    Resolution order:
    1. Exact match on canonical name (case-insensitive)
    2. Exact match on alias (case-insensitive)
    3. Fuzzy substring match on canonical names and aliases
    """
    q = query.strip().lower()
    if not q:
        return None
    # 1. Exact canonical match
    if q in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[q]
    # 2. Exact alias match
    if q in _ALIAS_INDEX:
        return _ALIAS_INDEX[q]
    # 3. Substring match — only match if query is a substring of a known name
    #    (not the reverse, to avoid "2-methyltetrahydrofuran" matching "tetrahydrofuran")
    candidates: list[tuple[str, int]] = []
    for canon_lower, canon in _CANONICAL_LOWER.items():
        if q in canon_lower:
            candidates.append((canon, len(canon_lower)))
    for alias_lower, canon in _ALIAS_INDEX.items():
        if q in alias_lower:
            candidates.append((canon, len(alias_lower)))
    if candidates:
        # Sort by name length (shorter = more specific match)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    return None


def _resolve_exact_name(query: str) -> str | None:
    q = query.strip().lower()
    if not q:
        return None
    if q in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[q]
    if q in _ALIAS_INDEX:
        return _ALIAS_INDEX[q]
    return None


def _float_or_none(raw: Any) -> float | None:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _load_local_price_catalog() -> dict[str, dict[str, Any]]:
    """Load TEA/LCA solvent prices from CSV and normalize them to USD/kg."""
    if not _TEA_LCA_SOLVENT_CSV.exists():
        return {}

    loaded: dict[str, dict[str, Any]] = {}
    try:
        with _TEA_LCA_SOLVENT_CSV.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_price = _float_or_none(row.get("price"))
                if raw_price is None:
                    continue
                price_usd_kg = raw_price / 1000.0
                if price_usd_kg <= 0:
                    continue
                candidates = [
                    (row.get("name_cosmobase") or "").strip(),
                    (row.get("name_biosteam") or "").strip(),
                ]
                keys: set[str] = set()
                display_name: str | None = None
                for candidate in candidates:
                    if not candidate:
                        continue
                    variants = {
                        candidate,
                        candidate.replace("_", " "),
                    }
                    for variant in variants:
                        cleaned = variant.strip()
                        if not cleaned:
                            continue
                        keys.add(cleaned.lower())
                        resolved = resolve_to_biosteam(cleaned) or _resolve_name(cleaned)
                        if resolved:
                            keys.add(resolved.lower())
                            if display_name is None:
                                display_name = resolved
                if not keys:
                    continue
                record = {
                    "solvent": display_name or (candidates[1] or candidates[0]).replace("_", " "),
                    "price_usd_kg": round(price_usd_kg, 4),
                    "price_source": "60_common_solvents-TEA-LCA.csv price column",
                    "price_region": "TEA/LCA reference",
                    "cas": (row.get("cas") or "").strip() or None,
                }
                for key in keys:
                    loaded[key] = record
    except Exception as exc:
        logger.warning("Failed to load TEA/LCA price catalog: %s", exc)
        return {}
    return loaded


def _lookup_csv_solvent_price(solvent: str) -> dict[str, Any] | None:
    q = solvent.strip()
    if not q:
        return None

    catalog = _load_local_price_catalog()
    candidates = [q.lower()]
    resolved = resolve_to_biosteam(q)
    if resolved:
        candidates.append(resolved.lower())
    canonical = _resolve_name(q)
    if canonical:
        candidates.append(canonical.lower())

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        hit = catalog.get(candidate)
        if hit:
            return hit
    return None


def lookup_local_solvent_market_data(solvent: str) -> dict[str, Any] | None:
    """Return curated local price/GWP metadata without any web fallback."""
    csv_entry = _lookup_csv_solvent_price(solvent)
    resolved_biosteam = resolve_to_biosteam(solvent)
    canonical = _resolve_name(solvent)
    csv_name = str((csv_entry or {}).get("solvent") or "").strip() or None
    csv_entry_name = _resolve_exact_name(csv_name) if csv_name else None
    entry_name: str | None = None
    if csv_entry_name is not None and csv_entry_name in _SOLVENT_DB:
        entry_name = csv_entry_name
    elif csv_entry is None and canonical is not None and canonical in _SOLVENT_DB:
        entry_name = canonical
    elif csv_entry is None and resolved_biosteam in _SOLVENT_DB:
        entry_name = resolved_biosteam
    entry = _SOLVENT_DB.get(entry_name) if entry_name is not None else None
    if csv_entry is None and entry is None:
        return None

    solvent_name = (
        (csv_entry or {}).get("solvent")
        or entry_name
        or resolved_biosteam
        or canonical
        or solvent
    )
    return {
        "solvent": solvent_name,
        "price_usd_kg": (csv_entry or {}).get("price_usd_kg")
        if (csv_entry or {}).get("price_usd_kg") is not None
        else (entry or {}).get("price_usd_kg"),
        "price_source": (csv_entry or {}).get("price_source") or (entry or {}).get("price_source"),
        "price_region": (csv_entry or {}).get("price_region") or (entry or {}).get("price_region"),
        "gwp_kg_co2e": (entry or {}).get("gwp_kg_co2e"),
        "gwp_source": (entry or {}).get("gwp_source"),
        "gwp_confidence": (entry or {}).get("gwp_confidence"),
        "cas": (csv_entry or {}).get("cas") or (entry or {}).get("cas"),
    }


# ---------------------------------------------------------------------------
# Web search helper (SerpAPI)
# ---------------------------------------------------------------------------
def _serpapi_search(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Run a SerpAPI Google search, returning organic results.
    Returns list of {title, snippet, link} dicts. Returns empty list if
    SERPAPI_API_KEY is not set or the request fails.
    """
    api_key = os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY")
    if not api_key:
        logger.debug("SERPAPI_KEY not set — skipping web search")
        return []
    try:
        import requests
    except ImportError:
        logger.debug("requests not installed — skipping web search")
        return []
    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "api_key": api_key,
                "num": num_results,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("SerpAPI search failed: %s", exc)
        return []
    results = []
    for item in data.get("organic_results", []):
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        })
    return results
# ---------------------------------------------------------------------------
# Price parsing from web search snippets
# ---------------------------------------------------------------------------
# Matches patterns like: $1.05/kg, $1050/MT, USD 1.05/kg, 1.05 USD/kg,
# $1,050/ton, $1050 per metric ton, etc.
_PRICE_PER_KG_RE = re.compile(
    r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:USD\s*)?(?:/kg|per\s+kg)",
    re.IGNORECASE,
)
_PRICE_PER_MT_RE = re.compile(
    r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:USD\s*)?(?:/(?:MT|ton(?:ne)?)|per\s+(?:metric\s+)?ton(?:ne)?)",
    re.IGNORECASE,
)
def _parse_price_from_snippets(snippets: list[str]) -> tuple[float | None, str | None]:
    """Extract price per kg from search result snippets.
    Returns (price_usd_per_kg, source_snippet) or (None, None).
    """
    for snippet in snippets:
        # Try $/kg first
        m = _PRICE_PER_KG_RE.search(snippet)
        if m:
            price = float(m.group(1).replace(",", ""))
            if 0.01 < price < 100:  # sanity check
                return price, snippet
        # Try $/MT and convert
        m = _PRICE_PER_MT_RE.search(snippet)
        if m:
            price_mt = float(m.group(1).replace(",", ""))
            if 10 < price_mt < 100_000:  # sanity check
                return round(price_mt / 1000, 3), snippet
    return None, None
# ---------------------------------------------------------------------------
# GWP parsing from web search snippets
# ---------------------------------------------------------------------------
_GWP_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*kg\s*CO2(?:\s*-?\s*eq(?:uiv)?)?(?:\s*/\s*kg|\s+per\s+kg)",
    re.IGNORECASE,
)
def _parse_gwp_from_snippets(snippets: list[str]) -> tuple[float | None, str | None]:
    """Extract GWP (kg CO2e/kg) from search result snippets."""
    for snippet in snippets:
        m = _GWP_RE.search(snippet)
        if m:
            gwp = float(m.group(1))
            if 0.01 < gwp < 50:  # sanity check
                return gwp, snippet
    return None, None
# ---------------------------------------------------------------------------
# Tool 1: lookup_solvent_price
# ---------------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
def lookup_solvent_price(solvent: str, region: str = "North America") -> str:
    """Look up the bulk industrial price of a solvent.
    Checks a curated database first, then falls back to web search
    (SerpAPI) for solvents not in the database.
    Args:
        solvent: Solvent name (common name, abbreviation, or CAS).
            Examples: "Acetone", "MEK", "THF", "Toluene"
        region: Price region (default "North America").
    Returns:
        JSON string with keys: solvent, price_usd_kg, currency, region,
        source, confidence, date
    """
    year = datetime.now().year
    # 1. Check built-in database
    entry = lookup_local_solvent_market_data(solvent)
    if entry and entry.get("price_usd_kg") is not None:
        return json.dumps({
            "solvent": entry.get("solvent") or solvent,
            "price_usd_kg": entry["price_usd_kg"],
            "currency": "USD",
            "region": entry.get("price_region", region),
            "source": entry.get("price_source"),
            "confidence": "high",
            "date": str(year),
            "cas": entry.get("cas"),
        })
    # 2. Web search fallback
    query = f'"{solvent}" bulk price per kg {year} industrial'
    results = _serpapi_search(query)
    snippets = [r["snippet"] for r in results if r.get("snippet")]
    price, source_snippet = _parse_price_from_snippets(snippets)
    if price is not None:
        source_url = results[0]["link"] if results else "web search"
        return json.dumps({
            "solvent": solvent,
            "price_usd_kg": price,
            "currency": "USD",
            "region": region,
            "source": f"Web search ({source_url})",
            "confidence": "web_search",
            "date": str(year),
        })
    # 3. Not found
    return json.dumps({
        "solvent": solvent,
        "price_usd_kg": None,
        "currency": "USD",
        "region": region,
        "source": None,
        "confidence": "unavailable",
        "date": str(year),
        "note": (
            f"Price for '{solvent}' not found in database or web search. "
            "Try a more common name, CAS number, or check ChemAnalyst/ECHEMI manually."
        ),
    })
# ---------------------------------------------------------------------------
# Tool 2: lookup_solvent_gwp
# ---------------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
def lookup_solvent_gwp(solvent: str) -> str:
    """Look up the cradle-to-gate GWP (Global Warming Potential) of a solvent.
    Checks a curated database first (16 BioSTEAM solvents + 10 common
    industrials). If not found, attempts web search. If still not found,
    returns a solvent-class average estimate with a warning.
    Args:
        solvent: Solvent name (common name, abbreviation, or CAS).
            Examples: "Acetone", "MEK", "THF", "Toluene"
    Returns:
        JSON string with keys: solvent, gwp_kg_co2e, unit, production_route,
        source, confidence
    """
    # 1. Check built-in database
    canonical = _resolve_name(solvent)
    if canonical and canonical in _SOLVENT_DB:
        entry = _SOLVENT_DB[canonical]
        return json.dumps({
            "solvent": canonical,
            "gwp_kg_co2e": entry["gwp_kg_co2e"],
            "unit": "kg CO2e / kg solvent",
            "production_route": "conventional (see source)",
            "source": entry["gwp_source"],
            "confidence": entry["gwp_confidence"],
            "cas": entry.get("cas"),
        })
    # 2. Web search fallback
    query = f'"{solvent}" GWP cradle-to-gate "kg CO2" per kg'
    results = _serpapi_search(query)
    snippets = [r["snippet"] for r in results if r.get("snippet")]
    gwp, source_snippet = _parse_gwp_from_snippets(snippets)
    if gwp is not None:
        source_url = results[0]["link"] if results else "web search"
        return json.dumps({
            "solvent": solvent,
            "gwp_kg_co2e": gwp,
            "unit": "kg CO2e / kg solvent",
            "production_route": "unknown (from web search)",
            "source": f"Web search ({source_url})",
            "confidence": "web_search",
        })
    # 3. Not found — provide class-average guidance
    class_guidance = {
        cls: f"{lo}-{hi} kg CO2e/kg"
        for cls, (lo, hi) in _GWP_CLASS_AVERAGES.items()
    }
    return json.dumps({
        "solvent": solvent,
        "gwp_kg_co2e": None,
        "unit": "kg CO2e / kg solvent",
        "production_route": None,
        "source": None,
        "confidence": "unavailable",
        "note": (
            f"GWP for '{solvent}' not found in database or web search. "
            "Solvent-specific LCA data is often paywalled (ecoinvent/GaBi). "
            "Use the class-average estimates below as rough guidance."
        ),
        "class_average_estimates": class_guidance,
    })
