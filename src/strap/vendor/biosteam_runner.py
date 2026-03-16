"""
BioSTEAM Simulation Runner

Manages subprocess lifecycle, parallelism, and result parsing for
BioSTEAM TEA/LCA simulations. Each simulation runs in an isolated
subprocess to avoid global state contamination.

Supports **any** thermosteam-resolvable solvent.  Known solvents use
validated data from Branch-TEA / ecoinvent; unknown solvents get
graceful fallbacks (price, dissolution temp from Solvent_Data.csv,
LCA impact factors from chemical-class averages).
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from strap.paths import get_data_path
from strap.solvent_registry import resolve_to_biosteam

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = Path(__file__).parent / "biosteam_worker.py"

_MAX_SUBPROCESS_STDOUT_BYTES = 10 * 1024 * 1024   # 10 MB
_MAX_SUBPROCESS_STDERR_BYTES = 2 * 1024 * 1024    # 2 MB

# ---------------------------------------------------------------------------
# Solvent_Data.csv loader — boiling points, CAS, and LogP for all solvents
# ---------------------------------------------------------------------------

_SOLVENT_DATA_CSV = get_data_path("Solvent_Data.csv")

# {normalised_name: {"bp_c": float|None, "cas": str, "name": str, "logp": float|None}}
_SOLVENT_CSV_DATA: dict[str, dict] = {}
# {cas_number: same dict} for CAS-based lookup
_SOLVENT_CSV_BY_CAS: dict[str, dict] = {}


def _load_solvent_csv() -> None:
    """Load boiling points, CAS numbers, and LogP from Solvent_Data.csv.

    Builds multiple lookup keys per entry:
    * Exact name (lowered)
    * CAS number
    * Parenthetical aliases — e.g. "Methylene Dichloride (Dichloromethane)"
      registers both "methylene dichloride (dichloromethane)" AND
      "dichloromethane" as keys.
    """
    if not _SOLVENT_DATA_CSV.exists():
        logger.warning(
            "Solvent_Data.csv not found at %s — fallbacks will use generic defaults",
            _SOLVENT_DATA_CSV,
        )
        return
    try:
        import re
        with open(_SOLVENT_DATA_CSV, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("Solvent name") or "").strip()
                if not name:
                    continue
                bp_str = (row.get("Bp (oC)") or "").strip()
                cas = (row.get("CAS number") or "").strip()
                logp_str = (row.get("LogP") or "").strip()
                bp_c: float | None = None
                logp: float | None = None
                try:
                    bp_c = float(bp_str) if bp_str else None
                except ValueError:
                    pass
                try:
                    logp = float(logp_str) if logp_str else None
                except ValueError:
                    pass
                entry = {
                    "bp_c": bp_c,
                    "cas": cas,
                    "name": name,
                    "logp": logp,
                }
                # Primary key: full name
                name_lower = name.lower()
                _SOLVENT_CSV_DATA[name_lower] = entry
                # CAS key
                if cas:
                    _SOLVENT_CSV_BY_CAS[cas] = entry
                # Extract parenthetical aliases, e.g.
                # "Methylene Dichloride (Dichloromethane)" → "dichloromethane"
                # "Tetrahydrofuran (THF)" → "thf"
                paren_aliases = re.findall(r"\(([^)]+)\)", name)
                for alias in paren_aliases:
                    alias_key = alias.strip().lower()
                    if alias_key and alias_key not in _SOLVENT_CSV_DATA:
                        _SOLVENT_CSV_DATA[alias_key] = entry
                # Also register name-before-parentheses as a key
                # "Tetrahydrofuran (THF)" → "tetrahydrofuran"
                if "(" in name:
                    base_name = name[:name.index("(")].strip().lower()
                    if base_name and base_name not in _SOLVENT_CSV_DATA:
                        _SOLVENT_CSV_DATA[base_name] = entry
        logger.info("Loaded %d solvent entries from Solvent_Data.csv", len(_SOLVENT_CSV_DATA))
    except Exception as exc:
        logger.error("Failed to load Solvent_Data.csv: %s", exc)


_load_solvent_csv()


def _csv_lookup(solvent: str) -> dict | None:
    """Look up solvent data from CSV (case-insensitive, exact match + CAS)."""
    key = solvent.lower().strip()
    # 1. Exact name match (includes parenthetical aliases)
    if key in _SOLVENT_CSV_DATA:
        return _SOLVENT_CSV_DATA[key]
    # 2. CAS number match
    if key in _SOLVENT_CSV_BY_CAS:
        return _SOLVENT_CSV_BY_CAS[key]
    # 3. Search key fully contained in a CSV name (but not vice versa)
    #    e.g. "n,n-dimethylformamide" in "n,n-dimethylformamide (dmf)"
    for csv_key, data in _SOLVENT_CSV_DATA.items():
        if key in csv_key and len(key) > 5:
            return data
    return None


# ---------------------------------------------------------------------------
# Curated solvent economic / LCA data (from web-research agent swarm)
# ---------------------------------------------------------------------------
# Loaded from data/solvent-econ-lca-summary.csv.
# Provides per-solvent bulk pricing and LCA impact factors collected from
# market reports, ecoinvent references, and published LCA studies.
# Used as a second-tier fallback: validated > curated > class-average > generic.

_CURATED_CSV = get_data_path("solvent-econ-lca-summary.csv")
_TEA_LCA_SOLVENT_CSV = get_data_path("60_common_solvents-TEA-LCA.csv")

# {normalised_name: {"price": float|None, "gwp": float|None, "htc": ..., "htnc": ..., "etox": ..., "class": str}}
_CURATED_BY_NAME: dict[str, dict] = {}
# {cas: same dict}
_CURATED_BY_CAS: dict[str, dict] = {}


def _load_curated_csv() -> None:
    """Load curated solvent prices and LCA factors from solvent-econ-lca-summary.csv."""
    if not _CURATED_CSV.exists():
        logger.debug("Curated solvent CSV not found at %s — skipping", _CURATED_CSV)
        return
    try:
        with open(_CURATED_CSV, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("solvent_name") or "").strip()
                cas = (row.get("cas") or "").strip()
                chem_class = (row.get("chemical_class") or "").strip()
                if not name:
                    continue

                def _float_or_none(val: str) -> float | None:
                    val = val.strip()
                    if not val:
                        return None
                    try:
                        return float(val)
                    except ValueError:
                        return None

                entry = {
                    "price": _float_or_none(row.get("price_usd_per_kg", "")),
                    "gwp": _float_or_none(row.get("gwp_kg_co2e_per_kg", "")),
                    "htc": _float_or_none(row.get("htc_ctuh_per_kg", "")),
                    "htnc": _float_or_none(row.get("htnc_ctuh_per_kg", "")),
                    "etox": _float_or_none(row.get("etox_ctue_per_kg", "")),
                    "class": chem_class,
                    "name": name,
                    "cas": cas,
                }
                # Key by normalised name (lowercase)
                _CURATED_BY_NAME[name.lower()] = entry
                # Also key by filename (which is the lowered / stripped form)
                fname = (row.get("filename") or "").strip().lower()
                if fname and fname not in _CURATED_BY_NAME:
                    _CURATED_BY_NAME[fname] = entry
                # Key by CAS
                if cas:
                    _CURATED_BY_CAS[cas] = entry
        logger.info("Loaded %d curated solvent entries from solvent-econ-lca-summary.csv", len(_CURATED_BY_NAME))
    except Exception as exc:
        logger.error("Failed to load curated solvent CSV: %s", exc)


_load_curated_csv()


def _curated_lookup(solvent: str) -> dict | None:
    """Look up curated economic/LCA data for a solvent.

    Resolution chain:
    1. Direct name match in curated CSV (case-insensitive)
    2. Resolve name → CAS via Solvent_Data.csv, then CAS → curated CSV
    3. Substring match (name longer than 5 chars contained in a curated key)
    """
    key = solvent.lower().strip()
    # 1. Direct name match
    if key in _CURATED_BY_NAME:
        return _CURATED_BY_NAME[key]
    # 2. CAS-based lookup: resolve name→CAS via Solvent_Data.csv, then CAS→curated
    csv_data = _csv_lookup(solvent)
    if csv_data and csv_data.get("cas"):
        cas = csv_data["cas"]
        if cas in _CURATED_BY_CAS:
            return _CURATED_BY_CAS[cas]
    # 3. Direct CAS match (if solvent string is itself a CAS)
    if key in _CURATED_BY_CAS:
        return _CURATED_BY_CAS[key]
    # 4. Substring match
    for curated_key, data in _CURATED_BY_NAME.items():
        if key in curated_key and len(key) > 5:
            return data
    return None


# ---------------------------------------------------------------------------
# LCA chemical-class averages (computed from validated ecoinvent/GaBi data)
# ---------------------------------------------------------------------------
# These are used as fallbacks when a solvent is NOT in _SOLVENT_LCA_IFS.
# Class averages are computed from the 16 core validated solvents.

_LCA_CLASS_AVERAGES: dict[str, dict[str, float]] = {
    "alkane":    {"solvent_gwp": 1.72, "solvent_htc": 4.05e-07, "solvent_htnc": 3.89e-07, "solvent_etox": 24.1},
    "aromatic":  {"solvent_gwp": 1.57, "solvent_htc": 3.20e-07, "solvent_htnc": 2.67e-07, "solvent_etox": 15.9},
    "alcohol":   {"solvent_gwp": 2.0,  "solvent_htc": 4.50e-07, "solvent_htnc": 4.00e-07, "solvent_etox": 25.0},
    "ketone":    {"solvent_gwp": 2.8,  "solvent_htc": 6.50e-07, "solvent_htnc": 6.00e-07, "solvent_etox": 35.0},
    "ester":     {"solvent_gwp": 4.90, "solvent_htc": 1.05e-06, "solvent_htnc": 1.30e-06, "solvent_etox": 63.9},
    "ether":     {"solvent_gwp": 4.50, "solvent_htc": 9.00e-07, "solvent_htnc": 1.20e-06, "solvent_etox": 60.0},
    "amide":     {"solvent_gwp": 3.50, "solvent_htc": 8.00e-07, "solvent_htnc": 9.50e-07, "solvent_etox": 48.0},
    "amine":     {"solvent_gwp": 3.71, "solvent_htc": 7.58e-07, "solvent_htnc": 8.89e-07, "solvent_etox": 50.2},
    "chlorinated": {"solvent_gwp": 3.30, "solvent_htc": 4.59e-07, "solvent_htnc": 7.13e-07, "solvent_etox": 30.1},
    "glycol":    {"solvent_gwp": 4.13, "solvent_htc": 9.68e-07, "solvent_htnc": 1.27e-06, "solvent_etox": 64.4},
    "sulfoxide": {"solvent_gwp": 2.80, "solvent_htc": 6.50e-07, "solvent_htnc": 7.90e-07, "solvent_etox": 39.0},
    "nitrile":   {"solvent_gwp": 3.50, "solvent_htc": 8.00e-07, "solvent_htnc": 9.50e-07, "solvent_etox": 48.0},
    "generic":   {"solvent_gwp": 3.00, "solvent_htc": 6.00e-07, "solvent_htnc": 7.00e-07, "solvent_etox": 40.0},
}


def _classify_solvent(solvent_name: str) -> str:
    """Classify a solvent into a chemical class for LCA fallback estimation.

    Uses keyword matching on the solvent name.  Returns one of the keys in
    ``_LCA_CLASS_AVERAGES``.
    """
    n = solvent_name.lower()

    # 1. Halogenated (highest priority — distinct LCA profile)
    if any(k in n for k in ("chloro", "bromo", "iodo", "fluoro", "freon")):
        return "chlorinated"

    # 2. Aromatics
    if any(k in n for k in (
        "benzene", "toluene", "xylene", "naphthalene", "phenyl",
        "styrene", "cumene", "aniline", "phenol", "pyridine",
        "indole", "quinoline", "anisole", "cresol",
    )):
        return "aromatic"

    # 3. Glycols / diols (before alcohols)
    if "glycol" in n or "diol" in n:
        return "glycol"

    # 4. Esters / lactones
    if any(k in n for k in (
        "acetate", "butyrate", "propionate", "formate",
        "phthalate", "benzoate", "acrylate", "lactone",
    )):
        return "ester"

    # 5. Ketones
    if any(k in n for k in (
        "ketone", "acetone", "butanone", "pentanone",
        "hexanone", "cyclohexanone", "acetophenone",
    )):
        return "ketone"

    # 6. Alcohols
    if any(k in n for k in (
        "methanol", "ethanol", "propanol", "butanol",
        "pentanol", "hexanol", "heptanol", "octanol",
        "cyclohexanol", "alcohol",
    )):
        return "alcohol"
    # Catch -ol suffix but avoid false positives on "toluol", "phenol" etc.
    if n.endswith("ol") and "tolu" not in n and "phen" not in n:
        return "alcohol"

    # 7. Ethers
    if any(k in n for k in (
        "ether", "furan", "dioxane", "dioxolane",
        "tetrahydrofuran", "tetrahydropyran", "methoxy",
        "ethoxy", "pyran",
    )):
        return "ether"

    # 8. Amides
    if any(k in n for k in ("amide", "formamide", "acetamide", "lactam", "pyrrolidone")):
        return "amide"

    # 9. Amines
    if any(k in n for k in ("amine", "piperidine", "pyrrolidine", "morpholine", "hydrazine")):
        return "amine"

    # 10. Nitriles
    if any(k in n for k in ("nitrile", "cyanide", "acetonitrile")):
        return "nitrile"

    # 11. Sulfoxides / sulfones
    if any(k in n for k in ("sulfoxide", "sulfone", "dmso", "sulfolane")):
        return "sulfoxide"

    # 12. Alkanes / cycloalkanes
    if any(k in n for k in (
        "hexane", "heptane", "octane", "nonane", "decane",
        "pentane", "butane", "propane", "cyclohexane",
        "cyclopentane", "methylcyclohexane", "dodecane",
    )):
        return "alkane"

    return "generic"


# ---------------------------------------------------------------------------
# Solvent / energy-case metadata (static, no BioSTEAM import needed)
# ---------------------------------------------------------------------------

# --- Core validated PE solvents (Branch-TEA) ---
_PE_SOLVENTS_CORE = [
    "sec-Butyl Acetate",
    "Isobutyl Acetate",
    "Tetrachloroethylene",
    "o-Chlorotoluene",
    "Methylcyclohexane",
    "Dodecanol",
    "Heptane",
    "Toluene",
    "Xylene",
]
# --- Extended PE/LDPE solvents from COMMON-SOLVENTS-DATABASE (thermosteam-validated) ---
_PE_SOLVENTS_EXTENDED = [
    "o-Xylene",
    "p-Xylene",
    "Cyclohexane",
    "Dodecane",
    "Hexane",
    "Benzene",
    "Acetone",
    "2-Butanone",
    "Ethyl acetate",
    "Tetrahydrofuran",
    "1-Propanol",
    "Ethanol",
    "Methanol",
    "Isopropanol",
    "tert-Butanol",
    "Cyclohexanol",
    "N,N-Dimethylformamide",
    "Diphenyl ether",
    "Acetylacetone",
    "2,3-Dihydropyran",
    "Tetrahydropyran",
    "Triethylamine",
    "Methyl acetate",
]
_PE_SOLVENTS = _PE_SOLVENTS_CORE + _PE_SOLVENTS_EXTENDED

_EVOH_SOLVENTS_E1 = ["Ethylene Glycol", "Pyridazine"]
_EVOH_SOLVENTS_E2 = [
    "butane-1,4-diol",
    "Diethanolamine",
    "Diethylene glycol",
    "Ethylene Glycol",
    "Propylene Glycol",
    "Pyridazine",
    "gamma-butyrolactone",
    # Extended EVOH solvents (>50% EVOH solubility from COMMON-SOLVENTS-DATABASE)
    "Dimethyl sulfoxide",
    "N,N-Dimethylformamide",
    "Triethylamine",
    "Methanol",
    "Ethanol",
    "Isopropanol",
]
_PET_SOLVENTS = [
    "Toluene",
    "Xylene",
    # Extended PET solvents (>30% PET solubility from COMMON-SOLVENTS-DATABASE)
    "Acetone",
    "N,N-Dimethylformamide",
    "Tetrahydrofuran",
    "2-Butanone",
    "Benzene",
]
_LDPE_SOLVENTS = list(_PE_SOLVENTS)  # LDPE dissolves in the same solvents as PE

# ---------------------------------------------------------------------------
# New-polymer solvent lists (PS / PP / PVC / PC)
# ---------------------------------------------------------------------------
# NOTE: PS, PP, and PVC are simulated using the PE process model (PE-proxy
# approximation) because thermosteam has no native oligomers for these
# polymers.  The solvent lists below reflect chemically appropriate solvents
# for each polymer; BioSTEAM process economics are approximate.
# Only solvents present in _SOLVENT_DEFAULTS / _SOLVENT_LCA_IFS are included
# so that LCA characterisation factors are always available.

# PS (polystyrene) — dissolves readily in aromatics and some polar solvents.
# Runs under PE-proxy model (PS → PE internally).
_PS_SOLVENTS = [
    "Toluene",
    "Xylene",
    "Tetrahydrofuran",
    "Acetone",
    "2-Butanone",
    "Cyclohexane",
    "Benzene",
    "Ethyl acetate",
]

# PP (polypropylene) — needs hot aromatic / cyclic solvents; limited solubility
# at ambient temperatures.  Runs under PE-proxy model (PP → PE internally).
_PP_SOLVENTS = [
    "Toluene",
    "Xylene",
    "Dodecane",       # proxy for Decalin (not in thermosteam DB)
    "Cyclohexane",
    "Tetrahydrofuran",
]

# PVC (poly(vinyl chloride)) — dissolves in polar aprotic solvents.
# Runs under PE-proxy model (PVC → PE internally).
_PVC_SOLVENTS = [
    "Tetrahydrofuran",
    "2-Butanone",     # proxy for Cyclohexanone (not in thermosteam DB)
    "N,N-Dimethylformamide",
    "Acetone",
]

# PC (polycarbonate) — dissolves in chlorinated and polar solvents.
# thermosteam ships a native PColigomer, so PC runs natively (no proxy).
# Dichloromethane is listed here for completeness but is on the chlorinated
# blocklist and may be filtered out in automated batch runs.
_PC_SOLVENTS = [
    "Dichloromethane",       # in chlorinated blocklist — may be filtered
    "Tetrahydrofuran",
    "N,N-Dimethylformamide",
    "Acetone",
    "Toluene",
]

_CHLORINATED_BLOCKLIST = [
    "Tetrachloroethylene", "o-Chlorotoluene",
    "Dichloromethane", "Chloroform",
]

# solvent: (price_usd_kg, dissolution_temp_c, gwp_if)
_SOLVENT_DEFAULTS: dict[str, tuple[float, float, float]] = {
    "sec-Butyl Acetate": (1.60, 110, 4.98),
    "Isobutyl Acetate": (1.60, 114, 4.81),
    "Tetrachloroethylene": (1.38, 120, 3.85),
    "o-Chlorotoluene": (2.40, 120, 2.74),
    "Methylcyclohexane": (1.55, 98, 2.55),
    "Dodecanol": (1.50, 120, 4.12),
    "Heptane": (1.42, 96, 0.897),
    "Toluene": (0.82, 110, 1.61),
    "Xylene": (0.84, 120, 1.52),
    "Ethylene Glycol": (0.53, 120, 2.7),
    "Pyridazine": (4.95, 120, 10.7),
    "butane-1,4-diol": (1.22, 120, 5.5),
    "Diethanolamine": (1.06, 120, 3.71),
    "Diethylene glycol": (0.59, 120, 3.15),
    "Propylene Glycol": (1.53, 120, 5.16),
    "gamma-butyrolactone": (2.58, 120, 6.54),
    # ── Extended solvents from COMMON-SOLVENTS-DATABASE ──────────────────
    # Prices: from solvent_lookup.py where available, else web research estimates.
    # GWP: from solvent_lookup.py where available, else class-average estimates.
    # Dissolution temps: min(BP_C - 2, 120). Low-BP solvents need pressure vessels.
    # Confidence: "estimated" unless sourced from Branch-TEA or ecoinvent.
    # --- Alkanes ---
    "Cyclohexane":             (0.90,  78,  1.2),
    "Dodecane":                (1.80, 120,  1.2),
    "Hexane":                  (0.85,  66,  0.9),
    # --- Aromatics ---
    "Benzene":                 (0.75,  78,  1.2),
    "o-Xylene":                (0.85, 120,  1.52),
    "p-Xylene":                (0.90, 120,  1.52),
    "Diphenyl ether":          (3.00, 120,  4.5),
    # --- Alcohols ---
    "Methanol":                (0.40,  62,  1.5),
    "Ethanol":                 (0.70,  76,  1.8),
    "Isopropanol":             (0.80,  80,  2.0),
    "1-Propanol":              (1.20,  95,  2.0),
    "tert-Butanol":            (1.40,  80,  2.2),
    "Cyclohexanol":            (1.30, 120,  2.5),
    # --- Ketones ---
    "Acetone":                 (1.05,  54,  2.55),
    "2-Butanone":              (1.39,  77,  3.2),
    "Acetylacetone":           (3.50, 120,  3.5),
    # --- Esters ---
    "Ethyl acetate":           (1.13,  75,  2.4),
    "Methyl acetate":          (0.90,  54,  2.2),
    # --- Ethers ---
    "Tetrahydrofuran":         (2.10,  64,  5.5),
    "Tetrahydropyran":         (8.00,  86,  4.0),
    "2,3-Dihydropyran":       (15.00,  83,  4.0),
    # --- Amides / Sulfoxides ---
    "N,N-Dimethylformamide":   (1.20, 120,  3.8),
    "Dimethyl sulfoxide":      (1.50, 120,  2.8),
    # --- Amines ---
    "Triethylamine":           (1.80,  86,  4.0),
    "Isopropylamine":          (2.00,  29,  4.0),  # BP 32°C — needs pressure vessel
    # --- Chlorinated (in blocklist — available for explicit use) ---
    "Dichloromethane":         (0.55,  37,  2.8),
    "Chloroform":              (0.45,  59,  3.0),
}

# Per-solvent LCA impact factors from Branch-TEA.ipynb reference notebook.
# Keys: solvent_gwp, solvent_htc, solvent_htnc, solvent_etox
# These are the *base* impact factors; the worker adds fixed offsets
# (e.g. +0.1563 for GWP burn credit) automatically.
_SOLVENT_LCA_IFS: dict[str, dict[str, float]] = {
    "sec-Butyl Acetate": {"solvent_gwp": 4.98, "solvent_htc": 1.04e-06, "solvent_htnc": 1.28e-06, "solvent_etox": 63},
    "Isobutyl Acetate":  {"solvent_gwp": 4.81, "solvent_htc": 1.06e-06, "solvent_htnc": 1.32e-06, "solvent_etox": 64.7},
    "Tetrachloroethylene": {"solvent_gwp": 3.85, "solvent_htc": 1.58e-07, "solvent_htnc": 4.89e-07, "solvent_etox": 8.74},
    "o-Chlorotoluene":   {"solvent_gwp": 2.74, "solvent_htc": 7.59e-07, "solvent_htnc": 9.37e-07, "solvent_etox": 51.4},
    "Methylcyclohexane":  {"solvent_gwp": 2.55, "solvent_htc": 5.43e-07, "solvent_htnc": 5.39e-07, "solvent_etox": 32.8},
    "Dodecanol":         {"solvent_gwp": 4.12, "solvent_htc": 1.20e-06, "solvent_htnc": 4.10e-06, "solvent_etox": 165},
    "Heptane":           {"solvent_gwp": 0.897, "solvent_htc": 2.67e-07, "solvent_htnc": 2.39e-07, "solvent_etox": 15.5},
    "Toluene":           {"solvent_gwp": 1.61, "solvent_htc": 3.31e-07, "solvent_htnc": 2.75e-07, "solvent_etox": 16.5},
    "Xylene":            {"solvent_gwp": 1.52, "solvent_htc": 3.09e-07, "solvent_htnc": 2.59e-07, "solvent_etox": 15.4},
    "Ethylene Glycol":   {"solvent_gwp": 2.7, "solvent_htc": 6.25e-07, "solvent_htnc": 7.65e-07, "solvent_etox": 40.7},
    "Pyridazine":        {"solvent_gwp": 10.7, "solvent_htc": 2.82e-06, "solvent_htnc": 4.07e-06, "solvent_etox": 286},
    # EVOH E2 solvents (5 additional — Ethylene Glycol & Pyridazine shared with E1 above)
    "butane-1,4-diol":   {"solvent_gwp": 5.5, "solvent_htc": 1.06e-06, "solvent_htnc": 1.43e-06, "solvent_etox": 68.5},
    "Diethanolamine":    {"solvent_gwp": 3.71, "solvent_htc": 7.58e-07, "solvent_htnc": 8.89e-07, "solvent_etox": 50.2},
    "Diethylene glycol": {"solvent_gwp": 3.15, "solvent_htc": 7.27e-07, "solvent_htnc": 8.93e-07, "solvent_etox": 47.5},
    "Propylene Glycol":  {"solvent_gwp": 5.16, "solvent_htc": 1.46e-06, "solvent_htnc": 1.98e-06, "solvent_etox": 101},
    "gamma-butyrolactone": {"solvent_gwp": 6.54, "solvent_htc": 1.36e-06, "solvent_htnc": 1.84e-06, "solvent_etox": 91.4},
    # ── Extended solvents (estimated LCA IFs — class-average scaling) ────
    # HTC/HTNC/ETOX are estimated by scaling from validated solvents in the
    # same chemical class. These should be replaced with ecoinvent/GaBi data
    # when available. GWP values marked (est) also need validation.
    # --- Alkanes (scaled from Heptane) ---
    "Cyclohexane":             {"solvent_gwp": 1.2,  "solvent_htc": 3.58e-07, "solvent_htnc": 3.20e-07, "solvent_etox": 20.7},
    "Dodecane":                {"solvent_gwp": 1.2,  "solvent_htc": 3.58e-07, "solvent_htnc": 3.20e-07, "solvent_etox": 20.7},
    "Hexane":                  {"solvent_gwp": 0.9,  "solvent_htc": 2.67e-07, "solvent_htnc": 2.39e-07, "solvent_etox": 15.5},
    # --- Aromatics (scaled from Toluene/Xylene) ---
    "Benzene":                 {"solvent_gwp": 1.2,  "solvent_htc": 2.50e-07, "solvent_htnc": 2.10e-07, "solvent_etox": 12.4},
    "o-Xylene":                {"solvent_gwp": 1.52, "solvent_htc": 3.09e-07, "solvent_htnc": 2.59e-07, "solvent_etox": 15.4},
    "p-Xylene":                {"solvent_gwp": 1.52, "solvent_htc": 3.09e-07, "solvent_htnc": 2.59e-07, "solvent_etox": 15.4},
    "Diphenyl ether":          {"solvent_gwp": 4.5,  "solvent_htc": 9.40e-07, "solvent_htnc": 1.26e-06, "solvent_etox": 63.0},
    # --- Alcohols (interpolated between alkane and glycol classes) ---
    "Methanol":                {"solvent_gwp": 1.5,  "solvent_htc": 3.50e-07, "solvent_htnc": 3.00e-07, "solvent_etox": 18.0},
    "Ethanol":                 {"solvent_gwp": 1.8,  "solvent_htc": 4.00e-07, "solvent_htnc": 3.50e-07, "solvent_etox": 22.0},
    "Isopropanol":             {"solvent_gwp": 2.0,  "solvent_htc": 4.50e-07, "solvent_htnc": 4.00e-07, "solvent_etox": 25.0},
    "1-Propanol":              {"solvent_gwp": 2.0,  "solvent_htc": 4.50e-07, "solvent_htnc": 4.00e-07, "solvent_etox": 25.0},
    "tert-Butanol":            {"solvent_gwp": 2.2,  "solvent_htc": 5.00e-07, "solvent_htnc": 4.50e-07, "solvent_etox": 28.0},
    "Cyclohexanol":            {"solvent_gwp": 2.5,  "solvent_htc": 5.50e-07, "solvent_htnc": 5.00e-07, "solvent_etox": 32.0},
    # --- Ketones (intermediate between esters and aromatics) ---
    "Acetone":                 {"solvent_gwp": 2.55, "solvent_htc": 5.50e-07, "solvent_htnc": 5.00e-07, "solvent_etox": 32.0},
    "2-Butanone":              {"solvent_gwp": 3.2,  "solvent_htc": 7.00e-07, "solvent_htnc": 6.50e-07, "solvent_etox": 40.0},
    "Acetylacetone":           {"solvent_gwp": 3.5,  "solvent_htc": 7.50e-07, "solvent_htnc": 7.00e-07, "solvent_etox": 44.0},
    # --- Esters (scaled from SBA/IBA) ---
    "Ethyl acetate":           {"solvent_gwp": 2.4,  "solvent_htc": 5.00e-07, "solvent_htnc": 6.10e-07, "solvent_etox": 30.0},
    "Methyl acetate":          {"solvent_gwp": 2.2,  "solvent_htc": 4.60e-07, "solvent_htnc": 5.60e-07, "solvent_etox": 27.5},
    # --- Ethers (scaled from GBL) ---
    "Tetrahydrofuran":         {"solvent_gwp": 5.5,  "solvent_htc": 1.15e-06, "solvent_htnc": 1.55e-06, "solvent_etox": 77.0},
    "Tetrahydropyran":         {"solvent_gwp": 4.0,  "solvent_htc": 8.30e-07, "solvent_htnc": 1.12e-06, "solvent_etox": 56.0},
    "2,3-Dihydropyran":        {"solvent_gwp": 4.0,  "solvent_htc": 8.30e-07, "solvent_htnc": 1.12e-06, "solvent_etox": 56.0},
    # --- Amides / Sulfoxides (scaled from Diethanolamine/EG) ---
    "N,N-Dimethylformamide":   {"solvent_gwp": 3.8,  "solvent_htc": 8.80e-07, "solvent_htnc": 1.08e-06, "solvent_etox": 53.0},
    "Dimethyl sulfoxide":      {"solvent_gwp": 2.8,  "solvent_htc": 6.50e-07, "solvent_htnc": 7.90e-07, "solvent_etox": 39.0},
    # --- Amines ---
    "Triethylamine":           {"solvent_gwp": 4.0,  "solvent_htc": 8.30e-07, "solvent_htnc": 1.12e-06, "solvent_etox": 56.0},
    "Isopropylamine":          {"solvent_gwp": 4.0,  "solvent_htc": 8.30e-07, "solvent_htnc": 1.12e-06, "solvent_etox": 56.0},
    # --- Chlorinated (scaled from Tetrachloroethylene) ---
    "Dichloromethane":         {"solvent_gwp": 2.8,  "solvent_htc": 1.15e-07, "solvent_htnc": 3.56e-07, "solvent_etox": 6.4},
    "Chloroform":              {"solvent_gwp": 3.0,  "solvent_htc": 1.23e-07, "solvent_htnc": 3.81e-07, "solvent_etox": 6.8},
}


def _load_tea_lca_process_defaults() -> dict[str, tuple[float, float, float]]:
    """Load price + default hot-process temperature defaults from the TEA/LCA CSV.

    Assumptions:
    - ``price`` is recorded in USD / metric ton and converted here to USD / kg.
    - ``th`` is the preferred heated-process temperature in degC.
    - ``th`` is clamped to remain below the normal-boiling-point ``bp`` at 1 atm.
    """
    if not _TEA_LCA_SOLVENT_CSV.exists():
        return {}

    loaded: dict[str, tuple[float, float, float]] = {}
    try:
        with open(_TEA_LCA_SOLVENT_CSV, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_name = (row.get("name_biosteam") or row.get("name_cosmobase") or "").strip()
                if not raw_name:
                    continue
                solvent_name = resolve_to_biosteam(raw_name) or raw_name
                try:
                    raw_price = float((row.get("price") or "").strip())
                    price_usd_per_kg = raw_price / 1000.0
                except ValueError:
                    continue
                try:
                    th_c = float((row.get("th") or "").strip())
                except ValueError:
                    th_c = 0.0
                try:
                    bp_c = float((row.get("bp") or "").strip())
                except ValueError:
                    bp_c = 0.0
                if th_c <= 0:
                    continue
                if bp_c > 1.0:
                    th_c = min(th_c, bp_c - 1.0)
                th_c = max(25.0, th_c)
                old_gwp = _SOLVENT_DEFAULTS.get(solvent_name, (0.0, 0.0, 0.0))[2]
                loaded[solvent_name] = (price_usd_per_kg, th_c, old_gwp)
    except Exception as exc:
        logger.error("Failed to load TEA/LCA solvent defaults CSV: %s", exc)
        return {}
    return loaded


_SOLVENT_DEFAULTS.update(_load_tea_lca_process_defaults())

# Natural gas CFs for C1/C3 (produced + combustion).
# From Branch-TEA.ipynb: 0.383 + 3.45805 = 3.84105, etc.
_NATURAL_GAS_CFS = {
    "natural_gas_gwp": 0.383 + 3.45805,
    "natural_gas_htc": 4.76e-10 + 2.4735e-07,
    "natural_gas_htnc": 4.44e-08 + 7.5175e-08,
    "natural_gas_etox": 0.908 + 5.7715,
}

# Water CFs (same for all solvents).
_WATER_CFS = {
    "water_gwp": 0.000127,
    "water_htc": 1.40e-10,
    "water_htnc": 7.96e-11,
    "water_etox": 0.00538,
}


def _build_lca_cfs(solvent: str, energy_case: str = "C1") -> dict[str, float]:
    """Build the lca_cfs dict for a solvent + energy case.

    Merges solvent IFs, natural-gas CFs, water CFs, and (for C2/C3)
    grid-electricity CFs.

    Three-tier fallback for solvent LCA impact factors:
    1. Validated data from ``_SOLVENT_LCA_IFS`` (Branch-TEA / ecoinvent)
    2. Curated data from ``solvent-econ-lca-summary.csv`` (web-research)
    3. Chemical-class averages from ``_LCA_CLASS_AVERAGES``
    """
    cfs: dict[str, float] = {}
    cfs.update(_NATURAL_GAS_CFS)
    cfs.update(_WATER_CFS)
    solvent_ifs = _SOLVENT_LCA_IFS.get(solvent)
    if solvent_ifs:
        cfs.update(solvent_ifs)
    else:
        # Tier 2: check curated CSV for per-solvent LCA factors
        curated = _curated_lookup(solvent)
        solvent_class = _classify_solvent(solvent)
        class_avg = _LCA_CLASS_AVERAGES[solvent_class]

        if curated and any(curated.get(k) is not None for k in ("gwp", "htc", "htnc", "etox")):
            # Build IFs from curated data, falling back to class averages per-indicator
            merged_ifs = {
                "solvent_gwp": curated["gwp"] if curated.get("gwp") is not None else class_avg["solvent_gwp"],
                "solvent_htc": curated["htc"] if curated.get("htc") is not None else class_avg["solvent_htc"],
                "solvent_htnc": curated["htnc"] if curated.get("htnc") is not None else class_avg["solvent_htnc"],
                "solvent_etox": curated["etox"] if curated.get("etox") is not None else class_avg["solvent_etox"],
            }
            cfs.update(merged_ifs)
            curated_keys = [k for k in ("gwp", "htc", "htnc", "etox") if curated.get(k) is not None]
            class_keys = [k for k in ("gwp", "htc", "htnc", "etox") if curated.get(k) is None]
            logger.info(
                "Solvent '%s' not in _SOLVENT_LCA_IFS — using curated data for %s, "
                "%s class avg for %s (GWP=%.2f, HTC=%.2e, HTNC=%.2e, ETOX=%.1f)",
                solvent, curated_keys, solvent_class, class_keys,
                merged_ifs["solvent_gwp"], merged_ifs["solvent_htc"],
                merged_ifs["solvent_htnc"], merged_ifs["solvent_etox"],
            )
        else:
            # Tier 3: pure class-average fallback
            cfs.update(class_avg)
            logger.info(
                "Solvent '%s' not in _SOLVENT_LCA_IFS or curated CSV — using %s class averages "
                "(GWP=%.2f, HTC=%.2e, HTNC=%.2e, ETOX=%.1f)",
                solvent, solvent_class,
                class_avg["solvent_gwp"], class_avg["solvent_htc"],
                class_avg["solvent_htnc"], class_avg["solvent_etox"],
            )

    # Grid electricity CFs apply to C2 and C3 (grid-connected cases).
    # C1 uses on-site CHP so electricity impacts are captured via NG CFs.
    if energy_case.upper() in ("C2", "C3"):
        cfs["electricity_gwp"] = 0.197
        cfs["electricity_htc"] = 3.12e-09
        cfs["electricity_htnc"] = 2.44e-07
        cfs["electricity_etox"] = 5.24
    return cfs

_ENERGY_CASES: dict[str, str] = {
    "C1": "CHP (on-site boiler + turbogenerator)",
    "C2": "Grid + AMCOR (no on-site utilities)",
    "C3": "Grid + Boiler (boiler but no turbogenerator)",
}


# ---------------------------------------------------------------------------
# 1. run_single_simulation
# ---------------------------------------------------------------------------

def run_single_simulation(config: dict, timeout: int = 120) -> dict:
    """Run one BioSTEAM simulation in an isolated subprocess.

    Args:
        config: Simulation parameters (see biosteam_worker.py input contract).
        timeout: Max seconds before killing subprocess (default 120).

    Returns:
        dict with 'success' key.  On success: tea/lca/operations data.
        On failure: 'error' and 'error_type' keys.
    """
    solvent = config.get("solvent", "unknown")
    energy = config.get("energy_case", "?")
    logger.info("Starting BioSTEAM simulation: solvent=%s energy=%s", solvent, energy)

    # Inject per-solvent LCA characterisation factors if not already provided
    if "lca_cfs" not in config:
        config = dict(config)  # don't mutate caller's dict
        config["lca_cfs"] = _build_lca_cfs(solvent, energy)

    json_str = json.dumps(config)

    try:
        result = subprocess.run(
            [sys.executable, str(_WORKER_SCRIPT), json_str],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Simulation timed out after %ds (solvent=%s)", timeout, solvent)
        return {
            "success": False,
            "error": f"Simulation timed out after {timeout}s",
            "error_type": "TimeoutExpired",
            "solvent": solvent,
            "energy_case": energy,
        }

    # Guard against runaway subprocess output (memory protection)
    if result.stdout and len(result.stdout) > _MAX_SUBPROCESS_STDOUT_BYTES:
        logger.warning(
            "BioSTEAM stdout truncated: %d -> %d bytes",
            len(result.stdout), _MAX_SUBPROCESS_STDOUT_BYTES,
        )
        result = subprocess.CompletedProcess(
            result.args, result.returncode,
            stdout=result.stdout[:_MAX_SUBPROCESS_STDOUT_BYTES],
            stderr=(result.stderr or "")[:_MAX_SUBPROCESS_STDERR_BYTES],
        )
    elif result.stderr and len(result.stderr) > _MAX_SUBPROCESS_STDERR_BYTES:
        result = subprocess.CompletedProcess(
            result.args, result.returncode,
            stdout=result.stdout,
            stderr=result.stderr[:_MAX_SUBPROCESS_STDERR_BYTES],
        )

    # Non-zero exit code
    if result.returncode != 0:
        stderr_lines = (result.stderr or "").strip().splitlines()
        err_msg = stderr_lines[-1] if stderr_lines else "Unknown subprocess error"
        logger.error(
            "Simulation failed (rc=%d, solvent=%s): %s",
            result.returncode, solvent, err_msg,
        )
        return {
            "success": False,
            "error": err_msg,
            "error_type": "SubprocessError",
            "returncode": result.returncode,
            "solvent": solvent,
            "energy_case": energy,
        }

    # Parse stdout JSON
    stdout = (result.stdout or "").strip()
    if not stdout:
        logger.error("Empty stdout from worker (solvent=%s)", solvent)
        return {
            "success": False,
            "error": "Worker produced no output",
            "error_type": "EmptyOutput",
            "stderr_snippet": (result.stderr or "")[:500],
            "solvent": solvent,
            "energy_case": energy,
        }

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error (solvent=%s): %s", solvent, exc)
        return {
            "success": False,
            "error": f"JSON parse error: {exc}",
            "error_type": "JSONDecodeError",
            "stdout_snippet": stdout[:500],
            "stderr_snippet": (result.stderr or "")[:500],
            "solvent": solvent,
            "energy_case": energy,
        }

    logger.info(
        "Simulation complete: solvent=%s success=%s",
        solvent, parsed.get("success"),
    )
    return parsed


# ---------------------------------------------------------------------------
# 2. run_batch_simulations
# ---------------------------------------------------------------------------

def run_batch_simulations(
    configs: list[dict],
    max_parallel: int = 3,
    timeout_per_sim: int = 120,
) -> list[dict]:
    """Run multiple BioSTEAM simulations, optionally in parallel.

    Each simulation runs in its own subprocess (no state contamination).
    Uses ThreadPoolExecutor for parallelism (each thread spawns a subprocess).

    Args:
        configs: List of simulation parameter dicts.
        max_parallel: Max concurrent subprocesses (default 3, capped at 4).
        timeout_per_sim: Timeout per simulation in seconds.

    Returns:
        List of result dicts in the same order as *configs*.
    """
    if not configs:
        logger.warning("run_batch_simulations called with empty configs list")
        return []

    workers = min(max_parallel, 4, len(configs))
    logger.info(
        "Running %d simulation(s) with %d parallel worker(s)",
        len(configs), workers,
    )

    # Pre-fill results list to preserve ordering
    results: list[dict | None] = [None] * len(configs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(run_single_simulation, cfg, timeout_per_sim): idx
            for idx, cfg in enumerate(configs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Unexpected error in batch item %d: %s", idx, exc)
                results[idx] = {
                    "success": False,
                    "error": str(exc),
                    "error_type": "ExecutorError",
                }

    # Safety: replace any remaining None slots
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "success": False,
                "error": "Result was not populated (internal error)",
                "error_type": "MissingResult",
            }

    succeeded = sum(1 for r in results if r.get("success"))
    logger.info(
        "Batch complete: %d/%d succeeded", succeeded, len(results),
    )
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 3. build_batch_configs
# ---------------------------------------------------------------------------

def build_batch_configs(
    solvents: list[str],
    target_plastic: str = "PE",
    energy_cases: list[str] | None = None,
    target_plastic_percent: float = 60,
    processing_capacity: float = 20_000,
    **kwargs: Any,
) -> list[dict]:
    """Generate config dicts for batch simulation.

    Creates one config per (solvent, energy_case) combination.

    Supports **any** solvent name.  Known solvents use validated data from
    ``_SOLVENT_DEFAULTS``; unknown solvents get graceful fallbacks:

    * **price** — $1.50/kg generic default
    * **dissolution_temperature_c** — min(BP − 10, 130) from
      ``Solvent_Data.csv``, or 110 °C if BP unavailable

    Args:
        solvents: List of solvent names.
        target_plastic: Target polymer (default 'PE').
        energy_cases: List of energy cases (default ['C1']).
        target_plastic_percent: wt% of target plastic in feed.
        processing_capacity: Plant capacity in kg/hr.
        **kwargs: Additional parameters applied to all configs.

    Returns:
        List of config dicts ready for *run_batch_simulations*.
    """
    if energy_cases is None:
        energy_cases = ["C1"]

    configs: list[dict] = []
    for solvent in solvents:
        for ec in energy_cases:
            cfg: dict[str, Any] = {
                "solvent": solvent,
                "target_plastic": target_plastic,
                "energy_case": ec,
                "target_plastic_percent": target_plastic_percent,
                "processing_capacity": processing_capacity,
            }
            # Apply solvent defaults (price, dissolution temp) if available
            defaults = _SOLVENT_DEFAULTS.get(solvent)
            if defaults:
                price, diss_temp, _gwp = defaults
                cfg.setdefault("solvent_price", price)
                cfg.setdefault("dissolution_temperature_c", diss_temp)
            else:
                # Unknown solvent — apply fallbacks
                # Tier 2: check curated CSV for web-researched price
                curated = _curated_lookup(solvent)
                if curated and curated.get("price") is not None:
                    cfg.setdefault("solvent_price", curated["price"])
                else:
                    cfg.setdefault("solvent_price", 1.50)
                csv_data = _csv_lookup(solvent)
                if csv_data and csv_data["bp_c"] is not None:
                    diss_temp_fb = min(csv_data["bp_c"] - 10, 130.0)
                    # Clamp to reasonable range [40, 130]
                    diss_temp_fb = max(40.0, diss_temp_fb)
                    cfg.setdefault("dissolution_temperature_c", diss_temp_fb)
                else:
                    cfg.setdefault("dissolution_temperature_c", 110.0)
            # Apply any extra kwargs
            cfg.update(kwargs)
            configs.append(cfg)

    logger.info(
        "Built %d configs: %d solvents x %d energy cases",
        len(configs), len(solvents), len(energy_cases),
    )
    return configs


# ---------------------------------------------------------------------------
# 4. get_supported_solvents
# ---------------------------------------------------------------------------

def get_supported_solvents() -> dict:
    """Return supported solvents, energy cases, and known limitations.

    Static metadata -- no BioSTEAM import required.
    """
    from strap.services.biosteam_service import (
        EVOH_SOLVENTS,
        EVOH_SOLVENTS_E2,
        LDPE_SOLVENTS,
        PC_SOLVENTS,
        PE_SOLVENTS,
        PET_SOLVENTS,
        PP_SOLVENTS,
        PS_SOLVENTS,
        PVC_SOLVENTS,
    )

    return {
        "pe_solvents": list(PE_SOLVENTS),
        "ldpe_solvents": list(LDPE_SOLVENTS),
        "evoh_solvents_e1": list(EVOH_SOLVENTS),
        "evoh_solvents_e2": list(EVOH_SOLVENTS_E2),
        "pet_solvents": list(PET_SOLVENTS),
        # New polymers (PS/PP/PVC use PE-proxy; PC is native)
        "ps_solvents": list(PS_SOLVENTS),
        "pp_solvents": list(PP_SOLVENTS),
        "pvc_solvents": list(PVC_SOLVENTS),
        "pc_solvents": list(PC_SOLVENTS),
        "energy_cases": dict(_ENERGY_CASES),
        "solvent_defaults": dict(_SOLVENT_DEFAULTS),
        "chlorinated_blocklist": list(_CHLORINATED_BLOCKLIST),
        "csv_solvents_loaded": len(_SOLVENT_CSV_DATA),
        "curated_solvents_loaded": len(_CURATED_BY_NAME),
        "catalog_source": "strap.services.biosteam_service",
        "known_limitations": [
            "Chlorinated solvents (TCE, OCT) may fail due to HCl not in property package",
            "Each simulation takes ~10-15 seconds in subprocess",
            "BioSTEAM requires subprocess isolation (global state contamination)",
            "PS, PP, PVC use the PE process model (PE-proxy) — approximate economics/LCA",
            "Solvents not in _SOLVENT_DEFAULTS check curated CSV, then fall back to $1.50/kg",
            "LCA 3-tier fallback: validated → curated CSV → chemical-class averages",
        ],
    }


# ---------------------------------------------------------------------------
# 5. format_results_table
# ---------------------------------------------------------------------------

def format_results_table(results: list[dict]) -> str:
    """Format batch results as a markdown table.

    Columns: Solvent | Energy Case | MSP ($/kg) | GWP | TCI ($M) | Status

    Failed simulations show the error message in the Status column.
    """
    if not results:
        return "_No simulation results to display._"

    lines: list[str] = [
        "| Solvent | Energy Case | MSP ($/kg) | GWP (kg CO2-eq/kg) | TCI ($M) | Status |",
        "|---------|-------------|------------|---------------------|----------|--------|",
    ]

    for r in results:
        solvent = r.get("solvent", r.get("config", {}).get("solvent", "?"))
        energy = r.get("energy_case", r.get("config", {}).get("energy_case", "?"))

        if r.get("success"):
            tea = r.get("tea", {})
            lca = r.get("lca", {})
            msp = tea.get("msp_usd_per_kg")
            gwp = lca.get("gwp_kg_co2e_per_kg")
            tci = tea.get("tci_usd")

            msp_str = f"{msp:.2f}" if msp is not None else "N/A"
            gwp_str = f"{gwp:.3f}" if gwp is not None else "N/A"
            tci_str = f"{tci / 1e6:.2f}" if tci is not None else "N/A"
            status = "OK"
        else:
            msp_str = "-"
            gwp_str = "-"
            tci_str = "-"
            err = r.get("error", "unknown error")
            # Truncate long error messages for table readability
            status = err[:60] + "..." if len(err) > 60 else err

        lines.append(
            f"| {solvent} | {energy} | {msp_str} | {gwp_str} | {tci_str} | {status} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. rank_results
# ---------------------------------------------------------------------------

_METRIC_KEYS: dict[str, tuple[str, str]] = {
    # metric_name: (top-level_section, key_within_section)
    "msp": ("tea", "msp_usd_per_kg"),
    "gwp": ("lca", "gwp_kg_co2e_per_kg"),
    "tci": ("tea", "tci_usd"),
    "energy": ("operations", "total_energy_mj_per_kg"),
}


# ---------------------------------------------------------------------------
# 6a. Sensitivity / uncertainty helpers
# ---------------------------------------------------------------------------

_DEFAULT_SWEEPABLE_PARAMS = [
    "solvent_price",
    "solvent_loss_pct",
    "dissolution_temperature_c",
    "precipitation_temperature_c",
    "feedstock_distance_km",
]


def _get_default_parameter_ranges(
    solvent: str,
    target_plastic: str = "PE",
) -> dict[str, tuple[float, float]]:
    """Return ``{param_name: (min, max)}`` for sweepable BioSTEAM parameters.

    Ranges are derived from ``_SOLVENT_DEFAULTS`` and process knowledge:

    * **solvent_price** – 0.7x to 1.5x the base price (historical volatility).
    * **solvent_loss_pct** – 0.001 % to 0.1 % (best-practice to poor recovery).
    * **dissolution_temperature_c** – base ± 20 °C, clamped to [90, BP − 2].
    * **precipitation_temperature_c** – 15 – 35 °C (seasonal ambient).
    * **feedstock_distance_km** – 0 – 200 km (local to regional sourcing).
    """
    defaults = _SOLVENT_DEFAULTS.get(solvent)
    if defaults:
        base_price, base_diss_temp, _gwp = defaults
    else:
        # Tier 2: check curated CSV for web-researched price
        curated = _curated_lookup(solvent)
        if curated and curated.get("price") is not None:
            base_price = curated["price"]
        else:
            base_price = 1.50
        csv_data = _csv_lookup(solvent)
        if csv_data and csv_data["bp_c"] is not None:
            base_diss_temp = min(csv_data["bp_c"] - 10, 130.0)
            base_diss_temp = max(40.0, base_diss_temp)
        else:
            base_diss_temp = 110.0

    # Dissolution temperature clamped to [90, boiling_point - 2]
    diss_lo = max(90.0, base_diss_temp - 20.0)
    diss_hi = base_diss_temp + 20.0

    return {
        "solvent_price": (round(base_price * 0.7, 4), round(base_price * 1.5, 4)),
        "solvent_loss_pct": (0.001, 0.1),
        "dissolution_temperature_c": (diss_lo, diss_hi),
        "precipitation_temperature_c": (15.0, 35.0),
        "feedstock_distance_km": (0.0, 200.0),
    }


def _build_monte_carlo_configs(
    solvent: str,
    target_plastic: str = "PE",
    energy_case: str = "C1",
    n_samples: int = 20,
    parameters: str | list[str] = "all",
    processing_capacity: float = 20_000,
) -> list[dict]:
    """Generate *n_samples* configs with uniform random sampling over parameter ranges.

    Each returned dict is a standard config passable to ``run_single_simulation()``.
    ``n_samples`` is capped at 50 to keep agent runtime practical.
    """
    import random

    cap = 50
    if n_samples > cap:
        logger.warning(
            "n_samples=%d exceeds cap (%d); clamping to %d",
            n_samples, cap, cap,
        )
        n_samples = cap

    ranges = _get_default_parameter_ranges(solvent, target_plastic)

    # Resolve which parameters to sweep
    if parameters == "all" or parameters == ["all"]:
        param_names = list(ranges.keys())
    elif isinstance(parameters, str):
        param_names = [p.strip() for p in parameters.split(",") if p.strip()]
    else:
        param_names = list(parameters)
    # Filter to only known sweepable params
    param_names = [p for p in param_names if p in ranges]

    # Base config values (midpoints for non-swept params)
    defaults = _SOLVENT_DEFAULTS.get(solvent)
    base_price = defaults[0] if defaults else 1.0
    base_diss = defaults[1] if defaults else 110.0

    configs: list[dict] = []
    for _ in range(n_samples):
        cfg: dict[str, Any] = {
            "solvent": solvent,
            "target_plastic": target_plastic,
            "energy_case": energy_case,
            "processing_capacity": processing_capacity,
            "solvent_price": base_price,
            "dissolution_temperature_c": base_diss,
            "precipitation_temperature_c": 25.0,
            "solvent_loss_pct": 0.01,
            "feedstock_distance_km": 0.0,
        }
        for p in param_names:
            lo, hi = ranges[p]
            cfg[p] = random.uniform(lo, hi)
        configs.append(cfg)

    return configs


def _build_sweep_configs(
    solvent: str,
    target_plastic: str = "PE",
    energy_case: str = "C1",
    parameter: str = "solvent_price",
    values: list[float] | None = None,
    processing_capacity: float = 20_000,
) -> list[dict]:
    """Generate one config per value in a parameter sweep.

    If *values* is ``None`` or empty, 5 evenly spaced values from the
    default range are generated automatically.
    """
    ranges = _get_default_parameter_ranges(solvent, target_plastic)
    if parameter not in ranges:
        raise ValueError(
            f"Unknown sweep parameter '{parameter}'. "
            f"Choose from: {list(ranges.keys())}"
        )

    if not values:
        lo, hi = ranges[parameter]
        import numpy as np
        values = list(np.linspace(lo, hi, 5))

    defaults = _SOLVENT_DEFAULTS.get(solvent)
    base_price = defaults[0] if defaults else 1.0
    base_diss = defaults[1] if defaults else 110.0

    configs: list[dict] = []
    for val in values:
        cfg: dict[str, Any] = {
            "solvent": solvent,
            "target_plastic": target_plastic,
            "energy_case": energy_case,
            "processing_capacity": processing_capacity,
            "solvent_price": base_price,
            "dissolution_temperature_c": base_diss,
            "precipitation_temperature_c": 25.0,
            "solvent_loss_pct": 0.01,
            "feedstock_distance_km": 0.0,
        }
        cfg[parameter] = val
        cfg["_sweep_param"] = parameter
        cfg["_sweep_value"] = val
        configs.append(cfg)

    return configs


def _build_tornado_configs(
    solvent: str,
    target_plastic: str = "PE",
    energy_case: str = "C1",
    parameters: str | list[str] = "all",
    processing_capacity: float = 20_000,
) -> tuple[dict, list[dict]]:
    """Build baseline + one-at-a-time (OAT) min/max configs for tornado analysis.

    Returns ``(baseline_config, oat_configs)`` where *oat_configs* has
    2 entries per parameter (min then max).  Each OAT config carries
    ``_tornado_param`` and ``_tornado_bound`` metadata keys.
    """
    ranges = _get_default_parameter_ranges(solvent, target_plastic)

    if parameters == "all" or parameters == ["all"]:
        param_names = list(ranges.keys())
    elif isinstance(parameters, str):
        param_names = [p.strip() for p in parameters.split(",") if p.strip()]
    else:
        param_names = list(parameters)
    param_names = [p for p in param_names if p in ranges]

    # Baseline uses midpoint of each parameter range
    defaults = _SOLVENT_DEFAULTS.get(solvent)
    base_price = defaults[0] if defaults else 1.0
    base_diss = defaults[1] if defaults else 110.0

    baseline: dict[str, Any] = {
        "solvent": solvent,
        "target_plastic": target_plastic,
        "energy_case": energy_case,
        "processing_capacity": processing_capacity,
        "solvent_price": base_price,
        "dissolution_temperature_c": base_diss,
        "precipitation_temperature_c": 25.0,
        "solvent_loss_pct": 0.01,
        "feedstock_distance_km": 0.0,
    }
    # Override baseline with midpoints for swept params
    for p in param_names:
        lo, hi = ranges[p]
        baseline[p] = (lo + hi) / 2.0

    oat_configs: list[dict] = []
    for p in param_names:
        lo, hi = ranges[p]
        for bound, val in [("min", lo), ("max", hi)]:
            cfg = dict(baseline)
            cfg[p] = val
            cfg["_tornado_param"] = p
            cfg["_tornado_bound"] = bound
            oat_configs.append(cfg)

    return baseline, oat_configs


def rank_results(results: list[dict], metric: str = "msp") -> list[dict]:
    """Rank successful results by a given metric (ascending = best first).

    Args:
        metric: One of 'msp', 'gwp', 'tci', 'energy'.

    Returns:
        Sorted list of result dicts. Failed simulations are appended at the
        end (unsorted).
    """
    if metric not in _METRIC_KEYS:
        logger.warning(
            "Unknown metric '%s'; falling back to 'msp'. "
            "Valid metrics: %s", metric, list(_METRIC_KEYS.keys()),
        )
        metric = "msp"

    section, key = _METRIC_KEYS[metric]

    def _sort_value(r: dict) -> float:
        if not r.get("success"):
            return float("inf")
        val = r.get(section, {}).get(key)
        return float(val) if val is not None else float("inf")

    return sorted(results, key=_sort_value)
