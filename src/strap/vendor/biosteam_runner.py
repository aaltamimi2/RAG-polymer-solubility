"""
BioSTEAM Simulation Runner

Manages subprocess lifecycle, parallelism, and result parsing for
BioSTEAM TEA/LCA simulations. Each simulation runs in an isolated
subprocess to avoid global state contamination.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = Path(__file__).parent / "biosteam_worker.py"

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
    """
    cfs: dict[str, float] = {}
    cfs.update(_NATURAL_GAS_CFS)
    cfs.update(_WATER_CFS)
    solvent_ifs = _SOLVENT_LCA_IFS.get(solvent)
    if solvent_ifs:
        cfs.update(solvent_ifs)
    else:
        logger.warning(
            "Solvent '%s' not in _SOLVENT_LCA_IFS — LCA solvent CFs will "
            "default to zero (only offsets applied). Results may be inaccurate.",
            solvent,
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
    return {
        "pe_solvents": list(_PE_SOLVENTS),
        "ldpe_solvents": list(_LDPE_SOLVENTS),
        "evoh_solvents_e1": list(_EVOH_SOLVENTS_E1),
        "evoh_solvents_e2": list(_EVOH_SOLVENTS_E2),
        "pet_solvents": list(_PET_SOLVENTS),
        # New polymers (PS/PP/PVC use PE-proxy; PC is native)
        "ps_solvents": list(_PS_SOLVENTS),
        "pp_solvents": list(_PP_SOLVENTS),
        "pvc_solvents": list(_PVC_SOLVENTS),
        "pc_solvents": list(_PC_SOLVENTS),
        "energy_cases": dict(_ENERGY_CASES),
        "solvent_defaults": dict(_SOLVENT_DEFAULTS),
        "chlorinated_blocklist": list(_CHLORINATED_BLOCKLIST),
        "known_limitations": [
            "Chlorinated solvents (TCE, OCT) fail due to HCl not in property package",
            "Each simulation takes ~10-15 seconds in subprocess",
            "BioSTEAM requires subprocess isolation (global state contamination)",
            "PS, PP, PVC use the PE process model (PE-proxy) — approximate economics/LCA",
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
