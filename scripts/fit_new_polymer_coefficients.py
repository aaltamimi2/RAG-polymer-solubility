#!/usr/bin/env python3
"""Fit ln(S%) = A + B/T_K + C/T_K^2 for ML-generated solubility data of novel polymers.

Pipeline:  ML thermal prediction -> COSMO-RS SLE -> solubility curve -> fit -> store
Output:    data/generated_coefficients.json  (merged on repeated runs)

Usage:
    # Single pair
    python scripts/fit_new_polymer_coefficients.py \
        --polymer "PLA" --psmiles "[*]OC(C)C(=O)[*]" --solvent toluene

    # All solvents for a polymer
    python scripts/fit_new_polymer_coefficients.py \
        --polymer "PLA" --psmiles "[*]OC(C)C(=O)[*]" --all-solvents

    # From existing COSMO-RS output
    python scripts/fit_new_polymer_coefficients.py \
        --from-csv path/to/cosmo_output.csv --polymer PLA --solvent toluene

    # Promote a dynamic entry to validated
    python scripts/fit_new_polymer_coefficients.py \
        --promote --polymer PLA --solvent toluene
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GENERATED_COEFF_PATH = DATA_DIR / "generated_coefficients.json"

# ---------------------------------------------------------------------------
# Model constants (same as fit_solubility_coefficients.py)
# ---------------------------------------------------------------------------
S_MIN = 1e-12
S_MAX = 100.0
R2_THRESHOLD = 0.98


# ---------------------------------------------------------------------------
# Core model (identical to the original script)
# ---------------------------------------------------------------------------
def _model(t_k: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """ln(S%) = A + B/T_K + C/T_K^2"""
    return a + b / t_k + c / t_k**2


def _r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _estimate_solubility_uncertainty(
    thermal_confidence: dict[str, float] | None,
) -> float | None:
    """Rough propagation of thermal-property uncertainty to solubility %.

    Uses a simple heuristic: weighted sum of normalised standard deviations
    for Tm, delta_Hf, and delta_Cp.  Returns None when no confidence info
    is available.
    """
    if thermal_confidence is None:
        return None
    tm_std = thermal_confidence.get("Tm_std_K", 0.0)
    hf_std = thermal_confidence.get("delta_Hf_std", 0.0)
    cp_std = thermal_confidence.get("delta_Cp_std", 0.0)
    # Heuristic weights (empirically tuned to COSMO-RS sensitivity)
    unc = 0.10 * tm_std + 0.002 * hf_std + 0.02 * cp_std
    return round(float(unc), 2) if unc > 0 else None


# ---------------------------------------------------------------------------
# 1.  fit_from_dataframe
# ---------------------------------------------------------------------------
def fit_from_dataframe(
    df: pd.DataFrame,
    polymer: str,
    solvent: str,
    thermal_confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit A, B, C coefficients from a DataFrame of temperature / solubility.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``temperature_c`` and ``solubility_pct``.
    polymer : str
        Polymer name for the output record.
    solvent : str
        Solvent name for the output record.
    thermal_confidence : dict, optional
        Keys ``Tm_std_K``, ``delta_Hf_std``, ``delta_Cp_std`` from the ML
        thermal-property predictor.

    Returns
    -------
    dict
        Coefficient entry following the existing schema plus provenance
        fields (``source``, ``tier``, ``thermal_confidence``,
        ``solubility_uncertainty_pct``, ``generated_at``).
    """
    df = df.copy()
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df["solubility_pct"] = pd.to_numeric(df["solubility_pct"], errors="coerce")
    df.dropna(subset=["temperature_c", "solubility_pct"], inplace=True)
    df.sort_values("temperature_c", inplace=True)

    temps_c_all = df["temperature_c"].values
    sols_all = df["solubility_pct"].values
    n_total = len(df)

    base: dict[str, Any] = {
        "polymer": polymer,
        "solvent": solvent,
        "source": "ml_cosmo",
        "tier": "dynamic",
        "thermal_confidence": thermal_confidence,
        "solubility_uncertainty_pct": _estimate_solubility_uncertainty(thermal_confidence),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Edge: all zero / near-zero ------------------------------------------
    if n_total == 0 or np.all(sols_all < 1e-10):
        return {
            **base,
            "category": "insoluble",
            "A": None, "B": None, "C": None,
            "r_squared": None,
            "n_points": int(n_total),
            "n_dropped_100": 0,
            "t_min_c": float(temps_c_all.min()) if n_total else None,
            "t_max_c": float(temps_c_all.max()) if n_total else None,
        }

    # Edge: all saturated --------------------------------------------------
    if np.all(sols_all >= 99.999):
        return {
            **base,
            "category": "saturated",
            "A": None, "B": None, "C": None,
            "r_squared": None,
            "n_points": int(n_total),
            "n_dropped_100": int(np.sum(sols_all == 100.0)),
            "t_min_c": float(temps_c_all.min()),
            "t_max_c": float(temps_c_all.max()),
        }

    # Drop exact 100.0 points (COSMO-RS NaN artefacts) --------------------
    mask = sols_all != 100.0
    n_dropped = int(np.sum(~mask))
    temps_c = temps_c_all[mask]
    sols = sols_all[mask]

    if len(sols) < 3:
        return {
            **base,
            "category": "saturated",
            "A": None, "B": None, "C": None,
            "r_squared": None,
            "n_points": int(n_total),
            "n_dropped_100": n_dropped,
            "t_min_c": float(temps_c_all.min()),
            "t_max_c": float(temps_c_all.max()),
        }

    # Clamp and fit --------------------------------------------------------
    s_clamped = np.clip(sols, S_MIN, S_MAX)
    ln_s = np.log(s_clamped)
    t_k = temps_c + 273.15

    try:
        popt, _ = curve_fit(_model, t_k, ln_s, p0=[0.0, 0.0, 0.0], maxfev=10_000)
        ln_s_pred = _model(t_k, *popt)
        r2 = _r_squared(ln_s, ln_s_pred)
    except Exception:
        logger.warning(
            "curve_fit failed for %s / %s — marking anomalous", polymer, solvent
        )
        r2 = -1.0
        popt = np.array([0.0, 0.0, 0.0])

    category = "fitted" if r2 >= R2_THRESHOLD else "anomalous"

    return {
        **base,
        "category": category,
        "A": round(float(popt[0]), 4),
        "B": round(float(popt[1]), 4),
        "C": round(float(popt[2]), 4),
        "r_squared": round(float(r2), 6),
        "n_points": int(len(sols)),
        "n_dropped_100": n_dropped,
        "t_min_c": float(temps_c.min()),
        "t_max_c": float(temps_c.max()),
    }


# ---------------------------------------------------------------------------
# 2.  fit_from_cosmo_output
# ---------------------------------------------------------------------------
def fit_from_cosmo_output(
    cosmo_output_path: str | Path,
    polymer: str,
    solvent: str,
    thermal_confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Read a COSMO-RS SLE output file and fit coefficients.

    Accepts CSV (comma- or tab-separated).  The file must contain columns
    that can be mapped to ``temperature_c`` and ``solubility_pct`` (case-
    insensitive partial match on *temperature* and *solubility*).
    """
    path = Path(cosmo_output_path)
    if not path.exists():
        raise FileNotFoundError(f"COSMO-RS output not found: {path}")

    # Try comma first, fall back to tab
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8", sep="\t")

    # Normalise column names ------------------------------------------------
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "temperature" in cl:
            col_map[c] = "temperature_c"
        elif "solubility" in cl:
            col_map[c] = "solubility_pct"
    df.rename(columns=col_map, inplace=True)

    required = {"temperature_c", "solubility_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot locate columns {missing} in {path}. "
            f"Found: {list(df.columns)}"
        )

    entry = fit_from_dataframe(df, polymer, solvent, thermal_confidence)
    entry["source"] = f"cosmo_file:{path.name}"
    return entry


# ---------------------------------------------------------------------------
# 3.  generate_and_fit  (full ML -> COSMO-RS -> fit pipeline)
# ---------------------------------------------------------------------------
def generate_and_fit(
    polymer_name: str,
    polymer_psmiles: str,
    solvent_name: str,
    polymer_cosmo: str | Path | None = None,
    solvent_cosmo: str | Path | None = None,
) -> dict[str, Any]:
    """Full pipeline: predict thermals -> COSMO-RS SLE -> fit coefficients.

    Requires ``strap.thermal_ml`` and ``strap.cosmo_interface`` to be
    importable (they are part of the STRAP v7 package).
    """
    # Late imports so the rest of the module works without these deps
    from strap.thermal_ml import predict_thermal_properties  # type: ignore[import-untyped]
    from strap.cosmo_interface import run_sle_calculation  # type: ignore[import-untyped]

    # Step 1 — ML thermal prediction ----------------------------------------
    logger.info("Predicting thermal properties for %s ...", polymer_name)
    thermal = predict_thermal_properties(polymer_psmiles)
    thermal_confidence: dict[str, float] | None = None
    if hasattr(thermal, "confidence") and thermal.confidence is not None:
        thermal_confidence = {
            "Tm_std_K": thermal.confidence.get("Tm_std_K", 0.0),
            "delta_Hf_std": thermal.confidence.get("delta_Hf_std", 0.0),
            "delta_Cp_std": thermal.confidence.get("delta_Cp_std", 0.0),
        }
    elif isinstance(thermal, dict) and "confidence" in thermal:
        thermal_confidence = {
            "Tm_std_K": thermal["confidence"].get("Tm_std_K", 0.0),
            "delta_Hf_std": thermal["confidence"].get("delta_Hf_std", 0.0),
            "delta_Cp_std": thermal["confidence"].get("delta_Cp_std", 0.0),
        }

    # Step 2 — COSMO-RS SLE calculation ------------------------------------
    logger.info("Running COSMO-RS SLE for %s / %s ...", polymer_name, solvent_name)
    sle_result = run_sle_calculation(
        polymer_name=polymer_name,
        polymer_psmiles=polymer_psmiles,
        solvent_name=solvent_name,
        polymer_cosmo=polymer_cosmo,
        solvent_cosmo=solvent_cosmo,
        thermal_properties=thermal,
    )

    # Expect sle_result to be a DataFrame (or have a .dataframe attr)
    if isinstance(sle_result, pd.DataFrame):
        df = sle_result
    elif hasattr(sle_result, "dataframe"):
        df = sle_result.dataframe
    elif hasattr(sle_result, "to_dataframe"):
        df = sle_result.to_dataframe()
    else:
        raise TypeError(
            f"Unexpected SLE result type {type(sle_result)}; "
            "expected DataFrame or object with .dataframe attribute"
        )

    # Step 3 — fit ----------------------------------------------------------
    entry = fit_from_dataframe(df, polymer_name, solvent_name, thermal_confidence)
    entry["source"] = "ml_cosmo"
    return entry


# ---------------------------------------------------------------------------
# 4.  batch_generate
# ---------------------------------------------------------------------------
# Default solvent list — mirrors the solvents in COMMON-SOLVENTS-DATABASE.csv
_DEFAULT_SOLVENTS: list[str] = [
    "toluene",
    "xylene",
    "chloroform",
    "THF",
    "dichloromethane",
    "acetone",
    "ethanol",
    "methanol",
    "water",
    "DMF",
    "DMSO",
    "hexane",
]


def batch_generate(
    polymer_name: str,
    polymer_psmiles: str,
    solvent_names: list[str] | None = None,
    polymer_cosmo: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run :func:`generate_and_fit` for multiple solvents.

    Parameters
    ----------
    polymer_name : str
    polymer_psmiles : str
    solvent_names : list[str], optional
        Solvents to test.  Defaults to :data:`_DEFAULT_SOLVENTS`.
    polymer_cosmo : path, optional
        Path to pre-computed COSMO file for the polymer.

    Returns
    -------
    list[dict]
        One coefficient entry per solvent.
    """
    solvents = solvent_names if solvent_names is not None else _DEFAULT_SOLVENTS
    entries: list[dict[str, Any]] = []

    for solvent in solvents:
        logger.info("--- %s / %s ---", polymer_name, solvent)
        try:
            entry = generate_and_fit(
                polymer_name=polymer_name,
                polymer_psmiles=polymer_psmiles,
                solvent_name=solvent,
                polymer_cosmo=polymer_cosmo,
            )
            entries.append(entry)
            logger.info(
                "  -> %s  R²=%s",
                entry["category"],
                entry.get("r_squared"),
            )
        except Exception:
            logger.exception("  FAILED for %s / %s", polymer_name, solvent)
    return entries


# ---------------------------------------------------------------------------
# 5.  save_generated_coefficients
# ---------------------------------------------------------------------------
def save_generated_coefficients(
    entries: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> Path:
    """Save (or merge) coefficient entries to the generated-coefficients JSON.

    If the file already exists the new entries are merged: existing records
    with the same ``(polymer, solvent)`` key are *replaced*; truly new
    records are appended.

    Returns the path that was written.
    """
    out = Path(output_path) if output_path else GENERATED_COEFF_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    existing_entries: list[dict[str, Any]] = []
    if out.exists():
        with open(out) as f:
            data = json.load(f)
        existing_entries = data.get("entries", [])

    # Build lookup by (polymer, solvent)
    lookup: dict[tuple[str, str], dict[str, Any]] = {
        (e["polymer"], e["solvent"]): e for e in existing_entries
    }
    for e in entries:
        lookup[(e["polymer"], e["solvent"])] = e

    merged = list(lookup.values())

    # Category counts
    categories: dict[str, int] = {}
    for e in merged:
        cat = e.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    output = {
        "description": (
            "ML-generated solubility interpolation coefficients: "
            "ln(S%) = A + B/T_K + C/T_K^2"
        ),
        "note": (
            "Generated via ML thermal prediction + COSMO-RS SLE. "
            "Exact S=100% points excluded before fitting."
        ),
        "source": "ml_cosmo pipeline",
        "model_path": str(out.relative_to(out.parent.parent)),
        "inference_module": "src/strap/tools/interpolation.py",
        "n_entries": len(merged),
        "categories": categories,
        "entries": merged,
    }

    with open(out, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Wrote %d entries to %s", len(merged), out)
    return out


# ---------------------------------------------------------------------------
# 6.  promote_to_validated
# ---------------------------------------------------------------------------
def promote_to_validated(
    polymer: str,
    solvent: str,
    output_path: str | Path | None = None,
) -> bool:
    """Change the tier of a (polymer, solvent) entry from 'dynamic' to 'validated'.

    Returns True if the entry was found and updated, False otherwise.
    """
    out = Path(output_path) if output_path else GENERATED_COEFF_PATH
    if not out.exists():
        logger.warning("Coefficients file does not exist: %s", out)
        return False

    with open(out) as f:
        data = json.load(f)

    updated = False
    for entry in data.get("entries", []):
        if entry["polymer"] == polymer and entry["solvent"] == solvent:
            if entry.get("tier") == "dynamic":
                entry["tier"] = "validated"
                entry["validated_at"] = datetime.now(timezone.utc).isoformat()
                updated = True
                logger.info("Promoted %s / %s to validated", polymer, solvent)
            else:
                logger.info(
                    "%s / %s already tier=%s", polymer, solvent, entry.get("tier")
                )
                updated = True  # entry exists, nothing to do
            break

    if not updated:
        logger.warning("No entry found for %s / %s", polymer, solvent)
        return False

    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit A, B, C solubility coefficients for new polymers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--polymer", required=False, help="Polymer name")
    p.add_argument("--psmiles", required=False, help="Polymer SMILES (pSMILES)")
    p.add_argument("--solvent", required=False, help="Single solvent name")
    p.add_argument(
        "--all-solvents",
        action="store_true",
        help="Fit against all default solvents",
    )
    p.add_argument(
        "--from-csv",
        metavar="PATH",
        help="Path to an existing COSMO-RS output CSV/TSV file",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Output JSON path (default: data/generated_coefficients.json)",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="Promote a (polymer, solvent) pair from dynamic to validated",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --promote mode --------------------------------------------------------
    if args.promote:
        if not args.polymer or not args.solvent:
            parser.error("--promote requires --polymer and --solvent")
        ok = promote_to_validated(args.polymer, args.solvent, args.output)
        return 0 if ok else 1

    # --from-csv mode -------------------------------------------------------
    if args.from_csv:
        if not args.polymer or not args.solvent:
            parser.error("--from-csv requires --polymer and --solvent")
        entry = fit_from_cosmo_output(args.from_csv, args.polymer, args.solvent)
        save_generated_coefficients([entry], args.output)
        _print_entry(entry)
        return 0

    # generate_and_fit / batch_generate mode --------------------------------
    if not args.polymer or not args.psmiles:
        parser.error("--polymer and --psmiles are required (unless using --from-csv)")

    if args.all_solvents:
        entries = batch_generate(args.polymer, args.psmiles)
    elif args.solvent:
        entry = generate_and_fit(args.polymer, args.psmiles, args.solvent)
        entries = [entry]
    else:
        parser.error("Provide --solvent or --all-solvents")
        return 1  # unreachable

    save_generated_coefficients(entries, args.output)
    for e in entries:
        _print_entry(e)

    return 0


def _print_entry(entry: dict[str, Any]) -> None:
    """Pretty-print a single coefficient entry to stdout."""
    print(
        f"  {entry['polymer']:>20s} / {entry['solvent']:<20s}  "
        f"cat={entry['category']:<10s}  "
        f"R²={entry.get('r_squared') or 'N/A':>10}  "
        f"tier={entry.get('tier', '?')}"
    )
    if entry.get("A") is not None:
        print(
            f"{'':>44s}A={entry['A']:.4f}  B={entry['B']:.4f}  C={entry['C']:.4f}"
        )


if __name__ == "__main__":
    sys.exit(main())
