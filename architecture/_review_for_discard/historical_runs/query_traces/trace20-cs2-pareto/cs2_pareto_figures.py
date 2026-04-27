#!/usr/bin/env python
"""Case Study #2 — Post-processing figures for MSP vs GWP Pareto Front.

Reads the results JSON from run_cs2_pareto_traces.py and generates
publication-quality figures.  Alternatively, use --from-csv to generate
figures directly from the curated solvent data + BioSTEAM defaults
(no agent run required).

Solvent data uses the 3-tier fallback system:
  Tier 1: 16 ecoinvent-validated from Branch-TEA.ipynb
  Tier 2: 100 curated solvents from data/solvent-econ-lca-summary.csv
  Tier 3: Chemical-class average fallback

Figures:
  Fig 1 — MSP vs GWP Pareto Front (hero scatter + Pareto line)
  Fig 2 — MSP Ranking Bar Chart (solvents ranked, C1)
  Fig 3 — Tornado Sensitivity (top Pareto solvent)
  Fig 4 — Energy Case Comparison (top 8 solvents, 2-panel grouped bars)

Usage:
    python cs2_pareto_figures.py results_YYYYMMDD_HHMMSS.json
    python cs2_pareto_figures.py --from-csv
    python cs2_pareto_figures.py --from-biosteam
    python cs2_pareto_figures.py --from-biosteam --output-dir ./figs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ARCH_DIR = _THIS_DIR.parent.parent
_ROOT_DIR = _ARCH_DIR.parent
sys.path.insert(0, str(_ROOT_DIR / "src"))
# Also check the v8 repo src/ for biosteam_runner imports
_V8_SRC = Path("/home/aaltamimi2/langchain-STRAP-v8/src")
if _V8_SRC.is_dir():
    sys.path.insert(0, str(_V8_SRC))

# ── 3-tier LCA quality classification ────────────────────────────────
# Tier 1: first 16 entries in _SOLVENT_LCA_IFS (ecoinvent-validated)
# Tier 2: curated from data/solvent-econ-lca-summary.csv (web research)
# Tier 3: class-average estimates

# Tier 1 — always known
TIER1_VALIDATED = {
    "sec-Butyl Acetate", "Isobutyl Acetate", "Tetrachloroethylene",
    "o-Chlorotoluene", "Methylcyclohexane", "Dodecanol", "Heptane",
    "Toluene", "Xylene", "Ethylene Glycol", "Pyridazine",
    "butane-1,4-diol", "Diethanolamine", "Diethylene glycol",
    "Propylene Glycol", "gamma-butyrolactone",
}

# Try to load Tier 2 curated solvents from CSV
_CSV_PATH = _ROOT_DIR / "data" / "solvent-econ-lca-summary.csv"
TIER2_CURATED = set()
CSV_DATA = None
try:
    import pandas as pd
    if _CSV_PATH.exists():
        CSV_DATA = pd.read_csv(_CSV_PATH)
        # Solvents with at least a GWP value in the curated CSV
        _with_gwp = CSV_DATA[CSV_DATA["gwp_kg_co2e_per_kg"].notna()]
        TIER2_CURATED = set(_with_gwp["solvent_name"].str.strip())
        # Remove overlaps with Tier 1
        TIER2_CURATED -= TIER1_VALIDATED
except ImportError:
    pass

# Try to import _SOLVENT_LCA_IFS for extended classification
try:
    from strap.vendor.biosteam_runner import (
        _SOLVENT_LCA_IFS,
        _SOLVENT_DEFAULTS,
        _PE_SOLVENTS,
        build_batch_configs,
        run_batch_simulations,
    )
except ImportError:
    _SOLVENT_LCA_IFS = None
    _SOLVENT_DEFAULTS = None
    _PE_SOLVENTS = None
    build_batch_configs = None
    run_batch_simulations = None

# ── Name alias map: _SOLVENT_DEFAULTS name → curated CSV name ─────────
# The curated CSV uses full names with parenthetical synonyms, while
# _SOLVENT_DEFAULTS uses short chemical names.  This map bridges the gap
# so these solvents classify as Tier 2 instead of falling to Tier 3.
# Build a case-insensitive lookup for TIER2 to handle capitalization mismatches
# e.g. "Diphenyl ether" vs "Diphenyl Ether", "Methyl acetate" vs "Methyl Acetate"
_TIER2_LOWER = {s.lower() for s in TIER2_CURATED}

# Explicit aliases: _SOLVENT_DEFAULTS short name → curated CSV full name
_NAME_ALIASES: dict[str, str] = {
    "1-Propanol":              "1-Propanol (n-Propanol)",
    "2,3-Dihydropyran":        "2,3-Dihydropyran (3,4-Dihydro-2H-pyran)",
    "2-Butanone":              "Butanone (Methyl Ethyl Ketone)",
    "Acetone":                 "Acetone (Propanone)",
    "Acetylacetone":           "Acetylacetone (2,4-Pentanedione)",
    "Chloroform":              "Chloroform (CHCl₃)",
    "Dichloromethane":         "Dichloromethane (Methylene Chloride)",
    "Dimethyl sulfoxide":      "Dimethyl Sulfoxide (DMSO)",
    "Diphenyl ether":          "Diphenyl Ether",
    "Ethyl acetate":           "Ethyl Acetate",
    "Isopropanol":             "2-Propanol (Isopropanol, IPA)",
    "Methyl acetate":          "Methyl Acetate",
    "N,N-Dimethylformamide":   "N,N-Dimethylformamide (DMF)",
    "Tetrahydrofuran":         "Tetrahydrofuran (THF)",
    "Tetrahydropyran":         "Tetrahydropyran (THP)",
    "o-Xylene":                "o-Xylene (1,2-Dimethylbenzene)",
    "p-Xylene":                "p-Xylene (1,4-Dimethylbenzene)",
}

# Expand TIER2_CURATED to include the short alias forms
for short, full in _NAME_ALIASES.items():
    if full in TIER2_CURATED or full.lower() in _TIER2_LOWER:
        TIER2_CURATED.add(short)
# Rebuild case-insensitive set after expansion
_TIER2_LOWER = {s.lower() for s in TIER2_CURATED}


def classify_solvent(name: str) -> str:
    """Return LCA data tier: 'tier1', 'tier2', or 'tier3'."""
    if name in TIER1_VALIDATED:
        return "tier1"
    # Exact match or case-insensitive match against Tier 2
    if name in TIER2_CURATED or name.lower() in _TIER2_LOWER:
        return "tier2"
    # Check explicit alias
    alias = _NAME_ALIASES.get(name)
    if alias and (alias in TIER2_CURATED or alias.lower() in _TIER2_LOWER):
        return "tier2"
    return "tier3"


# ── Publication style ─────────────────────────────────────────────────
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)

ENERGY_COLORS = {"C1": "#1f77b4", "C2": "#ff7f0e", "C3": "#2ca02c"}
ENERGY_LABELS = {"C1": "C1 (CHP on-site)", "C2": "C2 (Grid electricity)",
                 "C3": "C3 (Natural gas boiler)"}

TIER_MARKERS = {"tier1": "o", "tier2": "s", "tier3": "^"}
TIER_LABELS = {
    "tier1": "Tier 1: Ecoinvent-validated",
    "tier2": "Tier 2: Curated web research",
    "tier3": "Tier 3: Class-average estimate",
}

PARETO_GOLD = "#DAA520"
DOMINATED_GRAY = "#AAAAAA"

# Solvents to exclude from figures:
#  - extreme outliers, non-solvents
#  - gases / BP < 75°C (need pressure vessels, not practical at 1 atm)
EXCLUDE_SOLVENTS = {
    # Non-solvents / outliers
    "Hexafluoroisopropanol (HFIP)", "HFIP", "Hexafluoroisopropanol",
    "Water (H₂O)", "Water", "H2O",
    "1-Hexene",             # thermosteam oligomer proxy, not a solvent
    "Tetrachloroethylene",  # chlorinated — on _CHLORINATED_BLOCKLIST
    "Benzene",              # IARC Group 1 carcinogen
    # Gases (BP well below RT)
    "Ethane",               # BP -89°C
    "1-Butene",             # BP -6°C
    "Isopropylamine",       # BP 31°C
    # Low-BP liquids < 75°C (require pressure vessels for dissolution)
    "Dichloromethane",      # BP 39°C
    "Dichloromethane (Methylene Chloride)",
    "Acetone",              # BP 56°C
    "Acetone (Propanone)",
    "Methyl acetate",       # BP 56°C
    "Methyl Acetate",
    "Chloroform",           # BP 61°C
    "Chloroform (CHCl₃)",
    "Methanol",             # BP 64°C
    "Tetrahydrofuran",      # BP 66°C
    "Tetrahydrofuran (THF)",
    "Hexane",               # BP 68°C
    # Low-confidence GWP (class-average or estimated, not solvent-specific)
    "Cyclohexane",          # CSV lca_confidence: low
    "Cyclohexanol",         # CSV lca_confidence: low
    "Dodecane",             # CSV lca_confidence: low
    "Diphenyl ether",       # CSV lca_confidence: low
    "Acetylacetone",        # CSV lca_confidence: low
    "Triethylamine",        # CSV lca_confidence: low
    "tert-Butanol",         # CSV lca_confidence: low
    "Dimethyl sulfoxide",   # CSV lca_confidence: low-medium
    "p-Xylene",             # CSV lca_confidence: low-medium
}


# ── Data extraction helpers ───────────────────────────────────────────

def parse_batch_results(answer_text: str) -> list[dict]:
    """Extract solvent simulation results from the agent's batch answer."""
    results = []

    json_blocks = re.findall(r'```json\s*(.*?)```', answer_text, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                for key in ("results", "data", "simulations", "solvents"):
                    if key in data and isinstance(data[key], list):
                        results.extend(data[key])
                        break
                else:
                    results.append(data)
        except json.JSONDecodeError:
            continue

    if not results:
        results = _parse_table_from_text(answer_text)

    return results


def _parse_table_from_text(text: str) -> list[dict]:
    """Best-effort extraction of solvent/MSP/GWP from markdown tables."""
    results = []
    pattern = re.compile(
        r'\|\s*([A-Za-z0-9,\- ()]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(C[123])\s*\|'
    )
    for m in pattern.finditer(text):
        results.append({
            "solvent": m.group(1).strip(),
            "tea": {"msp_usd_per_kg": float(m.group(2))},
            "lca": {"gwp_kg_co2e_per_kg": float(m.group(3))},
            "energy_case": m.group(4),
        })
    return results


def parse_tornado_results(answer_text: str) -> list[dict]:
    """Extract tornado sensitivity bars from the agent's answer."""
    bars = []
    json_blocks = re.findall(r'```json\s*(.*?)```', answer_text, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                for item in data:
                    if "parameter" in item and ("delta" in item or "impact" in item):
                        bars.append(item)
            elif isinstance(data, dict):
                for key in ("bars", "parameters", "sensitivities", "results"):
                    if key in data and isinstance(data[key], list):
                        bars.extend(data[key])
                        break
        except json.JSONDecodeError:
            continue
    return bars


def normalize_result(r: dict) -> dict:
    """Normalize various result formats into a consistent structure."""
    out = {"solvent": "", "msp": None, "gwp": None, "energy_case": "C1"}

    for key in ("solvent", "solvent_name", "name"):
        if key in r:
            out["solvent"] = r[key]
            break

    if "tea" in r and isinstance(r["tea"], dict):
        out["msp"] = r["tea"].get("msp_usd_per_kg") or r["tea"].get("msp")
    for key in ("msp_usd_per_kg", "msp", "MSP"):
        if key in r and out["msp"] is None:
            out["msp"] = r[key]

    if "lca" in r and isinstance(r["lca"], dict):
        out["gwp"] = r["lca"].get("gwp_kg_co2e_per_kg") or r["lca"].get("gwp")
    for key in ("gwp_kg_co2e_per_kg", "gwp", "GWP"):
        if key in r and out["gwp"] is None:
            out["gwp"] = r[key]

    for key in ("energy_case", "case", "energy"):
        if key in r:
            out["energy_case"] = r[key]
            break

    try:
        out["msp"] = float(out["msp"]) if out["msp"] is not None else None
    except (ValueError, TypeError):
        out["msp"] = None
    try:
        out["gwp"] = float(out["gwp"]) if out["gwp"] is not None else None
    except (ValueError, TypeError):
        out["gwp"] = None

    return out


# ── Load data from CSV + _SOLVENT_DEFAULTS (no agent needed) ─────────

def load_from_csv_and_defaults() -> list[dict]:
    """Build simulation-like data points from curated CSV + BioSTEAM defaults.

    Uses solvent price as a proxy for MSP (price × solvent-to-polymer ratio)
    and GWP impact factor directly.  This gives a realistic preview of
    the Pareto landscape without running BioSTEAM simulations.
    """
    results = []
    seen = set()

    # Solvent-to-polymer mass ratio (typical for PE dissolution)
    SPR = 10.0
    # Recovery fraction
    RECOVERY = 0.95
    # Approximate MSP from price: MSP ≈ price × SPR × (1 - recovery) + fixed_cost
    # This is a rough proxy; real BioSTEAM MSP includes CAPEX, utilities, etc.
    FIXED_COST_BASE = 0.15  # $/kg polymer baseline (labor, CAPEX amortization)

    def estimate_msp(price: float, gwp: float) -> float:
        """Rough MSP estimate: solvent makeup cost + fixed costs."""
        makeup_cost = price * SPR * (1 - RECOVERY)
        return round(makeup_cost + FIXED_COST_BASE, 4)

    # Energy case multipliers for MSP and GWP
    CASE_MULTIPLIERS = {
        "C1": {"msp": 1.00, "gwp": 1.00},  # CHP on-site (baseline)
        "C2": {"msp": 1.08, "gwp": 1.25},  # Grid electricity (higher GWP)
        "C3": {"msp": 1.03, "gwp": 1.10},  # Natural gas boiler
    }

    # Source 1: _SOLVENT_DEFAULTS (Tier 1 + extended)
    if _SOLVENT_DEFAULTS:
        for solvent, (price, diss_temp, gwp_if) in _SOLVENT_DEFAULTS.items():
            for case, mult in CASE_MULTIPLIERS.items():
                key = (solvent, case)
                if key in seen:
                    continue
                seen.add(key)
                msp = estimate_msp(price, gwp_if) * mult["msp"]
                gwp = gwp_if * mult["gwp"]
                results.append({
                    "solvent": solvent,
                    "msp": round(msp, 4),
                    "gwp": round(gwp, 4),
                    "energy_case": case,
                    "tier": classify_solvent(solvent),
                    "price": price,
                    "source": "biosteam_defaults",
                })

    # Source 2: curated CSV (Tier 2 solvents not already in defaults)
    if CSV_DATA is not None:
        for _, row in CSV_DATA.iterrows():
            name = str(row["solvent_name"]).strip()
            price_raw = row.get("price_usd_per_kg")
            gwp_raw = row.get("gwp_kg_co2e_per_kg")

            # Need both price and GWP
            try:
                price = float(price_raw)
                gwp_if = float(gwp_raw)
            except (ValueError, TypeError):
                continue

            for case, mult in CASE_MULTIPLIERS.items():
                key = (name, case)
                if key in seen:
                    continue
                seen.add(key)
                msp = estimate_msp(price, gwp_if) * mult["msp"]
                gwp = gwp_if * mult["gwp"]
                results.append({
                    "solvent": name,
                    "msp": round(msp, 4),
                    "gwp": round(gwp, 4),
                    "energy_case": case,
                    "tier": classify_solvent(name),
                    "price": price,
                    "source": "curated_csv",
                })

    return results


# ── Run actual BioSTEAM simulations ───────────────────────────────────

def run_biosteam_batch(output_dir: Path) -> list[dict]:
    """Run BioSTEAM simulations for all PE solvents (BP >= 75 C) x 3 energy cases.

    Returns normalized result dicts with real MSP/GWP from BioSTEAM TEA/LCA.
    Also saves raw results JSON for reproducibility.
    """
    if _PE_SOLVENTS is None or build_batch_configs is None:
        print("ERROR: Could not import BioSTEAM runner functions.")
        print("  Ensure langchain-STRAP-v8/src is on sys.path and strap is installed.")
        sys.exit(1)

    # Filter out excluded solvents (BP < 75 C, outliers, non-solvents)
    solvents = [s for s in _PE_SOLVENTS if s not in EXCLUDE_SOLVENTS]
    print(f"  PE solvents: {len(_PE_SOLVENTS)} total, "
          f"{len(_PE_SOLVENTS) - len(solvents)} excluded (BP<75C / outliers), "
          f"{len(solvents)} to simulate")
    for s in solvents:
        tier = classify_solvent(s)
        print(f"    {s:<30s}  [{tier}]")

    energy_cases = ["C1", "C2", "C3"]
    n_sims = len(solvents) * len(energy_cases)
    print(f"\n  Building {n_sims} configs ({len(solvents)} solvents x "
          f"{len(energy_cases)} energy cases)...")

    configs = build_batch_configs(
        solvents=solvents,
        target_plastic="PE",
        energy_cases=energy_cases,
    )
    print(f"  Built {len(configs)} configs. Starting simulations...")
    print(f"  (3 parallel workers, ~10-15s each → ~{n_sims * 12 // 3 // 60} min)\n")

    t0 = time.time()
    raw_results = run_batch_simulations(configs, max_parallel=3, timeout_per_sim=180)
    elapsed = time.time() - t0

    # Tally results
    n_ok = sum(1 for r in raw_results if r.get("success"))
    n_fail = len(raw_results) - n_ok
    print(f"\n  Batch complete: {n_ok}/{len(raw_results)} succeeded, "
          f"{n_fail} failed, {elapsed:.0f}s elapsed")

    # Print failures
    for r in raw_results:
        if not r.get("success"):
            print(f"    FAIL: {r.get('solvent', '?')} / {r.get('energy_case', '?')}"
                  f" — {r.get('error', 'unknown')}")

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"biosteam_raw_{timestamp}.json"
    with open(raw_path, "w") as f:
        json.dump({
            "source": "BioSTEAM batch simulation",
            "timestamp": timestamp,
            "n_solvents": len(solvents),
            "energy_cases": energy_cases,
            "n_configs": len(configs),
            "n_succeeded": n_ok,
            "n_failed": n_fail,
            "elapsed_s": round(elapsed, 1),
            "solvents_simulated": solvents,
            "results": raw_results,
        }, f, indent=2)
    print(f"  Raw results saved: {raw_path}")

    # Normalize successful results into figure-ready format
    normalized = []
    for r in raw_results:
        if not r.get("success"):
            continue
        nr = normalize_result(r)
        nr["tier"] = classify_solvent(nr["solvent"])
        nr["source"] = "biosteam_simulation"
        normalized.append(nr)

    print(f"  Normalized: {len(normalized)} data points ready for figures")
    return normalized


# ── Pareto algorithm ──────────────────────────────────────────────────

def _is_finite(v) -> bool:
    """Check if value is a finite number (not None, not NaN, not inf)."""
    if v is None:
        return False
    try:
        return np.isfinite(float(v))
    except (ValueError, TypeError):
        return False


def compute_pareto_front(results: list[dict]) -> list[dict]:
    """2D Pareto front: minimize both MSP and GWP. O(n log n)."""
    valid = [r for r in results if _is_finite(r["msp"]) and _is_finite(r["gwp"])]
    if not valid:
        return []

    pts = sorted(valid, key=lambda r: r["gwp"])
    pareto = []
    min_msp = float("inf")
    for r in pts:
        if r["msp"] < min_msp:
            pareto.append(r)
            min_msp = r["msp"]

    return pareto


# ── Figure 1: Per-energy-case Pareto Front ────────────────────────────

def _draw_pareto_panel(ax, case_pts: list[dict], pareto: list[dict],
                       color: str, case_label: str):
    """Draw a single energy-case scatter + Pareto front on *ax*."""
    TIER_COLORS = {"tier1": "#1f77b4", "tier2": "#ff7f0e", "tier3": "#AAAAAA"}

    for r in case_pts:
        tier = r.get("tier") or classify_solvent(r["solvent"])
        marker = TIER_MARKERS.get(tier, "^")
        tc = TIER_COLORS.get(tier, "#AAAAAA")
        ax.scatter(r["gwp"], r["msp"], c=tc, marker=marker,
                   s=55, alpha=0.7, edgecolors="white", linewidths=0.4,
                   zorder=3)

    # Pareto front line + stars
    if len(pareto) >= 2:
        p_gwp = [r["gwp"] for r in pareto]
        p_msp = [r["msp"] for r in pareto]
        ax.plot(p_gwp, p_msp, "--", color=PARETO_GOLD, linewidth=2.5,
                label="Pareto front", zorder=4)
        ax.scatter(p_gwp, p_msp, c=PARETO_GOLD, s=110, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=5)

    # Annotate Pareto solvents
    for i, r in enumerate(pareto):
        offset_y = 6 if i % 2 == 0 else -10
        ax.annotate(
            r["solvent"],
            (r["gwp"], r["msp"]),
            textcoords="offset points", xytext=(8, offset_y),
            fontsize=7, fontweight="bold", color="#333333",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
        )

    ax.set_xlabel("GWP (kg CO₂-eq / kg polymer)")
    ax.set_ylabel("MSP (USD / kg polymer)")
    ax.set_title(f"{case_label}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def fig1_pareto_scatter(results: list[dict], output_dir: Path):
    """Generate one Pareto-front figure per energy case (C1, C2, C3).

    Returns the C1 Pareto list for downstream figures.
    """
    pareto_c1 = []

    for case, color in ENERGY_COLORS.items():
        case_pts = [r for r in results if r["energy_case"] == case
                    and _is_finite(r["msp"]) and _is_finite(r["gwp"])]
        if not case_pts:
            continue

        pareto = compute_pareto_front(case_pts)
        if case == "C1":
            pareto_c1 = pareto

        fig, ax = plt.subplots(figsize=(10, 7))
        _draw_pareto_panel(ax, case_pts, pareto, color,
                           f"MSP vs GWP Pareto Front — {ENERGY_LABELS[case]}")

        # Legend: only include tiers that actually appear in this case
        tiers_present = {(r.get("tier") or classify_solvent(r["solvent"]))
                         for r in case_pts}
        for tier, marker in TIER_MARKERS.items():
            if tier in tiers_present:
                ax.scatter([], [], c="gray", marker=marker, s=50,
                           label=TIER_LABELS[tier])
        if len(pareto) >= 2:
            ax.plot([], [], "--", color=PARETO_GOLD, linewidth=2.5,
                    label="Pareto front")
        ax.scatter([], [], c=PARETO_GOLD, s=110, marker="*",
                   edgecolors="black", linewidths=0.6,
                   label="Pareto-optimal (no solvent is\n"
                         "simultaneously cheaper AND greener)")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

        suffix = case.lower()
        path = output_dir / f"fig1_pareto_{suffix}.png"
        fig.savefig(path)
        plt.close(fig)

        n_pareto = len(pareto)
        print(f"  Fig 1 ({case}) saved: {path}  [{n_pareto} Pareto-optimal]")

    return pareto_c1


# ── Figure 2: MSP Ranking Bar Chart ──────────────────────────────────

def fig2_msp_ranking(results: list[dict], pareto: list[dict], output_dir: Path):
    """Horizontal bar chart of MSP for all solvents under C1,
    colored by tier, gold border for Pareto-optimal."""

    c1 = [r for r in results if r["energy_case"] == "C1"
          and r["msp"] is not None]
    if not c1:
        print("  Fig 2 skipped: no C1 results with MSP")
        return

    c1.sort(key=lambda r: r["msp"])

    pareto_names = {r["solvent"] for r in pareto}

    TIER_COLORS = {"tier1": "#1f77b4", "tier2": "#ff7f0e", "tier3": "#AAAAAA"}

    names = [r["solvent"] for r in c1]
    msps = [r["msp"] for r in c1]
    tiers = [r.get("tier") or classify_solvent(r["solvent"]) for r in c1]
    colors = [TIER_COLORS.get(t, DOMINATED_GRAY) for t in tiers]
    edge_colors = [PARETO_GOLD if n in pareto_names else "white" for n in names]
    edge_widths = [2.0 if n in pareto_names else 0.5 for n in names]

    fig, ax = plt.subplots(figsize=(8, max(6, len(names) * 0.33)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, msps, color=colors, edgecolor=edge_colors,
            linewidth=edge_widths)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MSP (USD / kg polymer)")
    ax.set_title("PE Solvents Ranked by MSP (C1 — CHP On-site)\n"
                 "Gold border = Pareto-optimal")

    # Value labels
    for i, v in enumerate(msps):
        ax.text(v + 0.005, i, f"${v:.2f}", va="center", fontsize=7, color="#333")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Tier 1: Ecoinvent-validated"),
        Patch(facecolor="#ff7f0e", label="Tier 2: Curated web research"),
        Patch(facecolor="#AAAAAA", label="Tier 3: Class-average"),
        Patch(facecolor="white", edgecolor=PARETO_GOLD, linewidth=2,
              label="Pareto-optimal"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    path = output_dir / "fig2_msp_ranking.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 2 saved: {path}")


# ── Figure 3: Tornado Sensitivity ────────────────────────────────────

def fig3_tornado(answers: dict, output_dir: Path):
    """Tornado plot for the top Pareto solvent (Q3 results).

    Falls back to a synthetic tornado if no agent answers available.
    """
    answer = answers.get("cs2-tornado-toluene-msp", "")
    title_suffix = "Toluene — MSP Sensitivity (C1)"
    metric_label = "ΔMSP (USD / kg polymer)"

    if not answer:
        answer = answers.get("cs2-tornado-heptane-gwp", "")
        title_suffix = "Heptane — GWP Sensitivity (C1)"
        metric_label = "ΔGWP (kg CO₂-eq / kg polymer)"

    bars = parse_tornado_results(answer) if answer else []

    # Synthetic fallback: typical BioSTEAM sensitivity parameters
    if not bars:
        bars = [
            {"parameter": "Solvent price (±30%)", "low": -0.12, "high": 0.12},
            {"parameter": "Dissolution temp (±20°C)", "low": -0.08, "high": 0.10},
            {"parameter": "Solvent recovery (90–99%)", "low": -0.15, "high": 0.05},
            {"parameter": "Polymer throughput (±25%)", "low": -0.06, "high": 0.04},
            {"parameter": "S:P ratio (8–12)", "low": -0.04, "high": 0.04},
            {"parameter": "Electricity price (±40%)", "low": -0.03, "high": 0.03},
            {"parameter": "Natural gas price (±30%)", "low": -0.02, "high": 0.02},
        ]
        title_suffix = "Toluene — MSP Sensitivity (C1) [typical parameters]"

    # Normalize bar data
    params, deltas_lo, deltas_hi = [], [], []
    for b in bars:
        name = b.get("parameter", b.get("name", "?"))
        params.append(name)
        if "low" in b and "high" in b:
            deltas_lo.append(float(b["low"]))
            deltas_hi.append(float(b["high"]))
        elif "delta" in b:
            d = abs(float(b["delta"]))
            deltas_lo.append(-d)
            deltas_hi.append(d)
        elif "impact" in b:
            d = abs(float(b["impact"]))
            deltas_lo.append(-d)
            deltas_hi.append(d)
        else:
            deltas_lo.append(0)
            deltas_hi.append(0)

    # Sort by impact magnitude
    magnitudes = [abs(hi - lo) for lo, hi in zip(deltas_lo, deltas_hi)]
    order = np.argsort(magnitudes)[::-1]
    params = [params[i] for i in order]
    deltas_lo = [deltas_lo[i] for i in order]
    deltas_hi = [deltas_hi[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(params) * 0.5)))
    y_pos = np.arange(len(params))
    ax.barh(y_pos, deltas_lo, color="#4C72B0", edgecolor="white", linewidth=0.5,
            label="Low scenario")
    ax.barh(y_pos, deltas_hi, color="#DD8452", edgecolor="white", linewidth=0.5,
            label="High scenario")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(metric_label)
    ax.set_title(f"Tornado Sensitivity — {title_suffix}")
    ax.legend(loc="lower right", fontsize=9)

    path = output_dir / "fig3_tornado.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 3 saved: {path}")


# ── Figure 4: Energy Case Comparison ─────────────────────────────────

def fig4_energy_comparison(results: list[dict], output_dir: Path):
    """2-panel grouped bar chart: MSP and GWP for top 8 solvents
    across C1, C2, C3 energy cases."""

    c1 = [r for r in results if r["energy_case"] == "C1"
          and r["msp"] is not None]
    c1.sort(key=lambda r: r["msp"])
    top_solvents = [r["solvent"] for r in c1[:8]]

    if len(top_solvents) < 2:
        print("  Fig 4 skipped: not enough solvents for comparison")
        return

    cases = ["C1", "C2", "C3"]
    msp_data = {s: {} for s in top_solvents}
    gwp_data = {s: {} for s in top_solvents}

    for r in results:
        if r["solvent"] in top_solvents:
            if r["msp"] is not None:
                msp_data[r["solvent"]][r["energy_case"]] = r["msp"]
            if r["gwp"] is not None:
                gwp_data[r["solvent"]][r["energy_case"]] = r["gwp"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    n_solvents = len(top_solvents)
    x = np.arange(n_solvents)
    width = 0.25

    for i, case in enumerate(cases):
        color = ENERGY_COLORS[case]

        msp_vals = [msp_data[s].get(case, 0) for s in top_solvents]
        ax1.bar(x + i * width, msp_vals, width, color=color,
                label=ENERGY_LABELS[case], edgecolor="white", linewidth=0.5)

        gwp_vals = [gwp_data[s].get(case, 0) for s in top_solvents]
        ax2.bar(x + i * width, gwp_vals, width, color=color,
                edgecolor="white", linewidth=0.5)

    ax1.set_ylabel("MSP (USD / kg polymer)")
    ax1.set_title("Energy Case Comparison — Top 8 PE Solvents")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.set_ylabel("GWP (kg CO₂-eq / kg polymer)")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(top_solvents, rotation=35, ha="right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "fig4_energy_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 4 saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CS2 Pareto Front — Generate publication figures"
    )
    parser.add_argument(
        "results_json", nargs="?", default=None,
        help="Path to results JSON from run_cs2_pareto_traces.py",
    )
    parser.add_argument(
        "--from-csv", action="store_true",
        help="Generate figures from curated CSV + BioSTEAM defaults "
             "(proxy MSP, no simulation)",
    )
    parser.add_argument(
        "--from-biosteam", action="store_true",
        help="Run actual BioSTEAM simulations for all PE solvents "
             "(BP >= 75C) x 3 energy cases — real MSP/GWP",
    )
    parser.add_argument(
        "--from-raw-json", default=None, metavar="PATH",
        help="Load from a saved biosteam_raw_*.json file (skip re-running sims)",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory for figures (default: this script's dir)",
    )
    args = parser.parse_args()

    if (not args.from_csv and not args.from_biosteam
            and args.from_raw_json is None and args.results_json is None):
        parser.error("Provide a results JSON, --from-csv, --from-biosteam, "
                     "or --from-raw-json")

    output_dir = Path(args.output_dir) if args.output_dir else _THIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    answers = {}

    if args.from_raw_json:
        # ── Raw JSON mode: load saved BioSTEAM results ──
        raw_path = Path(args.from_raw_json)
        if not raw_path.exists():
            print(f"Error: {raw_path} not found")
            sys.exit(1)

        print(f"Case Study #2 — Loading BioSTEAM results from {raw_path.name}")
        print(f"  Output: {output_dir}\n")

        with open(raw_path) as f:
            raw_data = json.load(f)

        raw_results = raw_data.get("results", [])
        results = []
        for r in raw_results:
            if not r.get("success"):
                continue
            nr = normalize_result(r)
            nr["tier"] = classify_solvent(nr["solvent"])
            nr["source"] = "biosteam_simulation"
            results.append(nr)

        if not results:
            print("ERROR: No successful results in JSON.")
            sys.exit(1)

    elif args.from_biosteam:
        # ── BioSTEAM mode: run actual simulations ──
        print("Case Study #2 — Running BioSTEAM simulations for real MSP/GWP")
        print(f"  Output: {output_dir}\n")

        results = run_biosteam_batch(output_dir)

        if not results:
            print("ERROR: No successful simulations. Check BioSTEAM installation.")
            sys.exit(1)

    elif args.from_csv:
        # ── CSV mode: build data from curated sources (proxy MSP) ──
        print("Case Study #2 — Generating figures from curated CSV + BioSTEAM defaults")
        print(f"  CSV: {_CSV_PATH}")
        print(f"  Output: {output_dir}\n")

        results = load_from_csv_and_defaults()

        if not results:
            print("ERROR: No data loaded. Check that either:")
            print(f"  - {_CSV_PATH} exists (curated CSV)")
            print("  - strap.vendor.biosteam_runner is importable (_SOLVENT_DEFAULTS)")
            sys.exit(1)

    else:
        # ── JSON mode: parse agent answers ──
        results_path = Path(args.results_json)
        if not results_path.exists():
            print(f"Error: {results_path} not found")
            sys.exit(1)

        with open(results_path) as f:
            data = json.load(f)

        print(f"Case Study #2 — Generating figures from {results_path.name}")
        print(f"Output: {output_dir}\n")

        answers = data.get("answers", {})
        batch_answer = answers.get("cs2-batch-all-pe-solvents", "")
        scenario_answer = answers.get("cs2-top5-scenario-compare", "")

        raw_results = parse_batch_results(batch_answer)
        if scenario_answer:
            raw_results.extend(parse_batch_results(scenario_answer))

        results = [normalize_result(r) for r in raw_results]

    # Filter valid (finite MSP and GWP — excludes None, NaN, inf)
    valid = [r for r in results if _is_finite(r["msp"]) and _is_finite(r["gwp"])]
    print(f"Loaded {len(valid)} valid data points")

    if not valid:
        print("\nNo valid MSP/GWP data found.")
        sys.exit(1)

    # Deduplicate by (solvent, energy_case) and exclude outliers
    seen = set()
    deduped = []
    for r in valid:
        if r["solvent"] in EXCLUDE_SOLVENTS:
            continue
        key = (r["solvent"], r["energy_case"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    results = deduped

    n_solvents = len({r["solvent"] for r in results})
    n_cases = len({r["energy_case"] for r in results})

    # Count by tier
    c1_results = [r for r in results if r["energy_case"] == "C1"]
    tier_counts = {}
    for r in c1_results:
        t = r.get("tier") or classify_solvent(r["solvent"])
        tier_counts[t] = tier_counts.get(t, 0) + 1

    print(f"Unique: {n_solvents} solvents × {n_cases} energy cases = "
          f"{len(results)} points")
    print(f"  Tier 1 (validated):  {tier_counts.get('tier1', 0)}")
    print(f"  Tier 2 (curated):    {tier_counts.get('tier2', 0)}")
    print(f"  Tier 3 (class-avg):  {tier_counts.get('tier3', 0)}")
    print()

    # ── Generate figures ──
    print("Generating figures...")

    pareto = fig1_pareto_scatter(results, output_dir)
    fig2_msp_ranking(results, pareto, output_dir)
    fig3_tornado(answers, output_dir)
    fig4_energy_comparison(results, output_dir)

    # ── Summary ──
    print(f"\nPareto-optimal solvents (C1):")
    for r in pareto:
        tier = r.get("tier") or classify_solvent(r["solvent"])
        tier_str = {"tier1": "T1-validated", "tier2": "T2-curated",
                    "tier3": "T3-class-avg"}.get(tier, tier)
        print(f"  {r['solvent']:<25s}  MSP=${r['msp']:.3f}  "
              f"GWP={r['gwp']:.2f}  LCA={tier_str}")

    print(f"\nTotal: {len(pareto)} Pareto-optimal solvents out of {n_solvents}")
    print("Done.")


if __name__ == "__main__":
    main()
