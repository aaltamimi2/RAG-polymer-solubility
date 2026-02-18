"""BioSTEAM TEA/LCA tools for rigorous STRAP process simulation.

Thin agent-facing wrappers around the BioSTEAM subprocess runner
(strap.vendor.biosteam_runner).  Each tool delegates simulation work
to the runner and formats results as JSON strings for the LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from strap.tools._helpers import safe_tool_wrapper

try:
    from strap.vendor.biosteam_runner import (
        run_single_simulation,
        run_batch_simulations,
        build_batch_configs,
        get_supported_solvents,
    )
except ImportError:
    run_single_simulation = None
    run_batch_simulations = None
    build_batch_configs = None
    get_supported_solvents = None

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Solvent expansion helpers
# ------------------------------------------------------------------
# Chlorinated solvents that fail in BioSTEAM (HCl not in property package)
_CHLORINATED_BLOCKLIST = frozenset({
    "Tetrachloroethylene", "o-Chlorotoluene",
    "Dichloromethane", "Chloroform",
})

# Core PE solvents from BioSTEAM plastics v0.1.4 (excluding chlorinated)
_PE_SOLVENTS_CORE = [
    "sec-Butyl Acetate", "Isobutyl Acetate", "Methylcyclohexane",
    "Dodecanol", "Heptane", "Toluene", "Xylene",
]

# Extended PE/LDPE solvents from COMMON-SOLVENTS-DATABASE (thermosteam-validated)
_PE_SOLVENTS_EXTENDED = [
    "o-Xylene", "p-Xylene", "Cyclohexane", "Dodecane", "Hexane",
    "Benzene", "Acetone", "2-Butanone", "Ethyl acetate",
    "Tetrahydrofuran", "1-Propanol", "Ethanol", "Methanol",
    "Isopropanol", "tert-Butanol", "Cyclohexanol",
    "N,N-Dimethylformamide", "Diphenyl ether", "Acetylacetone",
    "2,3-Dihydropyran", "Tetrahydropyran", "Triethylamine",
    "Methyl acetate",
]

_PE_SOLVENTS = _PE_SOLVENTS_CORE + _PE_SOLVENTS_EXTENDED

# EVOH solvents (E1 sequence — original high-selectivity)
_EVOH_SOLVENTS = [
    "Ethylene Glycol", "Pyridazine",
]

# All EVOH solvents (E2 sequence — broader set including extended)
_EVOH_SOLVENTS_E2 = [
    "butane-1,4-diol", "Diethanolamine", "Diethylene glycol",
    "Ethylene Glycol", "Propylene Glycol", "Pyridazine",
    "gamma-butyrolactone",
    # Extended EVOH solvents (>50% EVOH solubility)
    "Dimethyl sulfoxide", "N,N-Dimethylformamide", "Triethylamine",
    "Methanol", "Ethanol", "Isopropanol",
]

# PET solvents (aromatic/polar + extended from COMMON-SOLVENTS-DATABASE)
_PET_SOLVENTS = [
    "Toluene", "Xylene",
    # Extended PET solvents (>30% PET solubility)
    "Acetone", "N,N-Dimethylformamide", "Tetrahydrofuran",
    "2-Butanone", "Benzene",
]

# LDPE solvents — same as PE (both polyethylenes dissolve in the same solvents)
_LDPE_SOLVENTS = list(_PE_SOLVENTS)

_ALL_ENERGY_CASES = ["C1", "C2", "C3"]

from strap.solvent_registry import resolve_to_biosteam as _resolve_biosteam


def _expand_solvents(solvents_str: str, target_plastic: str) -> list[str]:
    """Parse comma-separated solvent string or expand shorthand keywords."""
    s = solvents_str.strip().lower()
    if s == "all_pe":
        return list(_PE_SOLVENTS)
    if s == "all_ldpe":
        return list(_LDPE_SOLVENTS)
    if s == "all_evoh":
        return list(_EVOH_SOLVENTS)
    if s == "all_evoh_e2":
        return list(_EVOH_SOLVENTS_E2)
    if s == "all_pet":
        return list(_PET_SOLVENTS)
    if s == "all":
        tp = target_plastic.upper()
        if tp == "EVOH":
            return list(_EVOH_SOLVENTS)
        if tp == "PET":
            return list(_PET_SOLVENTS)
        if tp == "LDPE":
            return list(_LDPE_SOLVENTS)
        return list(_PE_SOLVENTS)

    # Check if full string is a known alias (handles comma-containing names)
    bio = _resolve_biosteam(s)
    if bio:
        return [bio]

    # Split by comma and resolve each token against aliases
    parsed = [sv.strip() for sv in solvents_str.split(",") if sv.strip()]
    resolved = []
    for sv in parsed:
        alias = _resolve_biosteam(sv.strip())
        resolved.append(alias if alias else sv)
    return resolved


def _expand_energy_cases(cases_str: str) -> list[str]:
    """Parse comma-separated energy cases or expand 'all'."""
    s = cases_str.strip().lower()
    if s == "all":
        return list(_ALL_ENERGY_CASES)
    return [c.strip().upper() for c in cases_str.split(",") if c.strip()]


# ------------------------------------------------------------------
# Tool 1: Single simulation
# ------------------------------------------------------------------

@safe_tool_wrapper
def run_biosteam_simulation(
    solvent: str,
    target_plastic: str = "PE",
    energy_case: str = "C1",
    target_plastic_percent: float = 60,
    processing_capacity: float = 20000,
    dissolution_temp_c: float | None = None,
    precipitation_temp_c: float = 25,
    solvent_price: float | None = None,
) -> str:
    """Run a rigorous BioSTEAM STRAP process simulation for a single solvent.

    Returns TEA metrics (MSP, TCI, AOC), LCA metrics (GWP, HTC, HTNC, ETOX),
    and operational data (energy, water, waste) from a full BioSTEAM flowsheet.

    Args:
        solvent: Solvent name (e.g. "Toluene", "Xylene", "Heptane").
        target_plastic: Target plastic to recover — "PE", "LDPE", "EVOH", or "PET" (default "PE").
            LDPE uses the same solvents as PE. PET uses the same generic dissolution model
            with dynamic parameter registration.
        energy_case: Energy configuration "C1" (CHP), "C2" (Grid+AMCOR), or "C3" (Grid+Boiler). Default "C1".
        target_plastic_percent: Target plastic weight percent in feed, 0-100 (default 60).
        processing_capacity: Plant capacity in metric tons/year (default 20000).
        dissolution_temp_c: Dissolution temperature in Celsius (optional, uses BioSTEAM default if omitted).
        precipitation_temp_c: Precipitation temperature in Celsius (default 25).
        solvent_price: Override solvent price in $/kg (optional, uses BioSTEAM default if omitted).

    WHEN TO USE:
    - "Run BioSTEAM simulation for toluene"
    - "What is the MSP for PE recovery using xylene?"
    - "Simulate STRAP process with heptane at 20,000 MT/yr"
    - "TEA/LCA for PE dissolution in cyclohexane under C2 energy case"
    - "Run BioSTEAM for PET dissolution in Toluene"
    - "Run BioSTEAM for LDPE dissolution in Xylene"
    """
    if run_single_simulation is None:
        return "ERROR: biosteam_runner module not available. Install BioSTEAM dependencies."

    config = {
        "solvent": solvent,
        "target_plastic": target_plastic,
        "target_plastic_percent": target_plastic_percent,
        "processing_capacity": processing_capacity,
        "energy_case": energy_case,
        "precipitation_temperature_c": precipitation_temp_c,
    }
    if dissolution_temp_c is not None:
        config["dissolution_temperature_c"] = dissolution_temp_c
    if solvent_price is not None:
        config["solvent_price"] = solvent_price

    result = run_single_simulation(config)

    # Format display output
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown simulation error")
        return json.dumps({
            "display": f"**Simulation failed** for {solvent} ({target_plastic}, {energy_case}): {error_msg}",
            "data": result,
        }, indent=2)

    tea = result.get("tea", {})
    lca = result.get("lca", {})
    ops = result.get("operations", {})

    display = f"## BioSTEAM Simulation: {solvent} ({target_plastic}, {energy_case})\n\n"
    display += f"**Capacity:** {processing_capacity:,.0f} MT/yr | "
    display += f"**Feed:** {target_plastic_percent:.0f}% {target_plastic}\n\n"

    display += "### TEA Results\n"
    display += f"| Metric | Value |\n|--------|-------|\n"
    msp = tea.get("msp_usd_per_kg")
    tci = tea.get("tci_usd")
    aoc = tea.get("aoc_usd_per_yr")
    display += f"| MSP | ${msp:.4f}/kg |\n" if msp is not None else "| MSP | N/A |\n"
    display += f"| TCI | ${tci/1e6:.2f}M |\n" if tci is not None else "| TCI | N/A |\n"
    display += f"| AOC | ${aoc/1e6:.2f}M/yr |\n" if aoc is not None else "| AOC | N/A |\n"

    display += "\n### LCA Results\n"
    display += f"| Metric | Value |\n|--------|-------|\n"
    gwp = lca.get("gwp_kg_co2e_per_kg")
    display += f"| GWP | {gwp:.4f} kg CO2e/kg |\n" if gwp is not None else "| GWP | N/A |\n"
    htc = lca.get("htc_ctuh_per_kg")
    display += f"| HTC | {htc:.2e} CTUh/kg |\n" if htc is not None else "| HTC | N/A |\n"
    htnc = lca.get("htnc_ctuh_per_kg")
    display += f"| HTNC | {htnc:.2e} CTUh/kg |\n" if htnc is not None else "| HTNC | N/A |\n"
    etox = lca.get("etox_ctue_per_kg")
    display += f"| ETOX | {etox:.4f} CTUe/kg |\n" if etox is not None else "| ETOX | N/A |\n"

    display += "\n### Operational\n"
    total_e = ops.get("total_energy_mj_per_kg")
    display += f"- Total energy: {total_e:.2f} MJ/kg\n" if total_e is not None else ""
    runtime = result.get("runtime_seconds")
    display += f"- Simulation runtime: {runtime:.1f}s\n" if runtime is not None else ""

    return json.dumps({"display": display, "data": result}, indent=2)


# ------------------------------------------------------------------
# Tool 2: Batch simulation
# ------------------------------------------------------------------

@safe_tool_wrapper
def run_biosteam_batch(
    solvents: str,
    target_plastic: str = "PE",
    energy_cases: str = "C1",
    target_plastic_percent: float = 60,
    processing_capacity: float = 20000,
    max_parallel: int = 3,
) -> str:
    """Run BioSTEAM STRAP simulations across multiple solvents and/or energy cases.

    Batch mode: runs each (solvent, energy_case) combination and returns a
    ranked comparison table sorted by MSP.

    Args:
        solvents: Comma-separated solvent names, or "all_pe" (PE solvents),
            "all_ldpe" (LDPE solvents, same as PE), "all_evoh" (EVOH solvents),
            "all_pet" (PET solvents), or "all" (auto-select by target_plastic).
        target_plastic: Target plastic "PE", "LDPE", "EVOH", or "PET" (default "PE").
        energy_cases: Comma-separated energy cases or "all" for C1,C2,C3 (default "C1").
        target_plastic_percent: Target plastic weight percent 0-100 (default 60).
        processing_capacity: Plant capacity in MT/yr (default 20000).
        max_parallel: Maximum concurrent subprocess simulations (default 3).

    WHEN TO USE:
    - "Compare all PE solvents in BioSTEAM"
    - "Run batch simulation for toluene, xylene, heptane"
    - "Rank solvents by MSP across all energy cases"
    - "BioSTEAM comparison of all solvents under C1, C2, C3"
    - "Compare all PET solvents in BioSTEAM"
    - "Compare all LDPE solvents in BioSTEAM"
    """
    if run_single_simulation is None:
        return "ERROR: biosteam_runner module not available. Install BioSTEAM dependencies."

    solvent_list = _expand_solvents(solvents, target_plastic)
    case_list = _expand_energy_cases(energy_cases)

    if not solvent_list:
        return "ERROR: No solvents specified. Provide comma-separated names or 'all_pe'/'all_evoh'."
    if not case_list:
        return "ERROR: No energy cases specified. Provide 'C1', 'C2', 'C3', or 'all'."

    # Build configs for all combinations
    if build_batch_configs is not None:
        configs = build_batch_configs(
            solvents=solvent_list,
            energy_cases=case_list,
            target_plastic=target_plastic,
            target_plastic_percent=target_plastic_percent,
            processing_capacity=processing_capacity,
        )
    else:
        # Fallback: build configs manually
        configs = []
        for sv in solvent_list:
            for ec in case_list:
                configs.append({
                    "solvent": sv,
                    "target_plastic": target_plastic,
                    "target_plastic_percent": target_plastic_percent,
                    "processing_capacity": processing_capacity,
                    "energy_case": ec,
                })

    # Run all simulations
    if run_batch_simulations is not None:
        results = run_batch_simulations(configs, max_parallel=max_parallel)
    else:
        # Fallback: sequential execution
        results = []
        for cfg in configs:
            results.append(run_single_simulation(cfg))

    # Separate successes and failures
    successes = [r for r in results if r.get("success", False)]
    failures = [r for r in results if not r.get("success", False)]

    # Build display output
    display = f"## BioSTEAM Batch Results: {target_plastic}\n\n"
    display += f"**Solvents:** {', '.join(solvent_list)} | "
    display += f"**Energy cases:** {', '.join(case_list)} | "
    display += f"**Capacity:** {processing_capacity:,.0f} MT/yr\n\n"
    display += f"**Completed:** {len(successes)}/{len(configs)} simulations\n\n"

    if successes:
        # Sort by MSP ascending (lowest = best)
        successes.sort(
            key=lambda r: r.get("tea", {}).get("msp_usd_per_kg") or float("inf")
        )

        display += "### Rankings by MSP\n\n"
        display += "| Rank | Solvent | Case | MSP ($/kg) | TCI ($M) | GWP (kg CO2e/kg) |\n"
        display += "|------|---------|------|------------|----------|------------------|\n"

        for i, r in enumerate(successes, 1):
            tea = r.get("tea", {})
            lca = r.get("lca", {})
            msp_val = tea.get("msp_usd_per_kg")
            tci_val = tea.get("tci_usd")
            gwp_val = lca.get("gwp_kg_co2e_per_kg")

            msp_str = f"${msp_val:.4f}" if msp_val is not None else "N/A"
            tci_str = f"${tci_val / 1e6:.2f}M" if tci_val is not None else "N/A"
            gwp_str = f"{gwp_val:.4f}" if gwp_val is not None else "N/A"

            display += (
                f"| {i} | {r.get('solvent', '?')} | {r.get('energy_case', '?')} | "
                f"{msp_str} | {tci_str} | {gwp_str} |\n"
            )

        # Best / worst summary
        best = successes[0]
        display += f"\n**Best MSP:** {best.get('solvent')} ({best.get('energy_case')}) "
        best_msp = best.get("tea", {}).get("msp_usd_per_kg")
        display += f"at ${best_msp:.4f}/kg\n" if best_msp is not None else "\n"

        # GWP ranking
        gwp_sorted = sorted(
            successes,
            key=lambda r: r.get("lca", {}).get("gwp_kg_co2e_per_kg") or float("inf"),
        )
        best_gwp = gwp_sorted[0]
        display += f"**Lowest GWP:** {best_gwp.get('solvent')} ({best_gwp.get('energy_case')}) "
        best_gwp_val = best_gwp.get("lca", {}).get("gwp_kg_co2e_per_kg")
        display += f"at {best_gwp_val:.4f} kg CO2e/kg\n" if best_gwp_val is not None else "\n"

    if failures:
        display += f"\n### Failed Simulations ({len(failures)})\n\n"
        for f in failures:
            display += f"- {f.get('solvent', '?')} ({f.get('energy_case', '?')}): {f.get('error', 'unknown')}\n"

    structured_data = {
        "tool_name": "run_biosteam_batch",
        "success": len(successes) > 0,
        "total_configs": len(configs),
        "completed": len(successes),
        "failed": len(failures),
        "solvents": solvent_list,
        "energy_cases": case_list,
        "target_plastic": target_plastic,
        "processing_capacity": processing_capacity,
        "results": successes,
        "failures": [
            {"solvent": f.get("solvent"), "energy_case": f.get("energy_case"),
             "error": f.get("error")}
            for f in failures
        ],
    }

    return json.dumps({"display": display, "data": structured_data}, indent=2)


# ------------------------------------------------------------------
# Tool 3: Scenario comparison
# ------------------------------------------------------------------

@safe_tool_wrapper
def compare_biosteam_scenarios(
    scenarios_json: str,
) -> str:
    """Compare specific BioSTEAM STRAP scenarios side by side.

    Each scenario is a dict specifying at minimum {solvent, target_plastic,
    energy_case}. Additional optional keys match run_biosteam_simulation params.

    Args:
        scenarios_json: JSON array of scenario dicts. Each dict can contain:
            solvent (required), target_plastic (default "PE"),
            energy_case (default "C1"), target_plastic_percent (default 60),
            processing_capacity (default 20000), dissolution_temp_c,
            precipitation_temp_c (default 25), solvent_price.

    WHEN TO USE:
    - "Compare toluene C1 vs xylene C2 in BioSTEAM"
    - "Side-by-side BioSTEAM scenarios for different capacities"
    - "Which scenario has lowest MSP: toluene 20k or xylene 50k MT/yr?"
    """
    if run_single_simulation is None:
        return "ERROR: biosteam_runner module not available. Install BioSTEAM dependencies."

    try:
        scenarios = json.loads(scenarios_json)
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON in scenarios_json: {e}"

    if not isinstance(scenarios, list) or len(scenarios) < 1:
        return "ERROR: scenarios_json must be a JSON array with at least 1 scenario dict."

    # Run each scenario
    results = []
    for i, sc in enumerate(scenarios):
        if "solvent" not in sc:
            results.append({
                "success": False,
                "error": f"Scenario {i+1} missing required 'solvent' field",
                "scenario_index": i,
            })
            continue

        config = {
            "solvent": sc["solvent"],
            "target_plastic": sc.get("target_plastic", "PE"),
            "target_plastic_percent": sc.get("target_plastic_percent", 60),
            "processing_capacity": sc.get("processing_capacity", 20000),
            "energy_case": sc.get("energy_case", "C1"),
            "precipitation_temperature_c": sc.get("precipitation_temp_c", 25),
        }
        if sc.get("dissolution_temp_c") is not None:
            config["dissolution_temperature_c"] = sc["dissolution_temp_c"]
        if sc.get("solvent_price") is not None:
            config["solvent_price"] = sc["solvent_price"]

        result = run_single_simulation(config)
        result["scenario_index"] = i
        result["scenario_label"] = sc.get(
            "label",
            f"{sc['solvent']}/{sc.get('target_plastic', 'PE')}/{sc.get('energy_case', 'C1')}",
        )
        results.append(result)

    successes = [r for r in results if r.get("success", False)]
    failures = [r for r in results if not r.get("success", False)]

    # Build display
    display = f"## BioSTEAM Scenario Comparison\n\n"
    display += f"**Scenarios:** {len(scenarios)} | **Completed:** {len(successes)}\n\n"

    if successes:
        # MSP ranking
        msp_sorted = sorted(
            successes,
            key=lambda r: r.get("tea", {}).get("msp_usd_per_kg") or float("inf"),
        )

        display += "### Comparison Table (ranked by MSP)\n\n"
        display += "| Rank | Scenario | MSP ($/kg) | TCI ($M) | AOC ($M/yr) | GWP (kg CO2e/kg) |\n"
        display += "|------|----------|------------|----------|-------------|------------------|\n"

        for i, r in enumerate(msp_sorted, 1):
            tea = r.get("tea", {})
            lca = r.get("lca", {})
            label = r.get("scenario_label", "?")

            msp_val = tea.get("msp_usd_per_kg")
            tci_val = tea.get("tci_usd")
            aoc_val = tea.get("aoc_usd_per_yr")
            gwp_val = lca.get("gwp_kg_co2e_per_kg")

            msp_str = f"${msp_val:.4f}" if msp_val is not None else "N/A"
            tci_str = f"${tci_val / 1e6:.2f}M" if tci_val is not None else "N/A"
            aoc_str = f"${aoc_val / 1e6:.2f}M" if aoc_val is not None else "N/A"
            gwp_str = f"{gwp_val:.4f}" if gwp_val is not None else "N/A"

            display += f"| {i} | {label} | {msp_str} | {tci_str} | {aoc_str} | {gwp_str} |\n"

        # GWP ranking
        gwp_sorted = sorted(
            successes,
            key=lambda r: r.get("lca", {}).get("gwp_kg_co2e_per_kg") or float("inf"),
        )

        display += "\n### Rankings\n"
        best_msp = msp_sorted[0]
        display += f"- **Lowest MSP:** {best_msp.get('scenario_label')} "
        bm = best_msp.get("tea", {}).get("msp_usd_per_kg")
        display += f"(${bm:.4f}/kg)\n" if bm is not None else "\n"

        best_gwp = gwp_sorted[0]
        display += f"- **Lowest GWP:** {best_gwp.get('scenario_label')} "
        bg = best_gwp.get("lca", {}).get("gwp_kg_co2e_per_kg")
        display += f"({bg:.4f} kg CO2e/kg)\n" if bg is not None else "\n"

    if failures:
        display += f"\n### Failed Scenarios ({len(failures)})\n\n"
        for f in failures:
            display += f"- Scenario {f.get('scenario_index', '?') + 1}: {f.get('error', 'unknown')}\n"

    structured_data = {
        "tool_name": "compare_biosteam_scenarios",
        "success": len(successes) > 0,
        "scenarios_requested": len(scenarios),
        "completed": len(successes),
        "failed": len(failures),
        "results": results,
    }

    return json.dumps({"display": display, "data": structured_data}, indent=2)


# ------------------------------------------------------------------
# Tool 4: List supported solvents and configurations
# ------------------------------------------------------------------

@safe_tool_wrapper
def get_biosteam_solvents() -> str:
    """List all solvents and configurations supported by the BioSTEAM STRAP model.

    Returns supported solvents (PE, EVOH, and PET), energy case descriptions,
    default parameters, and known model limitations.

    WHEN TO USE:
    - "What solvents can BioSTEAM simulate?"
    - "List supported BioSTEAM configurations"
    - "What energy cases are available?"
    - "BioSTEAM model parameters and defaults"
    """
    # Try to get dynamic list from runner
    dynamic_solvents = None
    if get_supported_solvents is not None:
        try:
            dynamic_solvents = get_supported_solvents()
        except Exception:
            pass

    display = "## BioSTEAM STRAP Model: Supported Configurations\n\n"

    display += f"### PE Solvents ({len(_PE_SOLVENTS)})\n"
    display += f"**Core (Branch-TEA validated, {len(_PE_SOLVENTS_CORE)}):** "
    display += ", ".join(_PE_SOLVENTS_CORE) + "\n\n"
    display += f"**Extended (COMMON-SOLVENTS-DATABASE, {len(_PE_SOLVENTS_EXTENDED)}):** "
    display += ", ".join(_PE_SOLVENTS_EXTENDED) + "\n"
    display += "\n*Note: Chlorinated solvents excluded from batch expansions (HCl not in property package).*\n"

    display += f"\n### LDPE Solvents ({len(_LDPE_SOLVENTS)})\n"
    display += "*LDPE uses the same solvents as PE (both are polyethylenes).*\n"

    display += f"\n### EVOH Solvents — E1 Sequence ({len(_EVOH_SOLVENTS)})\n"
    for sv in _EVOH_SOLVENTS:
        display += f"- {sv}\n"

    display += f"\n### EVOH Solvents — E2 Sequence ({len(_EVOH_SOLVENTS_E2)})\n"
    for sv in _EVOH_SOLVENTS_E2:
        display += f"- {sv}\n"

    display += f"\n### PET Solvents ({len(_PET_SOLVENTS)})\n"
    for sv in _PET_SOLVENTS:
        display += f"- {sv}\n"
    display += "\n*PET uses the same generic dissolution model (define_dissolution) with dynamic parameter registration.*\n"

    display += "\n### Energy Cases\n"
    display += "| Case | Description |\n|------|-------------|\n"
    display += "| C1 | Combined Heat & Power (CHP) with turbogenerator |\n"
    display += "| C2 | Grid electricity + AMCOR heat (no boiler/turbine) |\n"
    display += "| C3 | Grid electricity + natural gas boiler (no turbine) |\n"

    display += "\n### Default Parameters\n"
    display += "| Parameter | Default | Unit |\n|-----------|---------|------|\n"
    display += "| target_plastic | PE | - |\n"
    display += "| target_plastic_percent | 60 | wt% |\n"
    display += "| processing_capacity | 20,000 | MT/yr |\n"
    display += "| energy_case | C1 | - |\n"
    display += "| precipitation_temp_c | 25 | C |\n"
    display += "| labor_cost | 120,000 | $/yr |\n"
    display += "| solvent_loss_pct | 0.01 | % |\n"

    display += "\n### Shorthand Keywords for Batch Mode\n"
    display += f"- `all_pe` -- all {len(_PE_SOLVENTS)} PE solvents (excludes chlorinated)\n"
    display += f"- `all_ldpe` -- all {len(_LDPE_SOLVENTS)} LDPE solvents (same as PE)\n"
    display += f"- `all_evoh` -- {len(_EVOH_SOLVENTS)} EVOH solvents (E1 sequence)\n"
    display += f"- `all_evoh_e2` -- {len(_EVOH_SOLVENTS_E2)} EVOH solvents (E2 sequence)\n"
    display += f"- `all_pet` -- all {len(_PET_SOLVENTS)} PET solvents\n"
    display += "- `all` (energy_cases) -- C1, C2, C3\n"

    display += "\n### Known Limitations\n"
    display += "- Chlorinated solvents (Tetrachloroethylene, o-Chlorotoluene) fail — excluded from expansions\n"
    display += "- Simulations run as isolated subprocesses (~10-15s each)\n"
    display += "- LCA characterisation factors use TRACI defaults unless overridden\n"

    structured_data = {
        "tool_name": "get_biosteam_solvents",
        "success": True,
        "pe_solvents_core": list(_PE_SOLVENTS_CORE),
        "pe_solvents_extended": list(_PE_SOLVENTS_EXTENDED),
        "pe_solvents": list(_PE_SOLVENTS),
        "ldpe_solvents": list(_LDPE_SOLVENTS),
        "evoh_solvents_e1": list(_EVOH_SOLVENTS),
        "evoh_solvents_e2": list(_EVOH_SOLVENTS_E2),
        "pet_solvents": list(_PET_SOLVENTS),
        "energy_cases": list(_ALL_ENERGY_CASES),
        "chlorinated_blocklist": list(_CHLORINATED_BLOCKLIST),
        "dynamic_solvents": dynamic_solvents,
    }

    return json.dumps({"display": display, "data": structured_data}, indent=2)


# ------------------------------------------------------------------
# Tool 5: Visualize BioSTEAM results
# ------------------------------------------------------------------

# Default market values (USD/kg) for MSP allocation
_POLYMER_MARKET_VALUES = {
    "PE": 1.10, "LDPE": 1.10, "HDPE": 1.20,
    "PP": 1.15, "PS": 1.30, "PVC": 0.90,
    "PET": 1.05, "EVOH": 4.50, "Nylon6": 2.80,
    "Nylon66": 3.00, "PMMA": 2.50,
}


@safe_tool_wrapper
def visualize_biosteam_results(
    results_json: str,
    chart_types: str = "all",
    output_dir: str = "./plots",
) -> str:
    """Generate matplotlib charts from BioSTEAM simulation results.

    Accepts the JSON output from any BioSTEAM tool (run_biosteam_simulation,
    run_biosteam_batch, compare_biosteam_scenarios, or run_biosteam_multi_polymer)
    and produces publication-quality charts.

    Args:
        results_json: JSON string from a BioSTEAM tool output's "data" field.
        chart_types: Comma-separated chart types to generate.
            Options: "cost_breakdown", "gwp_breakdown", "scenario_comparison", "all".
            Default "all".
        output_dir: Directory to save PNG files (default "./plots").

    WHEN TO USE:
    - "Visualize the BioSTEAM results"
    - "Create a cost breakdown chart from the simulation"
    - "Plot GWP comparison across solvents"
    - "Generate charts for the batch simulation results"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    try:
        data = json.loads(results_json) if isinstance(results_json, str) else results_json
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON: {e}"

    requested = set(t.strip() for t in chart_types.split(","))
    if "all" in requested:
        requested = {"cost_breakdown", "gwp_breakdown", "scenario_comparison"}

    saved_files = []

    # Extract results list from various tool output formats
    results_list = []
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            results_list = [r for r in data["results"] if r.get("success", False)]
        elif "tea" in data and data.get("success", False):
            results_list = [data]
        elif "per_polymer" in data:
            # Multi-polymer output
            results_list = [p.get("result", {}) for p in data["per_polymer"]
                           if p.get("result", {}).get("success", False)]
    elif isinstance(data, list):
        results_list = [r for r in data if r.get("success", False)]

    if not results_list:
        return json.dumps({
            "display": "No successful results to visualize.",
            "data": {"success": False, "error": "No successful simulation results found"},
        })

    # --- Cost breakdown bar chart ---
    if "cost_breakdown" in requested and results_list:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = []
        msp_vals = []
        tci_vals = []
        aoc_vals = []
        for r in results_list:
            tea = r.get("tea", {})
            label = r.get("solvent", r.get("scenario_label", "?"))
            ec = r.get("energy_case", "")
            if ec:
                label = f"{label}\n({ec})"
            labels.append(label)
            msp_vals.append(tea.get("msp_usd_per_kg") or 0)
            tci_vals.append((tea.get("tci_usd") or 0) / 1e6)
            aoc_vals.append((tea.get("aoc_usd_per_yr") or 0) / 1e6)

        x = range(len(labels))
        width = 0.25
        ax.bar([i - width for i in x], msp_vals, width, label="MSP ($/kg)", color="#2196F3")
        ax.bar(x, tci_vals, width, label="TCI ($M)", color="#FF9800")
        ax.bar([i + width for i in x], aoc_vals, width, label="AOC ($M/yr)", color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(fontsize=9)
        ax.set_title("Cost Breakdown", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")
        plt.tight_layout()
        path = os.path.join(output_dir, "biosteam_cost_breakdown.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

    # --- GWP breakdown bar chart ---
    if "gwp_breakdown" in requested and results_list:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = []
        gwp_vals = []
        htc_vals = []
        for r in results_list:
            lca = r.get("lca", {})
            label = r.get("solvent", r.get("scenario_label", "?"))
            ec = r.get("energy_case", "")
            if ec:
                label = f"{label}\n({ec})"
            labels.append(label)
            gwp_vals.append(lca.get("gwp_kg_co2e_per_kg") or 0)
            htc_vals.append((lca.get("htc_ctuh_per_kg") or 0) * 1e6)  # scale for visibility

        x = range(len(labels))
        ax.bar(x, gwp_vals, color="#E53935", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("GWP Comparison", fontsize=12, fontweight="bold")
        ax.set_ylabel("GWP (kg CO₂e/kg product)")
        plt.tight_layout()
        path = os.path.join(output_dir, "biosteam_gwp_breakdown.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

    # --- Scenario comparison (MSP vs GWP scatter) ---
    if "scenario_comparison" in requested and len(results_list) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in results_list:
            tea = r.get("tea", {})
            lca = r.get("lca", {})
            msp = tea.get("msp_usd_per_kg")
            gwp = lca.get("gwp_kg_co2e_per_kg")
            if msp is not None and gwp is not None:
                label = r.get("solvent", r.get("scenario_label", "?"))
                ax.scatter(gwp, msp, s=80, zorder=3)
                ax.annotate(label, (gwp, msp), fontsize=7,
                           textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("GWP (kg CO₂e/kg)")
        ax.set_ylabel("MSP ($/kg)")
        ax.set_title("Scenario Comparison: MSP vs GWP", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "biosteam_scenario_comparison.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)

    display = f"Generated {len(saved_files)} chart(s):\n"
    for p in saved_files:
        display += f"- {p}\n"

    return json.dumps({
        "display": display,
        "data": {"success": True, "charts": saved_files},
    }, indent=2)


# ------------------------------------------------------------------
# Tool 6: Multi-polymer sequential recovery
# ------------------------------------------------------------------

# Default parameters for standard 4-stage sequential recovery.
# From Branch-TEA.ipynb: target_plastic is always 'PE' for all stages.
# The stage labels describe the solvent *sequence*, not the target polymer.
_SEQUENTIAL_STAGE_DEFAULTS: dict[str, dict] = {
    "P1": {
        "target_plastic": "PE",
        "target_plastic_percent": 60,
        "processing_capacity": 20000,
        "description": "PE first recovery",
    },
    "E1": {
        "target_plastic": "PE",
        "target_plastic_percent": 10,
        "processing_capacity": 20000,
        "description": "EVOH first recovery (PE target)",
    },
    "E2": {
        "target_plastic": "PE",
        "target_plastic_percent": 25,
        "processing_capacity": 8000,
        "description": "EVOH second recovery (PE target)",
    },
    "P2": {
        "target_plastic": "PE",
        "target_plastic_percent": 66.667,
        "processing_capacity": 18000,
        "description": "PE second recovery",
    },
}


@safe_tool_wrapper
def run_biosteam_multi_polymer(
    polymers_json: str,
    energy_case: str = "C1",
    processing_capacity: float = 20000,
    allocation_method: str = "value",
    sequence_stages: str = "",
) -> str:
    """Run sequential BioSTEAM simulations for multi-polymer recovery and combine results.

    Chains one BioSTEAM subprocess run per polymer, then combines results
    with MSP allocation by polymer market value ratios. Reports blended MSP,
    combined TCI/AOC, weighted GWP, and per-polymer breakdown.

    Args:
        polymers_json: JSON array of polymer specs. Each dict must have:
            "polymer" (required): target plastic name (e.g. "PE", "EVOH").
            "solvent" (required): solvent name.
            Optional per-spec overrides: "target_plastic_percent",
            "processing_capacity", "dissolution_temp_c", "precipitation_temp_c".
        energy_case: Energy configuration "C1", "C2", or "C3" (default "C1").
        processing_capacity: Plant capacity in MT/yr (default 20000).
            Ignored when sequence_stages provides stage defaults.
        allocation_method: How to allocate combined MSP across polymers.
            "value" (default): weight by market value ratios.
            "mass": weight equally by mass fraction.
        sequence_stages: Comma-separated stage labels for standard 4-stage
            recovery: "P1,E1,E2,P2". Each label maps to reference notebook
            defaults (target_plastic_percent, processing_capacity). Per-spec
            overrides in polymers_json take priority over stage defaults.
            Empty string (default) = no stage defaults applied.

    WHEN TO USE:
    - "Run multi-polymer BioSTEAM for PE and EVOH recovery"
    - "Sequential recovery TEA for PE with toluene then EVOH with ethylene glycol"
    - "What is the blended MSP for a 2-polymer STRAP process?"
    - "Run the full P1,E1,E2,P2 sequence with different solvents"
    """
    if run_single_simulation is None:
        return "ERROR: biosteam_runner module not available. Install BioSTEAM dependencies."

    try:
        polymers = json.loads(polymers_json)
    except json.JSONDecodeError as e:
        return f"ERROR: Invalid JSON in polymers_json: {e}"

    if not isinstance(polymers, list) or len(polymers) < 1:
        return "ERROR: polymers_json must be a JSON array with at least 1 polymer spec."

    # Parse stage labels if provided
    stage_labels = (
        [s.strip().upper() for s in sequence_stages.split(",") if s.strip()]
        if sequence_stages else []
    )

    per_polymer_results = []
    total_tci = 0.0
    total_aoc = 0.0
    total_gwp_weighted = 0.0
    total_weight = 0.0

    for i, spec in enumerate(polymers):
        polymer = spec.get("polymer")
        solvent = spec.get("solvent")
        if not polymer or not solvent:
            per_polymer_results.append({
                "polymer": polymer or f"spec_{i}",
                "success": False,
                "error": "Missing required 'polymer' or 'solvent' field",
            })
            continue

        # Look up stage defaults when sequence_stages provided
        stage_defaults = {}
        stage_label = ""
        if stage_labels and i < len(stage_labels):
            stage_label = stage_labels[i]
            stage_defaults = _SEQUENTIAL_STAGE_DEFAULTS.get(stage_label, {})

        # Priority: per-spec override > stage default > function arg / fallback
        config = {
            "solvent": solvent,
            "target_plastic": stage_defaults.get("target_plastic", polymer),
            "target_plastic_percent": spec.get(
                "target_plastic_percent",
                stage_defaults.get("target_plastic_percent", 60),
            ),
            "processing_capacity": spec.get(
                "processing_capacity",
                stage_defaults.get("processing_capacity", processing_capacity),
            ),
            "energy_case": energy_case,
            "precipitation_temperature_c": spec.get("precipitation_temp_c", 25),
        }
        if spec.get("dissolution_temp_c") is not None:
            config["dissolution_temperature_c"] = spec["dissolution_temp_c"]

        result = run_single_simulation(config)
        result["polymer"] = polymer
        result["solvent"] = solvent

        if result.get("success", False):
            tea = result.get("tea", {})
            lca = result.get("lca", {})

            msp = tea.get("msp_usd_per_kg") or 0
            tci = tea.get("tci_usd") or 0
            aoc = tea.get("aoc_usd_per_yr") or 0
            gwp = lca.get("gwp_kg_co2e_per_kg") or 0

            # Weight for allocation
            if allocation_method == "value":
                market_val = _POLYMER_MARKET_VALUES.get(polymer.upper(), 1.0)
            else:
                market_val = 1.0  # equal mass weighting

            total_tci += tci
            total_aoc += aoc
            total_gwp_weighted += gwp * market_val
            total_weight += market_val

            per_polymer_results.append({
                "polymer": polymer,
                "solvent": solvent,
                "stage": stage_label,
                "success": True,
                "target_plastic_percent": config["target_plastic_percent"],
                "processing_capacity_mt_yr": config["processing_capacity"],
                "msp_usd_per_kg": msp,
                "tci_usd": tci,
                "aoc_usd_per_yr": aoc,
                "gwp_kg_co2e_per_kg": gwp,
                "weight": market_val,
                "result": result,
            })
        else:
            per_polymer_results.append({
                "polymer": polymer,
                "solvent": solvent,
                "stage": stage_label,
                "success": False,
                "error": result.get("error", "Unknown error"),
                "result": result,
            })

    # Compute blended metrics
    successes = [p for p in per_polymer_results if p.get("success", False)]
    n_ok = len(successes)

    blended_gwp = total_gwp_weighted / total_weight if total_weight > 0 else 0
    blended_msp = sum(p["msp_usd_per_kg"] * p["weight"] for p in successes) / total_weight if total_weight > 0 else 0

    # Build display
    display = f"## Multi-Polymer BioSTEAM Recovery\n\n"
    display += f"**Energy case:** {energy_case} | **Capacity:** {processing_capacity:,.0f} MT/yr\n"
    display += f"**Polymers:** {n_ok}/{len(polymers)} completed | "
    display += f"**Allocation:** {allocation_method}\n\n"

    if successes:
        display += "### Per-Polymer Results\n\n"
        display += "| Stage | Polymer | Solvent | Feed% | Cap (MT/yr) | MSP ($/kg) | TCI ($M) | AOC ($M/yr) | GWP |\n"
        display += "|-------|---------|---------|-------|-------------|------------|----------|-------------|-----|\n"
        for p in successes:
            display += (
                f"| {p.get('stage', '-')} | {p['polymer']} | {p['solvent']} | "
                f"{p.get('target_plastic_percent', '?')} | "
                f"{p.get('processing_capacity_mt_yr', '?'):,.0f} | "
                f"${p['msp_usd_per_kg']:.4f} | "
                f"${p['tci_usd']/1e6:.2f}M | "
                f"${p['aoc_usd_per_yr']/1e6:.2f}M | "
                f"{p['gwp_kg_co2e_per_kg']:.4f} |\n"
            )

        display += f"\n### Combined Metrics\n"
        display += f"- **Blended MSP:** ${blended_msp:.4f}/kg ({allocation_method}-weighted)\n"
        display += f"- **Combined TCI:** ${total_tci/1e6:.2f}M\n"
        display += f"- **Combined AOC:** ${total_aoc/1e6:.2f}M/yr\n"
        display += f"- **Weighted GWP:** {blended_gwp:.4f} kg CO2e/kg\n"

    failures = [p for p in per_polymer_results if not p.get("success", False)]
    if failures:
        display += f"\n### Failed ({len(failures)})\n"
        for f in failures:
            display += f"- {f['polymer']}: {f.get('error', 'unknown')}\n"

    structured_data = {
        "tool_name": "run_biosteam_multi_polymer",
        "success": n_ok > 0,
        "sequence_stages": stage_labels,
        "n_polymers": len(polymers),
        "completed": n_ok,
        "failed": len(failures),
        "energy_case": energy_case,
        "processing_capacity": processing_capacity,
        "allocation_method": allocation_method,
        "blended_msp_usd_per_kg": blended_msp,
        "combined_tci_usd": total_tci,
        "combined_aoc_usd_per_yr": total_aoc,
        "weighted_gwp_kg_co2e_per_kg": blended_gwp,
        "per_polymer": per_polymer_results,
    }

    return json.dumps({"display": display, "data": structured_data}, indent=2)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

def get_biosteam_tools() -> list:
    """Return all BioSTEAM TEA/LCA tools for subagent registration."""
    return [
        run_biosteam_simulation,
        run_biosteam_batch,
        compare_biosteam_scenarios,
        get_biosteam_solvents,
        visualize_biosteam_results,
        run_biosteam_multi_polymer,
    ]
