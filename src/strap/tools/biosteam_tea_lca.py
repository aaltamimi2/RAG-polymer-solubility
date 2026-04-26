"""BioSTEAM TEA/LCA tools for rigorous STRAP process simulation.
Thin agent-facing wrappers around the BioSTEAM subprocess runner
(strap.vendor.biosteam_runner).  Each tool delegates simulation work
to the runner and formats results as JSON strings for the LLM.
"""
from __future__ import annotations
import json
import logging
from typing import Optional
from strap.services.biosteam_service import (
    ALL_ENERGY_CASES as _ALL_ENERGY_CASES,
    CHLORINATED_BLOCKLIST as _CHLORINATED_BLOCKLIST,
    EVOH_SOLVENTS as _EVOH_SOLVENTS,
    EVOH_SOLVENTS_E2 as _EVOH_SOLVENTS_E2,
    LDPE_SOLVENTS as _LDPE_SOLVENTS,
    PC_SOLVENTS as _PC_SOLVENTS,
    PE_SOLVENTS as _PE_SOLVENTS,
    PE_SOLVENTS_CORE as _PE_SOLVENTS_CORE,
    PE_SOLVENTS_EXTENDED as _PE_SOLVENTS_EXTENDED,
    PET_SOLVENTS as _PET_SOLVENTS,
    POLYMER_MARKET_VALUES as _POLYMER_MARKET_VALUES,
    PP_SOLVENTS as _PP_SOLVENTS,
    PS_SOLVENTS as _PS_SOLVENTS,
    PVC_SOLVENTS as _PVC_SOLVENTS,
    SEQUENTIAL_STAGE_DEFAULTS as _SEQUENTIAL_STAGE_DEFAULTS,
    build_manual_batch_configs,
    build_single_config,
    expand_energy_cases as _expand_energy_cases,
    expand_solvents as _expand_solvents,
    extract_successful_results,
    json_tool_error,
    json_tool_response,
    parse_json_array,
    prioritize_batch_solvents,
    runner_unavailable_error,
)
from strap.tools._helpers import normalize_wsl_path, safe_tool_wrapper, save_plot
try:
    from strap.vendor.biosteam_runner import (
        run_single_simulation,
        run_batch_simulations,
        build_batch_configs,
        get_supported_solvents,
        _csv_lookup,
        _get_default_parameter_ranges,
        _build_monte_carlo_configs,
        _build_sweep_configs,
        _build_tornado_configs,
    )
except ImportError:
    run_single_simulation = None
    run_batch_simulations = None
    build_batch_configs = None
    get_supported_solvents = None
    _csv_lookup = None
    _get_default_parameter_ranges = None
    _build_monte_carlo_configs = None
    _build_sweep_configs = None
    _build_tornado_configs = None
logger = logging.getLogger(__name__)

_LARGE_BATCH_CONFIG_THRESHOLD = 12
_LARGE_BATCH_PER_SIM_TIMEOUT_S = 45
_LARGE_BATCH_WALL_BUDGET_S = 95
_LARGE_BATCH_MIN_SUCCESS_TARGET = 5


def _temperature_margin_metadata(solvent: str, dissolution_temp_c: float | None) -> dict:
    """Return atmospheric boiling-point margin metadata for a BioSTEAM case."""
    if dissolution_temp_c is None:
        return {}
    try:
        dissolution_temp = float(dissolution_temp_c)
    except (TypeError, ValueError):
        return {}

    metadata: dict = {"dissolution_temp_c": dissolution_temp}
    if _csv_lookup is None:
        return metadata
    try:
        solvent_data = _csv_lookup(solvent) or {}
    except Exception:
        solvent_data = {}
    bp_c = solvent_data.get("bp_c") if isinstance(solvent_data, dict) else None
    try:
        boiling_point_c = float(bp_c) if bp_c is not None else None
    except (TypeError, ValueError):
        boiling_point_c = None
    if boiling_point_c is None:
        return metadata

    margin_c = boiling_point_c - dissolution_temp
    metadata.update(
        {
            "boiling_point_c": boiling_point_c,
            "boiling_margin_c": margin_c,
            "operates_below_normal_boiling_point": margin_c > 0,
            "near_boiling_point": 0 <= margin_c < 5,
            "requires_pressurization_at_1atm": margin_c <= 0,
        }
    )
    return metadata


def _format_temperature_margin(metadata: dict) -> str:
    temp = metadata.get("dissolution_temp_c")
    if temp is None:
        return ""
    text = f"**Dissolution temperature:** {float(temp):.1f} C"
    bp = metadata.get("boiling_point_c")
    margin = metadata.get("boiling_margin_c")
    if bp is None or margin is None:
        return text
    text += f" | **Normal BP:** {float(bp):.1f} C | **1 atm margin:** {float(margin):.1f} C"
    if metadata.get("requires_pressurization_at_1atm"):
        text += (
            "\n\n**Feasibility caveat:** this setpoint is at or above the normal "
            "boiling point, so atmospheric operation is not valid without "
            "pressurization or a changed setpoint."
        )
    elif metadata.get("near_boiling_point"):
        text += (
            "\n\n**Feasibility caveat:** this is a narrow atmospheric-pressure "
            "margin and needs tight temperature control or pressure-rated operation."
        )
    return text
# ------------------------------------------------------------------
# Tool 1: Single simulation
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
    Accepts ANY thermosteam-resolvable solvent.  Known solvents use validated
    Branch-TEA data; unknown solvents get estimated defaults (price $1.50/kg,
    dissolution temp from Solvent_Data.csv, LCA class-average impact factors).
    Args:
        solvent: Solvent name (e.g. "Toluene", "Xylene", "Heptane").
            Accepts any solvent that thermosteam can resolve by name or CAS.
        target_plastic: Target plastic to recover — "PE", "LDPE", "EVOH", "PET",
            "PS", "PP", "PVC", or "PC" (default "PE").
            LDPE uses the same solvents as PE. PET uses the same generic dissolution model
            with dynamic parameter registration. PS, PP, and PVC use the PE process model
            as an approximation (PE-proxy) — results are order-of-magnitude estimates.
            PC uses the native thermosteam PColigomer.
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
        return runner_unavailable_error()
    config = build_single_config(
        solvent=solvent,
        target_plastic=target_plastic,
        target_plastic_percent=target_plastic_percent,
        processing_capacity=processing_capacity,
        energy_case=energy_case,
        dissolution_temp_c=dissolution_temp_c,
        precipitation_temp_c=precipitation_temp_c,
        solvent_price=solvent_price,
    )
    result = run_single_simulation(config)
    # Format display output
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown simulation error")
        result.setdefault("success", False)
        result.setdefault("error", error_msg)
        return json_tool_response(
            f"**Simulation failed** for {solvent} ({target_plastic}, {energy_case}): {error_msg}",
            result,
        )
    tea = result.get("tea", {})
    lca = result.get("lca", {})
    ops = result.get("operations", {})
    process_conditions = _temperature_margin_metadata(solvent, dissolution_temp_c)
    if process_conditions:
        result["process_conditions"] = process_conditions
    display = f"## BioSTEAM Simulation: {solvent} ({target_plastic}, {energy_case})\n\n"
    display += f"**Capacity:** {processing_capacity:,.0f} MT/yr | "
    display += f"**Feed:** {target_plastic_percent:.0f}% {target_plastic}\n\n"
    temperature_line = _format_temperature_margin(process_conditions)
    if temperature_line:
        display += temperature_line + "\n\n"
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
    return json_tool_response(display, result)
# ------------------------------------------------------------------
# Tool 2: Batch simulation
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
    Accepts ANY thermosteam-resolvable solvent names.  Unknown solvents get
    estimated defaults (price $1.50/kg, dissolution temp from Solvent_Data.csv,
    LCA class-average impact factors).
    Args:
        solvents: Comma-separated solvent names, or "all_pe" (PE solvents),
            "all_ldpe" (LDPE solvents, same as PE), "all_evoh" (EVOH solvents),
            "all_pet" (PET solvents), "all_ps" (PS solvents, PE-proxy),
            "all_pp" (PP solvents, PE-proxy), "all_pvc" (PVC solvents, PE-proxy),
            "all_pc" (PC solvents, native), or "all" (auto-select by target_plastic).
            Any other solvent name passes through to BioSTEAM for validation.
        target_plastic: Target plastic "PE", "LDPE", "EVOH", "PET", "PS", "PP",
            "PVC", or "PC" (default "PE"). PS/PP/PVC use the PE-proxy model
            (approximate). PC uses the native thermosteam PColigomer.
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
        return runner_unavailable_error()
    solvent_list = _expand_solvents(solvents, target_plastic)
    case_list = _expand_energy_cases(energy_cases)
    if not solvent_list:
        return json_tool_error(
            "No solvents specified. Provide comma-separated names or 'all_pe'/'all_evoh'.",
            tool_name="run_biosteam_batch",
        )
    if not case_list:
        return json_tool_error(
            "No energy cases specified. Provide 'C1', 'C2', 'C3', or 'all'.",
            tool_name="run_biosteam_batch",
        )
    solvent_list = prioritize_batch_solvents(solvent_list, target_plastic)
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
        configs = build_manual_batch_configs(
            solvents=solvent_list,
            energy_cases=case_list,
            target_plastic=target_plastic,
            target_plastic_percent=target_plastic_percent,
            processing_capacity=processing_capacity,
        )
    # Run all simulations
    partial_budget_note: str | None = None
    unattempted: list[dict[str, str]] = []
    if (
        run_batch_simulations is not None
        and len(configs) > _LARGE_BATCH_CONFIG_THRESHOLD
    ):
        # Large screening batches can otherwise exceed the end-to-end route timeout.
        # Run prioritized chunks and return the best completed subset within a fixed budget.
        # Publication-grade requirement: do not exit a partial batch before surfacing at
        # least a decision-quality minimum of successful scenarios unless the whole screen
        # has been exhausted.
        import time

        results = []
        workers = min(max_parallel, 4)
        start = time.monotonic()
        successes_so_far = 0
        for offset in range(0, len(configs), workers):
            elapsed = time.monotonic() - start
            over_budget = elapsed >= _LARGE_BATCH_WALL_BUDGET_S
            if over_budget and successes_so_far >= _LARGE_BATCH_MIN_SUCCESS_TARGET:
                break
            remaining_budget = _LARGE_BATCH_WALL_BUDGET_S - elapsed
            if over_budget:
                timeout_per_sim = 15
            else:
                timeout_per_sim = min(
                    _LARGE_BATCH_PER_SIM_TIMEOUT_S,
                    max(15, int(remaining_budget)),
                )
            chunk = configs[offset:offset + workers]
            results.extend(
                run_batch_simulations(
                    chunk,
                    max_parallel=workers,
                    timeout_per_sim=timeout_per_sim,
                )
            )
            successes_so_far = sum(1 for r in results if r.get("success", False))
            if successes_so_far >= _LARGE_BATCH_MIN_SUCCESS_TARGET and (
                over_budget
                or (time.monotonic() - start) >= (_LARGE_BATCH_WALL_BUDGET_S * 0.6)
            ):
                break

        attempted = len(results)
        if attempted < len(configs):
            unattempted = [
                {
                    "solvent": cfg.get("solvent", "?"),
                    "energy_case": cfg.get("energy_case", "?"),
                    "error": "Not executed within batch wall-clock budget",
                }
                for cfg in configs[attempted:]
            ]
            partial_budget_note = (
                f"Screened {attempted}/{len(configs)} cases within "
                f"{_LARGE_BATCH_WALL_BUDGET_S}s wall-clock budget."
            )
    elif run_batch_simulations is not None:
        results = run_batch_simulations(configs, max_parallel=max_parallel)
    else:
        # Fallback: sequential execution
        results = []
        for cfg in configs:
            results.append(run_single_simulation(cfg))
    # Separate successes and failures
    successes = [r for r in results if r.get("success", False)]
    failures = [r for r in results if not r.get("success", False)]
    failures.extend(unattempted)
    # Build display output
    display = f"## BioSTEAM Batch Results: {target_plastic}\n\n"
    display += f"**Solvents:** {', '.join(solvent_list)} | "
    display += f"**Energy cases:** {', '.join(case_list)} | "
    display += f"**Capacity:** {processing_capacity:,.0f} MT/yr\n\n"
    display += f"**Completed:** {len(successes)}/{len(configs)} simulations\n\n"
    if partial_budget_note:
        display += f"**Batch note:** {partial_budget_note}\n\n"
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
        "attempted": len(results),
        "partial": partial_budget_note is not None,
        "batch_note": partial_budget_note,
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
    return json_tool_response(display, structured_data)
# ------------------------------------------------------------------
# Tool 3: Scenario comparison
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
        return runner_unavailable_error()
    try:
        scenarios = parse_json_array(scenarios_json, field_name="scenarios_json")
    except ValueError as exc:
        return json_tool_error(str(exc), tool_name="compare_biosteam_scenarios")
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
        config = build_single_config(
            solvent=sc["solvent"],
            target_plastic=sc.get("target_plastic", "PE"),
            target_plastic_percent=sc.get("target_plastic_percent", 60),
            processing_capacity=sc.get("processing_capacity", 20000),
            energy_case=sc.get("energy_case", "C1"),
            dissolution_temp_c=sc.get("dissolution_temp_c"),
            precipitation_temp_c=sc.get("precipitation_temp_c", 25),
            solvent_price=sc.get("solvent_price"),
        )
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
    return json_tool_response(display, structured_data)
# ------------------------------------------------------------------
# Tool 4: List supported solvents and configurations
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
    display += "Candidate solvent lists are loaded from `data/60_common_solvents-TEA-LCA.csv`.\n"
    display += "For `all_*` batch keywords, STRAP uses the CSV-derived high-temperature candidate sets.\n\n"
    display += f"### PE Solvents ({len(_PE_SOLVENTS)})\n"
    display += ", ".join(_PE_SOLVENTS) + "\n"
    display += f"\n### LDPE Solvents ({len(_LDPE_SOLVENTS)})\n"
    display += ", ".join(_LDPE_SOLVENTS) + "\n"
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
    display += f"\n### PS Solvents — PE-proxy approximation ({len(_PS_SOLVENTS)})\n"
    for sv in _PS_SOLVENTS:
        display += f"- {sv}\n"
    display += "\n*PS (polystyrene) uses the PE process model as an approximation — economics/LCA are order-of-magnitude estimates.*\n"
    display += f"\n### PP Solvents — PE-proxy approximation ({len(_PP_SOLVENTS)})\n"
    for sv in _PP_SOLVENTS:
        display += f"- {sv}\n"
    display += "\n*PP (polypropylene) uses the PE process model as an approximation — economics/LCA are order-of-magnitude estimates.*\n"
    display += f"\n### PVC Solvents — PE-proxy approximation ({len(_PVC_SOLVENTS)})\n"
    for sv in _PVC_SOLVENTS:
        display += f"- {sv}\n"
    display += "\n*PVC (poly(vinyl chloride)) uses the PE process model as an approximation — economics/LCA are order-of-magnitude estimates.*\n"
    display += f"\n### PC Solvents — native thermosteam support ({len(_PC_SOLVENTS)})\n"
    for sv in _PC_SOLVENTS:
        display += f"- {sv}\n"
    display += "\n*PC (polycarbonate) uses the native PColigomer in thermosteam. Dichloromethane is on the chlorinated blocklist and may be filtered in batch runs.*\n"
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
    display += f"- `all_ps` -- all {len(_PS_SOLVENTS)} PS solvents (PE-proxy, approx.)\n"
    display += f"- `all_pp` -- all {len(_PP_SOLVENTS)} PP solvents (PE-proxy, approx.)\n"
    display += f"- `all_pvc` -- all {len(_PVC_SOLVENTS)} PVC solvents (PE-proxy, approx.)\n"
    display += f"- `all_pc` -- all {len(_PC_SOLVENTS)} PC solvents (native thermosteam)\n"
    display += "- `all` (energy_cases) -- C1, C2, C3\n"
    display += "\n### Any-Solvent Support\n"
    display += "The BioSTEAM tools accept **any** thermosteam-resolvable solvent, "
    display += "not just those in the predefined lists above. For unknown solvents:\n"
    display += "- **Price:** defaults to $1.50/kg\n"
    display += "- **Dissolution temp:** min(BP - 10, 130) from Solvent_Data.csv, or 110 C\n"
    display += "- **LCA factors:** class-average estimates based on chemical class\n"
    display += "\nThe predefined solvents use validated Branch-TEA data. "
    display += "Use other solvents for exploratory studies with estimated parameters.\n"
    display += "\n### Known Limitations\n"
    display += "- Chlorinated solvents (Tetrachloroethylene, o-Chlorotoluene) may fail — HCl not in property package\n"
    display += "- Simulations run as isolated subprocesses (~10-15s each)\n"
    display += "- LCA characterisation factors use TRACI defaults unless overridden\n"
    display += "- Unknown solvents use estimated LCA class-averages (order-of-magnitude accuracy)\n"
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
        # New polymers
        "ps_solvents": list(_PS_SOLVENTS),    # PE-proxy approximation
        "pp_solvents": list(_PP_SOLVENTS),    # PE-proxy approximation
        "pvc_solvents": list(_PVC_SOLVENTS),  # PE-proxy approximation
        "pc_solvents": list(_PC_SOLVENTS),    # native thermosteam PColigomer
        "energy_cases": list(_ALL_ENERGY_CASES),
        "chlorinated_blocklist": list(_CHLORINATED_BLOCKLIST),
        "dynamic_solvents": dynamic_solvents,
    }
    return json_tool_response(display, structured_data)
# ------------------------------------------------------------------
# Tool 5: Visualize BioSTEAM results
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
    output_dir = normalize_wsl_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    try:
        data = json.loads(results_json) if isinstance(results_json, str) else results_json
    except json.JSONDecodeError as e:
        return json_tool_error(f"Invalid JSON: {e}", tool_name="visualize_biosteam_results")
    requested = set(t.strip() for t in chart_types.split(","))
    if "all" in requested:
        requested = {"cost_breakdown", "gwp_breakdown", "scenario_comparison"}
    saved_files = []
    # Extract results list from various tool output formats
    results_list = extract_successful_results(data)
    if not results_list:
        return json_tool_error(
            "No successful simulation results found",
            tool_name="visualize_biosteam_results",
        )
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
        path = save_plot(fig, "biosteam_cost_breakdown", output_dir=output_dir, dpi=200)
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
        path = save_plot(fig, "biosteam_gwp_breakdown", output_dir=output_dir, dpi=200)
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
        path = save_plot(fig, "biosteam_scenario_comparison", output_dir=output_dir, dpi=200)
        saved_files.append(path)
    display = f"Generated {len(saved_files)} chart(s):\n"
    for p in saved_files:
        display += f"- {p}\n"
    return json_tool_response(display, {"success": True, "charts": saved_files})
# ------------------------------------------------------------------
# Tool 6: Multi-polymer sequential recovery
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
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
        return runner_unavailable_error()
    try:
        polymers = parse_json_array(polymers_json, field_name="polymers_json")
    except ValueError as exc:
        return json_tool_error(str(exc), tool_name="run_biosteam_multi_polymer")
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
    # --- Pass 1: validate specs and build configs (keep per-spec metadata) ---
    # Each entry in pending_items is either a ready-to-run dict or a pre-built
    # error record.  We process them in two lists so that invalid specs are
    # handled immediately while valid ones are batched.
    pending_items: list[dict] = []   # holds metadata + config for valid specs
    early_errors: list[tuple[int, dict]] = []  # (original_index, error_record)
    for i, spec in enumerate(polymers):
        polymer = spec.get("polymer")
        solvent = spec.get("solvent")
        if not polymer or not solvent:
            early_errors.append((i, {
                "polymer": polymer or f"spec_{i}",
                "success": False,
                "error": "Missing required 'polymer' or 'solvent' field",
            }))
            continue
        # Look up stage defaults when sequence_stages provided
        stage_defaults = {}
        stage_label = ""
        if stage_labels and i < len(stage_labels):
            stage_label = stage_labels[i]
            stage_defaults = _SEQUENTIAL_STAGE_DEFAULTS.get(stage_label, {})
        # Priority: per-spec override > stage default > function arg / fallback
        config = build_single_config(
            solvent=solvent,
            target_plastic=stage_defaults.get("target_plastic", polymer),
            target_plastic_percent=spec.get(
                "target_plastic_percent",
                stage_defaults.get("target_plastic_percent", 60),
            ),
            processing_capacity=spec.get(
                "processing_capacity",
                stage_defaults.get("processing_capacity", processing_capacity),
            ),
            energy_case=energy_case,
            dissolution_temp_c=spec.get("dissolution_temp_c"),
            precipitation_temp_c=spec.get("precipitation_temp_c", 25),
        )
        pending_items.append({
            "original_index": i,
            "polymer": polymer,
            "solvent": solvent,
            "stage_label": stage_label,
            "config": config,
        })
    # --- Pass 2: run all valid configs in parallel via ThreadPoolExecutor ---
    valid_configs = [item["config"] for item in pending_items]
    batch_results = run_batch_simulations(valid_configs, max_parallel=3)
    # --- Pass 3: merge batch results with per-spec metadata ---
    # Reconstruct per_polymer_results in original spec order.
    # Build a mapping from original index -> error record for early errors.
    error_by_index = {idx: rec for idx, rec in early_errors}
    # Build a mapping from original index -> (item metadata, raw result).
    result_by_index: dict[int, tuple[dict, dict]] = {}
    for item, raw_result in zip(pending_items, batch_results):
        result_by_index[item["original_index"]] = (item, raw_result)
    for i in range(len(polymers)):
        if i in error_by_index:
            per_polymer_results.append(error_by_index[i])
            continue
        item, result = result_by_index[i]
        polymer = item["polymer"]
        solvent = item["solvent"]
        stage_label = item["stage_label"]
        config = item["config"]
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
    return json_tool_response(display, structured_data)
# ------------------------------------------------------------------
# Tool 7: Monte Carlo uncertainty analysis
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
def run_biosteam_uncertainty(
    solvent: str,
    target_plastic: str = "PE",
    energy_case: str = "C1",
    n_samples: int = 20,
    processing_capacity: float = 20000,
    parameters: str = "all",
) -> str:
    """Run Monte Carlo uncertainty analysis on BioSTEAM STRAP process simulation.
    Samples N parameter sets from uniform distributions over physically
    meaningful ranges, runs each as a subprocess simulation, and reports
    percentile statistics (P5, P50, P95) for MSP, GWP, and TCI.
    Args:
        solvent: Solvent name (e.g. "Toluene").
        target_plastic: Target plastic "PE", "LDPE", "EVOH", "PET", etc. (default "PE").
        energy_case: Energy configuration "C1", "C2", or "C3" (default "C1").
        n_samples: Number of Monte Carlo samples (default 20, max 50).
        processing_capacity: Plant capacity in MT/yr (default 20000).
        parameters: Comma-separated parameter names to vary, or "all" (default "all").
            Available: solvent_price, solvent_loss_pct, dissolution_temperature_c,
            precipitation_temperature_c, feedstock_distance_km.
    WHEN TO USE:
    - "What is the uncertainty in MSP for toluene?"
    - "Monte Carlo analysis for PE recovery with xylene"
    - "Confidence interval for GWP using heptane"
    - "How confident are we in the BioSTEAM MSP estimate?"
    """
    if _build_monte_carlo_configs is None or run_batch_simulations is None:
        return runner_unavailable_error()
    import numpy as np
    configs = _build_monte_carlo_configs(
        solvent=solvent,
        target_plastic=target_plastic,
        energy_case=energy_case,
        n_samples=n_samples,
        parameters=parameters,
        processing_capacity=processing_capacity,
    )
    results = run_batch_simulations(configs, max_parallel=3)
    successes = [r for r in results if r.get("success", False)]
    failures = [r for r in results if not r.get("success", False)]
    if not successes:
        return json_tool_response(
            f"**Uncertainty analysis failed**: all {len(results)} simulations failed.\n"
            f"First error: {failures[0].get('error', 'unknown') if failures else 'unknown'}",
            {"success": False, "n_total": len(results), "n_failed": len(failures)},
        )
    # Extract metrics
    msp_vals = [r["tea"]["msp_usd_per_kg"] for r in successes if r.get("tea", {}).get("msp_usd_per_kg") is not None]
    gwp_vals = [r["lca"]["gwp_kg_co2e_per_kg"] for r in successes if r.get("lca", {}).get("gwp_kg_co2e_per_kg") is not None]
    tci_vals = [r["tea"]["tci_usd"] for r in successes if r.get("tea", {}).get("tci_usd") is not None]
    def _percentiles(vals):
        if not vals:
            return {"p5": None, "p50": None, "p95": None, "mean": None, "std": None}
        arr = np.array(vals)
        p5, p50, p95 = np.percentile(arr, [5, 50, 95])
        return {
            "p5": round(float(p5), 4),
            "p50": round(float(p50), 4),
            "p95": round(float(p95), 4),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
        }
    msp_stats = _percentiles(msp_vals)
    gwp_stats = _percentiles(gwp_vals)
    tci_stats = _percentiles(tci_vals)
    # Get parameter ranges used
    ranges = _get_default_parameter_ranges(solvent, target_plastic)
    if parameters != "all":
        param_list = [p.strip() for p in parameters.split(",") if p.strip()]
        ranges = {k: v for k, v in ranges.items() if k in param_list}
    # Build display
    display = f"## Monte Carlo Uncertainty: {solvent} ({target_plastic}, {energy_case})\n\n"
    display += f"**Samples:** {len(successes)}/{len(results)} succeeded | "
    display += f"**Capacity:** {processing_capacity:,.0f} MT/yr\n\n"
    display += "### Percentile Statistics\n\n"
    display += "| Metric | P5 | P50 (median) | P95 | Mean | Std Dev |\n"
    display += "|--------|----|--------------|-----|------|---------|\n"
    if msp_stats["p50"] is not None:
        display += (f"| MSP ($/kg) | ${msp_stats['p5']:.4f} | ${msp_stats['p50']:.4f} | "
                    f"${msp_stats['p95']:.4f} | ${msp_stats['mean']:.4f} | {msp_stats['std']:.4f} |\n")
    if gwp_stats["p50"] is not None:
        display += (f"| GWP (kg CO2e/kg) | {gwp_stats['p5']:.4f} | {gwp_stats['p50']:.4f} | "
                    f"{gwp_stats['p95']:.4f} | {gwp_stats['mean']:.4f} | {gwp_stats['std']:.4f} |\n")
    if tci_stats["p50"] is not None:
        display += (f"| TCI ($M) | ${tci_stats['p5']/1e6:.2f} | ${tci_stats['p50']/1e6:.2f} | "
                    f"${tci_stats['p95']/1e6:.2f} | ${tci_stats['mean']/1e6:.2f} | {tci_stats['std']/1e6:.2f} |\n")
    display += "\n### Parameter Ranges Sampled\n\n"
    display += "| Parameter | Min | Max | Units |\n"
    display += "|-----------|-----|-----|-------|\n"
    units = {
        "solvent_price": "$/kg", "solvent_loss_pct": "%",
        "dissolution_temperature_c": "C", "precipitation_temperature_c": "C",
        "feedstock_distance_km": "km",
    }
    for p, (lo, hi) in ranges.items():
        display += f"| {p} | {lo} | {hi} | {units.get(p, '-')} |\n"
    display += "\n### Interpretation\n"
    if msp_stats["p50"] is not None and msp_stats["p5"] is not None:
        spread = msp_stats["p95"] - msp_stats["p5"]
        pct_spread = spread / msp_stats["p50"] * 100 if msp_stats["p50"] else 0
        display += f"- MSP 90% confidence interval: ${msp_stats['p5']:.4f} – ${msp_stats['p95']:.4f} "
        display += f"(spread: ${spread:.4f}, {pct_spread:.1f}% of median)\n"
    if gwp_stats["p50"] is not None and gwp_stats["p5"] is not None:
        display += f"- GWP 90% confidence interval: {gwp_stats['p5']:.4f} – {gwp_stats['p95']:.4f} kg CO2e/kg\n"
    if failures:
        display += f"\n*{len(failures)} simulation(s) failed and were excluded.*\n"
    structured_data = {
        "tool_name": "run_biosteam_uncertainty",
        "success": True,
        "solvent": solvent,
        "target_plastic": target_plastic,
        "energy_case": energy_case,
        "n_samples": len(results),
        "n_succeeded": len(successes),
        "n_failed": len(failures),
        "msp_percentiles": msp_stats,
        "gwp_percentiles": gwp_stats,
        "tci_percentiles": tci_stats,
        "parameter_ranges": {k: {"min": v[0], "max": v[1]} for k, v in ranges.items()},
        "msp_values": msp_vals,
        "gwp_values": gwp_vals,
        "tci_values": [t / 1e6 for t in tci_vals],
    }
    return json_tool_response(display, structured_data)
# ------------------------------------------------------------------
# Tool 8: Parameter sweep
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
def run_biosteam_parameter_sweep(
    solvent: str,
    target_plastic: str = "PE",
    energy_case: str = "C1",
    parameter: str = "solvent_price",
    values: str = "",
    processing_capacity: float = 20000,
) -> str:
    """Sweep a single BioSTEAM parameter across a range and report MSP/GWP/TCI trend.
    Runs one simulation per value (default 5 evenly spaced) and returns a
    table showing how the chosen metric changes with the swept parameter.
    Args:
        solvent: Solvent name (e.g. "Toluene").
        target_plastic: Target plastic (default "PE").
        energy_case: Energy configuration (default "C1").
        parameter: Which parameter to sweep. Options: solvent_price,
            solvent_loss_pct, dissolution_temperature_c,
            precipitation_temperature_c, feedstock_distance_km.
        values: Comma-separated numeric values to sweep, or empty for auto 5-point range.
        processing_capacity: Plant capacity in MT/yr (default 20000).
    WHEN TO USE:
    - "How does MSP change with solvent price for toluene?"
    - "Sweep dissolution temperature from 90 to 130 for xylene"
    - "Parameter sweep of solvent loss for heptane"
    - "What happens to GWP if feedstock distance increases?"
    """
    if _build_sweep_configs is None or run_batch_simulations is None:
        return runner_unavailable_error()
    # Parse values string to list of floats
    parsed_values: list[float] | None = None
    if values and values.strip():
        try:
            parsed_values = [float(v.strip()) for v in values.split(",") if v.strip()]
        except ValueError:
            return json_tool_error(
                f"Could not parse values '{values}' as comma-separated numbers.",
                tool_name="run_biosteam_parameter_sweep",
            )
    configs = _build_sweep_configs(
        solvent=solvent,
        target_plastic=target_plastic,
        energy_case=energy_case,
        parameter=parameter,
        values=parsed_values,
        processing_capacity=processing_capacity,
    )
    results = run_batch_simulations(configs, max_parallel=3)
    # Pair configs with results and sort by swept value
    paired = list(zip(configs, results))
    paired.sort(key=lambda x: x[0].get("_sweep_value", 0))
    successes = [(c, r) for c, r in paired if r.get("success", False)]
    failures = [(c, r) for c, r in paired if not r.get("success", False)]
    if not successes:
        return json_tool_response(
            f"**Parameter sweep failed**: all {len(results)} simulations failed.",
            {"success": False},
        )
    units = {
        "solvent_price": "$/kg", "solvent_loss_pct": "%",
        "dissolution_temperature_c": "C", "precipitation_temperature_c": "C",
        "feedstock_distance_km": "km",
    }
    display = f"## Parameter Sweep: {parameter} for {solvent} ({target_plastic}, {energy_case})\n\n"
    display += f"**Capacity:** {processing_capacity:,.0f} MT/yr | "
    display += f"**Completed:** {len(successes)}/{len(paired)}\n\n"
    display += f"| {parameter} ({units.get(parameter, '-')}) | MSP ($/kg) | GWP (kg CO2e/kg) | TCI ($M) |\n"
    display += "|" + "-" * 30 + "|------------|------------------|----------|\n"
    sweep_data = []
    for cfg, r in successes:
        val = cfg.get("_sweep_value", 0)
        tea = r.get("tea", {})
        lca = r.get("lca", {})
        msp = tea.get("msp_usd_per_kg")
        gwp = lca.get("gwp_kg_co2e_per_kg")
        tci = tea.get("tci_usd")
        msp_str = f"${msp:.4f}" if msp is not None else "N/A"
        gwp_str = f"{gwp:.4f}" if gwp is not None else "N/A"
        tci_str = f"${tci/1e6:.2f}M" if tci is not None else "N/A"
        display += f"| {val:.4g} | {msp_str} | {gwp_str} | {tci_str} |\n"
        sweep_data.append({
            "parameter_value": val,
            "msp_usd_per_kg": msp,
            "gwp_kg_co2e_per_kg": gwp,
            "tci_usd": tci,
        })
    # Trend description
    if len(successes) >= 2:
        first_msp = successes[0][1].get("tea", {}).get("msp_usd_per_kg")
        last_msp = successes[-1][1].get("tea", {}).get("msp_usd_per_kg")
        if first_msp is not None and last_msp is not None:
            delta = last_msp - first_msp
            direction = "increases" if delta > 0 else "decreases"
            display += f"\n**Trend:** MSP {direction} by ${abs(delta):.4f}/kg "
            display += f"across the swept range of {parameter}.\n"
    if failures:
        display += f"\n*{len(failures)} value(s) failed and were excluded.*\n"
    structured_data = {
        "tool_name": "run_biosteam_parameter_sweep",
        "success": True,
        "solvent": solvent,
        "target_plastic": target_plastic,
        "energy_case": energy_case,
        "parameter": parameter,
        "n_values": len(paired),
        "n_succeeded": len(successes),
        "sweep_data": sweep_data,
    }
    return json_tool_response(display, structured_data)
# ------------------------------------------------------------------
# Tool 9: Tornado sensitivity analysis
# ------------------------------------------------------------------
@safe_tool_wrapper(structured_output=True)
def run_biosteam_tornado(
    solvent: str,
    target_plastic: str = "PE",
    energy_case: str = "C1",
    metric: str = "msp",
    parameters: str = "all",
    processing_capacity: float = 20000,
) -> str:
    """Run one-at-a-time (OAT) tornado sensitivity analysis for BioSTEAM STRAP process.
    Varies each parameter to its min and max while holding others at baseline,
    then ranks parameters by their impact on the chosen metric. Identifies
    which parameters drive economics or environmental impact the most.
    Args:
        solvent: Solvent name (e.g. "Toluene").
        target_plastic: Target plastic (default "PE").
        energy_case: Energy configuration (default "C1").
        metric: Metric to analyze — "msp", "gwp", or "tci" (default "msp").
        parameters: Comma-separated parameter names or "all" (default "all").
            Available: solvent_price, solvent_loss_pct, dissolution_temperature_c,
            precipitation_temperature_c, feedstock_distance_km.
        processing_capacity: Plant capacity in MT/yr (default 20000).
    WHEN TO USE:
    - "Which parameter drives MSP the most for toluene?"
    - "Tornado analysis for PE recovery with xylene"
    - "Sensitivity ranking for GWP with heptane"
    - "What drives the economics of the STRAP process?"
    """
    if _build_tornado_configs is None or run_batch_simulations is None:
        return runner_unavailable_error()
    baseline_cfg, oat_configs = _build_tornado_configs(
        solvent=solvent,
        target_plastic=target_plastic,
        energy_case=energy_case,
        parameters=parameters,
        processing_capacity=processing_capacity,
    )
    # Run baseline + all OAT configs
    all_configs = [baseline_cfg] + oat_configs
    results = run_batch_simulations(all_configs, max_parallel=3)
    baseline_result = results[0]
    oat_results = results[1:]
    if not baseline_result.get("success", False):
        return json_tool_response(
            f"**Tornado analysis failed**: baseline simulation failed.\n"
            f"Error: {baseline_result.get('error', 'unknown')}",
            {"success": False},
        )
    # Extract baseline metric value
    metric_map = {
        "msp": ("tea", "msp_usd_per_kg", "$/kg"),
        "gwp": ("lca", "gwp_kg_co2e_per_kg", "kg CO2e/kg"),
        "tci": ("tea", "tci_usd", "$"),
    }
    if metric not in metric_map:
        metric = "msp"
    section, key, unit = metric_map[metric]
    baseline_val = baseline_result.get(section, {}).get(key)
    if baseline_val is None:
        return json_tool_response(
            f"**Tornado analysis failed**: baseline has no {metric.upper()} value.",
            {"success": False},
        )
    # Match OAT min/max pairs by _tornado_param
    param_impacts: dict[str, dict] = {}
    for cfg, r in zip(oat_configs, oat_results):
        param = cfg.get("_tornado_param", "unknown")
        bound = cfg.get("_tornado_bound", "unknown")
        if not r.get("success", False):
            continue
        val = r.get(section, {}).get(key)
        if val is None:
            continue
        if param not in param_impacts:
            param_impacts[param] = {}
        param_impacts[param][bound] = val
    # Compute range and rank
    rankings = []
    for param, bounds in param_impacts.items():
        min_val = bounds.get("min")
        max_val = bounds.get("max")
        if min_val is not None and max_val is not None:
            range_val = abs(max_val - min_val)
            impact_pct = (range_val / abs(baseline_val)) * 100 if baseline_val != 0 else 0
            rankings.append({
                "parameter": param,
                "min_value": min_val,
                "max_value": max_val,
                "range": range_val,
                "impact_pct": round(impact_pct, 2),
                "baseline": baseline_val,
            })
    rankings.sort(key=lambda x: x["range"], reverse=True)
    # Get parameter ranges for display
    ranges = _get_default_parameter_ranges(solvent, target_plastic)
    # Build display
    display = f"## Tornado Sensitivity: {metric.upper()} for {solvent} ({target_plastic}, {energy_case})\n\n"
    if metric == "tci":
        display += f"**Baseline {metric.upper()}:** ${baseline_val/1e6:.2f}M | "
    else:
        display += f"**Baseline {metric.upper()}:** {baseline_val:.4f} {unit} | "
    display += f"**Capacity:** {processing_capacity:,.0f} MT/yr\n\n"
    display += "### Sensitivity Ranking (by impact on " + metric.upper() + ")\n\n"
    display += "| Rank | Parameter | Min Value | Max Value | Range | Impact (%) |\n"
    display += "|------|-----------|-----------|-----------|-------|------------|\n"
    for i, r in enumerate(rankings, 1):
        if metric == "tci":
            display += (f"| {i} | {r['parameter']} | ${r['min_value']/1e6:.2f}M | "
                        f"${r['max_value']/1e6:.2f}M | ${r['range']/1e6:.2f}M | "
                        f"{r['impact_pct']:.1f}% |\n")
        else:
            display += (f"| {i} | {r['parameter']} | {r['min_value']:.4f} | "
                        f"{r['max_value']:.4f} | {r['range']:.4f} | "
                        f"{r['impact_pct']:.1f}% |\n")
    display += "\n### Parameter Ranges Used\n\n"
    display += "| Parameter | Min | Max |\n|-----------|-----|-----|\n"
    for param in [r["parameter"] for r in rankings]:
        if param in ranges:
            lo, hi = ranges[param]
            display += f"| {param} | {lo} | {hi} |\n"
    if rankings:
        top = rankings[0]
        display += f"\n**Most influential parameter:** {top['parameter']} "
        display += f"({top['impact_pct']:.1f}% impact on {metric.upper()})\n"
    n_failed_oat = sum(1 for r in oat_results if not r.get("success", False))
    if n_failed_oat:
        display += f"\n*{n_failed_oat} OAT simulation(s) failed and were excluded.*\n"
    structured_data = {
        "tool_name": "run_biosteam_tornado",
        "success": True,
        "solvent": solvent,
        "target_plastic": target_plastic,
        "energy_case": energy_case,
        "metric": metric,
        "baseline_value": baseline_val,
        "n_parameters": len(rankings),
        "n_simulations": len(all_configs),
        "n_failed": n_failed_oat,
        "rankings": rankings,
        "parameter_ranges": {k: {"min": v[0], "max": v[1]} for k, v in ranges.items()},
    }
    return json_tool_response(display, structured_data)
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
        run_biosteam_uncertainty,
        run_biosteam_parameter_sweep,
        run_biosteam_tornado,
    ]
