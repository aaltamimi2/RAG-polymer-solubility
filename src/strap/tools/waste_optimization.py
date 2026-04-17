"""
Multi-layer Plastic Waste Optimization Tool.
Integrates BioSTEAM simulations with Pyomo superstructure optimization.
"""
import os
import json
import logging
import pandas as pd
from pathlib import Path

from strap.tools._helpers import safe_tool_wrapper
from strap.services.biosteam_service import json_tool_response, json_tool_error, build_single_config
from strap.vendor.biosteam_runner import run_single_simulation
from strap.waste_management.data_loader import load_all_data, STRAP_UNIT_COLS
from strap.waste_management.model import build_model
from strap.waste_management.solver import solve_single

logger = logging.getLogger(__name__)

# Human-readable scenario names
_SCENARIO_NAMES = {
    'A': 'Location 1: In-State & Out-of-State Facilities (Long-Distance Transport)',
    'B': 'Location 2: In-State Facilities Only (Short-Distance Transport)',
    'C': 'Location 3: Alternative Transport Configuration',
}

# BioSTEAM mapping to Excel metrics. Where exact mappings aren't directly available in standard JSON output, 
# we scale based on capacities or use the primary metric (like GWP for all GHG).
def _map_biosteam_to_strap_row(strap_data_row, res_json, capacity_tons_yr):
    tea = res_json.get("tea", {})
    lca = res_json.get("lca", {})
    ops = res_json.get("operations", {})

    # Energy: MJ/kg -> MJ/yr
    energy_mj_kg = ops.get("total_energy_mj_per_kg") or 0
    strap_data_row["Total Energy Consumed [MJ/yr]"] = energy_mj_kg * capacity_tons_yr * 1000

    # GHG: kg CO2e/kg -> tons CO2e/yr
    gwp_kg = lca.get("gwp_kg_co2e_per_kg") or 0
    gwp_tons_yr = gwp_kg * capacity_tons_yr
    strap_data_row["GWP [tonCO2e/yr]"] = gwp_tons_yr
    strap_data_row["Total Direct GHG emissions [Scope 1] [metric tons CO2 equivalent [tCO2e/yr]]"] = gwp_tons_yr
    strap_data_row["Total Energy indirect GHG emissions (Scope 2) [metric tons CO2 equivalent (t CO2e/yr)]"] = 0

    # Water / Waste
    water_m3_yr = ops.get("water_consumed_m3_yr") or 0
    strap_data_row["Water consumed/discarded [m3/yr]"] = water_m3_yr
    waste_kg_yr = ops.get("waste_generated_kg_yr") or 0
    strap_data_row["Waste generated - Non Hazardous [kg/yr]"] = waste_kg_yr

    # Cost — guard all against None
    capex_usd = tea.get("tci_usd") or 0
    aoc_usd_yr = tea.get("aoc_usd_per_yr") or 0
    if capex_usd > 0:
        strap_data_row["CAPEX [USD/yr]"] = capex_usd / 10  # simple 10-yr annualisation
    if aoc_usd_yr > 0:
        strap_data_row["OPEX [USD/yr]"] = aoc_usd_yr

    # Toxicity
    strap_data_row["Human toxicity cancer [CTUh/yr]"] = (lca.get("htc_ctuh_per_kg") or 0) * capacity_tons_yr * 1000
    strap_data_row["Human toxicity non-cancer [CTUh/yr]"] = (lca.get("htnc_ctuh_per_kg") or 0) * capacity_tons_yr * 1000
    strap_data_row["Ecotoxicity [CTUe/yr]"] = (lca.get("etox_ctue_per_kg") or 0) * capacity_tons_yr * 1000

    return strap_data_row

def _update_excel_with_biosteam(feed: float, pe_fraction: float, evoh_fraction: float, excel_path: Path):
    """Run dynamic BioSTEAM simulations based on feed scaling and update the Excel file."""
    capacity_pe = max(feed * pe_fraction, 1)
    capacity_evoh = max(feed * evoh_fraction, 1)
    
    df = pd.read_excel(excel_path, sheet_name="StrapScenario3 Units")
    unique_simulations = {}
    
    for _, row in df.iterrows():
        w = row.get("Wash number")
        p = row.get("Polymer")
        s = row.get("Solvents")
        if pd.isna(w) or pd.isna(p) or pd.isna(s): continue
        
        cap = capacity_pe if p == "PE" else capacity_evoh
        key = (p, s, cap)
        
        if key not in unique_simulations:
            config = build_single_config(
                solvent=s,
                target_plastic=p,
                target_plastic_percent=100.0,
                processing_capacity=cap,
                energy_case="C1",
            )
            try:
                res = run_single_simulation(config)
                unique_simulations[key] = res
            except Exception as e:
                logger.error(f"Failed BioSTEAM simulation for {p} in {s}: {e}")
                unique_simulations[key] = {"success": False}
    
    for i, row in df.iterrows():
        p = row.get("Polymer")
        s = row.get("Solvents")
        if pd.isna(p) or pd.isna(s): continue
        
        cap = capacity_pe if p == "PE" else capacity_evoh
        res = unique_simulations.get((p, s, cap), {})
        
        if res.get("success", False):
            df.iloc[i] = _map_biosteam_to_strap_row(row, res, cap)
            
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        df.to_excel(writer, sheet_name="StrapScenario3 Units", index=False)


@safe_tool_wrapper(structured_output=True)
def run_waste_management_optimization(
    feed: float,
    pe_fraction: float,
    pet_fraction: float,
    n6_fraction: float,
    evoh_fraction: float,
    scenario: str = 'A',
    objective: str = 'max_profit'
) -> str:
    """Run the PIW multi-layer plastic waste optimization model.
    This tools recalculates costs and operational parameters using BioSTEAM based on the specified input 
    fractions and total feed, updates the base Excel data, runs the optimization over all simulated pathways 
    using the available solvers, and returns the optimal configuration.
    
    Args:
        feed: Total mixed plastic feed in tonnes/year (e.g. 8000).
        pe_fraction: Fraction of Polyethylene (PE) in the feed (0.0 to 1.0).
        pet_fraction: Fraction of Polyethylene terephthalate (PET) in the feed.
        n6_fraction: Fraction of Nylon-6 (N6) in the feed.
        evoh_fraction: Fraction of Ethylene vinyl alcohol (EVOH) in the feed.
            (Note: fractions must sum to 1.0)
        scenario: Location scenario 'A', 'B', or 'C'. Default is 'A'.
        objective: 'max_profit', 'min_emissions', or 'max_circularity'. Default 'max_profit'.
    
    WHEN TO USE:
    - "Optimize waste management for a plant with 8000 feed composed of 60% PE, 20% PET..."
    - "Evaluate the maximum profit of processing 10000 tons of 50/50 PE/EVOH waste"
    """
    try:
        # 1. Parse and validate constraints
        assert feed > 0, "Feed must be greater than 0"
        total_frac = pe_fraction + pet_fraction + n6_fraction + evoh_fraction
        assert abs(total_frac - 1.0) < 0.01, f"Fractions must sum to 1.0, got {total_frac}"
        
        # Paths
        base_dir = Path(__file__).resolve().parent.parent / "waste_management"
        excel_path = base_dir / "Data for model_Scenarios.xlsx"
        
        if not excel_path.exists():
            return json_tool_error(f"Excel file not found at {excel_path}", tool_name="run_waste_management_optimization")
        
        # 2. Run BioSTEAM and update the Excel file
        _update_excel_with_biosteam(feed, pe_fraction, evoh_fraction, excel_path)
        
        # 3. Optimize Using Pyomo and SCIP
        scen_keys = {
            'A': {'other_sheet': 'Othertech w TransportA', 'distances': {'strap':0,'lf':9.2,'we':151,'py':1034,'gas_er':0,'gas_h2':2036,'gas_h2cc':2036}},
            'B': {'other_sheet': 'Othertech w TransportB', 'distances': {'strap':0,'lf':9.2,'we':151,'py':76.1,'gas_er':0,'gas_h2':76.1,'gas_h2cc':76.1}},
            'C': {'other_sheet': 'Othertech w TransportA', 'distances': {'strap':0,'lf':9.2,'we':151,'py':1034,'gas_er':0,'gas_h2':2036,'gas_h2cc':2036}}
        }
        if scenario not in scen_keys: 
            scenario = 'A'
            
        CONFIG = {
            'Feed': feed,
            'PE_f': pe_fraction,
            'PET_f': pet_fraction,
            'N6_f': n6_fraction,
            'EV_f': evoh_fraction,
            'Cpe': 1173,      # USD/tonne
            'Cevoh': 8100,
            'Cwte': 259.57,
            'UB_energy': 6.26e7,
            'UB_ghg': 21303.35985408156,
            'UB_withdrawal': 14468.80855,
            'UB_waste': 1.92e6,
            'fc_t': 3.01,
            'vc_t': 0.07,
            'products_heat': 583.33,
            'products_electricity': 724.3693,
            'price_heat': 0.13,
            'price_elec': 0.0996,
            'Cgas_pw': 110,
            'ce_weights': {'energy':0.20,'ghg':0.20,'water':0.20,'waste':0.20,'subs':0.20},
            'distances': scen_keys[scenario]['distances'],
        }
        
        data = load_all_data(
            excel_path=excel_path,
            strap_sheet='StrapScenario3 Units',
            other_sheet=scen_keys[scenario]['other_sheet'],
            p_strap=1.0,
        )
        
        m = build_model(data, CONFIG)
        # User specified to use SCIP solver
        try:
            results = solve_single(m, objective, solver_name="scip")
        except Exception as e:
            logger.warning(f"Failed to run SCIP solver: {e}. Falling back to default tools.")
            results = solve_single(m, objective, solver_name="gurobi")
        # circularity_score is already normalized 0-1 by solver.extract_results()

        display = f"## Multi-layer Plastic Optimization Results\n\n"
        display += f"**Objective:** {objective} | **Scenario:** {_SCENARIO_NAMES.get(scenario, scenario)}\n"
        display += f"**Feed:** {feed} tonnes/year ({pe_fraction*100:.0f}% PE, {pet_fraction*100:.0f}% PET, {n6_fraction*100:.0f}% N6, {evoh_fraction*100:.0f}% EVOH)\n\n"
        
        display += "### Optimal Technology Pathways Selected\n"
        display += f"- **Stage 1 (Separation):** {', '.join(results.get('stage1_tech', [])) or 'None'}\n"
        display += f"- **Stage 2 (Conversion):** {', '.join(results.get('stage2_tech', [])) or 'None'}\n"
        display += f"- **Stage 3 (End of Life):** {', '.join(results.get('stage3_tech', [])) or 'None'}\n"
        
        display += "\n### Chosen STRAP Solvents\n"
        display += f"- **Wash 1 (PE Target):** {', '.join(results.get('wash1_selection', [])) or 'None'}\n"
        display += f"- **Wash 2 (EVOH Target):** {', '.join(results.get('wash2_selection', [])) or 'None'}\n"
        
        display += "\n### Economic and Environmental Impact\n"
        display += f"- **Total Profit:** ${results.get('profit', 0):,.2f}\n"
        display += f"- **Emissions:** {results.get('emissions', 0):,.2f} tCO2e/yr\n"
        display += f"- **Circularity Score (0–1):** {results.get('circularity_score', 0):.4f}\n"
        display += f"- **Capital Cost:** ${results.get('capital_cost', 0):,.2f}\n"
        display += f"- **Operational Cost:** ${results.get('operational_cost', 0):,.2f}\n"
        
        # Remove any non-serializable objects from results before returning as JSON
        clean_results = {k: str(v) if not isinstance(v, (int, float, str, list, dict, bool)) else v for k, v in results.items()}
        
        return json_tool_response(display, clean_results)
        
    except Exception as e:
        logger.exception("Error in run_waste_management_optimization")
        return json_tool_error(str(e), tool_name="run_waste_management_optimization")


@safe_tool_wrapper(structured_output=True)
def run_pareto_optimization(
    feed: float,
    pe_fraction: float,
    pet_fraction: float,
    n6_fraction: float,
    evoh_fraction: float,
    pareto_pair: str = 'profit_vs_emissions',
    scenario: str = 'A',
    n_points: int = 20,
) -> str:
    """Run epsilon-constraint multi-objective optimization to generate a Pareto front.

    This tool performs multi-objective optimization as described in the PIW paper,
    generating a Pareto trade-off curve between two objectives using the
    epsilon-constraint method. It sweeps one objective as a constraint while
    optimizing the other, producing a set of Pareto-optimal solutions.

    Args:
        feed: Total mixed plastic feed in tonnes/year (e.g. 8000).
        pe_fraction: Fraction of Polyethylene (PE) in the feed (0.0 to 1.0).
        pet_fraction: Fraction of Polyethylene terephthalate (PET) in the feed.
        n6_fraction: Fraction of Nylon-6 (N6) in the feed.
        evoh_fraction: Fraction of Ethylene vinyl alcohol (EVOH) in the feed.
            (Note: fractions must sum to 1.0)
        pareto_pair: Which two objectives to trade off. Options:
            'profit_vs_emissions' — Max profit s.t. emissions ≤ ε  (default)
            'profit_vs_circularity' — Max profit s.t. CE ≥ ε
            'emissions_vs_circularity' — Min emissions s.t. CE ≥ ε
        scenario: Location scenario 'A', 'B', or 'C'. Default is 'A'.
        n_points: Number of Pareto points to generate (default 20, max 50).

    WHEN TO USE:
    - "Generate a Pareto front of profit vs emissions"
    - "Show the trade-off between circularity and profit"
    - "Run multi-objective optimization"
    - "How does profit change as I tighten the emissions constraint?"
    """
    from strap.waste_management.solver import (
        solve_single,
        pareto_profit_vs_emissions,
        pareto_profit_vs_ce,
        pareto_emissions_vs_ce,
    )

    try:
        # 1. Validate
        assert feed > 0, "Feed must be greater than 0"
        total_frac = pe_fraction + pet_fraction + n6_fraction + evoh_fraction
        assert abs(total_frac - 1.0) < 0.01, f"Fractions must sum to 1.0, got {total_frac}"
        n_points = max(5, min(n_points, 50))

        valid_pairs = ('profit_vs_emissions', 'profit_vs_circularity', 'emissions_vs_circularity')
        if pareto_pair not in valid_pairs:
            pareto_pair = 'profit_vs_emissions'

        # 2. Scenario config
        base_dir = Path(__file__).resolve().parent.parent / "waste_management"
        excel_path = base_dir / "Data for model_Scenarios.xlsx"

        if not excel_path.exists():
            return json_tool_error(f"Excel file not found at {excel_path}", tool_name="run_pareto_optimization")

        # 2. Run BioSTEAM and update the Excel file dynamically
        _update_excel_with_biosteam(feed, pe_fraction, evoh_fraction, excel_path)

        scen_keys = {
            'A': {'other_sheet': 'Othertech w TransportA', 'distances': {'strap':0,'lf':9.2,'we':151,'py':1034,'gas_er':0,'gas_h2':2036,'gas_h2cc':2036}},
            'B': {'other_sheet': 'Othertech w TransportB', 'distances': {'strap':0,'lf':9.2,'we':151,'py':76.1,'gas_er':0,'gas_h2':76.1,'gas_h2cc':76.1}},
            'C': {'other_sheet': 'Othertech w TransportA', 'distances': {'strap':0,'lf':9.2,'we':151,'py':1034,'gas_er':0,'gas_h2':2036,'gas_h2cc':2036}},
        }
        if scenario not in scen_keys:
            scenario = 'A'

        CONFIG = {
            'Feed': feed,
            'PE_f': pe_fraction,
            'PET_f': pet_fraction,
            'N6_f': n6_fraction,
            'EV_f': evoh_fraction,
            'Cpe': 1173, 'Cevoh': 8100, 'Cwte': 259.57,
            'UB_energy': 6.26e7,
            'UB_ghg': 21303.35985408156,
            'UB_withdrawal': 14468.80855,
            'UB_waste': 1.92e6,
            'fc_t': 3.01, 'vc_t': 0.07,
            'products_heat': 583.33, 'products_electricity': 724.3693,
            'price_heat': 0.13, 'price_elec': 0.0996, 'Cgas_pw': 110,
            'ce_weights': {'energy':0.20,'ghg':0.20,'water':0.20,'waste':0.20,'subs':0.20},
            'distances': scen_keys[scenario]['distances'],
        }

        data = load_all_data(
            excel_path=excel_path,
            strap_sheet='StrapScenario3 Units',
            other_sheet=scen_keys[scenario]['other_sheet'],
            p_strap=1.0,
        )

        # 3. Solve anchor points (ideal/non-ideal) for the two objectives
        from strap.waste_management.model import build_model as _build_model

        # --- Anchor: max_profit ---
        m_profit = _build_model(data, CONFIG)
        res_profit = solve_single(m_profit, 'max_profit', solver_name='gurobi')
        if res_profit is None:
            m_profit = _build_model(data, CONFIG)
            res_profit = solve_single(m_profit, 'max_profit', solver_name='scip')

        # --- Anchor: min_emissions ---
        m_emit = _build_model(data, CONFIG)
        res_emit = solve_single(m_emit, 'min_emissions', solver_name='gurobi')
        if res_emit is None:
            m_emit = _build_model(data, CONFIG)
            res_emit = solve_single(m_emit, 'min_emissions', solver_name='scip')

        # --- Anchor: max_circularity ---
        m_ce = _build_model(data, CONFIG)
        res_ce = solve_single(m_ce, 'max_circularity', solver_name='gurobi')
        if res_ce is None:
            m_ce = _build_model(data, CONFIG)
            res_ce = solve_single(m_ce, 'max_circularity', solver_name='scip')

        # Extract anchor values
        profit_ideal = res_profit['profit'] if res_profit else 0
        emit_ideal = res_emit['emissions'] if res_emit else 0
        emit_from_profit = res_profit['emissions'] if res_profit else emit_ideal
        ce_ideal = res_ce['CE'] if res_ce else 0
        ce_from_profit = res_profit['CE'] if res_profit else 0

        # 4. Run epsilon-constraint sweep
        m_pareto = _build_model(data, CONFIG)

        if pareto_pair == 'profit_vs_emissions':
            df = pareto_profit_vs_emissions(
                m_pareto, emit_ideal, emit_from_profit, n_points=n_points, solver_name='gurobi',
            )
            x_label, y_label = 'emissions', 'profit'
        elif pareto_pair == 'profit_vs_circularity':
            df = pareto_profit_vs_ce(
                m_pareto, ce_from_profit, ce_ideal, n_points=n_points, solver_name='gurobi',
            )
            x_label, y_label = 'CE', 'profit'
        else:  # emissions_vs_circularity
            df = pareto_emissions_vs_ce(
                m_pareto, ce_from_profit, ce_ideal, n_points=n_points, solver_name='gurobi',
            )
            x_label, y_label = 'CE', 'emissions'

        # 5. Build display output
        display = f"## Pareto Front: {pareto_pair.replace('_', ' ').title()}\n\n"
        display += f"**Scenario:** {_SCENARIO_NAMES.get(scenario, scenario)} | **Points:** {len(df)}/{n_points}\n"
        display += f"**Feed:** {feed} tonnes/yr ({pe_fraction*100:.0f}% PE, {pet_fraction*100:.0f}% PET, {n6_fraction*100:.0f}% N6, {evoh_fraction*100:.0f}% EVOH)\n\n"

        # Anchor points summary
        display += "### Anchor (Ideal) Solutions\n"
        if res_profit:
            display += f"- **Max Profit:** ${res_profit['profit']:,.0f} | Emissions: {res_profit['emissions']:,.0f} tCO2e/yr | Circularity: {res_profit['circularity_score']:.4f}\n"
        if res_emit:
            display += f"- **Min Emissions:** ${res_emit['profit']:,.0f} | Emissions: {res_emit['emissions']:,.0f} tCO2e/yr | Circularity: {res_emit['circularity_score']:.4f}\n"
        if res_ce:
            display += f"- **Max Circularity:** ${res_ce['profit']:,.0f} | Emissions: {res_ce['emissions']:,.0f} tCO2e/yr | Circularity: {res_ce['circularity_score']:.4f}\n"

        # Pareto table  (show up to 10 selected points across the front)
        if len(df) > 0:
            display += f"\n### Pareto Front ({len(df)} feasible points)\n"
            display += "| # | Profit ($) | Emissions (tCO2e/yr) | Circularity (0–1) | Stage 1 | Stage 2 | Stage 3 |\n"
            display += "|---|-----------|---------------------|-------------------|---------|---------|--------|\n"
            step = max(1, len(df) // 10)
            for idx in range(0, len(df), step):
                row = df.iloc[idx]
                ce_norm = max(0.0, min(row.get('CE', 0) / 1_000_000.0, 1.0))
                display += f"| {idx+1} | {row.get('profit', 0):,.0f} | {row.get('emissions', 0):,.0f} | {ce_norm:.4f} | {', '.join(row.get('stage1', []))} | {', '.join(row.get('stage2', []))} | {', '.join(row.get('stage3', []))} |\n"
        else:
            display += "\n**No feasible Pareto points found.** The model may be infeasible for this configuration.\n"

        display += "\n> **Tip:** You can also run `profit_vs_circularity` or `emissions_vs_circularity` Pareto fronts. Just ask!\n"

        # Build clean JSON result
        pareto_data = df.to_dict(orient='records') if len(df) > 0 else []
        clean_results = {
            'pareto_pair': pareto_pair,
            'scenario': scenario,
            'n_feasible': len(df),
            'anchors': {
                'max_profit': {k: v for k, v in (res_profit or {}).items() if isinstance(v, (int, float, str, list))},
                'min_emissions': {k: v for k, v in (res_emit or {}).items() if isinstance(v, (int, float, str, list))},
                'max_circularity': {k: v for k, v in (res_ce or {}).items() if isinstance(v, (int, float, str, list))},
            },
            'pareto_front': pareto_data[:20],  # cap JSON size
        }

        return json_tool_response(display, clean_results)

    except Exception as e:
        logger.exception("Error in run_pareto_optimization")
        return json_tool_error(str(e), tool_name="run_pareto_optimization")
