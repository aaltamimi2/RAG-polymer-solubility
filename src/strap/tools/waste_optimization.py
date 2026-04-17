"""
Multi-layer Plastic Waste Optimization Tool.
Integrates BioSTEAM simulations with Pyomo superstructure optimization.
"""
import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from strap.tools._helpers import safe_tool_wrapper
from strap.services.biosteam_service import json_tool_response, json_tool_error, build_single_config
from strap.vendor.biosteam_runner import run_single_simulation
from strap.waste_management.data_loader import load_all_data, STRAP_UNIT_COLS
from strap.waste_management.model import build_model
from strap.waste_management.solver import solve_single

logger = logging.getLogger(__name__)

_NUMERIC_WORKBOOK_COLUMNS = tuple(STRAP_UNIT_COLS.values())

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
    temp_dir: Path | None = None
    try:
        # 1. Parse and validate constraints
        assert feed > 0, "Feed must be greater than 0"
        total_frac = pe_fraction + pet_fraction + n6_fraction + evoh_fraction
        assert abs(total_frac - 1.0) < 0.01, f"Fractions must sum to 1.0, got {total_frac}"
        
        # Paths
        base_dir = Path(__file__).resolve().parent.parent / "waste_management"
        source_excel_path = base_dir / "Data for model_Scenarios.xlsx"

        if not source_excel_path.exists():
            return json_tool_error(
                f"Excel file not found at {source_excel_path}",
                tool_name="run_waste_management_optimization",
            )

        # Operate on an isolated workbook copy so concurrent runs do not mutate
        # the packaged source asset in place.
        temp_dir = Path(tempfile.mkdtemp(prefix="strap_waste_opt_"))
        excel_path = temp_dir / source_excel_path.name
        shutil.copy2(source_excel_path, excel_path)
        
        # Define capacities
        capacity_pe = max(feed * pe_fraction, 1)  # minimum 1 ton to avoid biosteam div/0
        capacity_evoh = max(feed * evoh_fraction, 1)
        
        # 2. Run BioSTEAM and update the Excel file
        df = pd.read_excel(excel_path, sheet_name="StrapScenario3 Units")
        for column in _NUMERIC_WORKBOOK_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
        
        # We need to simulate unique solvents used in the Excel
        # To save time, we group by Polymer and Solvent, run BioSTEAM once per (Polymer, Solvent), and update all matched Rows.
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
                    target_plastic_percent=100.0, # pure feed entering that stage
                    processing_capacity=cap, # MT per year
                    energy_case="C1",
                )
                try:
                    res = run_single_simulation(config)
                    unique_simulations[key] = res
                except Exception as e:
                    logger.error(f"Failed BioSTEAM simulation for {p} in {s}: {e}")
                    unique_simulations[key] = {"success": False}
        
        # Apply updates back to Dataframe
        for i, row in df.iterrows():
            p = row.get("Polymer")
            s = row.get("Solvents")
            if pd.isna(p) or pd.isna(s): continue
            
            cap = capacity_pe if p == "PE" else capacity_evoh
            res = unique_simulations.get((p, s, cap), {})
            
            if res.get("success", False):
                updated_row = _map_biosteam_to_strap_row(row.copy(), res, cap)
                for column, value in updated_row.items():
                    df.at[i, column] = value
        
        # Save updated sheet using ExcelWriter (this preserves other sheets ideally, but doing pd.ExcelWriter requires care not to wipe others)
        # We will use openpyxl to append/replace the sheet safely
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            df.to_excel(writer, sheet_name="StrapScenario3 Units", index=False)
            
        
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
            logger.warning("Failed to run SCIP solver: %s. Falling back to available solvers.", e)
            results = solve_single(m, objective, solver_name=None)
        if not results:
            return json_tool_error(
                "Optimization model did not return a feasible solution.",
                tool_name="run_waste_management_optimization",
            )
        # Normalize circularity score (CE) to 0‑1 range as per paper
        raw_ce = results.get('CE', 0)
        circularity_score = max(0.0, min(raw_ce / 1_000_000.0, 1.0))
        results['raw_circularity_score'] = raw_ce
        results['circularity_score'] = circularity_score
        
        display = f"## Multi-layer Plastic Optimization Results\n\n"
        display += f"**Objective:** {objective} | **Scenario:** {scenario}\n"
        display += f"**Feed:** {feed} tonnes/year ({pe_fraction*100}% PE, {pet_fraction*100}% PET, {n6_fraction*100}% N6, {evoh_fraction*100}% EVOH)\n\n"
        
        display += "### Optimal Technology Pathways Selected\n"
        display += f"- **Stage 1 (Separation):** {results.get('stage1_tech', [])}\n"
        display += f"- **Stage 2 (Conversion):** {results.get('stage2_tech', [])}\n"
        display += f"- **Stage 3 (End of Life):** {results.get('stage3_tech', [])}\n"
        
        display += "\n### Chosen STRAP Solvents\n"
        display += f"- **Wash 1 (PE Target):** {results.get('wash1_selection', [])}\n"
        display += f"- **Wash 2 (EVOH Target):** {results.get('wash2_selection', [])}\n"
        
        display += "\n### Economic and Environmental Impact\n"
        display += f"- **Total Profit:** ${results.get('profit', 0):,.2f}\n"
        display += f"- **Emissions:** {results.get('emissions', 0):,.2f} tCO2\n"
        # Show circularity appropriately based on selected objective
        if objective == 'max_circularity':
            display += f"- **Circularity (0‑1):** {results.get('circularity_score', 0):.4f}\n"
        else:
            display += f"- **Circularity (0‑1):** {results.get('circularity_score', 0):.4f}\n"
        display += f"- **Capital Cost:** ${results.get('capital_cost', 0):,.2f}\n"
        display += f"- **Operational Cost:** ${results.get('operational_cost', 0):,.2f}\n"
        
        # Remove any non-serializable objects from results before returning as JSON
        # Pyomo objects might be in dictionary, ensure all are primitives
        clean_results = {
            k: str(v) if not isinstance(v, (int, float, str, list, dict, bool)) else v
            for k, v in results.items()
        }

        return json_tool_response(display, clean_results)
        
    except Exception as e:
        logger.exception("Error in run_waste_management_optimization")
        return json_tool_error(str(e), tool_name="run_waste_management_optimization")
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
