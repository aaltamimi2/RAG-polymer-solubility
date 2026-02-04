"""STRAP (Solvent-Targeted Recovery and Precipitation) process analysis tools."""

from __future__ import annotations

import json
import logging
import asyncio
from typing import List, Dict, Optional, Any

from strap.tools._helpers import safe_tool_wrapper, truncate_output

try:
    from strap.vendor import tea_lca
except ImportError:
    tea_lca = None

logger = logging.getLogger(__name__)


def _get_plot_url(filepath: str) -> str:
    """Convert filepath to displayable format."""
    return f"Plot saved: `{filepath}`"


# ------------------------------------------------------------------
# analyze_strap_process
# ------------------------------------------------------------------

@safe_tool_wrapper
async def analyze_strap_process(
    polymers: List[str],
    feedstock_composition: Dict[str, float] = None,
    capacity_mt_yr: float = 10000.0,
    recovery_solvents: Dict[str, str] = None
) -> str:
    """Run full STRAP TEA/LCA analysis for multi-polymer recovery from plastic waste.

    Args:
        polymers: List of polymers to recover (e.g., ['PE', 'PET', 'EVOH'])
        feedstock_composition: Polymer fractions (e.g., {'PE': 0.8, 'PET': 0.1, 'EVOH': 0.1})
        capacity_mt_yr: Plant capacity in metric tons/year (default: 10000)
        recovery_solvents: Dict mapping polymer to solvent (e.g., {'PS': 'propanone'})

    WHEN TO USE:
    - "Run STRAP analysis for PE/EVOH at 10,000 mt/yr"
    - "Full TEA/LCA for polymer recovery from plastic waste"
    - "Run TEA with solvents: PS->propanone, PP->cyclohexane"
    """
    # Build feedstock composition if not provided
    if feedstock_composition is None:
        n = len(polymers)
        feedstock_composition = {p: 1.0 / n for p in polymers}

    # Normalize composition
    total = sum(feedstock_composition.values())
    feedstock_composition = {k: v / total for k, v in feedstock_composition.items()}

    # Select solvents for each polymer - use custom if provided, else auto-select
    recovery_steps = []
    for polymer in polymers:
        polymer_upper = polymer.upper()

        # Check for custom solvent mapping first
        if recovery_solvents and polymer_upper in recovery_solvents:
            solvent = recovery_solvents[polymer_upper]
        elif recovery_solvents and polymer.lower() in recovery_solvents:
            solvent = recovery_solvents[polymer.lower()]
        elif polymer_upper in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
            # Auto-select from defaults
            compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[polymer_upper]
            solvent = compatible[0] if compatible else 'xylene'
        else:
            solvent = 'xylene'  # Default solvent

        recovery_steps.append({
            'polymer': polymer_upper,
            'solvent': solvent,
            'recover': True
        })

    # Run full analysis
    results = tea_lca.run_full_strap_analysis(
        feedstock_composition=feedstock_composition,
        recovery_steps=recovery_steps,
        capacity_mt_yr=capacity_mt_yr,
        scenario_name=f"STRAP-{'-'.join(polymers)}"
    )

    # Format output - clean simple format
    scenario_name = results.get('scenario', {}).get('name', f"STRAP-{'-'.join(polymers)}")

    output = "STRAP PROCESS ANALYSIS\n\n"
    output += f"Scenario: {scenario_name}\n"
    output += f"Capacity: {capacity_mt_yr:,.0f} metric tons/year\n\n"

    # Feedstock composition
    output += "FEEDSTOCK COMPOSITION\n"
    for p, frac in feedstock_composition.items():
        output += f"{p}: {frac*100:.1f}%\n"
    output += "\n"

    # Recovery steps
    output += "RECOVERY STEPS\n"
    for step in recovery_steps:
        output += f"{step['polymer']} -> {step['solvent'].title()}\n"
    output += "\n"

    # TEA Results
    tea_econ = results['tea'].get('economics', results['tea'])
    tci_millions = tea_econ.get('tci_millions', tea_econ.get('total_capital_investment_usd', 0) / 1e6)
    capital = results['tea'].get('capital', {})

    output += "CAPITAL COSTS\n"
    output += f"Total Capital (TCI): ${tci_millions:.2f}M\n"
    output += f"Equipment Cost: ${capital.get('total_equipment_cost_usd', 0)/1e6:.2f}M\n\n"

    output += "OPERATING ECONOMICS\n"
    output += f"Unit Operating Cost (UOC): ${tea_econ.get('unit_operating_cost_usd_kg', 0):.4f}/kg\n"
    output += f"Annual Operating Cost: ${tea_econ.get('annual_operating_cost_usd', 0)/1e6:.2f}M/yr\n"
    output += f"Annual Revenue: ${tea_econ.get('annual_revenue_usd', 0)/1e6:.2f}M/yr\n"
    output += f"Net Annual Profit: ${tea_econ.get('net_annual_profit_usd', 0)/1e6:.2f}M/yr\n"
    output += f"Simple Payback: {tea_econ.get('simple_payback_years', 0):.2f} years\n"
    output += f"ROI: {tea_econ.get('return_on_investment_pct', 0):.1f}%\n\n"

    # MSP Results
    msp_data = results.get('msp', {})
    msp_by_polymer = msp_data.get('msp_by_polymer_usd_kg', msp_data)
    output += "MINIMUM SELLING PRICE (NPV=0 @ 15% IRR)\n"
    for polymer, price in msp_by_polymer.items():
        output += f"{polymer}: ${price:.4f}/kg\n"
    if 'msp_weighted_avg_usd_kg' in msp_data:
        output += f"Weighted Avg: ${msp_data['msp_weighted_avg_usd_kg']:.4f}/kg\n"
    output += "\n"

    # LCA Results
    lca_data = results.get('lca', {})
    lca_by_polymer = lca_data.get('by_polymer', {})
    virgin_comp = lca_data.get('virgin_comparison', {})

    output += "LIFE CYCLE ASSESSMENT (GWP kg CO2eq/kg)\n"
    for polymer in polymers:
        pu = polymer.upper()
        if pu in lca_by_polymer:
            strap_gwp = lca_by_polymer[pu].get('gwp_kg_co2eq', 0)
            virgin_gwp = tea_lca.LCA_EMISSION_FACTORS['virgin_gwp'].get(pu, 2.0)
            reduction = virgin_comp.get(pu, {}).get('gwp_reduction_pct', 0)
            output += f"{pu}: STRAP={strap_gwp:.3f}, Virgin={virgin_gwp:.3f}, Reduction={reduction:.1f}%\n"
    output += "\n"

    # GWP Breakdown
    gwp_breakdown = lca_data.get('gwp_breakdown', {})
    if gwp_breakdown:
        output += "GWP BREAKDOWN BY SOURCE (kg CO2eq/kg)\n"
        for polymer, breakdown in gwp_breakdown.items():
            sources = ", ".join([f"{k.replace('_', ' ').title()}={v:.3f}" for k, v in breakdown.items()])
            output += f"{polymer}: {sources}\n"

    # Build structured data for programmatic access
    # Extract GWP values by polymer
    gwp_by_polymer = {}
    virgin_gwp = {}
    gwp_reduction_pct = {}
    for polymer in polymers:
        pu = polymer.upper()
        if pu in lca_by_polymer:
            gwp_by_polymer[pu] = lca_by_polymer[pu].get('gwp_kg_co2eq', 0)
            virgin_gwp[pu] = tea_lca.LCA_EMISSION_FACTORS['virgin_gwp'].get(pu, 2.0)
            gwp_reduction_pct[pu] = virgin_comp.get(pu, {}).get('gwp_reduction_pct', 0)

    structured_data = {
        "tool_name": "analyze_strap_process",
        "success": True,
        "polymers": [p.upper() for p in polymers],
        "feedstock_composition": feedstock_composition,
        "capacity_mt_yr": capacity_mt_yr,
        "tci_millions": tci_millions,
        "equipment_cost_millions": capital.get('total_equipment_cost_usd', 0) / 1e6,
        "unit_operating_cost": tea_econ.get('unit_operating_cost_usd_kg', 0),
        "annual_operating_cost_millions": tea_econ.get('annual_operating_cost_usd', 0) / 1e6,
        "annual_revenue_millions": tea_econ.get('annual_revenue_usd', 0) / 1e6,
        "net_annual_profit_millions": tea_econ.get('net_annual_profit_usd', 0) / 1e6,
        "simple_payback_years": tea_econ.get('simple_payback_years', 0),
        "roi_pct": tea_econ.get('return_on_investment_pct', 0),
        "msp_by_polymer": msp_by_polymer,
        "msp_weighted_avg": msp_data.get('msp_weighted_avg_usd_kg'),
        "gwp_by_polymer": gwp_by_polymer,
        "virgin_gwp": virgin_gwp,
        "gwp_reduction_pct": gwp_reduction_pct,
        "recovery_steps": [{"polymer": s['polymer'], "solvent": s['solvent']} for s in recovery_steps],
    }

    # Return structured JSON
    return json.dumps({"display": output, "data": structured_data}, ensure_ascii=False)


# ------------------------------------------------------------------
# calculate_strap_msp
# ------------------------------------------------------------------

@safe_tool_wrapper
async def calculate_strap_msp(
    polymers: List[str],
    feedstock_composition: Dict[str, float] = None,
    capacity_mt_yr: float = 10000.0,
    target_irr: float = 0.15
) -> str:
    """Calculate Minimum Selling Price (MSP) where NPV=0 at target IRR for recovered polymers.

    Args:
        polymers: List of polymers to recover (e.g., ['PE', 'EVOH'])
        feedstock_composition: Polymer fractions (optional, defaults to equal split)
        capacity_mt_yr: Plant capacity in metric tons/year (default: 10000)
        target_irr: Target internal rate of return (default: 0.15 = 15%)

    WHEN TO USE:
    - "What's the minimum selling price for STRAP recovered PE?"
    - "Calculate MSP at 15% IRR for polymer recovery"
    - "Break-even price for STRAP recycled EVOH"
    """
    # Build feedstock composition
    if feedstock_composition is None:
        n = len(polymers)
        feedstock_composition = {p.upper(): 1.0 / n for p in polymers}

    # Normalize
    total = sum(feedstock_composition.values())
    feedstock_composition = {k: v / total for k, v in feedstock_composition.items()}

    # Auto-select solvents
    recovery_steps = []
    for polymer in polymers:
        pu = polymer.upper()
        if pu in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
            compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[pu]
            solvent = compatible[0] if compatible else 'xylene'
        else:
            solvent = 'xylene'
        recovery_steps.append({'polymer': pu, 'solvent': solvent, 'recover': True})

    # Calculate MSP
    msp_results = tea_lca.calculate_msp(
        capacity_mt_yr=capacity_mt_yr,
        feedstock_composition=feedstock_composition,
        recovery_steps=recovery_steps,
        target_irr=target_irr
    )

    # Get market prices for comparison
    market_prices = tea_lca.DEFAULT_POLYMER_PROPS.recovered_prices

    # Extract MSP by polymer from nested structure
    msp_by_polymer = msp_results.get('msp_by_polymer_usd_kg', msp_results)

    output = "# Minimum Selling Price (MSP) Analysis\n\n"
    output += f"**Target IRR:** {target_irr*100:.0f}%\n"
    output += f"**Capacity:** {capacity_mt_yr:,.0f} mt/yr\n\n"

    output += "| Polymer | MSP ($/kg) | Market Price ($/kg) | Margin |\n"
    output += "|---------|------------|---------------------|--------|\n"
    for polymer in polymers:
        pu = polymer.upper()
        msp = msp_by_polymer.get(pu, 0)
        market = market_prices.get(pu, 1.0)
        margin = market - msp
        margin_pct = (margin / msp * 100) if msp > 0 else 0
        output += f"| {pu} | ${msp:.4f} | ${market:.2f} | ${margin:.2f} ({margin_pct:+.1f}%) |\n"

    if 'msp_weighted_avg_usd_kg' in msp_results:
        output += f"\n**Weighted Average MSP:** ${msp_results['msp_weighted_avg_usd_kg']:.4f}/kg\n"

    output += "\n### Interpretation\n"
    output += "- MSP < Market Price: Project is economically viable\n"
    output += "- Positive margin indicates potential profit at market prices\n"
    output += f"- Calculated at {target_irr*100:.0f}% IRR over 20-year project life\n"

    # Build structured data
    margins = {}
    for polymer in polymers:
        pu = polymer.upper()
        msp = msp_by_polymer.get(pu, 0)
        market = market_prices.get(pu, 1.0)
        margins[pu] = market - msp

    structured_data = {
        "tool_name": "calculate_strap_msp",
        "success": True,
        "polymers": [p.upper() for p in polymers],
        "capacity_mt_yr": capacity_mt_yr,
        "target_irr": target_irr,
        "msp_by_polymer": msp_by_polymer,
        "msp_weighted_avg": msp_results.get('msp_weighted_avg_usd_kg'),
        "market_prices": {p.upper(): market_prices.get(p.upper(), 1.0) for p in polymers},
        "margins": margins,
        "recovery_steps": [{"polymer": s['polymer'], "solvent": s['solvent']} for s in recovery_steps],
    }

    return json.dumps({"display": output, "data": structured_data}, ensure_ascii=False)


# ------------------------------------------------------------------
# compare_strap_scenarios
# ------------------------------------------------------------------

@safe_tool_wrapper
async def compare_strap_scenarios(
    scenario_configs: List[Dict[str, Any]]
) -> str:
    """Compare multiple STRAP scenarios on TEA/LCA metrics with rankings and recommendations.

    Args:
        scenario_configs: List of dicts with keys: name, polymers, feedstock_composition,
            capacity_mt_yr, recovery_solvents (all optional except polymers)

    WHEN TO USE:
    - "Compare PE-only vs PE+EVOH recovery scenarios"
    - "Which STRAP configuration is most profitable?"
    - "Rank scenarios by ROI and carbon footprint"
    """
    scenarios = []
    for config in scenario_configs:
        polymers = config.get('polymers', ['PE'])
        capacity = config.get('capacity_mt_yr', 10000)

        # Build feedstock composition
        fc = config.get('feedstock_composition')
        if fc is None:
            n = len(polymers)
            fc = {p.upper(): 1.0 / n for p in polymers}

        # Normalize
        total = sum(fc.values())
        fc = {k: v / total for k, v in fc.items()}

        # Get custom solvents if provided, else auto-select
        custom_solvents = config.get('recovery_solvents', {})
        recovery_steps = []
        for polymer in polymers:
            pu = polymer.upper()
            # Check for custom solvent mapping first
            if custom_solvents and pu in custom_solvents:
                solvent = custom_solvents[pu]
            elif custom_solvents and polymer.lower() in custom_solvents:
                solvent = custom_solvents[polymer.lower()]
            elif pu in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
                compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[pu]
                solvent = compatible[0] if compatible else 'xylene'
            else:
                solvent = 'xylene'
            recovery_steps.append({'polymer': pu, 'solvent': solvent, 'recover': True})

        scenario = tea_lca.build_strap_scenario(
            name=config.get('name', f"Scenario-{len(scenarios)+1}"),
            feedstock_composition=fc,
            recovery_sequence=recovery_steps,
            capacity_mt_yr=capacity,
            description=config.get('description', '')
        )
        scenarios.append(scenario)

    # Compare scenarios
    comparison = tea_lca.compare_scenarios(scenarios)

    # Format output
    output = "# STRAP Scenario Comparison\n\n"
    output += f"Comparing {len(scenarios)} scenarios\n\n"

    # Comparison table - use 'comparison_table' key
    comparison_table = comparison.get('comparison_table', comparison.get('results', []))
    output += "## Economic Comparison\n\n"
    output += "| Scenario | Capacity | TCI ($M) | UOC ($/kg) | ROI (%) | Payback (yr) |\n"
    output += "|----------|----------|----------|------------|---------|-------------|\n"
    for row in comparison_table:
        output += f"| {row['name']} | {row['capacity_mt_yr']:,.0f} | "
        output += f"{row['tci_millions']:.2f} | {row['uoc_usd_kg']:.4f} | "
        output += f"{row['roi_pct']:.1f} | {row['payback_years']:.2f} |\n"
    output += "\n"

    # Rankings - use direct keys from comparison
    output += "## Rankings\n\n"
    if 'best_roi' in comparison:
        output += f"- **Best ROI:** {comparison['best_roi']}\n"
    if 'lowest_uoc' in comparison:
        output += f"- **Lowest UOC:** {comparison['lowest_uoc']}\n"
    if 'best_payback' in comparison:
        output += f"- **Fastest Payback:** {comparison['best_payback']}\n"

    # Generate comparison visualizations
    output += "\n## Comparison Visualizations\n\n"
    viz_paths = {}

    try:
        # Economics comparison bar chart (UOC, ROI, Payback)
        econ_path = tea_lca.plot_scenario_economics_comparison(comparison_table)
        if econ_path:
            viz_paths['economics'] = econ_path
            output += f"- **Economics Comparison**: `{_get_plot_url(econ_path)}`\n"
    except Exception as e:
        logger.warning(f"Could not generate economics comparison chart: {e}")

    try:
        # GWP comparison bar chart
        gwp_path = tea_lca.plot_scenario_gwp_comparison(scenario_configs)
        if gwp_path:
            viz_paths['gwp'] = gwp_path
            output += f"- **GWP/LCA Comparison**: `{_get_plot_url(gwp_path)}`\n"
    except Exception as e:
        logger.warning(f"Could not generate GWP comparison chart: {e}")

    if viz_paths:
        output += f"\nGenerated {len(viz_paths)} comparison visualization(s).\n"
    else:
        output += "\n*No visualizations generated.*\n"

    # Build structured data for feedback loop evaluation
    # Extract best cost metrics for feedback condition check
    best_uoc = min([row['uoc_usd_kg'] for row in comparison_table]) if comparison_table else None
    avg_uoc = sum([row['uoc_usd_kg'] for row in comparison_table]) / len(comparison_table) if comparison_table else None

    structured_data = {
        "tool_name": "compare_strap_scenarios",
        "success": True,
        "scenarios_compared": len(scenarios),
        "cost_per_kg": best_uoc,  # For feedback loop check
        "avg_cost_per_kg": avg_uoc,
        "comparison_table": [
            {
                "name": row['name'],
                "solvent": row.get('solvent', 'unknown'),
                "cost_per_kg_usd": row['uoc_usd_kg'],
                "roi_pct": row['roi_pct'],
                "payback_years": row['payback_years'],
                "capacity_mt_yr": row['capacity_mt_yr'],
                "tci_millions": row['tci_millions'],
            }
            for row in comparison_table
        ],
        "rankings": {
            "best_roi": comparison.get('best_roi'),
            "lowest_uoc": comparison.get('lowest_uoc'),
            "best_payback": comparison.get('best_payback'),
        },
        "visualizations": viz_paths,
    }

    return json.dumps({"display": output, "data": structured_data}, ensure_ascii=False)


# ------------------------------------------------------------------
# generate_strap_visualizations
# ------------------------------------------------------------------

@safe_tool_wrapper
async def generate_strap_visualizations(
    polymers: List[str],
    feedstock_composition: Dict[str, float] = None,
    capacity_mt_yr: float = 10000.0,
    chart_type: str = "all"
) -> str:
    """Generate STRAP visualizations (scale economics, cost breakdown, comparison).

    Args:
        polymers: List of polymers to recover (e.g., ['PE', 'EVOH'])
        feedstock_composition: Polymer fractions (optional, defaults to equal split)
        capacity_mt_yr: Plant capacity in metric tons/year (default: 10000)
        chart_type: "all", "scale_economics", "cost_breakdown", or "comparison" (default: "all")

    WHEN TO USE:
    - "Generate STRAP visualizations for PE/EVOH"
    - "Show all STRAP charts for multilayer recycling"
    - "Show scale economics for STRAP PE/EVOH recovery"
    - "How does plant size affect STRAP economics?"
    - "Plot UOC vs capacity for polymer recycling"
    """
    # Build feedstock composition
    if feedstock_composition is None:
        n = len(polymers)
        feedstock_composition = {p.upper(): 1.0 / n for p in polymers}

    # Normalize
    total = sum(feedstock_composition.values())
    feedstock_composition = {k: v / total for k, v in feedstock_composition.items()}

    # Auto-select solvents
    recovery_steps = []
    for polymer in polymers:
        pu = polymer.upper()
        if pu in tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents:
            compatible = tea_lca.DEFAULT_POLYMER_PROPS.compatible_solvents[pu]
            solvent = compatible[0] if compatible else 'xylene'
        else:
            solvent = 'xylene'
        recovery_steps.append({'polymer': pu, 'solvent': solvent, 'recover': True})

    # Generate visualizations
    all_plots = tea_lca.generate_strap_visualizations(
        feedstock_composition=feedstock_composition,
        recovery_steps=recovery_steps,
        capacity_mt_yr=capacity_mt_yr,
        scenario_name=f"STRAP-{'-'.join(polymers)}"
    )

    # Filter by chart_type if specific
    if chart_type != "all":
        all_plots = {name: path for name, path in all_plots.items() if chart_type in name}

    output = "## STRAP Visualizations\n\n"
    output += f"**Polymers:** {', '.join(polymers)}\n"
    output += f"**Capacity:** {capacity_mt_yr:,.0f} mt/yr\n\n"
    output += "### Generated Plots\n\n"
    for name, path in all_plots.items():
        output += f"- **{name.replace('_', ' ').title()}**: `{path}`\n"

    output += "\n### Plot Descriptions\n"
    output += "- **Scale Economics**: UOC and TCI curves across capacity range\n"
    output += "- **Cost Breakdown**: Operating and capital cost breakdown\n"
    output += "- **Comparison**: STRAP vs virgin polymer metrics\n"

    return output
