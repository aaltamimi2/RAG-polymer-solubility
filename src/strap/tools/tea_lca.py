"""Techno-Economic Analysis (TEA) and Life Cycle Assessment (LCA) tools."""

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


@safe_tool_wrapper
async def analyze_solvent_recovery_tea(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> str:
    """Run TEA for solvent recovery, calculating capital/operating costs and economic metrics.

    Args:
        solvent: Solvent name (e.g., 'toluene', 'acetone', 'ethanol')
        polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
        solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
        recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95)
        process_temp_c: Process temperature in Celsius (default: 80)

    WHEN TO USE:
    - "What's the cost of recovering toluene?"
    - "TEA for LDPE separation at 100 kg/hr"
    - "Calculate payback period for solvent recovery"
    """
    results = tea_lca.run_full_tea_analysis(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )
    display = tea_lca.format_tea_results(results)

    # Build structured data
    capital = results.get('capital_costs', {})
    operating = results.get('operating_costs', {})
    economics = results.get('economics', {})

    structured_data = {
        "tool_name": "analyze_solvent_recovery_tea",
        "success": True,
        "solvent": solvent,
        "throughput_kg_hr": polymer_throughput_kg_hr,
        "recovery_rate": recovery_fraction,
        "temperature": process_temp_c,
        "cost_per_kg": economics.get('cost_per_kg_polymer_usd'),
        "total_capex": capital.get('total_capex'),
        "annual_opex": operating.get('total_annual'),
        "payback_years": economics.get('simple_payback_years'),
        "cost_breakdown": {
            "energy": operating.get('energy_cost_annual', 0),
            "labor": operating.get('labor_cost_annual', 0),
            "solvent_makeup": operating.get('solvent_makeup_cost_annual', 0),
        },
    }

    return json.dumps({"display": display, "data": structured_data}, ensure_ascii=False)


@safe_tool_wrapper
async def analyze_solvent_recovery_lca(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0
) -> str:
    """Run LCA for solvent recovery, calculating emissions, energy use, and environmental impact.

    Args:
        solvent: Solvent name (e.g., 'toluene', 'acetone', 'ethanol')
        polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
        solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
        recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95)
        process_temp_c: Process temperature in Celsius (default: 80)

    WHEN TO USE:
    - "What's the carbon footprint of using toluene?"
    - "LCA for solvent recovery"
    - "How much CO2 does the separation process emit?"
    """
    results = tea_lca.run_full_lca_analysis(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )
    display = tea_lca.format_lca_results(results)

    # Build structured data
    emissions = results.get('emissions', {})
    energy = results.get('energy', {})

    structured_data = {
        "tool_name": "analyze_solvent_recovery_lca",
        "success": True,
        "solvent": solvent,
        "throughput_kg_hr": polymer_throughput_kg_hr,
        "recovery_rate": recovery_fraction,
        "co2_kg_per_kg": emissions.get('total_co2_kg_per_kg'),
        "energy_mj_per_kg": energy.get('total_mj_per_kg'),
        "emissions_breakdown": emissions,
        "energy_breakdown": energy,
    }

    return json.dumps({"display": display, "data": structured_data})


@safe_tool_wrapper
async def compare_solvents_tea_lca(
    solvents: List[str],
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0,
    generate_charts: bool = True
) -> str:
    """Compare multiple solvents on TEA/LCA metrics with rankings and optional charts.

    Args:
        solvents: List of solvents to compare (e.g., ['toluene', 'acetone', 'ethanol'])
        polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
        solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
        recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95)
        process_temp_c: Process temperature in Celsius (default: 80)
        generate_charts: Generate comparison visualizations (default: True)

    WHEN TO USE:
    - "Compare toluene vs acetone for cost and emissions"
    - "Which solvent is cheapest and greenest?"
    - "Rank solvents by cost and carbon footprint"
    - "Plot cost vs emissions for multiple solvents"
    - "Show comparison chart for TEA/LCA"
    """
    results = tea_lca.compare_solvents_tea_lca(
        solvents=solvents,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )
    display = tea_lca.format_comparison_results(results)

    # Build structured data
    comparison = results.get('comparison', [])
    rankings = results.get('rankings', {})

    structured_data = {
        "tool_name": "compare_solvents_tea_lca",
        "success": True,
        "solvents": solvents,
        "throughput_kg_hr": polymer_throughput_kg_hr,
        "recovery_rate": recovery_fraction,
        "comparison": comparison,
        "best_by_cost": rankings.get('by_cost', [None])[0] if rankings.get('by_cost') else None,
        "best_by_emissions": rankings.get('by_emissions', [None])[0] if rankings.get('by_emissions') else None,
        "best_overall": rankings.get('overall', [None])[0] if rankings.get('overall') else None,
        "cost_ranking": rankings.get('by_cost', []),
        "emissions_ranking": rankings.get('by_emissions', []),
    }

    # Generate comparison visualizations if requested
    if generate_charts:
        try:
            plots = tea_lca.generate_comparison_visualizations(
                solvents=solvents,
                polymer_throughput_kg_hr=polymer_throughput_kg_hr
            )
            chart_info = f"\n\n## Solvent Comparison Visualizations\n\n"
            chart_info += f"Comparing: {', '.join([s.title() for s in solvents])}\n\n"
            chart_info += f"Generated {len(plots)} visualizations:\n\n"
            for name, path in plots.items():
                chart_info += f"- **{name.replace('_', ' ').title()}**: `{path}`\n"
            display += chart_info
            structured_data["visualizations"] = plots
        except Exception as e:
            logger.warning(f"Could not generate comparison charts: {e}")

    return json.dumps({"display": display, "data": structured_data}, ensure_ascii=False)


@safe_tool_wrapper
async def generate_tea_lca_visualizations(
    solvent: str,
    polymer_throughput_kg_hr: float = 100.0,
    solvent_to_polymer_ratio: float = 10.0,
    recovery_fraction: float = 0.95,
    process_temp_c: float = 80.0,
    scope: str = "tea",
    chart_type: str = "all"
) -> str:
    """Generate TEA and/or LCA visualizations for solvent recovery analysis.

    Args:
        solvent: Solvent name (e.g., 'toluene', 'acetone', 'ethanol')
        polymer_throughput_kg_hr: Polymer processing rate in kg/hr (default: 100)
        solvent_to_polymer_ratio: Mass ratio of solvent to polymer (default: 10:1)
        recovery_fraction: Solvent recovery efficiency 0-1 (default: 0.95)
        process_temp_c: Process temperature in Celsius (default: 80)
        scope: "tea", "lca", or "both" (default: "tea")
        chart_type: "all", "sensitivity", "cashflow", "cost_breakdown", "energy_flow", or "emissions" (default: "all")

    WHEN TO USE:
    - "Show TEA charts for toluene"
    - "Visualize emissions breakdown for DMF"
    - "Generate cashflow diagram" / "Show sensitivity analysis"
    - "Generate all TEA and LCA visualizations"
    """
    all_plots = {}
    viz_kwargs = dict(
        solvent=solvent,
        polymer_throughput_kg_hr=polymer_throughput_kg_hr,
        solvent_to_polymer_ratio=solvent_to_polymer_ratio,
        recovery_fraction=recovery_fraction,
        process_temp_c=process_temp_c
    )

    # TEA visualizations
    if scope in ("tea", "both"):
        tea_plots = tea_lca.generate_all_tea_visualizations(**viz_kwargs)
        if chart_type == "all":
            all_plots.update(tea_plots)
        else:
            for name, path in tea_plots.items():
                if chart_type in name:
                    all_plots[name] = path

    # LCA visualizations
    if scope in ("lca", "both"):
        lca_plots = tea_lca.generate_all_lca_visualizations(**viz_kwargs)
        if chart_type == "all":
            all_plots.update(lca_plots)
        else:
            for name, path in lca_plots.items():
                if chart_type in name:
                    all_plots[name] = path

    scope_label = {"tea": "TEA", "lca": "LCA", "both": "TEA & LCA"}[scope]
    result = f"## {scope_label} Visualizations for {solvent.title()} Recovery\n\n"
    result += f"Generated {len(all_plots)} visualizations:\n\n"
    for name, path in all_plots.items():
        result += f"- **{name.replace('_', ' ').title()}**: `{path}`\n"

    return result
