"""Phase 1-2: model-free routing sweep + REAL direct fast-path execution.

Iterates parametric query families (solubility x solvents x temperatures, HSP,
TEA/LCA/BioSTEAM, Pareto, chains, research controls) through:
  - the deterministic fallback router (offline route for every query)
  - the direct fast path, EXECUTING real core tools (DuckDB + interpolation)
    when a shape matches — still zero model/API calls.

Writes JSON results and prints a summary of expectation mismatches.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from strap.direct_fast_path import try_direct_tool_fast_path
from strap.route_planner import clear_active_route_plans, fallback_route_plan
from strap.routing_classifier import explain_direct_answer_query
from strap.testing_utils import block_model_access

OUT = _ROOT / "architecture" / "test_results" / "subagent_validation_offline_20260701"

POLYMERS = ["LDPE", "HDPE", "PP", "PS", "PVC", "EVOH", "PET"]
SOLVENTS = ["toluene", "xylene", "dodecane", "THF", "dichloromethane", "o-xylene", "decalin", "DMSO"]
TEMPS = [25, 60, 100, 120, 140]


def build_cases() -> list[dict]:
    cases: list[dict] = []

    def add(family, query, expect_mode=None, expect_primary=None, expect_fast_path=None):
        cases.append({
            "family": family,
            "query": query,
            "expect_mode": expect_mode,
            "expect_primary": expect_primary,
            "expect_fast_path": expect_fast_path,
        })

    # --- Family A: point/range solubility lookups (should stay direct + fast path) ---
    for p, s, t in [
        ("LDPE", "toluene", 100), ("HDPE", "xylene", 120), ("PP", "dodecane", 140),
        ("PS", "THF", 25), ("PVC", "THF", 60), ("EVOH", "DMSO", 120), ("PET", "DMSO", 140),
        ("PP", "o-xylene", 120),
    ]:
        add("A_point_solubility",
            f"What is the solubility of {p} in {s} at {t} C?",
            expect_mode="direct", expect_fast_path=True)
    # decalin has no interpolation data — correct behavior is direct mode with
    # graceful fall-through to the orchestrator (no deterministic execution).
    add("A_point_solubility", "What is the solubility of LDPE in decalin at 100 C?",
        expect_mode="direct", expect_fast_path=None)
    for p, s, lo, hi in [("LDPE", "dodecane", 25, 140), ("PP", "xylene", 60, 140), ("PS", "toluene", 25, 100)]:
        add("A_range_solubility",
            f"Show the solubility of {p} in {s} from {lo} to {hi} C.",
            expect_mode="direct", expect_fast_path=True)

    # --- Family B: solvent-candidate lookups with ceilings (direct) ---
    for p in ["LDPE", "PP", "PVC"]:
        add("B_candidates_single",
            f"What solvents dissolve {p}?", expect_mode="direct", expect_fast_path=True)
    add("B_candidates_multi",
        "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, identify solvents "
        "that are promising for dissolving any one of the components below 100 deg C.",
        expect_mode="direct", expect_fast_path=True)
    add("B_candidates_multi",
        "Give candidate solvents for HDPE/PP/PS below 120 C.",
        expect_mode="direct", expect_fast_path=True)

    # --- Family C: HSP / statistics-ml ---
    add("C_hsp", "Evaluate PS versus PVC using Hansen solubility parameters and compare RED values for THF.",
        expect_mode="specialists", expect_primary="statistics-ml")
    add("C_hsp", "Make an HSP RED heatmap for EVOH and PET in polar aprotic solvents.",
        expect_mode="specialists", expect_primary="statistics-ml")
    add("C_hsp", "Predict the glass transition temperature for a new polyamide and report confidence.",
        expect_mode="specialists", expect_primary="statistics-ml")
    add("C_hsp", "Run a statistical correlation between solvent LogP and PS solubility with confidence intervals.",
        expect_mode="specialists", expect_primary="statistics-ml")

    # --- Family D: separation / process design ---
    add("D_separation", "Generate the best separation sequence for LDPE/EVOH/PET below 100 C.",
        expect_mode="specialists", expect_primary="separation-engineer")
    add("D_separation", "Rank solvent selectivity for LDPE over PET at 90 C and identify the best selective solvent.",
        expect_mode="specialists", expect_primary="separation-engineer")
    add("D_separation", "Plan a sequential separation for HDPE, PP, and PS with antisolvent precipitation options.",
        expect_mode="specialists", expect_primary="separation-engineer")

    # --- Family E: TEA/LCA BioSTEAM with parameter variations ---
    biosteam_variants = [
        "Run a techno-economic analysis for recovering PE from a PE/EVOH multilayer using toluene.",
        "Estimate the MSP for LDPE recovery with dodecane at 8000 tonnes per year under the CHP energy configuration.",
        "Compare CAPEX and OPEX for STRAP processing of PET/EVOH at 5000 vs 20000 tonnes per year.",
        "What is the GWP per kg of recovered polymer for PE dissolution in toluene under Grid versus Grid+Boiler energy scenarios?",
        "Run BioSTEAM for a 60% PE / 40% EVOH feed at 8000 tpy and report MSP, TCI, AOC, and GWP.",
        "Do a life cycle assessment of the STRAP process for polypropylene recovery with xylene, including emissions breakdown.",
        "Batch-screen toluene, xylene, and dodecane for PE recovery economics and rank by MSP.",
        "Simulate multi-polymer sequential recovery of PE then EVOH and report blended MSP and combined TCI.",
    ]
    for q in biosteam_variants:
        add("E_biosteam", q, expect_mode="specialists", expect_primary="biosteam-analyst")

    # --- Family F: optimization / Pareto with parameter variations ---
    pareto_variants = [
        "Optimize the processing pathway for maximum profit for a PE/EVOH/PET feed.",
        "Generate the Pareto frontier of total cost versus emissions for a 60/40 PE/EVOH feed at 8000 tpy.",
        "Maximize circularity subject to profit >= 0 for multilayer film waste and show the trade-off frontier.",
        "Run the waste-management superstructure with at least 1 STRAP wash step and up to 2 wash steps, maximizing profit.",
        "Sweep feed compositions from 20% to 80% PE and produce Pareto slices of profit versus GWP for each composition.",
        "Which processing pathway minimizes emissions for a PP/PS/PVC mixed stream? No Pareto analysis needed.",
    ]
    for q in pareto_variants:
        add("F_optimization", q, expect_mode="specialists", expect_primary="optimization-engineer")

    # --- Family G: chained workflows ---
    add("G_chain", "Design a separation sequence for LDPE/PP/EVOH below 120 C, then estimate MSP and GWP for the route.",
        expect_mode="specialists", expect_primary="separation-engineer")
    add("G_chain", "Screen contaminant removal for PFAS in recycled LDPE, then qualify solvents for the separation route.",
        expect_mode="specialists", expect_primary="contaminant-removal-analyst")
    add("G_chain", "Find the optimal separation sequence for PE/EVOH and then create a separation tree plot.",
        expect_mode="specialists", expect_primary="separation-engineer")

    # --- Family H: research controls (must not be hijacked by domain tokens) ---
    add("H_research", "Find recent journal articles on techno-economic analysis of solvent-based polyolefin recycling.",
        expect_mode="specialists", expect_primary="scholar-researcher")
    add("H_research", "Search patents about Hansen solubility parameter screening for polymer separation.",
        expect_mode="specialists", expect_primary="patent-researcher")
    add("H_research", "What do our indexed documents say about BioSTEAM TEA assumptions? Cite retrieved chunks.",
        expect_mode="specialists", expect_primary="rag-analyst")

    # --- Family I: safety ---
    add("I_safety", "Show the safety card for toluene at 100 C operating temperature.",
        expect_fast_path=True)
    add("I_safety", "Compare GSK solvent sustainability scores for toluene, xylene, and DMSO.",
        expect_mode="specialists", expect_primary="safety-analyst")

    # --- Family J: temperature-constraint phrasing variants on one query ---
    for phrase in ["below 100 C", "under 100C", "up to 100 deg C", "at or below 100 °C", "below 212 F"]:
        add("J_temp_variants",
            f"What solvents dissolve HDPE {phrase}?",
            expect_mode="direct", expect_fast_path=True)

    return cases


def run() -> dict:
    results = []
    mismatches = []
    with block_model_access():
        for case in build_cases():
            clear_active_route_plans()
            query = case["query"]
            plan = fallback_route_plan(query)
            direct = explain_direct_answer_query(query)

            fast_path_result = None
            fast_error = None
            try:
                fp = try_direct_tool_fast_path(query)
                if fp is not None:
                    fast_path_result = {
                        "tool_name": fp.tool_name,
                        "display_chars": len(fp.display or ""),
                        "display_head": (fp.display or "")[:160],
                        "route_decision": fp.route_decision,
                    }
            except Exception as exc:  # noqa: BLE001 - collect, don't crash sweep
                fast_error = f"{type(exc).__name__}: {exc}"

            row = {
                **case,
                "fallback_mode": plan.mode,
                "fallback_steps": plan.subagent_names(),
                "direct_reason": direct.get("reason"),
                "fast_path": fast_path_result,
                "fast_path_error": fast_error,
            }

            problems = []
            if fast_error:
                problems.append(f"fast path raised: {fast_error}")
            if case["expect_mode"] and plan.mode != case["expect_mode"]:
                problems.append(f"mode {plan.mode} != expected {case['expect_mode']}")
            if case["expect_primary"] and (
                not plan.subagent_names() or plan.subagent_names()[0] != case["expect_primary"]
            ):
                problems.append(
                    f"primary {plan.subagent_names()[:1]} != expected {case['expect_primary']}"
                )
            if case["expect_fast_path"] is True and fast_path_result is None:
                problems.append("expected fast-path execution, got none")
            if case["expect_fast_path"] is True and fast_path_result and not fast_path_result["display_chars"]:
                problems.append("fast path returned empty display")
            # sanity: response should mention the polymer it was asked about
            if fast_path_result:
                mentioned = [p for p in POLYMERS if re.search(rf"\b{p}\b", query)]
                display_upper = (fast_path_result["display_head"] or "").upper()
                if mentioned and not any(p in display_upper for p in mentioned):
                    row["note"] = "display head does not mention requested polymer (may be fine for tables)"

            row["problems"] = problems
            if problems:
                mismatches.append(row)
            results.append(row)

    summary = {
        "total": len(results),
        "with_problems": len(mismatches),
        "fast_path_executions": sum(1 for r in results if r["fast_path"]),
        "fast_path_errors": sum(1 for r in results if r["fast_path_error"]),
        "by_family": {},
    }
    for row in results:
        fam = summary["by_family"].setdefault(row["family"], {"total": 0, "problems": 0})
        fam["total"] += 1
        fam["problems"] += bool(row["problems"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1_routing_sweep.json").write_text(json.dumps(
        {"summary": summary, "results": results}, indent=2))

    print(json.dumps(summary, indent=2))
    for row in mismatches:
        print(f"\nPROBLEM [{row['family']}] {row['query'][:90]}")
        for problem in row["problems"]:
            print(f"   - {problem}")
        print(f"   fallback={row['fallback_mode']}/{row['fallback_steps']} reason={row['direct_reason']}")
    return summary


if __name__ == "__main__":
    run()
