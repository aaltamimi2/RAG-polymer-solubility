"""Phase 5: typed-runtime BioSTEAM / Pareto compile + execution sweep (no API).

Validates, per parameter variant:
  - compile: mode, workflow, extracted tool args (polymer, solvent, capacity,
    energy case incl. named configs, wash steps), artifact contracts,
    recorded assumptions
  - runtime: run_typed_runtime under enforce_selected with the deterministic
    TEST callable registry — asserts which queries execute in the typed lane
    vs defer to the specialist (legacy_fallback)
  - research-token queries: the production route-plan gate must refuse
    interception (compile-level strictness alone would dead-end them)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from strap.planning.compiler import compile_request
from strap.planning.config import (
    DEFAULT_SELECTED_ENFORCEMENT_ARTIFACTS,
    DEFAULT_SELECTED_ENFORCEMENT_WORKFLOWS,
    PlannerConfig,
)
from strap.planning.runtime import run_typed_runtime

from strap.route_planner import RoutePlan, RoutePlanner, RouteStep
from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware
from strap.testing_utils import block_model_access

OUT = _ROOT / "architecture" / "test_results" / "subagent_validation_offline_20260701"
CONFIG = PlannerConfig(
    mode="enforce_selected",
    selected_enforcement_artifacts=set(DEFAULT_SELECTED_ENFORCEMENT_ARTIFACTS),
    selected_enforcement_workflows=set(DEFAULT_SELECTED_ENFORCEMENT_WORKFLOWS),
)

BANK_10_1 = (
    "For a mixed plastic feedstock of 8000 tonnes/year composed of 60% PE and 40% EVOH under "
    "scenario A, have the separation engineer propose the top 3 solvent candidates per polymer "
    "using the dynamic-programming planner and include separation plots for the shortlisted PE "
    "and EVOH solvents. Then pass exactly those shortlisted candidates to the optimization "
    "engineer to maximize profit, requiring at least 1 STRAP wash step and allowing up to 2 "
    "wash steps. Finally, have the visualization specialist create a separation tree plot and "
    "an optimization figure summarizing the routed result. Report the selected solvents, total "
    "profit, total cost, circularity, and save all plots."
)

CASES: list[dict] = [
    dict(name="tea_c1_explicit",
         query="Run BioSTEAM TEA/LCA for LDPE with Cyclohexane under energy case C1 and report MSP, TCI, AOC, and GWP.",
         expect_mode="single_agent", expect_args={"energy_case": "C1", "target_plastic": "LDPE"},
         expect_runtime="executed"),
    dict(name="tea_named_chp",
         query="Run BioSTEAM TEA/LCA for LDPE with dodecane at 8000 tonnes per year under the CHP energy configuration; report MSP and GWP.",
         expect_mode="single_agent", expect_args={"energy_case": "C1", "processing_capacity": 8000},
         expect_runtime="executed"),
    dict(name="tea_named_grid_boiler",
         query="Simulate the BioSTEAM STRAP process for PET in DMSO under the Grid+Boiler energy scenario and give techno-economic results.",
         expect_mode="single_agent", expect_args={"energy_case": "C3", "target_plastic": "PET"},
         expect_runtime="executed"),
    dict(name="tea_no_energy_defaults_c1",
         query="Run a BioSTEAM TEA for PE recovery using toluene and report MSP, TCI, AOC, and GWP.",
         expect_mode="single_agent", expect_args={"energy_case": "C1"},
         expect_assumption="energy_case", expect_runtime="executed"),
    dict(name="tea_plain_tea_wording",
         query="Give me a techno-economic analysis of PE recovery with toluene under energy case C2.",
         expect_mode="single_agent", expect_args={"energy_case": "C2"},
         expect_runtime="executed"),
    dict(name="tea_plot",
         query="Run a BioSTEAM TEA and LCA for PE in toluene under case C1 and plot the cost and GWP breakdowns.",
         expect_mode="planned_workflow", expect_artifacts={"biosteam_tea_lca_result", "biosteam_tea_lca_plot"},
         expect_runtime="executed"),
    dict(name="tea_batch_defers_to_specialist",
         query="Batch-screen toluene, xylene, and dodecane for PE recovery TEA under case C1 and rank by MSP.",
         expect_mode="single_agent", expect_assumption="typed_enforcement",
         expect_runtime="legacy_fallback"),
    dict(name="tea_multi_polymer_defers",
         query="Run BioSTEAM multi-polymer sequential recovery for PE then EVOH with toluene under case C1 and report blended MSP.",
         expect_assumption="typed_enforcement", expect_runtime="legacy_fallback"),
    dict(name="pareto_cost_vs_emissions",
         query="Generate the Pareto frontier of total cost versus emissions for a 60% PE / 40% EVOH feed at 8000 tpy.",
         expect_artifacts_any={"optimization_pareto_front", "optimization_pareto_landscape"},
         expect_runtime="executed"),
    # NOTE: direct Pareto slices (no separation stage) currently dispatch to
    # point optimization — slices are reachable only via the routed workflow.
    # Recorded as a findings item; this case uses the routed phrasing.
    dict(name="pareto_slices_routed",
         query="Have the separation engineer shortlist solvents for a PE/EVOH feed at 8000 tpy using the dynamic-programming planner, then run the optimization for fixed feed compositions of 20% PE, 50% PE, and 80% PE producing Pareto slices of profit versus GWP with one PNG per composition.",
         expect_workflow="routed_optimization_slices",
         expect_artifacts_any={"optimization_pareto_slices", "optimization_pareto_slices_plot"},
         expect_runtime="executed"),
    dict(name="routed_optimization_bank_query",
         query=BANK_10_1,
         expect_workflow="routed_optimization",
         expect_args_any={"1": 1, "2": 2},
         expect_runtime="executed"),
    dict(name="pareto_landscape",
         query="Show the full Pareto landscape of all feasible points for total cost versus emissions for a feed of 50% PE and 50% EVOH at 10000 tonnes per year.",
         expect_artifacts_any={"optimization_pareto_landscape"},
         expect_runtime="executed"),
    # --- plan-driven intent: NO biosteam keywords anywhere in the query ---
    dict(name="tea_no_keywords_keyword_only_misses",
         query="What would it cost per kilogram to recover PE from packaging waste using toluene at 8000 tonnes per year under case C1?",
         expect_compile_status="unsupported", expect_runtime="legacy_fallback"),
    dict(name="tea_no_keywords_plan_driven",
         query="What would it cost per kilogram to recover PE from packaging waste using toluene at 8000 tonnes per year under case C1?",
         plan_deliverables=["biosteam_tea_lca_result"],
         expect_mode="single_agent", expect_artifacts={"biosteam_tea_lca_result"},
         expect_runtime="executed"),
]

RESEARCH_CASES = [
    ("research_tea_tokens", "Search Web of Science for recent TEA/LCA studies of solvent-based plastics recycling.", "scholar-researcher"),
    ("rag_biosteam_tokens", "Search the literature RAG for BioSTEAM TEA/LCA assumptions used in STRAP economics papers.", "rag-analyst"),
]


def _flatten(obj) -> str:
    return json.dumps(obj, default=str).lower()


def _static_registry_for(plan) -> dict:
    """Deterministic stub wrappers for every callable the plan requires.

    Validates executor mechanics (dependency order, authorization, output
    contract verification, ledger) without running real simulations.
    """
    import tempfile

    from strap.planning.runtime_wrappers import make_static_artifact_wrapper

    registry = {}
    tmpdir = Path(tempfile.mkdtemp(prefix="phase5_artifacts_"))
    for step in plan.steps if plan else []:
        if not step.allowed_tools:
            continue
        overrides = {}
        for out in step.output_contracts or []:
            for contract in out.artifact_contracts:
                if contract.path_policy == "required":
                    path = tmpdir / f"{contract.artifact_type}.png"
                    path.write_bytes(b"stub")
                    overrides[contract.artifact_type] = [str(path)]
        registry[step.allowed_tools[0]] = make_static_artifact_wrapper(output_paths=overrides or None)
    return registry


def run() -> dict:
    rows, problems = [], []

    with block_model_access():
        for case in CASES:
            context = (
                {"plan_requested_artifact_types": list(case["plan_deliverables"])}
                if case.get("plan_deliverables") else None
            )
            result = compile_request(case["query"], context=context)
            plan = result.plan
            registry = _static_registry_for(plan)
            args_flat = _flatten([getattr(s, "tool_args_template", {}) for s in (plan.steps if plan else [])])
            artifacts = sorted({
                contract.artifact_type
                for s in (plan.steps if plan else [])
                for out in (s.output_contracts or [])
                for contract in out.artifact_contracts
            }) if plan else []
            assumptions = [a.key for a in getattr(plan, "assumptions", [])] if plan else []

            runtime = run_typed_runtime(case["query"], config=CONFIG, context=context,
                                        callable_registry=registry, persist=False)

            row = {
                "name": case["name"], "query": case["query"][:90],
                "compile_status": result.status, "mode": getattr(plan, "mode", None),
                "workflow_id": getattr(plan, "workflow_id", None),
                "artifacts": artifacts, "assumptions": assumptions,
                "runtime_status": runtime.status, "runtime_reason": (runtime.reason or "")[:110],
            }

            checks = []
            if case.get("expect_compile_status") and result.status != case["expect_compile_status"]:
                checks.append(f"compile={result.status} expected {case['expect_compile_status']}")
            if case.get("expect_mode") and row["mode"] != case["expect_mode"]:
                checks.append(f"mode={row['mode']} expected {case['expect_mode']}")
            if case.get("expect_workflow") and row["workflow_id"] != case["expect_workflow"]:
                checks.append(f"workflow={row['workflow_id']} expected {case['expect_workflow']}")
            for key, value in (case.get("expect_args") or {}).items():
                token = f'"{key}": "{value}"'.lower() if isinstance(value, str) else f'"{key}": {value}'
                if token not in args_flat:
                    checks.append(f"args missing {key}={value}")
            for key, value in (case.get("expect_args_any") or {}).items():
                if str(value).lower() not in args_flat:
                    checks.append(f"args missing ~{key}={value}")
            for artifact in case.get("expect_artifacts") or ():
                if artifact not in artifacts:
                    checks.append(f"missing artifact {artifact}")
            if case.get("expect_artifacts_any") and not set(case["expect_artifacts_any"]) & set(artifacts):
                checks.append(f"none of {sorted(case['expect_artifacts_any'])} present")
            if case.get("expect_assumption") and case["expect_assumption"] not in assumptions:
                checks.append(f"assumption {case['expect_assumption']} not recorded")
            if case.get("expect_runtime") and runtime.status != case["expect_runtime"]:
                checks.append(f"runtime={runtime.status} expected {case['expect_runtime']} ({row['runtime_reason']})")

            row["problems"] = checks
            problems.extend(f"{case['name']}: {c}" for c in checks)
            rows.append(row)

        # Research-token queries: production stack protection = route-plan gate.
        for name, query, specialist in RESEARCH_CASES:
            plan = RoutePlan(query=query, mode="specialists",
                             steps=(RouteStep(subagent=specialist),))
            planner = RoutePlanner(backend=None)
            planner._cache[query.strip().casefold().replace("  ", " ")] = plan  # simulate planner decision
            from strap.route_planner import activate_route_plan, normalize_query_key
            planner._cache.clear()
            planner._cache[normalize_query_key(query)] = plan
            activate_route_plan(plan)
            middleware = TypedRuntimeMiddleware(route_planner=planner)
            gate_permits = middleware._plan_permits_typed_runtime(query)
            compile_result = compile_request(query)
            rows.append({
                "name": name, "query": query[:90],
                "compile_status": compile_result.status,
                "gate_permits_interception": gate_permits,
                "problems": ["route-plan gate failed to refuse interception"] if gate_permits else [],
            })
            if gate_permits:
                problems.append(f"{name}: gate failed to refuse")

    doc = {
        "summary": {
            "cases": len(rows),
            "with_problems": sum(1 for r in rows if r["problems"]),
        },
        "problems": problems,
        "cases": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase5_biosteam_pareto_sweep.json").write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc["summary"], indent=2))
    for row in rows:
        marker = "FAIL" if row["problems"] else "ok"
        print(f"[{marker}] {row['name']:<34} compile={row.get('compile_status'):<22} "
              f"mode={row.get('mode')} runtime={row.get('runtime_status', row.get('gate_permits_interception'))}")
        for check in row["problems"]:
            print(f"       - {check}")
    return doc


if __name__ == "__main__":
    run()
