from __future__ import annotations

import json

from strap.planning.compiler import CompileResult, PlannerBackend, compile_request, compile_shadow_diagnostics
from strap.planning.capability_registry import exported_tool_names
from strap.planning.query_bank import validated_query_bank_rows


FIXED_TIME = "2026-04-26T00:00:00+00:00"


class StubBackend:
    planner_model_id = "stub/planner"

    def __init__(self, payload):
        self.payload = payload

    def propose_plan_payload(self, query, facts):
        return self.payload


def _step_snapshot(result: CompileResult) -> list[dict[str, object]]:
    assert result.plan is not None
    return [
        {
            "step_id": step.step_id,
            "role": step.role,
            "execution_kind": step.execution_kind,
            "allowed_tools": step.allowed_tools,
            "outputs": [
                artifact.artifact_type
                for contract in step.output_contracts
                for artifact in contract.artifact_contracts
            ],
            "depends_on": step.depends_on,
        }
        for step in result.plan.steps
    ]


def test_compile_safety_card_snapshot():
    result = compile_request(
        "Show a safety card for THF at 60 C.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.compiler_version
    assert result.plan.capability_registry_version
    assert result.extracted_facts["solvents"] == ["Tetrahydrofuran"]
    assert result.extracted_facts["temperatures_c"] == [60.0]
    assert _step_snapshot(result) == [
        {
            "step_id": "safety_assessment",
            "role": "safety-analyst",
            "execution_kind": "tool",
            "allowed_tools": ["get_solvent_safety_card"],
            "outputs": ["solvent_safety_card"],
            "depends_on": [],
        }
    ]


def test_fact_extractor_accepts_deg_c_temperature_notation():
    from strap.planning.extractors import extract_facts

    facts = extract_facts("Identify LDPE/EVOH/PET solvent candidates below 100 deg C.")

    assert facts.temperatures_c == [100.0]


def test_fact_extractor_normalizes_fahrenheit_kelvin_and_wrapped_paths(tmp_path):
    from strap.planning.extractors import extract_facts

    query = (
        "Identify LDPE/EVOH/PET solvent candidates below 212 fahrenehit and save to "
        f"{tmp_path}/case-1/01-ldpe-evoh-p\n"
        "  et\n"
        "    -solubility/json."
    )

    facts = extract_facts(query)

    assert abs(facts.temperatures_c[0] - 100.0) < 1e-6
    assert facts.output_dir == str(tmp_path / "case-1" / "01-ldpe-evoh-pet-solubility" / "json")

    fahrenheit = extract_facts("Identify LDPE/EVOH/PET solvent candidates below 212 degrees Fahrenheit.")
    kelvin = extract_facts("Identify LDPE/EVOH/PET solvent candidates below 373.15 degrees Kelvin.")
    assert abs(fahrenheit.temperatures_c[0] - 100.0) < 1e-6
    assert abs(kelvin.temperatures_c[0] - 100.0) < 1e-6


def test_compile_hsp_heatmap_snapshot():
    result = compile_request(
        "Use the Hansen model to screen polyolefins against nonpolar solvents and show the RED heatmap.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert _step_snapshot(result) == [
        {
            "step_id": "hsp_screen",
            "role": "statistics-ml",
            "execution_kind": "tool",
            "allowed_tools": ["screen_hsp_solubility_matrix"],
            "outputs": ["hsp_red_heatmap"],
            "depends_on": [],
        }
    ]
    args = result.plan.steps[0].tool_args_template
    assert args["polymer_category"] == "polyolefins"
    assert args["solvent_polarity"] == "nonpolar"


def test_compile_hsp_compatibility_heatmap_does_not_become_separation_heatmap():
    result = compile_request(
        "Screen nylons against polar aprotic solvents using HSP and return the batch compatibility heatmap.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.extracted_facts["requested_artifact_types"] == ["hsp_red_heatmap"]
    assert result.extracted_facts["hsp_polymer_category"] == "nylons"
    assert result.extracted_facts["hsp_solvent_polarity"] == "polar aprotic"
    assert _step_snapshot(result)[0]["allowed_tools"] == ["screen_hsp_solubility_matrix"]


def test_compile_hsp_heatmap_propagates_output_dir():
    result = compile_request(
        "Use HSP to screen polyolefins against nonpolar solvents and save the RED heatmap to /tmp/hsp_case.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.steps[0].tool_args_template["output_dir"] == "/tmp/hsp_case"


def test_compile_separation_tree_snapshot():
    result = compile_request(
        "Create a separation tree for LDPE, EVOH, and PET at 100 C and save to /tmp/separation_case.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.final_response_contract.require_paths is True
    assert _step_snapshot(result) == [
        {
            "step_id": "plot_separation_visualization",
            "role": "visualization-specialist",
            "execution_kind": "tool",
            "allowed_tools": ["create_separation_tree_plot"],
            "outputs": ["separation_tree_plot"],
            "depends_on": [],
        }
    ]
    assert result.plan.steps[0].tool_args_template["output_dir"] == "/tmp/separation_case"


def test_compile_selectivity_heatmap_snapshot():
    result = compile_request(
        "Create a selectivity heatmap for LDPE, EVOH, and PET with Cyclohexane and Toluene at 100 C.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.final_response_contract.require_paths is True
    assert result.extracted_facts["solvents"] == ["Toluene", "Cyclohexane"]
    assert _step_snapshot(result) == [
        {
            "step_id": "plot_separation_visualization",
            "role": "visualization-specialist",
            "execution_kind": "tool",
            "allowed_tools": ["create_selectivity_heatmap"],
            "outputs": ["separation_selectivity_heatmap"],
            "depends_on": [],
        }
    ]


def test_compile_biosteam_tea_lca_snapshot():
    result = compile_request(
        "Estimate CAPEX/OPEX/GWP for a STRAP process case: LDPE recovered with Cyclohexane "
        "at 79.7 C, 8000 tonnes/year total feed capacity, 60 wt% LDPE in the feed, under C2.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.extracted_facts["energy_case"] == "C2"
    assert _step_snapshot(result) == [
        {
            "step_id": "run_biosteam_tea_lca",
            "role": "biosteam-analyst",
            "execution_kind": "tool",
            "allowed_tools": ["run_biosteam_simulation"],
            "outputs": ["biosteam_tea_lca_result"],
            "depends_on": [],
        }
    ]
    args = result.plan.steps[0].tool_args_template
    assert args["solvent"] == "Cyclohexane"
    assert args["target_plastic"] == "LDPE"
    assert args["energy_case"] == "C2"
    assert args["processing_capacity"] == 8000.0
    assert args["target_plastic_percent"] == 60.0
    assert args["dissolution_temp_c"] == 79.7


def test_compile_biosteam_plot_snapshot():
    result = compile_request(
        "Estimate BioSTEAM TEA/LCA for PET with Toluene under energy case C1 and create a chart in /tmp/biosteam_case.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.mode == "planned_workflow"
    assert _step_snapshot(result) == [
        {
            "step_id": "run_biosteam_tea_lca",
            "role": "biosteam-analyst",
            "execution_kind": "tool",
            "allowed_tools": ["run_biosteam_simulation"],
            "outputs": ["biosteam_tea_lca_result"],
            "depends_on": [],
        },
        {
            "step_id": "plot_biosteam_tea_lca",
            "role": "visualization-specialist",
            "execution_kind": "tool",
            "allowed_tools": ["visualize_biosteam_results"],
            "outputs": ["biosteam_tea_lca_plot"],
            "depends_on": ["run_biosteam_tea_lca"],
        },
    ]
    assert result.plan.steps[1].tool_args_template["output_dir"] == "/tmp/biosteam_case"


def test_compile_biosteam_defaults_energy_case_with_assumption():
    """No energy case in the query compiles with the tool default (C1) recorded
    as an explicit assumption instead of dead-ending in clarification."""
    result = compile_request("Estimate CAPEX/OPEX/GWP for LDPE recovered with Cyclohexane.", created_at=FIXED_TIME)

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.steps[0].tool_args_template["energy_case"] == "C1"
    assumption = next(a for a in result.plan.assumptions if a.key == "energy_case")
    assert assumption.value == "C1"
    assert assumption.source == "default"


def test_compile_biosteam_accepts_named_energy_configurations():
    """CHP / Grid+Boiler vocabulary (as advertised in the subagent description)
    maps to C1/C3 instead of failing extraction."""
    chp = compile_request(
        "Run BioSTEAM TEA for PE with toluene under the CHP energy configuration.",
        created_at=FIXED_TIME,
    )
    assert chp.status == "compiled"
    assert chp.plan.steps[0].tool_args_template["energy_case"] == "C1"

    boiler = compile_request(
        "Run BioSTEAM TEA for PE with toluene under the Grid+Boiler energy scenario.",
        created_at=FIXED_TIME,
    )
    assert boiler.status == "compiled"
    assert boiler.plan.steps[0].tool_args_template["energy_case"] == "C3"


def test_compile_biosteam_multi_solvent_defers_to_specialist():
    """Batch solvent comparisons compile but are not typed-enforcement
    targets — the biosteam-analyst specialist owns run_biosteam_batch."""
    result = compile_request(
        "Batch-screen toluene, xylene, and dodecane for PE recovery TEA under case C1 and rank by MSP.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert any(a.key == "typed_enforcement" and a.value == "deferred_to_specialist"
               for a in result.plan.assumptions)
    required = {
        contract.artifact_type
        for step in result.plan.steps
        for out in step.output_contracts
        for contract in out.artifact_contracts
        if contract.required
    }
    assert "biosteam_tea_lca_result" not in required


def test_compile_biosteam_intent_wins_over_generic_optimize_word():
    result = compile_request(
        "Optimize BioSTEAM TEA/LCA for LDPE with Cyclohexane under C2.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.intent_family == "biosteam_tea_lca"
    assert _step_snapshot(result)[0]["allowed_tools"] == ["run_biosteam_simulation"]


def test_compile_biosteam_waste_feedstock_phrase_still_routes_to_biosteam():
    result = compile_request(
        "Estimate BioSTEAM TEA/LCA for LDPE waste with Cyclohexane under C2.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.intent_family == "biosteam_tea_lca"
    assert _step_snapshot(result)[0]["allowed_tools"] == ["run_biosteam_simulation"]


def test_compile_optimized_biosteam_waste_feedstock_phrase_still_routes_to_biosteam():
    result = compile_request(
        "Optimize BioSTEAM TEA/LCA for LDPE waste feedstock with Cyclohexane under C2.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.intent_family == "biosteam_tea_lca"
    assert _step_snapshot(result)[0]["allowed_tools"] == ["run_biosteam_simulation"]


def test_compile_biosteam_landfill_waste_phrase_still_routes_to_biosteam():
    result = compile_request(
        "Estimate BioSTEAM TEA/LCA for LDPE landfill waste with Cyclohexane under C2.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.intent_family == "biosteam_tea_lca"
    assert _step_snapshot(result)[0]["allowed_tools"] == ["run_biosteam_simulation"]


def test_compile_waste_management_biosteam_pareto_stays_optimizer_owned():
    result = compile_request(
        "Run waste management Pareto optimization with BioSTEAM TEA/LCA assumptions for "
        "8000 tonnes/year composed of 60% LDPE and 40% EVOH under scenario A.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.intent_family == "optimization"
    assert _step_snapshot(result)[0]["allowed_tools"] == ["run_waste_management_pareto"]
    assert _step_snapshot(result)[0]["outputs"] == ["optimization_pareto_front", "optimization_pareto_landscape"]


def test_compile_direct_optimization_snapshot():
    query = (
        "Optimize waste management for a mixed plastic feedstock of 8000 tonnes/year composed of "
        "60% PE and 40% EVOH under scenario A. Restrict the candidate solvents to Toluene or "
        "Heptane for PE and Pyridazine or Ethylene Glycol for EVOH. Maximize profit, require at "
        "least 1 STRAP wash step and allow up to 2 wash steps."
    )
    result = compile_request(query, created_at=FIXED_TIME)

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.mode == "single_agent"
    assert result.extracted_facts["feed_capacity_tpy"] == 8000.0
    assert result.extracted_facts["feed_composition"] == {"PE": 0.6, "EVOH": 0.4}
    assert _step_snapshot(result) == [
        {
            "step_id": "optimize_point",
            "role": "optimization-engineer",
            "execution_kind": "tool",
            "allowed_tools": ["run_waste_management_optimization"],
            "outputs": ["optimization_point_result"],
            "depends_on": [],
        }
    ]


def test_compile_no_pareto_defaults_to_max_profit_point_optimization():
    result = compile_request(
        "Optimize waste management for 8000 tonnes/year 60% PE 40% EVOH. I do not want Pareto.",
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.extracted_facts["pareto_negated"] is True
    assert result.extracted_facts["requested_artifact_types"] == ["optimization_point_result"]
    step = result.plan.steps[0]
    assert step.allowed_tools == ["run_waste_management_optimization"]
    assert step.tool_args_template["objective"] == "max_profit"


def test_compile_single_objective_aliases_choose_requested_objective():
    cases = {
        "Run max circularity for 8000 tonnes/year 60% PE 40% EVOH": "max_circularity",
        "Minimize emissions for 8000 tonnes/year 60% PE 40% EVOH waste management": "min_emissions",
        "Minimize total cost for 8000 tonnes/year 60% PE 40% EVOH": "min_total_cost",
    }

    for query, objective in cases.items():
        result = compile_request(query, created_at=FIXED_TIME)
        assert result.status == "compiled"
        step = result.plan.steps[0]
        assert step.allowed_tools == ["run_waste_management_optimization"]
        assert step.tool_args_template["objective"] == objective
        assert result.extracted_facts["requested_artifact_types"] == ["optimization_point_result"]


def test_compile_routed_no_pareto_uses_point_optimization_after_handoff():
    query = (
        "For 8000 tonnes/year composed of 60% LDPE and 40% EVOH, have the separation engineer "
        "propose the top 4 solvent candidates per polymer using dynamic programming, then pass "
        "those candidates to the optimization engineer. Do not run Pareto; return the single "
        "max-profit route."
    )
    result = compile_request(query, created_at=FIXED_TIME)

    assert result.status == "compiled"
    assert result.plan.mode == "planned_workflow"
    steps = result.plan.steps
    assert steps[2].step_id == "optimize_point"
    assert steps[2].allowed_tools == ["run_waste_management_optimization"]
    assert steps[2].tool_args_template["objective"] == "max_profit"
    assert _step_snapshot(result)[2]["outputs"] == ["optimization_point_result"]


def test_compile_routed_pareto_snapshot():
    query = (
        "For a mixed plastic feedstock of 8000 tonnes/year composed of 20% LDPE, 60% EVOH, and "
        "20% PET under scenario A, have the separation engineer propose the top 12 solvent "
        "candidates per polymer using the dynamic-programming planner with temperature "
        "recommendations. Then pass those candidates to the optimization engineer to run a "
        "cost-vs-circularity Pareto landscape with 100 points, requiring at least 1 STRAP wash "
        "and allowing up to 2 washes. Finally, plot all feasible points and highlight the frontier."
    )
    result = compile_request(query, created_at=FIXED_TIME)

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.mode == "planned_workflow"
    assert result.extracted_facts["feed_composition"] == {"PE": 0.2, "EVOH": 0.6, "PET": 0.2}
    assert _step_snapshot(result) == [
        {
            "step_id": "separation_candidates",
            "role": "separation-engineer",
            "execution_kind": "tool",
            "allowed_tools": ["plan_multiple_separation_schemes"],
            "outputs": ["separation_topk_sequences", "optimization_stage_candidates"],
            "depends_on": [],
        },
        {
            "step_id": "build_optimization_handoff",
            "role": "handoff_adapter",
            "execution_kind": "handoff_adapter",
            "allowed_tools": ["build_handoff"],
            "outputs": ["optimization_stage_candidates", "handoff_payload"],
            "depends_on": ["separation_candidates"],
        },
        {
            "step_id": "optimize_pareto",
            "role": "optimization-engineer",
            "execution_kind": "tool",
            "allowed_tools": ["run_waste_management_pareto"],
            "outputs": ["optimization_pareto_front", "optimization_pareto_landscape"],
            "depends_on": ["build_optimization_handoff"],
        },
        {
            "step_id": "plot_optimization",
            "role": "visualization-specialist",
            "execution_kind": "tool",
            "allowed_tools": ["plot_optimization_pareto_front"],
            "outputs": ["optimization_pareto_plot"],
            "depends_on": ["optimize_pareto"],
        },
    ]


def test_compile_multislice_snapshot():
    query = (
        "For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year under scenario A, have the "
        "separation engineer propose the top 12 solvent candidates per polymer using the "
        "dynamic-programming planner with temperature recommendations. Then run cost-vs-circularity "
        "Pareto landscape optimization for five fixed feed compositions: 20/60/20, 34/33/33, "
        "60/20/20, 20/20/60, and 5/5/90. Require at least 1 STRAP wash and allow up to 2 washes. "
        "Save one PNG per composition and one combined comparison plot showing all feasible points."
    )
    result = compile_request(query, created_at=FIXED_TIME)

    assert result.status == "compiled"
    assert result.extracted_facts["composition_slices"] == [
        {"PE": 0.2, "EVOH": 0.6, "PET": 0.2},
        {"PE": 0.34, "EVOH": 0.33, "PET": 0.33},
        {"PE": 0.6, "EVOH": 0.2, "PET": 0.2},
        {"PE": 0.2, "EVOH": 0.2, "PET": 0.6},
        {"PE": 0.05, "EVOH": 0.05, "PET": 0.9},
    ]
    assert _step_snapshot(result)[2] == {
        "step_id": "optimize_slices",
        "role": "optimization-engineer",
        "execution_kind": "tool",
        "allowed_tools": ["run_waste_management_pareto_slices"],
        "outputs": [
            "optimization_pareto_slices",
            "optimization_pareto_front",
            "optimization_pareto_landscape",
            "sidecar_file",
        ],
        "depends_on": ["build_optimization_handoff"],
    }
    assert _step_snapshot(result)[3]["allowed_tools"] == ["plot_optimization_pareto_slices"]


def test_validated_p0_query_bank_rows_compile_and_include_named_tools():
    exported_named_failures: list[str] = []
    artifact_failures: list[str] = []
    exported = exported_tool_names()
    for row in validated_query_bank_rows(priority="P0"):
        result = compile_request(row.query, created_at=FIXED_TIME)
        assert result.status == "compiled", f"{row.sheet_name}:{row.row_number}: {result}"
        assert result.plan is not None
        planned_tools = {
            tool
            for step in result.plan.steps
            for tool in step.allowed_tools
        }
        expected_tools = row.expected_tool_names(exported)
        missing = set(expected_tools) - planned_tools
        if missing:
            exported_named_failures.append(f"{row.sheet_name}:{row.row_number}:{sorted(missing)}")
        planned_artifacts = {
            artifact.artifact_type
            for step in result.plan.steps
            for contract in step.output_contracts
            for artifact in contract.artifact_contracts
        }
        expected_artifacts = row.expected_artifact_types()
        missing_artifacts = set(expected_artifacts) - planned_artifacts
        if missing_artifacts:
            artifact_failures.append(f"{row.sheet_name}:{row.row_number}:{sorted(missing_artifacts)}")
    assert not exported_named_failures
    assert not artifact_failures


def test_compile_missing_required_inputs_returns_clarification_result():
    result = compile_request("Optimize waste management for PE and EVOH.", created_at=FIXED_TIME)

    assert result.status == "clarification_required"
    assert result.plan is not None
    assert result.plan.mode == "clarification_required"
    assert {item.name for item in result.plan.missing_inputs} == {"feed_capacity_tpy", "feed_composition_json"}


def test_backend_malformed_json_is_invalid():
    result = compile_request("anything", planner_backend=StubBackend("{not json"), created_at=FIXED_TIME)

    assert result.status == "invalid"
    assert result.validation_errors


def test_backend_unknown_tool_is_invalid():
    payload = {
        "mode": "single_tool_or_specialist",
        "intent_family": "safety",
        "complexity": "moderate",
        "steps": [
            {
                "step_id": "bad",
                "label": "Bad",
                "role": "safety-analyst",
                "execution_kind": "tool",
                "allowed_tools": ["not_a_tool"],
                "output_contracts": [
                    {
                        "contract_id": "out",
                        "artifact_contracts": [{"artifact_type": "solvent_safety_card"}],
                    }
                ],
            }
        ],
    }
    result = compile_request("anything", planner_backend=StubBackend(payload), created_at=FIXED_TIME)

    assert result.status == "invalid"
    assert any("unknown tool not_a_tool" in error for error in result.validation_errors)


def test_backend_unknown_artifact_is_invalid():
    payload = {
        "mode": "single_tool_or_specialist",
        "intent_family": "safety",
        "complexity": "moderate",
        "steps": [
            {
                "step_id": "bad",
                "label": "Bad",
                "role": "safety-analyst",
                "execution_kind": "tool",
                "allowed_tools": ["get_solvent_safety_card"],
                "output_contracts": [
                    {
                        "contract_id": "out",
                        "artifact_contracts": [{"artifact_type": "not_an_artifact"}],
                    }
                ],
            }
        ],
    }
    result = compile_request("anything", planner_backend=StubBackend(payload), created_at=FIXED_TIME)

    assert result.status == "invalid"
    assert any("unknown output artifact not_an_artifact" in error for error in result.validation_errors)


def test_backend_role_tool_mismatch_is_invalid():
    payload = {
        "mode": "single_tool_or_specialist",
        "intent_family": "optimization",
        "complexity": "moderate",
        "steps": [
            {
                "step_id": "bad",
                "label": "Bad",
                "role": "safety-analyst",
                "execution_kind": "tool",
                "allowed_tools": ["run_waste_management_pareto"],
                "output_contracts": [
                    {
                        "contract_id": "out",
                        "artifact_contracts": [{"artifact_type": "optimization_pareto_front"}],
                    }
                ],
                "tool_args_template": {
                    "feed_capacity_tpy": 8000,
                    "feed_composition_json": {"PE": 0.6, "EVOH": 0.4},
                    "x_metric": "total_cost",
                    "y_metric": "emissions",
                },
            }
        ],
    }
    result = compile_request("anything", planner_backend=StubBackend(payload), created_at=FIXED_TIME)

    assert result.status == "invalid"
    assert any("not allowed for role safety-analyst" in error for error in result.validation_errors)


def test_backend_missing_output_contract_is_invalid():
    payload = {
        "mode": "single_tool_or_specialist",
        "intent_family": "safety",
        "complexity": "moderate",
        "steps": [
            {
                "step_id": "bad",
                "label": "Bad",
                "role": "safety-analyst",
                "execution_kind": "tool",
                "allowed_tools": ["get_solvent_safety_card"],
                "output_contracts": [],
            }
        ],
    }
    result = compile_request("anything", planner_backend=StubBackend(payload), created_at=FIXED_TIME)

    assert result.status == "invalid"
    assert result.validation_errors


def test_backend_bad_dependency_is_invalid():
    payload = {
        "mode": "planned_workflow",
        "intent_family": "mixed_workflow",
        "complexity": "complex",
        "steps": [
            {
                "step_id": "first",
                "label": "First",
                "role": "safety-analyst",
                "execution_kind": "tool",
                "allowed_tools": ["get_solvent_safety_card"],
                "depends_on": ["future"],
                "output_contracts": [
                    {
                        "contract_id": "out",
                        "artifact_contracts": [{"artifact_type": "solvent_safety_card"}],
                    }
                ],
            },
            {
                "step_id": "future",
                "label": "Future",
                "role": "safety-analyst",
                "execution_kind": "tool",
                "allowed_tools": ["get_solvent_safety_card"],
                "output_contracts": [
                    {
                        "contract_id": "out2",
                        "artifact_contracts": [{"artifact_type": "solvent_safety_card"}],
                    }
                ],
            },
        ],
    }
    result = compile_request("anything", planner_backend=StubBackend(payload), created_at=FIXED_TIME)

    assert result.status == "invalid"
    assert result.validation_errors


def test_backend_missing_required_tool_input_is_invalid():
    payload = {
        "mode": "single_agent",
        "intent_family": "optimization",
        "complexity": "moderate",
        "steps": [
            {
                "step_id": "optimize",
                "label": "Optimize",
                "role": "optimization-engineer",
                "execution_kind": "tool",
                "allowed_tools": ["run_waste_management_optimization"],
                "output_contracts": [
                    {
                        "contract_id": "out",
                        "artifact_contracts": [{"artifact_type": "optimization_point_result"}],
                    }
                ],
                "tool_args_template": {"objective": "max_profit"},
            }
        ],
    }
    result = compile_request("anything", planner_backend=StubBackend(payload), created_at=FIXED_TIME)

    assert result.status == "invalid"
    assert any("missing required input feed_capacity_tpy" in error for error in result.validation_errors)


def test_backend_null_or_empty_required_tool_inputs_are_invalid():
    payload = {
        "mode": "single_agent",
        "intent_family": "optimization",
        "complexity": "moderate",
        "steps": [
            {
                "step_id": "optimize",
                "label": "Optimize",
                "role": "optimization-engineer",
                "execution_kind": "tool",
                "allowed_tools": ["run_waste_management_optimization"],
                "output_contracts": [
                    {
                        "contract_id": "out",
                        "artifact_contracts": [{"artifact_type": "optimization_point_result"}],
                    }
                ],
                "tool_args_template": {
                    "feed_capacity_tpy": None,
                    "feed_composition_json": {},
                    "objective": "max_profit",
                },
            }
        ],
    }
    result = compile_request("anything", planner_backend=StubBackend(payload), created_at=FIXED_TIME)

    assert result.status == "invalid"
    assert any("missing required input feed_capacity_tpy" in error for error in result.validation_errors)
    assert any("missing required input feed_composition_json" in error for error in result.validation_errors)


def test_shadow_diagnostics_are_structured_and_non_user_visible():
    result = compile_request("Show a safety card for THF at 60 C.", created_at=FIXED_TIME)
    diagnostics = compile_shadow_diagnostics(result)

    assert diagnostics["status"] == "compiled"
    assert diagnostics["plan"]["steps"][0]["allowed_tools"] == ["get_solvent_safety_card"]
    assert "user_visible_message" not in diagnostics
    json.dumps(diagnostics)
