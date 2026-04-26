from __future__ import annotations

import json
from pathlib import Path

from strap.planning.config import PlannerConfig
from strap.planning.executor import StepCallableResult
from strap.planning.models import (
    ArtifactContract,
    ArtifactFrame,
    ExecutionLedger,
    InputContract,
    OutputContract,
    PlanStep,
)
from strap.planning.runtime import run_typed_runtime
from strap.planning.runtime_production_wrappers import (
    get_production_runtime_callable_registry,
    wrap_biosteam_simulation,
    wrap_biosteam_visualization,
    wrap_dp_state_map_plot,
    wrap_hsp_red_heatmap,
    wrap_hsp_single_pair,
    wrap_optimization_handoff,
    wrap_optimization_pareto_plot,
    wrap_separation_selectivity_heatmap,
    wrap_separation_topk,
    wrap_separation_tree_plot,
    wrap_solvent_safety_card,
    wrap_solvent_safety_comparison,
    wrap_waste_management_pareto,
)


FIXED_TIME = "2026-04-26T00:00:00+00:00"


def _envelope(tool_name: str, **data):
    payload = {"success": True, "tool_name": tool_name}
    payload.update(data)
    return json.dumps({"display": "ok", "data": payload})


def _output(*artifact_types: str, path_required: bool = False) -> list[OutputContract]:
    return [
        OutputContract(
            contract_id="out",
            artifact_contracts=[
                ArtifactContract(
                    artifact_type=artifact_type,
                    path_policy="required" if path_required else "optional",
                )
                for artifact_type in artifact_types
            ],
        )
    ]


def _step(
    step_id: str,
    tool_name: str,
    *artifact_types: str,
    input_contracts: list[InputContract] | None = None,
    tool_args_template: dict | None = None,
    execution_kind: str = "tool",
    path_required: bool = False,
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        label=step_id,
        role="handoff_adapter" if execution_kind == "handoff_adapter" else "direct_tool",
        execution_kind=execution_kind,  # type: ignore[arg-type]
        allowed_tools=[tool_name],
        input_contracts=input_contracts or [],
        output_contracts=_output(*artifact_types, path_required=path_required),
        tool_args_template=tool_args_template or {},
    )


def _ledger(*artifacts: ArtifactFrame) -> ExecutionLedger:
    return ExecutionLedger(
        plan_id="plan_test",
        run_id="run_test",
        status="running",
        started_at=FIXED_TIME,
        artifacts=list(artifacts),
    )


def _payload_artifact(artifact_type: str, source_step_id: str, payload: dict) -> ArtifactFrame:
    return ArtifactFrame(
        artifact_id=f"{source_step_id}:{artifact_type}",
        artifact_type=artifact_type,
        source_step_id=source_step_id,
        validation_summary={"payload": payload},
    )


def test_production_registry_excludes_solubility_plotter():
    registry = get_production_runtime_callable_registry()

    assert "plot_dynamic_programming_separation_options" in registry
    assert "plot_solubility_vs_temperature" not in registry


def test_dp_state_map_wrapper_forces_state_map_only(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    rank_plot = tmp_path / "rank1.png"
    state_map = tmp_path / "ldpe_evoh_pet_state_map.png"
    rank_plot.write_text("rank")
    state_map.write_text("state")

    def fake_dp_tool(**kwargs):
        captured.update(kwargs)
        return _envelope(
            "plot_dynamic_programming_separation_options",
            plot_paths=[str(rank_plot), str(state_map)],
        )

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "plot_dynamic_programming_separation_options", fake_dp_tool)
    step = _step(
        "plot_state_map",
        "plot_dynamic_programming_separation_options",
        "separation_dp_state_map",
        tool_args_template={"polymers": ["LDPE", "EVOH", "PET"], "temperature": 100.0, "output_dir": str(tmp_path)},
        path_required=True,
    )

    result = wrap_dp_state_map_plot(step, _ledger())

    assert result.success is True
    assert captured["include_sequence_plots"] is False
    assert captured["include_state_map"] is True
    assert captured["include_objective_paths"] is False
    assert result.artifacts[0].artifact_type == "separation_dp_state_map"
    assert result.artifacts[0].output_paths == [str(state_map)]


def test_dp_state_map_wrapper_fails_closed_without_state_map_evidence(monkeypatch, tmp_path: Path):
    rank_plot = tmp_path / "rank1.png"
    rank_plot.write_text("rank")

    def fake_dp_tool(**kwargs):
        return _envelope("plot_dynamic_programming_separation_options", plot_paths=[str(rank_plot)])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "plot_dynamic_programming_separation_options", fake_dp_tool)
    step = _step(
        "plot_state_map",
        "plot_dynamic_programming_separation_options",
        "separation_dp_state_map",
        tool_args_template={"polymers": "LDPE,EVOH,PET"},
        path_required=True,
    )

    result = wrap_dp_state_map_plot(step, _ledger())

    assert result.success is False
    assert result.artifacts == []


def test_runtime_dp_state_map_real_wrapper_receives_normalized_unc_output_dir(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    state_map = tmp_path / "dp_state_map.png"
    state_map.write_text("state")

    def fake_dp_tool(**kwargs):
        captured.update(kwargs)
        return _envelope("plot_dynamic_programming_separation_options", plot_paths=[str(state_map)])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "plot_dynamic_programming_separation_options", fake_dp_tool)
    raw_unc = (
        r"\\wsl.localhost\Ubuntu-20.04\home\aaltamimi2"
        r"\langchain-STRAP-v9-contaminants\plots"
    )
    result = run_typed_runtime(
        f'Plot the dynamic-programming state map for LDPE/EVOH/PET and save to "{raw_unc}".',
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_dp_state_map"}),
        callable_registry=get_production_runtime_callable_registry(),
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result.status == "executed"
    assert captured["output_dir"] == "/home/aaltamimi2/langchain-STRAP-v9-contaminants/plots"


def test_separation_topk_wrapper_preserves_requested_top_six(monkeypatch):
    captured: dict[str, object] = {}

    def fake_planner(**kwargs):
        captured.update(kwargs)
        return _envelope(
            "plan_multiple_separation_schemes",
            top_k_sequences=[{"rank": index + 1, "sequence": ["LDPE", "EVOH"]} for index in range(kwargs["n_variants"])],
            polymer_solvent_candidates={},
        )

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "plan_multiple_separation_schemes", fake_planner)
    step = _step(
        "separation_candidates",
        "plan_multiple_separation_schemes",
        "separation_topk_sequences",
        "optimization_stage_candidates",
        tool_args_template={"polymers": ["LDPE", "EVOH", "PET"], "top_k_per_polymer": 6},
    )

    result = wrap_separation_topk(step, _ledger())

    assert result.success is True
    assert captured["n_variants"] == 6
    assert result.data["n_sequences"] == 6


def test_handoff_wrapper_consumes_ledger_payload_not_query_text(monkeypatch):
    source_payload = {
        "polymers": ["LDPE", "EVOH"],
        "top_k_sequences": [
            {
                "rank": 1,
                "sequence": ["LDPE", "EVOH"],
                "solvent_mapping": {"LDPE": "Cyclohexane"},
                "steps": [{"polymer": "LDPE", "solvent": "Cyclohexane", "temperature_c": 79.7}],
            }
        ],
        "source_user_query": "source query",
    }
    source_artifact = _payload_artifact("optimization_stage_candidates", "separation_candidates", source_payload)

    def fake_adapter(source_record, *, scope_user_query=None):
        assert source_record.payload == source_payload
        assert scope_user_query == "source query"
        return (
            "optimization.stage_candidates.v1",
            {"constraint_mode": "ranked_soft", "stages": [{"stage_id": "candidate_pool_pe"}]},
            "task prompt",
        )

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "_adapt_separation_to_optimization", fake_adapter)
    step = _step(
        "build_optimization_handoff",
        "build_handoff",
        "optimization_stage_candidates",
        "handoff_payload",
        execution_kind="handoff_adapter",
        tool_args_template={"source_step_id": "separation_candidates"},
    )

    result = wrap_optimization_handoff(step, _ledger(source_artifact))

    assert result.success is True
    assert {artifact.artifact_type for artifact in result.artifacts} == {"optimization_stage_candidates", "handoff_payload"}
    payload = result.artifacts[0].validation_summary["payload"]
    assert payload["handoff_contract"] == "optimization.stage_candidates.v1"
    assert result.artifacts[0].source_handoff_ids


def test_safety_card_wrapper_requires_safety_profile(monkeypatch):
    captured: dict[str, object] = {}

    def fake_card(**kwargs):
        captured.update(kwargs)
        return _envelope("get_solvent_safety_card", solvent_name="THF", safety_profile={"identity": {"name": "THF"}})

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "get_solvent_safety_card", fake_card)
    step = _step(
        "safety_assessment",
        "get_solvent_safety_card",
        "solvent_safety_card",
        tool_args_template={"solvent_name": "THF", "operating_temp_c": 60},
    )

    result = wrap_solvent_safety_card(step, _ledger())

    assert result.success is True
    assert captured["solvent_name"] == "THF"
    assert result.artifacts[0].artifact_type == "solvent_safety_card"


def test_safety_card_wrapper_fails_closed_without_profile(monkeypatch):
    def fake_card(**kwargs):
        return _envelope("get_solvent_safety_card", solvent_name="THF")

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "get_solvent_safety_card", fake_card)
    step = _step("safety_assessment", "get_solvent_safety_card", "solvent_safety_card", tool_args_template={"solvent_name": "THF"})

    result = wrap_solvent_safety_card(step, _ledger())

    assert result.success is False
    assert result.artifacts == []


def test_safety_comparison_wrapper_requires_profiles(monkeypatch):
    captured: dict[str, object] = {}

    def fake_compare(**kwargs):
        captured.update(kwargs)
        return _envelope("compare_solvent_safety_cards", profiles=[{"identity": {"name": "THF"}}])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "compare_solvent_safety_cards", fake_compare)
    step = _step(
        "safety_assessment",
        "compare_solvent_safety_cards",
        "solvent_safety_comparison",
        tool_args_template={"solvents": ["THF", "Toluene"]},
    )

    result = wrap_solvent_safety_comparison(step, _ledger())

    assert result.success is True
    assert captured["solvent_names"] == "THF,Toluene"
    assert result.artifacts[0].artifact_type == "solvent_safety_comparison"


def test_hsp_single_pair_wrapper_requires_red_evidence(monkeypatch, tmp_path: Path):
    plot_path = tmp_path / "hsp.png"
    plot_path.write_text("hsp")
    captured: dict[str, object] = {}
    plot_dirs: list[str] = []

    def fake_hsp(**kwargs):
        captured.update(kwargs)
        return _envelope(
            "predict_solubility_ml",
            analysis_type="hsp_binary_screen",
            red=0.72,
            probability=0.88,
            artifacts=[str(plot_path)],
        )

    def fake_set_plots_dir(path: str) -> str:
        plot_dirs.append(path)
        return "/old/plots"

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "predict_solubility_ml", fake_hsp)
    monkeypatch.setattr(wrappers, "set_plots_dir", fake_set_plots_dir)
    step = _step(
        "hsp_screen",
        "predict_solubility_ml",
        "hsp_single_pair_summary",
        tool_args_template={"polymer": "PET", "solvent": "Toluene", "temperature_c": 25, "output_dir": str(tmp_path)},
    )

    result = wrap_hsp_single_pair(step, _ledger())

    assert result.success is True
    assert captured["polymer_name"] == "PET"
    assert captured["solvent_name"] == "Toluene"
    assert plot_dirs == [str(tmp_path), "/old/plots"]
    assert result.artifacts[0].output_paths == [str(plot_path)]


def test_hsp_single_pair_wrapper_fails_closed_without_red(monkeypatch):
    def fake_hsp(**kwargs):
        return _envelope("predict_solubility_ml", analysis_type="hsp_binary_screen")

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "predict_solubility_ml", fake_hsp)
    step = _step("hsp_screen", "predict_solubility_ml", "hsp_single_pair_summary", tool_args_template={"polymer": "PET", "solvent": "Toluene"})

    result = wrap_hsp_single_pair(step, _ledger())

    assert result.success is False
    assert result.artifacts == []


def test_hsp_heatmap_wrapper_requires_results_and_artifact_path(monkeypatch, tmp_path: Path):
    plot_path = tmp_path / "hsp_heatmap.png"
    plot_path.write_text("heatmap")
    captured: dict[str, object] = {}
    plot_dirs: list[str] = []

    def fake_matrix(**kwargs):
        captured.update(kwargs)
        return _envelope(
            "screen_hsp_solubility_matrix",
            analysis_type="hsp_binary_screen",
            results=[{"polymer": "PE", "solvent": "Hexane", "red": 0.9}],
            artifacts=[str(plot_path)],
        )

    def fake_set_plots_dir(path: str) -> str:
        plot_dirs.append(path)
        return "/old/plots"

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "screen_hsp_solubility_matrix", fake_matrix)
    monkeypatch.setattr(wrappers, "set_plots_dir", fake_set_plots_dir)
    step = _step(
        "hsp_screen",
        "screen_hsp_solubility_matrix",
        "hsp_red_heatmap",
        tool_args_template={"polymer_category": "polyolefins", "solvent_polarity": "nonpolar", "output_dir": str(tmp_path)},
        path_required=True,
    )

    result = wrap_hsp_red_heatmap(step, _ledger())

    assert result.success is True
    assert captured["polymer_category"] == "polyolefins"
    assert captured["solvent_polarity"] == "nonpolar"
    assert plot_dirs == [str(tmp_path), "/old/plots"]
    assert result.artifacts[0].output_paths == [str(plot_path)]


def test_hsp_heatmap_wrapper_fails_closed_without_artifact_path(monkeypatch):
    def fake_matrix(**kwargs):
        return _envelope("screen_hsp_solubility_matrix", analysis_type="hsp_binary_screen", results=[{"red": 0.9}], artifacts=[])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "screen_hsp_solubility_matrix", fake_matrix)
    step = _step("hsp_screen", "screen_hsp_solubility_matrix", "hsp_red_heatmap", path_required=True)

    result = wrap_hsp_red_heatmap(step, _ledger())

    assert result.success is False
    assert result.artifacts == []


def test_hsp_heatmap_runtime_copies_output_path_to_diagnostics(monkeypatch, tmp_path: Path):
    plot_path = tmp_path / "hsp_heatmap.png"
    plot_path.write_text("heatmap")

    def fake_matrix(**kwargs):
        return _envelope(
            "screen_hsp_solubility_matrix",
            analysis_type="hsp_binary_screen",
            results=[{"polymer": "PE", "solvent": "Hexane", "red": 0.9}],
            artifacts=[str(plot_path)],
        )

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "screen_hsp_solubility_matrix", fake_matrix)
    result = run_typed_runtime(
        "Use the Hansen model to screen polyolefins against nonpolar solvents and show the RED heatmap.",
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"hsp_red_heatmap"}),
        callable_registry=get_production_runtime_callable_registry(),
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result.status == "executed"
    assert result.manifest is not None
    assert Path(result.manifest.produced_file_copies[str(plot_path)]).exists()


def test_separation_tree_wrapper_requires_plot_paths(monkeypatch, tmp_path: Path):
    rank1 = tmp_path / "separation_sequence_rank1.png"
    topk = tmp_path / "separation_topk_comparison.png"
    rank1.write_text("rank1")
    topk.write_text("topk")
    captured: dict[str, object] = {}

    def fake_tree(**kwargs):
        captured.update(kwargs)
        return _envelope("create_separation_tree_plot", plot_paths=[str(rank1), str(topk)])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "create_separation_tree_plot", fake_tree)
    step = _step(
        "plot_separation_tree",
        "create_separation_tree_plot",
        "separation_tree_plot",
        tool_args_template={"polymers": ["LDPE", "EVOH", "PET"], "temperature": 100, "output_dir": str(tmp_path)},
        path_required=True,
    )

    result = wrap_separation_tree_plot(step, _ledger())

    assert result.success is True
    assert captured["polymers"] == "LDPE,EVOH,PET"
    assert captured["output_dir"] == str(tmp_path)
    assert result.artifacts[0].artifact_type == "separation_tree_plot"
    assert result.artifacts[0].output_paths == [str(rank1), str(topk)]


def test_separation_tree_wrapper_fails_closed_without_paths(monkeypatch):
    def fake_tree(**kwargs):
        return _envelope("create_separation_tree_plot", plot_paths=[])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "create_separation_tree_plot", fake_tree)
    step = _step("plot_separation_tree", "create_separation_tree_plot", "separation_tree_plot", tool_args_template={"polymers": "LDPE,EVOH"})

    result = wrap_separation_tree_plot(step, _ledger())

    assert result.success is False
    assert result.artifacts == []


def test_selectivity_heatmap_wrapper_requires_filepath(monkeypatch, tmp_path: Path):
    plot_path = tmp_path / "selectivity_heatmap.png"
    plot_path.write_text("heatmap")
    captured: dict[str, object] = {}
    plot_dirs: list[str] = []

    def fake_heatmap(**kwargs):
        captured.update(kwargs)
        return _envelope("create_selectivity_heatmap", filepath=str(plot_path), matrix_rows=2)

    def fake_set_plots_dir(path: str) -> str:
        plot_dirs.append(path)
        return "/old/plots"

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "create_selectivity_heatmap", fake_heatmap)
    monkeypatch.setattr(wrappers, "set_plots_dir", fake_set_plots_dir)
    step = _step(
        "plot_selectivity_heatmap",
        "create_selectivity_heatmap",
        "separation_selectivity_heatmap",
        tool_args_template={
            "polymers": ["LDPE", "EVOH", "PET"],
            "solvents": ["Cyclohexane", "Toluene"],
            "temperature": 100,
            "output_dir": str(tmp_path),
        },
        path_required=True,
    )

    result = wrap_separation_selectivity_heatmap(step, _ledger())

    assert result.success is True
    assert captured["polymers"] == "LDPE,EVOH,PET"
    assert captured["solvents"] == "Cyclohexane,Toluene"
    assert plot_dirs == [str(tmp_path), "/old/plots"]
    assert result.artifacts[0].artifact_type == "separation_selectivity_heatmap"
    assert result.artifacts[0].output_paths == [str(plot_path)]


def test_selectivity_heatmap_wrapper_fails_closed_without_filepath(monkeypatch):
    def fake_heatmap(**kwargs):
        return _envelope("create_selectivity_heatmap", matrix_rows=2)

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "create_selectivity_heatmap", fake_heatmap)
    step = _step(
        "plot_selectivity_heatmap",
        "create_selectivity_heatmap",
        "separation_selectivity_heatmap",
        tool_args_template={"polymers": "LDPE,EVOH"},
        path_required=True,
    )

    result = wrap_separation_selectivity_heatmap(step, _ledger())

    assert result.success is False
    assert result.artifacts == []


def test_selectivity_heatmap_runtime_copies_output_path_to_diagnostics(monkeypatch, tmp_path: Path):
    plot_path = tmp_path / "selectivity_heatmap.png"
    plot_path.write_text("heatmap")

    def fake_heatmap(**kwargs):
        return _envelope("create_selectivity_heatmap", filepath=str(plot_path), matrix_rows=2)

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "create_selectivity_heatmap", fake_heatmap)
    result = run_typed_runtime(
        f'Create a selectivity heatmap for LDPE, EVOH, and PET with Cyclohexane and Toluene and save to "{tmp_path}".',
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_selectivity_heatmap"}),
        callable_registry=get_production_runtime_callable_registry(),
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result.status == "executed"
    assert result.manifest is not None
    assert Path(result.manifest.produced_file_copies[str(plot_path)]).exists()


def test_biosteam_simulation_wrapper_requires_tea_lca_evidence(monkeypatch):
    captured: dict[str, object] = {}

    def fake_biosteam(**kwargs):
        captured.update(kwargs)
        return _envelope(
            "run_biosteam_simulation",
            solvent="Cyclohexane",
            target_plastic="LDPE",
            energy_case="C2",
            tea={"msp_usd_per_kg": 0.95, "tci_usd": 66_000_000, "aoc_usd_per_yr": 2_600_000},
            lca={"gwp_kg_co2e_per_kg": 0.875},
        )

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "run_biosteam_simulation", fake_biosteam)
    step = _step(
        "run_biosteam_tea_lca",
        "run_biosteam_simulation",
        "biosteam_tea_lca_result",
        tool_args_template={
            "solvent": "Cyclohexane",
            "target_plastic": "LDPE",
            "energy_case": "C2",
            "processing_capacity": 8000,
            "target_plastic_percent": 60,
            "dissolution_temp_c": 79.7,
        },
    )

    result = wrap_biosteam_simulation(step, _ledger())

    assert result.success is True
    assert captured["solvent"] == "Cyclohexane"
    assert captured["target_plastic"] == "LDPE"
    assert captured["energy_case"] == "C2"
    assert result.artifacts[0].artifact_type == "biosteam_tea_lca_result"


def test_biosteam_simulation_wrapper_fails_closed_without_tea(monkeypatch):
    def fake_biosteam(**kwargs):
        return _envelope("run_biosteam_simulation", solvent="Cyclohexane", lca={"gwp_kg_co2e_per_kg": 0.875})

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "run_biosteam_simulation", fake_biosteam)
    step = _step(
        "run_biosteam_tea_lca",
        "run_biosteam_simulation",
        "biosteam_tea_lca_result",
        tool_args_template={"solvent": "Cyclohexane", "target_plastic": "LDPE", "energy_case": "C2"},
    )

    result = wrap_biosteam_simulation(step, _ledger())

    assert result.success is False
    assert result.artifacts == []


def test_biosteam_simulation_wrapper_fails_closed_when_one_tea_metric_missing(monkeypatch):
    def fake_biosteam(**kwargs):
        return _envelope(
            "run_biosteam_simulation",
            solvent="Cyclohexane",
            tea={"msp_usd_per_kg": 0.95, "tci_usd": 66_000_000},
            lca={"gwp_kg_co2e_per_kg": 0.875},
        )

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "run_biosteam_simulation", fake_biosteam)
    step = _step(
        "run_biosteam_tea_lca",
        "run_biosteam_simulation",
        "biosteam_tea_lca_result",
        tool_args_template={"solvent": "Cyclohexane", "target_plastic": "LDPE", "energy_case": "C2"},
    )

    result = wrap_biosteam_simulation(step, _ledger())

    assert result.success is False
    assert "aoc_usd_per_yr" in (result.error or "")
    assert result.artifacts == []


def test_biosteam_visualization_wrapper_uses_structured_payload(monkeypatch, tmp_path: Path):
    chart_path = tmp_path / "biosteam_cost_breakdown.png"
    chart_path.write_text("chart")
    captured: dict[str, object] = {}
    payload = {
        "success": True,
        "solvent": "Cyclohexane",
        "target_plastic": "LDPE",
        "energy_case": "C2",
        "tea": {"msp_usd_per_kg": 0.95},
        "lca": {"gwp_kg_co2e_per_kg": 0.875},
    }

    def fake_visualize(**kwargs):
        captured.update(kwargs)
        return _envelope("visualize_biosteam_results", charts=[str(chart_path)])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "visualize_biosteam_results", fake_visualize)
    step = _step(
        "plot_biosteam_tea_lca",
        "visualize_biosteam_results",
        "biosteam_tea_lca_plot",
        input_contracts=[InputContract(artifact_type="biosteam_tea_lca_result", source_step_id="run_biosteam_tea_lca")],
        tool_args_template={"source_step_id": "run_biosteam_tea_lca", "output_dir": str(tmp_path)},
        path_required=True,
    )

    result = wrap_biosteam_visualization(step, _ledger(_payload_artifact("biosteam_tea_lca_result", "run_biosteam_tea_lca", payload)))

    assert result.success is True
    assert json.loads(captured["results_json"]) == payload
    assert captured["output_dir"] == str(tmp_path)
    assert result.artifacts[0].artifact_type == "biosteam_tea_lca_plot"
    assert result.artifacts[0].output_paths == [str(chart_path)]


def test_biosteam_visualization_wrapper_fails_closed_without_chart(monkeypatch):
    def fake_visualize(**kwargs):
        return _envelope("visualize_biosteam_results", charts=[])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "visualize_biosteam_results", fake_visualize)
    payload = {"success": True, "tea": {"msp_usd_per_kg": 1.0}, "lca": {"gwp_kg_co2e_per_kg": 0.5}}
    step = _step(
        "plot_biosteam_tea_lca",
        "visualize_biosteam_results",
        "biosteam_tea_lca_plot",
        input_contracts=[InputContract(artifact_type="biosteam_tea_lca_result", source_step_id="run_biosteam_tea_lca")],
        path_required=True,
    )

    result = wrap_biosteam_visualization(step, _ledger(_payload_artifact("biosteam_tea_lca_result", "run_biosteam_tea_lca", payload)))

    assert result.success is False
    assert result.artifacts == []


def test_biosteam_runtime_runs_fake_simulation_and_plot(monkeypatch, tmp_path: Path):
    chart_path = tmp_path / "biosteam_cost_breakdown.png"
    chart_path.write_text("chart")

    def fake_biosteam(**kwargs):
        return _envelope(
            "run_biosteam_simulation",
            solvent=kwargs["solvent"],
            target_plastic=kwargs["target_plastic"],
            energy_case=kwargs["energy_case"],
            tea={"msp_usd_per_kg": 0.95, "tci_usd": 66_000_000, "aoc_usd_per_yr": 2_600_000},
            lca={"gwp_kg_co2e_per_kg": 0.875},
        )

    def fake_visualize(**kwargs):
        return _envelope("visualize_biosteam_results", charts=[str(chart_path)])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "run_biosteam_simulation", fake_biosteam)
    monkeypatch.setattr(wrappers, "visualize_biosteam_results", fake_visualize)
    result = run_typed_runtime(
        f"Estimate BioSTEAM TEA/LCA for LDPE with Cyclohexane under C2 and create a chart in {tmp_path}.",
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"biosteam_tea_lca_plot"}),
        callable_registry=get_production_runtime_callable_registry(),
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result.status == "executed"
    assert result.ledger is not None
    assert {artifact.artifact_type for artifact in result.ledger.artifacts} == {
        "biosteam_tea_lca_result",
        "biosteam_tea_lca_plot",
    }
    assert result.manifest is not None
    assert Path(result.manifest.produced_file_copies[str(chart_path)]).exists()


def test_pareto_wrapper_uses_stage_candidates_from_ledger(monkeypatch):
    stage_payload = {"constraint_mode": "ranked_soft", "fallback_policy": "fail_closed", "route_pool_mode": "exact"}
    captured: dict[str, object] = {}

    def fake_pareto(**kwargs):
        captured.update(kwargs)
        return _envelope(
            "run_waste_management_pareto",
            analysis_type="pareto_front",
            x_metric="total_cost",
            y_metric="circularity",
            points=[{"total_cost": 1.0, "circularity_score": 0.5}],
        )

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "run_waste_management_pareto", fake_pareto)
    step = _step(
        "optimize_pareto",
        "run_waste_management_pareto",
        "optimization_pareto_front",
        "optimization_pareto_landscape",
        input_contracts=[InputContract(artifact_type="optimization_stage_candidates", source_step_id="build_optimization_handoff")],
        tool_args_template={
            "feed_capacity_tpy": 8000,
            "feed_composition_json": {"PE": 0.6, "EVOH": 0.2, "PET": 0.2},
            "scenario": "A",
            "x_metric": "total_cost",
            "y_metric": "circularity",
        },
    )
    ledger = _ledger(_payload_artifact("optimization_stage_candidates", "build_optimization_handoff", stage_payload))

    result = wrap_waste_management_pareto(step, ledger)

    assert result.success is True
    assert captured["feed"] == 8000
    assert captured["feed_composition_json"] == {"PE": 0.6, "EVOH": 0.2, "PET": 0.2}
    assert captured["stage_candidates_json"] == stage_payload
    assert captured["constraint_mode"] == "ranked_soft"


def test_pareto_plot_wrapper_uses_optimizer_payload_from_ledger(monkeypatch, tmp_path: Path):
    optimizer_payload = {
        "analysis_type": "pareto_front",
        "x_metric": "total_cost",
        "y_metric": "circularity",
        "points": [{"total_cost": 1.0, "circularity_score": 0.5}],
    }
    plot_path = tmp_path / "pareto.png"
    plot_path.write_text("plot")
    captured: dict[str, object] = {}

    def fake_plot(**kwargs):
        captured.update(kwargs)
        return _envelope("plot_optimization_pareto_front", plot_paths=[str(plot_path)])

    import strap.planning.runtime_production_wrappers as wrappers

    monkeypatch.setattr(wrappers, "plot_optimization_pareto_front", fake_plot)
    step = _step(
        "plot_optimization",
        "plot_optimization_pareto_front",
        "optimization_pareto_plot",
        input_contracts=[InputContract(artifact_type="optimization_pareto_landscape", source_step_id="optimize_pareto")],
        tool_args_template={"source_step_id": "optimize_pareto", "plot_mode": "landscape", "output_dir": str(tmp_path)},
        path_required=True,
    )
    ledger = _ledger(_payload_artifact("optimization_pareto_landscape", "optimize_pareto", optimizer_payload))

    result = wrap_optimization_pareto_plot(step, ledger)

    assert result.success is True
    assert captured["pareto_result_json"] == optimizer_payload
    assert captured["plot_mode"] == "landscape"
    assert captured["output_dir"] == str(tmp_path)
    assert result.artifacts[0].output_paths == [str(plot_path)]
