from __future__ import annotations

import json
from pathlib import Path

from strap.planning.compiler import compile_request
from strap.planning.config import PlannerConfig
from strap.planning.models import ExecutionLedger
from strap.planning.runtime import run_typed_runtime
from strap.planning.runtime_paths import normalize_runtime_path
from strap.planning.runtime_persistence import RuntimeArtifactManifest
from strap.planning.runtime_wrappers import (
    get_runtime_callable_registry,
    make_static_artifact_wrapper,
    wrap_legacy_callable,
)


FIXED_TIME = "2026-04-26T00:00:00+00:00"


def _multislice_query() -> str:
    return (
        "For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year under scenario A, have the "
        "separation engineer propose the top 12 solvent candidates per polymer using the "
        "dynamic-programming planner with temperature recommendations. Then run cost-vs-circularity "
        "Pareto landscape optimization for five fixed feed compositions: 20/60/20, 34/33/33, "
        "60/20/20, 20/20/60, and 5/5/90. Require at least 1 STRAP wash and allow up to 2 washes. "
        "Save one PNG per composition and one combined comparison plot showing all feasible points."
    )


def test_dp_state_map_query_now_compiles_to_enforceable_plot_plan():
    result = compile_request("Plot the dynamic-programming state map for LDPE/EVOH/PET.", created_at=FIXED_TIME)

    assert result.status == "compiled"
    assert result.plan is not None
    assert result.plan.steps[0].allowed_tools == ["plot_dynamic_programming_separation_options"]
    assert [
        artifact.artifact_type
        for contract in result.plan.steps[0].output_contracts
        for artifact in contract.artifact_contracts
    ] == ["separation_dp_state_map"]


def test_query_level_unc_save_path_is_normalized_into_step_args():
    raw_unc = (
        r"\\wsl.localhost\Ubuntu-20.04\home\aaltamimi2"
        r"\langchain-STRAP-v9-contaminants\plots"
    )
    result = compile_request(
        f'Plot the dynamic-programming state map for LDPE/EVOH/PET and save to "{raw_unc}".',
        created_at=FIXED_TIME,
    )

    assert result.status == "compiled"
    assert result.extracted_facts["output_dir"] == "/home/aaltamimi2/langchain-STRAP-v9-contaminants/plots"
    assert result.plan is not None
    assert result.plan.steps[0].tool_args_template["output_dir"] == (
        "/home/aaltamimi2/langchain-STRAP-v9-contaminants/plots"
    )


def test_routed_pareto_plot_step_gets_normalized_output_dir_and_stem():
    raw_unc = (
        r"\\wsl.localhost\Ubuntu-20.04\home\aaltamimi2"
        r"\langchain-STRAP-v9-contaminants\plots\ldpe60_evoh20_pet20.png"
    )
    query = (
        "For a mixed plastic feedstock of 8000 tonnes/year composed of 60% LDPE, 20% EVOH, and 20% PET "
        "under scenario A, have the separation engineer propose the top 12 solvent candidates per polymer using the "
        "dynamic-programming planner with temperature recommendations. Then run cost-vs-circularity Pareto landscape "
        f'optimization with 100 points, requiring at least 1 STRAP wash and allowing up to 2 washes. Save the plot to "{raw_unc}".'
    )
    result = compile_request(query, created_at=FIXED_TIME)

    assert result.status == "compiled"
    assert result.plan is not None
    plot_step = next(step for step in result.plan.steps if step.step_id == "plot_optimization")
    assert plot_step.tool_args_template["output_dir"] == "/home/aaltamimi2/langchain-STRAP-v9-contaminants/plots"
    assert plot_step.tool_args_template["output_stem"] == "ldpe60_evoh20_pet20"


def test_context_output_dir_overrides_query_save_path(tmp_path: Path):
    raw_unc = (
        r"\\wsl.localhost\Ubuntu-20.04\home\aaltamimi2"
        r"\langchain-STRAP-v9-contaminants\plots"
    )
    override = tmp_path / "context_output"
    result = compile_request(
        f'Plot the dynamic-programming state map for LDPE/EVOH/PET and save to "{raw_unc}".',
        context={"output_dir": str(override)},
        created_at=FIXED_TIME,
    )

    assert result.extracted_facts["output_dir"] == str(override)
    assert result.plan is not None
    assert result.plan.steps[0].tool_args_template["output_dir"] == str(override)


def test_runtime_off_and_shadow_modes_fall_back_without_execution(tmp_path: Path):
    def should_not_run(step, ledger):  # pragma: no cover - assertion guard
        raise AssertionError("typed runtime should not execute in off/shadow mode")

    for mode in ("off", "shadow"):
        result = run_typed_runtime(
            "Plot the dynamic-programming state map for LDPE/EVOH/PET.",
            config=PlannerConfig(mode=mode),
            callable_registry={"plot_dynamic_programming_separation_options": should_not_run},
            output_root=str(tmp_path),
            created_at=FIXED_TIME,
        )

        assert result.status == "legacy_fallback"
        assert result.ledger is None
        assert result.manifest is None


def test_selected_compile_failure_does_not_silently_fall_back(tmp_path: Path):
    result = run_typed_runtime(
        "Optimize waste management for PE and EVOH.",
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_point_result"},
        ),
        callable_registry={},
        output_root=str(tmp_path),
        created_at=FIXED_TIME,
    )

    assert result.status == "typed_failure"
    assert result.selected is True
    assert result.compile_result.status == "clarification_required"
    assert result.manifest is not None
    assert Path(result.manifest.files["compile_result"]).exists()


def test_selected_workflow_compile_failure_does_not_silently_fall_back(tmp_path: Path):
    result = run_typed_runtime(
        "Have the separation engineer propose LDPE/EVOH/PET candidates, then pass them to optimization for a Pareto plot.",
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_workflows={"routed_optimization"},
        ),
        callable_registry={},
        output_root=str(tmp_path),
        created_at=FIXED_TIME,
    )

    assert result.status == "typed_failure"
    assert result.selected is True
    assert result.compile_result.status == "clarification_required"
    assert result.plan is not None
    assert {item.name for item in result.plan.missing_inputs} == {"feed_capacity_tpy", "feed_composition_json"}
    assert result.manifest is not None


def test_selected_missing_wrapper_fails_before_execution(tmp_path: Path):
    calls: list[str] = []

    def tracked_wrapper(step, ledger):
        calls.append(step.allowed_tools[0])
        return make_static_artifact_wrapper()(step, ledger)

    result = run_typed_runtime(
        _multislice_query(),
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_slices"},
        ),
        callable_registry={
            "plan_multiple_separation_schemes": tracked_wrapper,
            "build_handoff": tracked_wrapper,
            "plot_optimization_pareto_slices": tracked_wrapper,
        },
        output_root=str(tmp_path),
        created_at=FIXED_TIME,
    )

    assert result.status == "typed_failure"
    assert "missing_wrapper:run_waste_management_pareto_slices" in result.diagnostics
    assert calls == []


def test_runtime_executes_fake_dp_state_map_and_persists_models(tmp_path: Path):
    plot_path = tmp_path / "state_map.png"
    plot_path.write_text("png")
    result = run_typed_runtime(
        "Plot the dynamic-programming state map for LDPE/EVOH/PET.",
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"separation_dp_state_map"},
        ),
        callable_registry=get_runtime_callable_registry(
            wrappers={
                "plot_dynamic_programming_separation_options": make_static_artifact_wrapper(
                    output_paths={"separation_dp_state_map": [str(plot_path)]}
                )
            }
        ),
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result.status == "executed"
    assert result.ledger is not None
    assert result.ledger.status == "succeeded"
    assert result.manifest is not None
    manifest = RuntimeArtifactManifest.model_validate_json(Path(result.manifest.files["manifest"]).read_text())
    ledger = ExecutionLedger.model_validate_json(Path(manifest.files["ledger"]).read_text())
    assert ledger.status == "succeeded"
    assert Path(manifest.produced_file_copies[str(plot_path)]).exists()


def test_runtime_wrapper_sees_normalized_query_output_dir(tmp_path: Path):
    raw_unc = (
        r"\\wsl.localhost\Ubuntu-20.04\home\aaltamimi2"
        r"\langchain-STRAP-v9-contaminants\plots"
    )
    seen_args: dict[str, object] = {}
    plot_path = tmp_path / "state_map.png"
    plot_path.write_text("png")

    def capture_args(step, ledger):
        seen_args.update(step.tool_args_template)
        return make_static_artifact_wrapper(
            output_paths={"separation_dp_state_map": [str(plot_path)]}
        )(step, ledger)

    result = run_typed_runtime(
        f'Plot the dynamic-programming state map for LDPE/EVOH/PET and save to "{raw_unc}".',
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"separation_dp_state_map"},
        ),
        callable_registry={"plot_dynamic_programming_separation_options": capture_args},
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result.status == "executed"
    assert seen_args["output_dir"] == "/home/aaltamimi2/langchain-STRAP-v9-contaminants/plots"


def test_runtime_executes_multislice_with_slice_callable_only(tmp_path: Path):
    calls: list[str] = []
    plot_path = tmp_path / "slices.png"
    plot_path.write_text("png")

    def tracked(step, ledger):
        calls.append(step.allowed_tools[0])
        return make_static_artifact_wrapper(
            output_paths={"optimization_pareto_slices_plot": [str(plot_path)]}
        )(step, ledger)

    result = run_typed_runtime(
        _multislice_query(),
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_slices"},
        ),
        callable_registry=get_runtime_callable_registry(
            wrappers={
                "plan_multiple_separation_schemes": tracked,
                "build_handoff": tracked,
                "run_waste_management_pareto_slices": tracked,
                "plot_optimization_pareto_slices": tracked,
            }
        ),
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result.status == "executed"
    assert "run_waste_management_pareto_slices" in calls
    assert "run_waste_management_pareto" not in calls
    assert result.manifest is not None
    assert Path(result.manifest.files["plan"]).exists()
    assert Path(result.manifest.files["artifacts"]).exists()


def test_wsl_unc_output_path_normalizes_to_linux_path():
    raw = (
        r"\\wsl.localhost\Ubuntu-20.04\home\aaltamimi2"
        r"\langchain-STRAP-v9-contaminants\plots"
    )

    assert normalize_runtime_path(raw) == "/home/aaltamimi2/langchain-STRAP-v9-contaminants/plots"


def test_legacy_wrapper_does_not_mint_artifacts_without_explicit_evidence(tmp_path: Path):
    output_path = tmp_path / "plot.png"
    output_path.write_text("png")
    raw_path = str(output_path).replace("/", "\\")
    wrapper = wrap_legacy_callable(lambda **kwargs: {"plot_paths": [raw_path], "large_payload": {"ignored": True}})
    compiled = compile_request("Plot the dynamic-programming state map for LDPE/EVOH/PET.", created_at=FIXED_TIME)
    assert compiled.plan is not None
    step = compiled.plan.steps[0]

    result = wrapper(step, None)  # type: ignore[arg-type]

    assert result.artifacts == []
    assert result.data == {"legacy_result_type": "dict", "artifact_types_declared": []}
    json.dumps(result.model_dump(mode="json"))


def test_legacy_wrapper_uses_explicit_artifact_evidence_and_normalizes_paths(tmp_path: Path):
    output_path = tmp_path / "plot.png"
    output_path.write_text("png")
    raw_path = str(output_path).replace("/", "\\")
    wrapper = wrap_legacy_callable(
        lambda **kwargs: {"plot_paths": [raw_path], "large_payload": {"ignored": True}},
        artifact_types={"separation_dp_state_map"},
    )
    compiled = compile_request("Plot the dynamic-programming state map for LDPE/EVOH/PET.", created_at=FIXED_TIME)
    assert compiled.plan is not None
    step = compiled.plan.steps[0]

    result = wrapper(step, None)  # type: ignore[arg-type]

    assert result.artifacts[0].output_paths == [normalize_runtime_path(raw_path)]
    assert result.data == {
        "legacy_result_type": "dict",
        "artifact_types_declared": ["separation_dp_state_map"],
    }
    json.dumps(result.model_dump(mode="json"))
