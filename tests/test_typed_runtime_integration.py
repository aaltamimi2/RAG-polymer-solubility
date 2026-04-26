from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from strap.planning.config import PlannerConfig
from strap.planning.executor import StepCallableResult
from strap.planning.models import ArtifactFrame
from strap.planning.runtime_wrappers import make_static_artifact_wrapper
from strap.planning.typed_runtime_integration import (
    TypedRuntimeMiddleware,
    format_typed_runtime_failure,
    format_typed_runtime_success,
    maybe_run_typed_runtime,
    summarize_typed_runtime_progress,
)


FIXED_TIME = "2026-04-26T00:00:00+00:00"


def test_integration_off_and_shadow_return_none(tmp_path: Path):
    def should_not_run(step, ledger):  # pragma: no cover - assertion guard
        raise AssertionError("typed runtime should not execute")

    for mode in ("off", "shadow"):
        result = maybe_run_typed_runtime(
            "Plot the dynamic-programming state map for LDPE/EVOH/PET.",
            config=PlannerConfig(mode=mode),
            callable_registry={"plot_dynamic_programming_separation_options": should_not_run},
            output_root=str(tmp_path),
            created_at=FIXED_TIME,
        )
        assert result is None


def test_integration_selected_dp_executes_with_registered_wrapper(tmp_path: Path):
    plot_path = tmp_path / "state_map.png"
    plot_path.write_text("png")

    def wrapper(step, ledger):
        return StepCallableResult(
            artifacts=[
                ArtifactFrame(
                    artifact_id="plot_state_map:separation_dp_state_map",
                    artifact_type="separation_dp_state_map",
                    source_step_id=step.step_id,
                    output_paths=[str(plot_path)],
                )
            ]
        )

    result = maybe_run_typed_runtime(
        "Plot the dynamic-programming state map for LDPE/EVOH/PET.",
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_dp_state_map"}),
        callable_registry={"plot_dynamic_programming_separation_options": wrapper},
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result is not None
    assert result.status == "executed"
    assert result.manifest is not None
    assert "separation_dp_state_map" in format_typed_runtime_success(result)
    progress = summarize_typed_runtime_progress(result)
    assert progress.status == "executed"
    assert progress.completed_steps == ["plot_state_map"]
    assert progress.produced_artifact_paths == [str(plot_path)]


def test_integration_selected_missing_wrapper_returns_typed_failure(tmp_path: Path):
    result = maybe_run_typed_runtime(
        "Plot the dynamic-programming state map for LDPE/EVOH/PET.",
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_dp_state_map"}),
        callable_registry={},
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result is not None
    assert result.status == "typed_failure"
    assert "missing_wrapper:plot_dynamic_programming_separation_options" in result.diagnostics
    assert result.manifest is not None
    assert Path(result.manifest.files["manifest"]).exists()
    summary = format_typed_runtime_failure(result)
    assert "Failed phase: missing_wrapper" in summary
    assert "Diagnostic bundle:" in summary


def test_integration_failure_summary_includes_failed_step_and_checks(tmp_path: Path):
    def wrong_artifact(step, ledger):
        return StepCallableResult(
            artifacts=[
                ArtifactFrame(
                    artifact_id="plot_state_map:solubility_curve",
                    artifact_type="solubility_curve",
                    source_step_id=step.step_id,
                )
            ]
        )

    result = maybe_run_typed_runtime(
        "Plot the dynamic-programming state map for LDPE/EVOH/PET.",
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_dp_state_map"}),
        callable_registry={"plot_dynamic_programming_separation_options": wrong_artifact},
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result is not None
    assert result.status == "typed_failure"
    summary = format_typed_runtime_failure(result)
    assert "Failed phase: verification" in summary
    assert "Failed step: plot_state_map" in summary
    assert "artifact_type_mismatch" in summary
    assert "Diagnostic bundle:" in summary


def test_typed_runtime_middleware_short_circuits_selected_query(tmp_path: Path):
    plot_path = tmp_path / "state_map.png"
    plot_path.write_text("png")
    handler_called = False

    def wrapper(step, ledger):
        return StepCallableResult(
            artifacts=[
                ArtifactFrame(
                    artifact_id="plot_state_map:separation_dp_state_map",
                    artifact_type="separation_dp_state_map",
                    source_step_id=step.step_id,
                    output_paths=[str(plot_path)],
                )
            ]
        )

    def handler(request):  # pragma: no cover - assertion guard
        nonlocal handler_called
        handler_called = True
        return ModelResponse(result=[AIMessage(content="legacy")])

    middleware = TypedRuntimeMiddleware(
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_dp_state_map"}),
        callable_registry={"plot_dynamic_programming_separation_options": wrapper},
        output_root=str(tmp_path / "runs"),
    )
    response = middleware.wrap_model_call(
        SimpleNamespace(messages=[HumanMessage(content="Plot the dynamic-programming state map for LDPE/EVOH/PET.")]),
        handler,
    )

    assert handler_called is False
    assert response.result[0].additional_kwargs["strap_origin"] == "typed_runtime"
    assert response.result[0].additional_kwargs["strap_typed_runtime_status"] == "executed"
    assert response.result[0].additional_kwargs["strap_runtime_progress"]["completed_steps"] == ["plot_state_map"]
    assert "separation_dp_state_map" in response.result[0].content


def test_typed_runtime_middleware_answers_prior_artifact_followup(tmp_path: Path):
    plot_path = tmp_path / "red_heatmap.png"
    diagnostic_copy = tmp_path / "runs" / "produced_files" / "red_heatmap.png"
    prior_artifact = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap",
        artifact_type="hsp_red_heatmap",
        source_step_id="hsp_screen",
        output_paths=[str(plot_path)],
    )
    prior_message = AIMessage(
        content="Typed runtime completed.",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "executed",
            "strap_plan_id": "plan_hsp",
            "strap_workflow_id": "hsp_red_heatmap",
            "strap_runtime_progress": {
                "status": "executed",
                "produced_artifact_paths": [str(plot_path)],
                "failed_checks": [],
                "diagnostic_bundle_path": str(tmp_path / "runs"),
            },
            "strap_manifest": {
                "run_id": "run_hsp",
                "run_dir": str(tmp_path / "runs"),
                "files": {"manifest": str(tmp_path / "runs" / "manifest.json")},
                "produced_file_copies": {str(plot_path): str(diagnostic_copy)},
                "created_at": FIXED_TIME,
            },
            "strap_run_ledger": {
                "plan_id": "plan_hsp",
                "run_id": "run_hsp",
                "status": "succeeded",
                "started_at": FIXED_TIME,
                "completed_at": FIXED_TIME,
                "step_records": [],
                "artifacts": [prior_artifact.model_dump(mode="json")],
                "repairs": [],
                "final_contract_status": {},
            },
        },
    )
    handler_called = False

    def handler(request):  # pragma: no cover - assertion guard
        nonlocal handler_called
        handler_called = True
        return ModelResponse(result=[AIMessage(content="legacy")])

    middleware = TypedRuntimeMiddleware(
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"hsp_red_heatmap"}),
        callable_registry={},
        output_root=str(tmp_path / "runs"),
    )
    response = middleware.wrap_model_call(
        SimpleNamespace(
            messages=[
                HumanMessage(content="Use the Hansen model to show the RED heatmap."),
                prior_message,
                HumanMessage(content="Where did that RED heatmap get saved?"),
            ]
        ),
        handler,
    )

    assert handler_called is False
    message = response.result[0]
    assert message.additional_kwargs["strap_origin"] == "typed_runtime_followup"
    assert message.additional_kwargs["strap_typed_runtime_status"] == "answered_from_prior_artifacts"
    assert str(plot_path) in message.content
    assert str(diagnostic_copy) in message.content


def test_typed_runtime_middleware_off_and_shadow_do_not_answer_prior_followups(tmp_path: Path):
    prior_artifact = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap",
        artifact_type="hsp_red_heatmap",
        source_step_id="hsp_screen",
        output_paths=[str(tmp_path / "red_heatmap.png")],
    )
    prior_message = AIMessage(
        content="Typed runtime completed.",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "executed",
            "strap_plan_id": "plan_hsp",
            "strap_workflow_id": "hsp_red_heatmap",
            "strap_manifest": {
                "run_id": "run_hsp",
                "run_dir": str(tmp_path / "runs"),
                "files": {},
                "produced_file_copies": {},
                "created_at": FIXED_TIME,
            },
            "strap_run_ledger": {
                "plan_id": "plan_hsp",
                "run_id": "run_hsp",
                "status": "succeeded",
                "started_at": FIXED_TIME,
                "completed_at": FIXED_TIME,
                "step_records": [],
                "artifacts": [prior_artifact.model_dump(mode="json")],
                "repairs": [],
                "final_contract_status": {},
            },
        },
    )

    for mode in ("off", "shadow"):
        handler_called = False

        def handler(request):
            nonlocal handler_called
            handler_called = True
            return ModelResponse(result=[AIMessage(content=f"legacy {mode}")])

        middleware = TypedRuntimeMiddleware(config=PlannerConfig(mode=mode), callable_registry={}, output_root=str(tmp_path / "runs"))
        response = middleware.wrap_model_call(
            SimpleNamespace(messages=[prior_message, HumanMessage(content="Where did that heatmap get saved?")]),
            handler,
        )

        assert handler_called is True
        assert response.result[0].content == f"legacy {mode}"


def test_typed_runtime_middleware_does_not_swallow_new_artifact_request(tmp_path: Path):
    handler_called = False

    def handler(request):
        nonlocal handler_called
        handler_called = True
        return ModelResponse(result=[AIMessage(content="legacy")])

    middleware = TypedRuntimeMiddleware(
        config=PlannerConfig(mode="off"),
        callable_registry={},
        output_root=str(tmp_path / "runs"),
    )
    response = middleware.wrap_model_call(
        SimpleNamespace(
            messages=[
                AIMessage(
                    content="Typed runtime completed.",
                    additional_kwargs={
                        "strap_origin": "typed_runtime",
                        "strap_typed_runtime_status": "executed",
                        "strap_runtime_progress": {"produced_artifact_paths": [str(tmp_path / "prior.png")]},
                    },
                ),
                HumanMessage(
                    content=(
                        "Screen nylons against polar aprotic solvents using HSP and return the batch "
                        "compatibility heatmap."
                    )
                ),
            ]
        ),
        handler,
    )

    assert handler_called is True
    assert response.result[0].content == "legacy"


def test_typed_runtime_middleware_answers_optimizer_why_followup_without_rerun(tmp_path: Path):
    payload_artifact = ArtifactFrame(
        artifact_id="optimize_pareto:optimization_pareto_landscape",
        artifact_type="optimization_pareto_landscape",
        source_step_id="optimize_pareto",
        output_paths=[str(tmp_path / "pareto_payload.json")],
        validation_summary={
            "payload": {
                "analysis_type": "pareto_front",
                "x_metric": "total_cost",
                "y_metric": "emissions",
                "n_points_feasible": 1,
                "n_points_raw_feasible": 1,
                "points": [
                    {
                        "point_id": 1,
                        "total_cost": 1000.0,
                        "emissions": 120.0,
                        "stage3_tech": ["lf"],
                        "route_id": "route_1",
                        "wash1_selection": ["PE-Cyclohexane"],
                    }
                ],
            }
        },
    )
    prior_message = AIMessage(
        content="Typed runtime completed.",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "executed",
            "strap_plan_id": "plan_routed",
            "strap_workflow_id": "routed_optimization",
            "strap_run_plan": {
                "plan_id": "plan_routed",
                "workflow_id": "routed_optimization",
                "global_constraints": {"requested_artifact_types": ["optimization_pareto_landscape"]},
                "steps": [],
            },
            "strap_run_ledger": {
                "plan_id": "plan_routed",
                "run_id": "run_routed",
                "status": "succeeded",
                "started_at": FIXED_TIME,
                "completed_at": FIXED_TIME,
                "step_records": [],
                "artifacts": [payload_artifact.model_dump(mode="json")],
                "repairs": [],
                "final_contract_status": {},
            },
            "strap_manifest": {
                "run_id": "run_routed",
                "run_dir": str(tmp_path / "prior"),
                "files": {},
                "produced_file_copies": {},
                "created_at": FIXED_TIME,
            },
        },
    )
    handler_called = False

    def should_not_execute(step, ledger):  # pragma: no cover - assertion guard
        raise AssertionError("explanatory follow-up should not execute typed runtime")

    def handler(request):  # pragma: no cover - assertion guard
        nonlocal handler_called
        handler_called = True
        return ModelResponse(result=[AIMessage(content="legacy")])

    middleware = TypedRuntimeMiddleware(
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_landscape", "optimization_pareto_plot"},
            selected_enforcement_workflows={"routed_optimization"},
        ),
        callable_registry={
            "plan_multiple_separation_schemes": should_not_execute,
            "build_handoff": should_not_execute,
            "run_waste_management_pareto": should_not_execute,
            "plot_optimization_pareto_front": should_not_execute,
        },
        output_root=str(tmp_path / "runs"),
    )
    response = middleware.wrap_model_call(
        SimpleNamespace(messages=[prior_message, HumanMessage(content="Why did that optimization choose landfill?")]),
        handler,
    )

    assert handler_called is False
    message = response.result[0]
    assert message.additional_kwargs["strap_origin"] == "typed_runtime_followup"
    assert message.additional_kwargs["strap_followup_reason"] == "optimization_answer_from_prior_artifacts"
    assert "no new run was started" in message.content
    assert "Landfill check: the verified frontier includes lf/landfill in a process stage." in message.content


def test_typed_runtime_middleware_hydrates_same_pareto_rerun(tmp_path: Path):
    plot_path = tmp_path / "rerun" / "pareto.png"
    plot_path.parent.mkdir()
    plot_path.write_text("png")
    captured_plot_args: dict[str, object] = {}
    payload_artifact = ArtifactFrame(
        artifact_id="optimize_pareto:optimization_pareto_landscape",
        artifact_type="optimization_pareto_landscape",
        source_step_id="optimize_pareto",
        output_paths=[str(tmp_path / "payload.json")],
    )
    prior_message = AIMessage(
        content="Typed runtime completed.",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "executed",
            "strap_plan_id": "plan_routed",
            "strap_workflow_id": "routed_optimization",
            "strap_run_plan": {
                "plan_id": "plan_routed",
                "workflow_id": "routed_optimization",
                "user_query": "prior routed optimization",
                "global_constraints": {
                    "feed_capacity_tpy": 8000.0,
                    "feed_composition": {"PE": 0.6, "EVOH": 0.4},
                    "scenario": "A",
                    "top_k_per_polymer": 6,
                    "n_points": 16,
                    "min_washes": 1,
                    "max_washes": 2,
                    "metrics": ["total_cost", "emissions"],
                    "requested_artifact_types": [
                        "separation_topk_sequences",
                        "optimization_stage_candidates",
                        "optimization_pareto_front",
                        "optimization_pareto_landscape",
                        "optimization_pareto_plot",
                    ],
                },
                "steps": [
                    {
                        "step_id": "optimize_pareto",
                        "allowed_tools": ["run_waste_management_pareto"],
                        "tool_args_template": {
                            "feed_capacity_tpy": 8000.0,
                            "feed_composition_json": {"PE": 0.6, "EVOH": 0.4},
                            "scenario": "A",
                            "x_metric": "total_cost",
                            "y_metric": "emissions",
                            "objective": "pareto",
                            "n_points": 16,
                            "min_washes": 1,
                            "max_washes": 2,
                        },
                    }
                ],
            },
            "strap_run_ledger": {
                "plan_id": "plan_routed",
                "run_id": "run_routed",
                "status": "succeeded",
                "started_at": FIXED_TIME,
                "completed_at": FIXED_TIME,
                "step_records": [],
                "artifacts": [payload_artifact.model_dump(mode="json")],
                "repairs": [],
                "final_contract_status": {},
            },
            "strap_manifest": {
                "run_id": "run_routed",
                "run_dir": str(tmp_path / "prior"),
                "files": {},
                "produced_file_copies": {},
                "created_at": FIXED_TIME,
            },
        },
    )

    def wrapper(step, ledger):
        if step.step_id == "plot_optimization":
            captured_plot_args.update(step.tool_args_template)
        return make_static_artifact_wrapper(
            output_paths={"optimization_pareto_plot": [str(plot_path)]}
        )(step, ledger)

    handler_called = False

    def handler(request):  # pragma: no cover - assertion guard
        nonlocal handler_called
        handler_called = True
        return ModelResponse(result=[AIMessage(content="legacy")])

    middleware = TypedRuntimeMiddleware(
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_plot", "optimization_pareto_landscape"},
            selected_enforcement_workflows={"routed_optimization"},
        ),
        callable_registry={
            "plan_multiple_separation_schemes": wrapper,
            "build_handoff": wrapper,
            "run_waste_management_pareto": wrapper,
            "plot_optimization_pareto_front": wrapper,
        },
        output_root=str(tmp_path / "runs"),
    )
    response = middleware.wrap_model_call(
        SimpleNamespace(
            messages=[
                prior_message,
                HumanMessage(
                    content=(
                        f'Generate the same Pareto landscape again with 8 points and save it to "{tmp_path / "rerun"}".'
                    )
                ),
            ]
        ),
        handler,
    )

    assert handler_called is False
    message = response.result[0]
    assert message.additional_kwargs["strap_origin"] == "typed_runtime"
    assert message.additional_kwargs["strap_typed_runtime_status"] == "executed"
    assert message.additional_kwargs["strap_workflow_id"] == "routed_optimization"
    assert message.additional_kwargs["strap_runtime_progress"]["completed_steps"] == [
        "separation_candidates",
        "build_optimization_handoff",
        "optimize_pareto",
        "plot_optimization",
    ]
    assert captured_plot_args["output_dir"] == str(tmp_path / "rerun")
    assert str(plot_path) in message.content


def test_typed_runtime_success_formats_frontier_points_table(tmp_path: Path):
    payload_path = tmp_path / "pareto_payload.json"
    payload_path.write_text("{}")
    plot_path = tmp_path / "pareto.png"
    plot_path.write_text("png")

    def optimize_wrapper(step, ledger):
        payload = {
            "analysis_type": "pareto_front",
            "x_metric": "total_cost",
            "y_metric": "emissions",
            "n_points_feasible": 1,
            "n_points_raw_feasible": 1,
            "points": [
                {
                    "point_id": 1,
                    "total_cost": 1000.0,
                    "emissions": 120.0,
                    "stage3_tech": ["lf"],
                    "route_id": "route_1",
                    "wash1_selection": ["PE-Cyclohexane"],
                }
            ],
        }
        return StepCallableResult(
            artifacts=[
                ArtifactFrame(
                    artifact_id=f"{step.step_id}:optimization_pareto_front",
                    artifact_type="optimization_pareto_front",
                    source_step_id=step.step_id,
                    output_paths=[str(payload_path)],
                    validation_summary={"payload": payload},
                ),
                ArtifactFrame(
                    artifact_id=f"{step.step_id}:optimization_pareto_landscape",
                    artifact_type="optimization_pareto_landscape",
                    source_step_id=step.step_id,
                    output_paths=[str(payload_path)],
                    validation_summary={"payload": payload},
                ),
            ]
        )

    result = maybe_run_typed_runtime(
        (
            "Run a Pareto optimization plot for 8000 tonnes/year composed of 60% PE and 40% EVOH "
            "under scenario A with 8 points."
        ),
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_landscape", "optimization_pareto_plot"},
        ),
        callable_registry={
            "run_waste_management_pareto": optimize_wrapper,
            "plot_optimization_pareto_front": make_static_artifact_wrapper(
                output_paths={"optimization_pareto_plot": [str(plot_path)]}
            ),
        },
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result is not None
    assert result.status == "executed"
    summary = format_typed_runtime_success(result)
    assert "| # | total_cost | emissions | stages | route | washes |" in summary
    assert "| 1 | 1,000 | 120 | lf/landfill | route_1 | wash1=PE-Cyclohexane |" in summary


def test_progress_deduplicates_shared_optimizer_payload_paths(tmp_path: Path):
    payload_path = tmp_path / "pareto_payload.json"
    payload_path.write_text("{}")
    plot_path = tmp_path / "pareto.png"
    plot_path.write_text("png")

    result = maybe_run_typed_runtime(
        (
            "Run a Pareto optimization plot for 8000 tonnes/year composed of 60% PE and 40% EVOH "
            "under scenario A with 8 points."
        ),
        config=PlannerConfig(
            mode="enforce_selected",
            selected_enforcement_artifacts={"optimization_pareto_landscape", "optimization_pareto_plot"},
        ),
        callable_registry={
            "run_waste_management_pareto": make_static_artifact_wrapper(
                output_paths={
                    "optimization_pareto_front": [str(payload_path)],
                    "optimization_pareto_landscape": [str(payload_path)],
                }
            ),
            "plot_optimization_pareto_front": make_static_artifact_wrapper(
                output_paths={"optimization_pareto_plot": [str(plot_path)]}
            ),
        },
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result is not None
    assert result.status == "executed"
    progress = summarize_typed_runtime_progress(result)
    assert progress.produced_artifact_paths == [str(payload_path), str(plot_path)]
    assert result.manifest is not None
    assert list(result.manifest.produced_file_copies) == [str(payload_path), str(plot_path)]


def test_typed_runtime_middleware_metadata_matches_synthesis_validation_failure(monkeypatch, tmp_path: Path):
    plot_path = tmp_path / "state_map.png"
    plot_path.write_text("png")

    def wrapper(step, ledger):
        return StepCallableResult(
            artifacts=[
                ArtifactFrame(
                    artifact_id="plot_state_map:separation_dp_state_map",
                    artifact_type="separation_dp_state_map",
                    source_step_id=step.step_id,
                    output_paths=[str(plot_path)],
                )
            ]
        )

    import strap.planning.typed_runtime_integration as integration

    class FailedValidation:
        status = "failed"
        reason = "forced validation failure"
        failed_checks = ["forced_check"]

    monkeypatch.setattr(integration, "validate_final_synthesis_sources", lambda *args, **kwargs: FailedValidation())
    middleware = TypedRuntimeMiddleware(
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_dp_state_map"}),
        callable_registry={"plot_dynamic_programming_separation_options": wrapper},
        output_root=str(tmp_path / "runs"),
    )
    response = middleware.wrap_model_call(
        SimpleNamespace(messages=[HumanMessage(content="Plot the dynamic-programming state map for LDPE/EVOH/PET.")]),
        lambda request: ModelResponse(result=[AIMessage(content="legacy")]),
    )

    assert response.result[0].additional_kwargs["strap_typed_runtime_status"] == "typed_failure"
    assert "Typed runtime failed." in response.result[0].content


def test_typed_runtime_middleware_falls_through_when_unselected(tmp_path: Path):
    handler_called = False

    def handler(request):
        nonlocal handler_called
        handler_called = True
        return ModelResponse(result=[AIMessage(content="legacy")])

    middleware = TypedRuntimeMiddleware(
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_dp_state_map"}),
        callable_registry={},
        output_root=str(tmp_path / "runs"),
    )
    response = middleware.wrap_model_call(
        SimpleNamespace(messages=[HumanMessage(content="What is the solubility of LDPE in cyclohexane at 80 C?")]),
        handler,
    )

    assert handler_called is True
    assert response.result[0].content == "legacy"


def test_typed_runtime_middleware_complex_unselected_request_still_falls_through(tmp_path: Path):
    handler_called = False
    complex_query = (
        "For a mixed plastic feedstock of 8000 tonnes/year composed of 60% LDPE, 20% EVOH, and 20% PET "
        "under scenario A, have the separation engineer propose candidates and then run a Pareto optimization plot."
    )

    def handler(request):
        nonlocal handler_called
        handler_called = True
        return ModelResponse(result=[AIMessage(content="legacy complex route")])

    middleware = TypedRuntimeMiddleware(
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"separation_dp_state_map"}),
        callable_registry={},
        output_root=str(tmp_path / "runs"),
    )
    response = middleware.wrap_model_call(SimpleNamespace(messages=[HumanMessage(content=complex_query)]), handler)

    assert handler_called is True
    assert response.result[0].content == "legacy complex route"


def test_create_dissolve_agent_includes_typed_runtime_before_routing(monkeypatch):
    import strap.agent as agent_module
    from strap.planning.typed_runtime_integration import TypedRuntimeMiddleware
    from strap.routing import RoutingMiddleware

    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent_module, "init_chat_model", lambda name: object())
    monkeypatch.setattr(agent_module, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(agent_module, "_build_subagents", lambda overrides=None: [])
    monkeypatch.setattr(agent_module, "get_core_tools", lambda: [])
    monkeypatch.setattr(agent_module, "get_result_extractor_tools", lambda: [])
    monkeypatch.setattr(agent_module, "FilesystemBackend", lambda root_dir: object())

    agent_module.create_dissolve_agent()

    middleware = captured["middleware"]
    typed_index = next(i for i, item in enumerate(middleware) if isinstance(item, TypedRuntimeMiddleware))
    routing_index = next(i for i, item in enumerate(middleware) if isinstance(item, RoutingMiddleware))
    assert typed_index < routing_index


def test_integration_selected_compile_failure_persists_diagnostics(tmp_path: Path):
    result = maybe_run_typed_runtime(
        "Optimize waste management for PE and EVOH.",
        config=PlannerConfig(mode="enforce_selected", selected_enforcement_artifacts={"optimization_point_result"}),
        callable_registry={},
        output_root=str(tmp_path / "runs"),
        created_at=FIXED_TIME,
    )

    assert result is not None
    assert result.status == "typed_failure"
    assert result.compile_result.status == "clarification_required"
    assert result.manifest is not None
    assert Path(result.manifest.files["request"]).exists()
    assert Path(result.manifest.files["compile_result"]).exists()
    assert "Failed phase: compile" in format_typed_runtime_failure(result)
