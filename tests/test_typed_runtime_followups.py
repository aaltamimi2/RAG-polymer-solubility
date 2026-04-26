from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from strap.planning.models import ArtifactFrame
from strap.planning.typed_runtime_followups import maybe_answer_typed_runtime_followup


def _typed_message(
    artifact: ArtifactFrame,
    *,
    status: str = "executed",
    run_dir: str = "/tmp/typed-run",
    copies: dict[str, str] | None = None,
) -> AIMessage:
    return AIMessage(
        content="typed runtime completed",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": status,
            "strap_plan_id": "plan_test",
            "strap_workflow_id": "workflow_test",
            "strap_runtime_progress": {
                "status": status,
                "produced_artifact_paths": list(artifact.output_paths),
                "failed_checks": [],
                "diagnostic_bundle_path": run_dir,
            },
            "strap_manifest": {
                "run_id": "run_test",
                "run_dir": run_dir,
                "files": {"manifest": f"{run_dir}/manifest.json"},
                "produced_file_copies": copies or {},
                "created_at": "2026-04-26T00:00:00+00:00",
            },
            "strap_run_ledger": {
                "plan_id": "plan_test",
                "run_id": "run_test",
                "status": "succeeded",
                "started_at": "2026-04-26T00:00:00+00:00",
                "completed_at": "2026-04-26T00:00:01+00:00",
                "step_records": [],
                "artifacts": [artifact.model_dump(mode="json")],
                "repairs": [],
                "final_contract_status": {},
            },
        },
    )


def _typed_failure_message() -> AIMessage:
    return AIMessage(
        content="Typed runtime failed.",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "typed_failure",
            "strap_plan_id": "plan_failure",
            "strap_workflow_id": None,
            "strap_runtime_progress": {
                "status": "typed_failure",
                "produced_artifact_paths": [],
                "failed_checks": ["callable_reported_failure", "required_artifact_missing"],
                "diagnostic_bundle_path": "/tmp/typed-failure",
            },
            "strap_manifest": {
                "run_id": "run_failure",
                "run_dir": "/tmp/typed-failure",
                "files": {"manifest": "/tmp/typed-failure/manifest.json"},
                "produced_file_copies": {},
                "created_at": "2026-04-26T00:00:00+00:00",
            },
            "strap_run_ledger": {
                "plan_id": "plan_failure",
                "run_id": "run_failure",
                "status": "failed",
                "started_at": "2026-04-26T00:00:00+00:00",
                "completed_at": "2026-04-26T00:00:01+00:00",
                "step_records": [
                    {
                        "step_id": "hsp_screen",
                        "status": "failed",
                        "attempt": 1,
                        "callable_name": "predict_solubility_ml",
                        "verification_status": "failed",
                        "failed_checks": ["callable_reported_failure", "required_artifact_missing"],
                        "artifact_ids": [],
                        "input_artifact_ids": [],
                        "output_artifact_ids": [],
                    }
                ],
                "artifacts": [],
                "repairs": [],
                "final_contract_status": {},
            },
        },
    )


def _pareto_payload() -> dict:
    return {
        "analysis_type": "pareto_front",
        "x_metric": "total_cost",
        "y_metric": "emissions",
        "n_points_feasible": 2,
        "n_points_raw_feasible": 3,
        "points": [
            {
                "point_id": 1,
                "total_cost": 1000.0,
                "emissions": 120.0,
                "stage3_tech": ["lf"],
                "stage3_variants": ["lf"],
                "route_id": "route_1",
                "wash1_selection": ["PE-Cyclohexane"],
                "wash2_selection": ["EVOH-Ethylene Glycol"],
            },
            {
                "point_id": 2,
                "total_cost": 1250.0,
                "emissions": 90.0,
                "stage3_tech": ["wte"],
                "route_id": "route_2",
                "wash1_selection": ["PE-Toluene"],
            },
        ],
        "all_feasible_points": [{"point_id": 1}, {"point_id": 2}, {"point_id": 3}],
    }


def _typed_optimization_message() -> AIMessage:
    payload_artifact = ArtifactFrame(
        artifact_id="optimize_pareto:optimization_pareto_landscape",
        artifact_type="optimization_pareto_landscape",
        source_step_id="optimize_pareto",
        output_paths=["/tmp/pareto_payload.json"],
        validation_summary={"payload": _pareto_payload()},
    )
    artifact = ArtifactFrame(
        artifact_id="plot_optimization:optimization_pareto_plot",
        artifact_type="optimization_pareto_plot",
        source_step_id="plot_optimization",
        output_paths=["/tmp/pareto.png"],
    )
    return AIMessage(
        content="Typed runtime completed.",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "executed",
            "strap_plan_id": "plan_routed",
            "strap_workflow_id": "routed_optimization",
            "strap_run_plan": {
                "plan_id": "plan_routed",
                "workflow_id": "routed_optimization",
                "global_constraints": {},
                "steps": [],
            },
            "strap_manifest": {
                "run_id": "run_routed",
                "run_dir": "/tmp/routed-run",
                "files": {"manifest": "/tmp/routed-run/manifest.json"},
                "produced_file_copies": {},
                "created_at": "2026-04-26T00:00:00+00:00",
            },
            "strap_run_ledger": {
                "plan_id": "plan_routed",
                "run_id": "run_routed",
                "status": "succeeded",
                "started_at": "2026-04-26T00:00:00+00:00",
                "completed_at": "2026-04-26T00:00:01+00:00",
                "step_records": [
                    {
                        "step_id": "separation_candidates",
                        "status": "succeeded",
                        "attempt": 1,
                        "callable_name": "plan_multiple_separation_schemes",
                        "verification_status": "passed",
                        "failed_checks": [],
                        "artifact_ids": [],
                        "input_artifact_ids": [],
                        "output_artifact_ids": [],
                    },
                    {
                        "step_id": "build_optimization_handoff",
                        "status": "succeeded",
                        "attempt": 1,
                        "callable_name": "build_handoff",
                        "verification_status": "passed",
                        "failed_checks": [],
                        "artifact_ids": [],
                        "input_artifact_ids": [],
                        "output_artifact_ids": [],
                    },
                    {
                        "step_id": "optimize_pareto",
                        "status": "succeeded",
                        "attempt": 1,
                        "callable_name": "run_waste_management_pareto",
                        "verification_status": "passed",
                        "failed_checks": [],
                        "artifact_ids": [],
                        "input_artifact_ids": [],
                        "output_artifact_ids": [],
                    },
                    {
                        "step_id": "plot_optimization",
                        "status": "succeeded",
                        "attempt": 1,
                        "callable_name": "plot_optimization_pareto_front",
                        "verification_status": "passed",
                        "failed_checks": [],
                        "artifact_ids": [],
                        "input_artifact_ids": [],
                        "output_artifact_ids": [],
                    },
                ],
                "artifacts": [payload_artifact.model_dump(mode="json"), artifact.model_dump(mode="json")],
                "repairs": [],
                "final_contract_status": {},
            },
        },
    )


def test_path_followup_returns_prior_hsp_heatmap_path():
    artifact = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap",
        artifact_type="hsp_red_heatmap",
        source_step_id="hsp_screen",
        output_paths=["/tmp/red_heatmap.png"],
        validation_summary={
            "payload": {
                "results": [{"polymer": "LDPE", "solvent": "dodecane"}],
                "polymer_resolution": {"category": "polyolefin"},
                "solvent_polarity": "nonpolar",
            }
        },
    )

    decision = maybe_answer_typed_runtime_followup(
        "Where did that RED heatmap get saved?",
        [_typed_message(artifact)],
    )

    assert decision.should_answer is True
    assert decision.reason == "path_status_from_prior_artifacts"
    assert "/tmp/red_heatmap.png" in (decision.response_text or "")
    assert decision.progress["matched_artifact_types"] == ["hsp_red_heatmap"]


def test_diagnostic_bundle_followup_returns_copy_path():
    artifact = ArtifactFrame(
        artifact_id="plot_biosteam:biosteam_tea_lca_plot",
        artifact_type="biosteam_tea_lca_plot",
        source_step_id="plot_biosteam",
        output_paths=["/tmp/tea_lca.png"],
    )
    message = _typed_message(
        artifact,
        run_dir="/tmp/typed-bundle",
        copies={"/tmp/tea_lca.png": "/tmp/typed-bundle/produced_files/tea_lca.png"},
    )

    decision = maybe_answer_typed_runtime_followup(
        "Where is the BioSTEAM TEA/LCA plot and diagnostic bundle?",
        [message],
    )

    assert decision.should_answer is True
    assert "/tmp/tea_lca.png" in (decision.response_text or "")
    assert "/tmp/typed-bundle/produced_files/tea_lca.png" in (decision.response_text or "")
    assert "/tmp/typed-bundle" in (decision.response_text or "")


def test_summary_followup_formats_two_recent_heatmaps_without_legacy():
    first = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap:1",
        artifact_type="hsp_red_heatmap",
        source_step_id="hsp_screen",
        output_paths=["/tmp/polyolefins_red.png"],
        validation_summary={"payload": {"results": [{"a": 1}, {"a": 2}]}},
    )
    second = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap:2",
        artifact_type="hsp_red_heatmap",
        source_step_id="hsp_screen",
        output_paths=["/tmp/nylons_red.png"],
        validation_summary={"payload": {"results": [{"a": 1}]}},
    )

    decision = maybe_answer_typed_runtime_followup(
        "Summarize what we learned from the two heatmap requests and include paths.",
        [_typed_message(first), _typed_message(second)],
    )

    assert decision.should_answer is True
    assert "Typed runtime artifact summary:" in (decision.response_text or "")
    assert "/tmp/nylons_red.png" in (decision.response_text or "")
    assert "/tmp/polyolefins_red.png" in (decision.response_text or "")
    assert "results_count: 1" in (decision.response_text or "")
    assert "results_count: 2" in (decision.response_text or "")


def test_summary_with_artifact_hint_still_includes_recent_typed_failure():
    heatmap = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap",
        artifact_type="hsp_red_heatmap",
        source_step_id="hsp_screen",
        output_paths=["/tmp/red_heatmap.png"],
        validation_summary={"payload": {"results": [{"a": 1}]}},
    )

    decision = maybe_answer_typed_runtime_followup(
        "Summarize what we learned from the heatmap request and the GVL check.",
        [_typed_message(heatmap), _typed_failure_message()],
    )

    assert decision.should_answer is True
    assert "hsp_red_heatmap" in (decision.response_text or "")
    assert "typed_runtime_failure" in (decision.response_text or "")
    assert "required_artifact_missing" in (decision.response_text or "")
    assert "/tmp/typed-failure" in (decision.response_text or "")


def test_runtime_status_followup_reports_steps_tools_and_provenance():
    decision = maybe_answer_typed_runtime_followup(
        "What typed-runtime steps completed, and did the run use the routed Pareto plan rather than a direct solubility or safety path?",
        [_typed_optimization_message()],
    )

    assert decision.should_answer is True
    assert decision.reason == "runtime_status_from_prior_plan"
    assert "Completed steps: separation_candidates, build_optimization_handoff, optimize_pareto, plot_optimization" in (
        decision.response_text or ""
    )
    assert "run_waste_management_pareto" in (decision.response_text or "")
    assert "routed Pareto optimization workflow" in (decision.response_text or "")
    assert "no direct solubility or safety fast-path" in (decision.response_text or "")
    assert decision.progress["completed_steps"] == [
        "separation_candidates",
        "build_optimization_handoff",
        "optimize_pareto",
        "plot_optimization",
    ]


def test_optimizer_frontier_followup_renders_points_table_from_payload():
    decision = maybe_answer_typed_runtime_followup(
        "What were the frontier points from that optimization?",
        [_typed_optimization_message()],
    )

    assert decision.should_answer is True
    assert decision.reason == "optimization_answer_from_prior_artifacts"
    assert "no new run was started" in (decision.response_text or "")
    assert "| # | total_cost | emissions | stages | route | washes |" in (decision.response_text or "")
    assert "| 1 | 1,000 | 120 | lf/landfill | route_1 | wash1=PE-Cyclohexane; wash2=EVOH-Ethylene Glycol |" in (
        decision.response_text or ""
    )


def test_optimizer_why_followup_answers_from_payload_without_hydration_shape():
    decision = maybe_answer_typed_runtime_followup(
        "Why did that optimization choose landfill?",
        [_typed_optimization_message()],
    )

    assert decision.should_answer is True
    assert decision.reason == "optimization_answer_from_prior_artifacts"
    assert "Landfill check: the verified frontier includes lf/landfill in a process stage." in (
        decision.response_text or ""
    )
    assert "Interpretation limit:" in (decision.response_text or "")


def test_new_artifact_request_is_not_swallowed_by_followup_resolver():
    artifact = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap",
        artifact_type="hsp_red_heatmap",
        output_paths=["/tmp/red_heatmap.png"],
    )

    decision = maybe_answer_typed_runtime_followup(
        "Screen nylons against polar aprotic solvents using HSP and return the batch compatibility heatmap.",
        [_typed_message(artifact)],
    )

    assert decision.should_answer is False


def test_new_saved_artifact_request_is_not_swallowed_by_followup_resolver():
    artifact = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap",
        artifact_type="hsp_red_heatmap",
        output_paths=["/tmp/red_heatmap.png"],
    )

    decision = maybe_answer_typed_runtime_followup(
        'Screen nylons against polar aprotic solvents using HSP and return the batch compatibility heatmap. Save it to "/tmp/hsp_followup".',
        [_typed_message(artifact)],
    )

    assert decision.should_answer is False


def test_new_lookup_request_with_list_is_not_swallowed_by_followup_resolver():
    artifact = ArtifactFrame(
        artifact_id="hsp_screen:hsp_red_heatmap",
        artifact_type="hsp_red_heatmap",
        output_paths=["/tmp/red_heatmap.png"],
    )

    decision = maybe_answer_typed_runtime_followup(
        "List available solvents for LDPE.",
        [_typed_message(artifact)],
    )

    assert decision.should_answer is False


def test_no_prior_typed_artifacts_returns_false():
    decision = maybe_answer_typed_runtime_followup(
        "Where did that plot get saved?",
        [AIMessage(content="legacy answer")],
    )

    assert decision.should_answer is False
    assert decision.reason == "no_prior_typed_runtime_artifacts"


def test_formatter_uses_generic_fallback_for_non_specialized_artifacts():
    artifact = ArtifactFrame(
        artifact_id="state_map:separation_dp_state_map",
        artifact_type="separation_dp_state_map",
        source_step_id="plot_state_map",
        output_paths=["/tmp/state_map.png"],
    )
    decision = maybe_answer_typed_runtime_followup(
        "Summarize the separation state map output.",
        [_typed_message(artifact)],
    )

    assert decision.should_answer is True
    assert "separation_dp_state_map" in (decision.response_text or "")
    assert "/tmp/state_map.png" in (decision.response_text or "")


def test_followup_can_load_artifacts_from_manifest_file(tmp_path: Path):
    artifact = ArtifactFrame(
        artifact_id="pareto:optimization_pareto_plot",
        artifact_type="optimization_pareto_plot",
        source_step_id="plot_pareto",
        output_paths=["/tmp/pareto.png"],
    )
    artifacts_path = tmp_path / "artifacts.json"
    artifacts_path.write_text(f"[{artifact.model_dump_json()}]", encoding="utf-8")
    message = AIMessage(
        content="typed runtime completed",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "executed",
            "strap_manifest": {
                "run_id": "run_test",
                "run_dir": str(tmp_path),
                "files": {"artifacts": str(artifacts_path), "manifest": str(tmp_path / "manifest.json")},
                "produced_file_copies": {},
                "created_at": "2026-04-26T00:00:00+00:00",
            },
        },
    )

    decision = maybe_answer_typed_runtime_followup("Give me the Pareto plot path.", [message])

    assert decision.should_answer is True
    assert "/tmp/pareto.png" in (decision.response_text or "")
