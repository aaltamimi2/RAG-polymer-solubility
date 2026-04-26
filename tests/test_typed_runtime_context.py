from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from strap.planning.compiler import compile_request
from strap.planning.models import ArtifactFrame
from strap.planning.typed_runtime_context import (
    collect_recent_typed_runtime_snapshots,
    maybe_hydrate_context_for_selected_followup,
)


FIXED_TIME = "2026-04-26T00:00:00+00:00"


def _optimization_plan() -> dict:
    return {
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
    }


def _optimization_ledger(artifacts: list[ArtifactFrame]) -> dict:
    return {
        "plan_id": "plan_routed",
        "run_id": "run_routed",
        "status": "succeeded",
        "started_at": FIXED_TIME,
        "completed_at": FIXED_TIME,
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
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        "repairs": [],
        "final_contract_status": {},
    }


def _optimization_message(*, run_dir: str = "/tmp/run") -> AIMessage:
    payload = ArtifactFrame(
        artifact_id="optimize_pareto:optimization_pareto_landscape",
        artifact_type="optimization_pareto_landscape",
        source_step_id="optimize_pareto",
        output_paths=["/tmp/pareto_payload.json"],
    )
    plot = ArtifactFrame(
        artifact_id="plot_optimization:optimization_pareto_plot",
        artifact_type="optimization_pareto_plot",
        source_step_id="plot_optimization",
        output_paths=["/tmp/pareto.png"],
    )
    return AIMessage(
        content="typed runtime completed",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "executed",
            "strap_plan_id": "plan_routed",
            "strap_workflow_id": "routed_optimization",
            "strap_run_plan": _optimization_plan(),
            "strap_run_ledger": _optimization_ledger([payload, plot]),
            "strap_manifest": {
                "run_id": "run_routed",
                "run_dir": run_dir,
                "files": {},
                "produced_file_copies": {},
                "created_at": FIXED_TIME,
            },
        },
    )


def test_collects_snapshot_from_typed_message_metadata():
    snapshots = collect_recent_typed_runtime_snapshots([_optimization_message()])

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.plan_id == "plan_routed"
    assert snapshot.workflow_id == "routed_optimization"
    assert [artifact.artifact_type for artifact in snapshot.artifacts] == [
        "optimization_pareto_landscape",
        "optimization_pareto_plot",
    ]


def test_collects_snapshot_from_manifest_files(tmp_path: Path):
    artifacts = [
        ArtifactFrame(
            artifact_id="plot_optimization:optimization_pareto_plot",
            artifact_type="optimization_pareto_plot",
            output_paths=["/tmp/pareto.png"],
        )
    ]
    (tmp_path / "plan.json").write_text(json.dumps(_optimization_plan()), encoding="utf-8")
    (tmp_path / "ledger.json").write_text(json.dumps(_optimization_ledger(artifacts)), encoding="utf-8")
    (tmp_path / "artifacts.json").write_text(
        json.dumps([artifact.model_dump(mode="json") for artifact in artifacts]),
        encoding="utf-8",
    )
    message = AIMessage(
        content="typed runtime completed",
        additional_kwargs={
            "strap_origin": "typed_runtime",
            "strap_typed_runtime_status": "executed",
            "strap_manifest": {
                "run_id": "run_routed",
                "run_dir": str(tmp_path),
                "files": {
                    "plan": str(tmp_path / "plan.json"),
                    "ledger": str(tmp_path / "ledger.json"),
                    "artifacts": str(tmp_path / "artifacts.json"),
                },
                "produced_file_copies": {},
                "created_at": FIXED_TIME,
            },
        },
    )

    snapshot = collect_recent_typed_runtime_snapshots([message])[0]

    assert snapshot.workflow_id == "routed_optimization"
    assert snapshot.ledger is not None
    assert snapshot.artifacts[0].artifact_type == "optimization_pareto_plot"


def test_hydrates_same_pareto_rerun_with_current_turn_overrides():
    hydration = maybe_hydrate_context_for_selected_followup(
        'Generate the same Pareto landscape again with 8 points and save it to "/tmp/rerun".',
        [_optimization_message()],
    )

    assert hydration is not None
    assert hydration.context["feed_capacity_tpy"] == 8000.0
    assert hydration.context["feed_composition"] == {"PE": 0.6, "EVOH": 0.4}
    assert hydration.context["scenario"] == "A"
    assert hydration.context["prior_workflow_id"] == "routed_optimization"

    compiled = compile_request(
        'Generate the same Pareto landscape again with 8 points and save it to "/tmp/rerun".',
        context=hydration.context,
        created_at=FIXED_TIME,
    )

    assert compiled.status == "compiled"
    assert compiled.extracted_facts["n_points"] == 8
    assert compiled.extracted_facts["output_dir"] == "/tmp/rerun"
    assert compiled.plan is not None
    assert compiled.plan.workflow_id == "routed_optimization"
    plot_step = next(step for step in compiled.plan.steps if step.step_id == "plot_optimization")
    assert plot_step.tool_args_template["output_dir"] == "/tmp/rerun"


def test_does_not_hydrate_without_reuse_marker_or_compatible_typed_run():
    assert maybe_hydrate_context_for_selected_followup(
        "Generate a Pareto landscape with 8 points.",
        [_optimization_message()],
    ) is None
    assert maybe_hydrate_context_for_selected_followup(
        "Generate the same Pareto landscape again.",
        [AIMessage(content="legacy")],
    ) is None


def test_does_not_hydrate_interpretive_optimization_followup():
    assert maybe_hydrate_context_for_selected_followup(
        "Why did that optimization choose landfill?",
        [_optimization_message()],
    ) is None
    assert maybe_hydrate_context_for_selected_followup(
        "Why did that optimization choose landfill? Answer from optimizer artifacts only; do not rerun.",
        [_optimization_message()],
    ) is None
