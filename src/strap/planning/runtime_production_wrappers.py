"""Production wrappers for selected typed-runtime workflows.

Unlike ``runtime_wrappers.py``, this module is not generic. Each wrapper calls
one concrete tool/adapter and only emits artifacts that are evidenced by the
structured tool output or by ledger artifacts from prior steps.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strap.handoff_adapters import _adapt_separation_to_optimization
from strap.handoff_models import HandoffRecord, HandoffScope
from strap.planning.executor import StepCallable, StepCallableResult
from strap.planning.models import ArtifactFrame, ExecutionLedger, PlanStep
from strap.planning.runtime_paths import normalize_runtime_path, slugify_run_component
from strap.tools.biosteam_tea_lca import run_biosteam_simulation, visualize_biosteam_results
from strap.tools.ml_prediction import predict_solubility_ml, screen_hsp_solubility_matrix
from strap.tools.safety_card import compare_solvent_safety_cards, get_solvent_safety_card
from strap.tools._helpers import set_plots_dir
from strap.tools.separation_visualization_tools import (
    create_selectivity_heatmap,
    create_separation_tree_plot,
    plot_dynamic_programming_separation_options,
)
from strap.tools.sequence_planning_tools import plan_multiple_separation_schemes
from strap.tools.visualization import plot_optimization_pareto_front, plot_optimization_pareto_slices
from strap.tools.waste_optimization import run_waste_management_pareto, run_waste_management_pareto_slices


PRODUCTION_WRAPPER_VERSION = "2026.04.pr6"


def _parse_tool_envelope(raw: Any, *, tool_name: str) -> tuple[bool, dict[str, Any], str]:
    """Return ``success, data, display`` from a STRAP tool response."""
    parsed: Any = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return True, {"raw_display": raw, "tool_name": tool_name}, raw
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict):
        data = dict(parsed["data"])
        display = str(parsed.get("display") or "")
    elif isinstance(parsed, dict):
        data = dict(parsed)
        display = str(parsed.get("display") or "")
    else:
        return False, {"raw_result_type": type(raw).__name__, "tool_name": tool_name}, ""
    success = bool(data.get("success", not data.get("error")))
    data.setdefault("tool_name", tool_name)
    return success, data, display


def _artifact(
    step: PlanStep,
    artifact_type: str,
    *,
    payload: dict[str, Any] | None = None,
    output_paths: list[str] | None = None,
    source_handoff_ids: list[str] | None = None,
    suffix: str | None = None,
) -> ArtifactFrame:
    normalized_paths = [normalize_runtime_path(path) for path in (output_paths or []) if str(path).strip()]
    return ArtifactFrame(
        artifact_id=f"{step.step_id}:{artifact_type}{':' + suffix if suffix else ''}",
        artifact_type=artifact_type,
        source_step_id=step.step_id,
        source_handoff_ids=source_handoff_ids or [],
        output_paths=normalized_paths,
        validation_summary={
            "normalized_by": "production_typed_runtime_wrapper",
            "wrapper_version": PRODUCTION_WRAPPER_VERSION,
            "payload": payload,
        },
    )


def _latest_artifact(
    ledger: ExecutionLedger,
    artifact_type: str,
    *,
    source_step_id: str | None = None,
) -> ArtifactFrame | None:
    for artifact in reversed(ledger.artifacts):
        if artifact.artifact_type != artifact_type:
            continue
        if source_step_id and artifact.source_step_id != source_step_id:
            continue
        return artifact
    return None


def _artifact_payload(artifact: ArtifactFrame | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    payload = artifact.validation_summary.get("payload")
    if isinstance(payload, dict):
        return payload
    for raw_path in artifact.output_paths:
        path = Path(normalize_runtime_path(raw_path))
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict) and isinstance(loaded.get("data"), dict):
            return loaded["data"]
        if isinstance(loaded, dict):
            return loaded
    return None


def _list_arg(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set | frozenset):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _json_or_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _state_map_paths(paths: list[Any]) -> list[str]:
    candidates = [str(path) for path in paths if str(path).strip()]
    return [
        path
        for path in candidates
        if "state_map" in Path(path).name.lower() or "dp_state_map" in Path(path).name.lower()
    ]


def wrap_dp_state_map_plot(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    polymers = args.get("polymers")
    if isinstance(polymers, list):
        polymers = ",".join(str(polymer) for polymer in polymers)
    temperature = args.get("temperature", args.get("temperature_c", 100.0))
    raw = plot_dynamic_programming_separation_options(
        polymers=str(polymers or ""),
        temperature=float(temperature),
        output_dir=args.get("output_dir"),
        objectives=str(args.get("objectives") or "selectivity"),
        include_sequence_plots=False,
        include_state_map=True,
        include_objective_paths=False,
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="plot_dynamic_programming_separation_options")
    paths = _state_map_paths(list(data.get("plot_paths") or []))
    if not success or not paths:
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "DP state-map tool did not produce a state-map path."),
        )
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "separation_dp_state_map", payload=data, output_paths=paths)],
        data={"tool_name": "plot_dynamic_programming_separation_options", "plot_paths": paths},
    )


def wrap_separation_topk(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    polymers = args.get("polymers")
    if isinstance(polymers, list):
        polymers = ",".join(str(polymer) for polymer in polymers)
    top_k = args.get("top_k_per_polymer") or args.get("top_k") or args.get("n_variants") or 2
    try:
        n_variants = max(1, min(int(top_k), 12))
    except (TypeError, ValueError):
        n_variants = 2
    raw = plan_multiple_separation_schemes(
        polymers=str(polymers or ""),
        temperature=float(args.get("temperature", args.get("temperature_c", 120.0))),
        n_variants=n_variants,
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="plan_multiple_separation_schemes")
    if not success or not data.get("top_k_sequences"):
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "Separation planner did not produce top_k_sequences."),
        )
    if args.get("source_user_query"):
        data.setdefault("source_user_query", args["source_user_query"])
    return StepCallableResult(
        success=True,
        artifacts=[
            _artifact(step, "separation_topk_sequences", payload=data),
            _artifact(step, "optimization_stage_candidates", payload=data),
        ],
        data={"tool_name": "plan_multiple_separation_schemes", "n_sequences": len(data.get("top_k_sequences") or [])},
    )


def wrap_solvent_safety_card(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    solvent_name = str(args.get("solvent_name") or args.get("solvent") or "").strip()
    if not solvent_name:
        return StepCallableResult(success=False, error="Safety-card request is missing solvent_name.")
    raw = get_solvent_safety_card(
        solvent_name=solvent_name,
        operating_temp_c=args.get("operating_temp_c") or args.get("temperature_c"),
        include_pubchem=bool(args.get("include_pubchem", True)),
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="get_solvent_safety_card")
    if not success or not isinstance(data.get("safety_profile"), dict):
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "Safety-card tool did not return a safety_profile."),
        )
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "solvent_safety_card", payload=data)],
        data={"tool_name": "get_solvent_safety_card", "solvent_name": data.get("solvent_name")},
    )


def wrap_solvent_safety_comparison(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    solvents = args.get("solvents") or args.get("solvent_names") or []
    solvent_names = ",".join(str(item) for item in _list_arg(solvents))
    if not solvent_names:
        return StepCallableResult(success=False, error="Safety-comparison request is missing solvents.")
    raw = compare_solvent_safety_cards(
        solvent_names=solvent_names,
        operating_temp_c=args.get("operating_temp_c") or args.get("temperature_c"),
        include_pubchem=bool(args.get("include_pubchem", True)),
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="compare_solvent_safety_cards")
    profiles = data.get("profiles")
    if not success or not isinstance(profiles, list) or not profiles:
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "Safety comparison did not return profiles."),
        )
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "solvent_safety_comparison", payload=data)],
        data={"tool_name": "compare_solvent_safety_cards", "n_profiles": len(profiles)},
    )


def wrap_hsp_single_pair(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    polymer = str(args.get("polymer") or args.get("polymer_name") or "").strip()
    solvent = str(args.get("solvent") or args.get("solvent_name") or "").strip()
    if not polymer or not solvent:
        return StepCallableResult(success=False, error="HSP single-pair request is missing polymer or solvent.")
    output_dir = args.get("output_dir")
    previous_plot_dir: str | None = None
    try:
        if output_dir:
            previous_plot_dir = set_plots_dir(normalize_runtime_path(str(output_dir)))
        raw = predict_solubility_ml(
            polymer_name=polymer,
            solvent_name=solvent,
            temperature=float(args.get("temperature", args.get("temperature_c", 25.0))),
            generate_visualizations=bool(args.get("generate_visualizations", True)),
        )
    finally:
        if previous_plot_dir is not None:
            set_plots_dir(previous_plot_dir)
    success, data, _display = _parse_tool_envelope(raw, tool_name="predict_solubility_ml")
    if not success or data.get("analysis_type") != "hsp_binary_screen" or "red" not in data:
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "HSP single-pair output is missing RED/HSP evidence."),
        )
    paths = [str(path) for path in data.get("artifacts") or []]
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "hsp_single_pair_summary", payload=data, output_paths=paths)],
        data={"tool_name": "predict_solubility_ml", "red": data.get("red"), "probability": data.get("probability")},
    )


def wrap_hsp_red_heatmap(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    output_dir = args.get("output_dir")
    previous_plot_dir: str | None = None
    try:
        if output_dir:
            previous_plot_dir = set_plots_dir(normalize_runtime_path(str(output_dir)))
        raw = screen_hsp_solubility_matrix(
            polymers=args.get("polymers") or args.get("polymer"),
            polymer_category=args.get("polymer_category"),
            solvents=args.get("solvents") or args.get("solvent"),
            solvent_category=args.get("solvent_category"),
            solvent_polarity=args.get("solvent_polarity"),
            temperature_c=float(args.get("temperature_c", args.get("temperature", 25.0))),
            generate_visualization=bool(args.get("generate_visualization", True)),
        )
    finally:
        if previous_plot_dir is not None:
            set_plots_dir(previous_plot_dir)
    success, data, _display = _parse_tool_envelope(raw, tool_name="screen_hsp_solubility_matrix")
    paths = [str(path) for path in data.get("artifacts") or []]
    if not success or data.get("analysis_type") != "hsp_binary_screen" or not isinstance(data.get("results"), list) or not paths:
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "HSP matrix output is missing results or heatmap artifact paths."),
        )
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "hsp_red_heatmap", payload=data, output_paths=paths)],
        data={"tool_name": "screen_hsp_solubility_matrix", "n_results": len(data.get("results") or []), "plot_paths": paths},
    )


def wrap_separation_tree_plot(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    polymers = args.get("polymers")
    if isinstance(polymers, list):
        polymers = ",".join(str(polymer) for polymer in polymers)
    if not str(polymers or "").strip():
        return StepCallableResult(success=False, error="Separation-tree request is missing polymers.")
    raw = create_separation_tree_plot(
        polymers=str(polymers),
        temperature=float(args.get("temperature", args.get("temperature_c", 120.0))),
        output_dir=args.get("output_dir"),
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="create_separation_tree_plot")
    paths = [str(path) for path in data.get("plot_paths") or [] if str(path).strip()]
    if not paths:
        paths = [str(path) for path in [data.get("rank1_plot"), data.get("topk_plot")] if str(path or "").strip()]
    if not success or not paths:
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "Separation-tree tool did not produce plot paths."),
        )
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "separation_tree_plot", payload=data, output_paths=paths)],
        data={"tool_name": "create_separation_tree_plot", "plot_paths": paths},
    )


def wrap_separation_selectivity_heatmap(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    polymers = args.get("polymers")
    solvents = args.get("solvents")
    if isinstance(polymers, list):
        polymers = ",".join(str(polymer) for polymer in polymers)
    if isinstance(solvents, list):
        solvents = ",".join(str(solvent) for solvent in solvents)
    if not str(polymers or "").strip():
        return StepCallableResult(success=False, error="Selectivity-heatmap request is missing polymers.")

    output_dir = args.get("output_dir")
    previous_plot_dir: str | None = None
    try:
        if output_dir:
            previous_plot_dir = set_plots_dir(normalize_runtime_path(str(output_dir)))
        raw = create_selectivity_heatmap(
            polymers=str(polymers),
            solvents=str(solvents or ""),
            temperature=float(args.get("temperature", args.get("temperature_c", 100.0))),
        )
    finally:
        if previous_plot_dir is not None:
            set_plots_dir(previous_plot_dir)

    success, data, _display = _parse_tool_envelope(raw, tool_name="create_selectivity_heatmap")
    path_candidates = [
        data.get("filepath"),
        *(data.get("plot_paths") or []),
        *(data.get("artifacts") or []),
    ]
    paths = [str(path) for path in path_candidates if str(path or "").strip()]
    if not success or not paths:
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "Selectivity-heatmap tool did not produce a plot path."),
        )
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "separation_selectivity_heatmap", payload=data, output_paths=paths)],
        data={"tool_name": "create_selectivity_heatmap", "plot_paths": paths},
    )


def wrap_biosteam_simulation(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    args = dict(step.tool_args_template)
    solvent = str(args.get("solvent") or "").strip()
    target_plastic = str(args.get("target_plastic") or "").strip()
    energy_case = str(args.get("energy_case") or "").strip()
    if not solvent or not target_plastic or not energy_case:
        return StepCallableResult(success=False, error="BioSTEAM simulation is missing solvent, target_plastic, or energy_case.")
    raw = run_biosteam_simulation(
        solvent=solvent,
        target_plastic=target_plastic,
        energy_case=energy_case,
        target_plastic_percent=float(args.get("target_plastic_percent", 60)),
        processing_capacity=float(args.get("processing_capacity", args.get("feed_capacity_tpy", 20000))),
        dissolution_temp_c=args.get("dissolution_temp_c") or args.get("temperature_c"),
        precipitation_temp_c=float(args.get("precipitation_temp_c", 25)),
        solvent_price=args.get("solvent_price"),
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="run_biosteam_simulation")
    tea = data.get("tea")
    lca = data.get("lca")
    if not success or not isinstance(tea, dict) or not isinstance(lca, dict):
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "BioSTEAM simulation did not return TEA/LCA evidence."),
        )
    missing_tea = [key for key in ("msp_usd_per_kg", "tci_usd", "aoc_usd_per_yr") if tea.get(key) is None]
    if missing_tea:
        return StepCallableResult(success=False, data=data, error=f"BioSTEAM TEA result is missing: {', '.join(missing_tea)}.")
    if lca.get("gwp_kg_co2e_per_kg") is None:
        return StepCallableResult(success=False, data=data, error="BioSTEAM LCA result is missing GWP.")
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "biosteam_tea_lca_result", payload=data)],
        data={
            "tool_name": "run_biosteam_simulation",
            "solvent": data.get("solvent", solvent),
            "target_plastic": data.get("target_plastic", target_plastic),
            "energy_case": data.get("energy_case", energy_case),
        },
    )


def wrap_biosteam_visualization(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    payload = _source_payload_for_plot(step, ledger, {"biosteam_tea_lca_result"})
    if not payload:
        return StepCallableResult(success=False, error="No BioSTEAM TEA/LCA payload found in ledger.")
    raw = visualize_biosteam_results(
        results_json=json.dumps(payload),
        chart_types=str(step.tool_args_template.get("chart_types") or "all"),
        output_dir=str(step.tool_args_template.get("output_dir") or "./plots"),
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="visualize_biosteam_results")
    paths = [str(path) for path in data.get("charts") or data.get("plot_paths") or [] if str(path).strip()]
    if not success or not paths:
        return StepCallableResult(
            success=False,
            data=data,
            error=str(data.get("error") or "BioSTEAM visualization did not produce chart paths."),
        )
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "biosteam_tea_lca_plot", payload=data, output_paths=paths)],
        data={"tool_name": "visualize_biosteam_results", "plot_paths": paths},
    )


def wrap_optimization_handoff(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    source_step_id = str(step.tool_args_template.get("source_step_id") or "separation_candidates")
    source_artifact = (
        _latest_artifact(ledger, "optimization_stage_candidates", source_step_id=source_step_id)
        or _latest_artifact(ledger, "separation_topk_sequences", source_step_id=source_step_id)
    )
    source_payload = _artifact_payload(source_artifact)
    if not source_payload:
        return StepCallableResult(success=False, error="No structured separation payload found in ledger.")

    handoff_id = f"typed_{uuid.uuid4().hex[:12]}"
    scope = HandoffScope(
        invocation_id="typed_runtime",
        run_id=ledger.run_id,
        thread_id="typed_runtime",
    )
    source_record = HandoffRecord(
        handoff_id=handoff_id,
        scope=scope,
        producer="separation-engineer",
        consumer="optimization-engineer",
        contract="separation-engineer.result.v1",
        status="ok",
        payload=source_payload,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    contract, handoff_payload, task_prompt = _adapt_separation_to_optimization(
        source_record,
        scope_user_query=str(step.tool_args_template.get("source_user_query") or source_payload.get("source_user_query") or ""),
    )
    handoff_payload = dict(handoff_payload)
    handoff_payload["handoff_contract"] = contract
    handoff_payload["task_prompt"] = task_prompt
    return StepCallableResult(
        success=True,
        artifacts=[
            _artifact(step, "handoff_payload", payload=handoff_payload, source_handoff_ids=[handoff_id]),
            _artifact(step, "optimization_stage_candidates", payload=handoff_payload, source_handoff_ids=[handoff_id]),
        ],
        data={
            "handoff_contract": contract,
            "source_handoff_id": handoff_id,
            "candidate_counts_by_polymer": handoff_payload.get("candidate_counts_by_polymer"),
        },
    )


def _stage_candidates_from_ledger(step: PlanStep, ledger: ExecutionLedger) -> dict[str, Any] | None:
    for contract in step.input_contracts:
        if contract.artifact_type not in {"optimization_stage_candidates", "handoff_payload"}:
            continue
        artifact = _latest_artifact(ledger, contract.artifact_type, source_step_id=contract.source_step_id)
        payload = _artifact_payload(artifact)
        if payload:
            return payload
    return (
        _artifact_payload(_latest_artifact(ledger, "optimization_stage_candidates"))
        or _artifact_payload(_latest_artifact(ledger, "handoff_payload"))
    )


def _base_optimization_kwargs(step: PlanStep, ledger: ExecutionLedger) -> tuple[dict[str, Any], dict[str, Any] | None]:
    args = dict(step.tool_args_template)
    handoff_payload = _stage_candidates_from_ledger(step, ledger)
    feed = args.get("feed", args.get("feed_capacity_tpy"))
    if feed is None and handoff_payload:
        feed = handoff_payload.get("feed_capacity_tpy")
    stage_candidates = handoff_payload or args.get("stage_candidates_json")
    kwargs: dict[str, Any] = {
        "feed": feed,
        "scenario": args.get("scenario", "A"),
        "x_metric": args.get("x_metric", "total_cost"),
        "y_metric": args.get("y_metric", "circularity"),
        "n_points": int(args.get("n_points") or 100),
        "candidate_solvents": args.get("candidate_solvents") or args.get("solvent_shortlist"),
        "polymer_solvent_filters_json": args.get("polymer_solvent_filters_json"),
        "stage_candidates_json": stage_candidates,
        "constraint_mode": args.get("constraint_mode") or (handoff_payload or {}).get("constraint_mode"),
        "fallback_policy": args.get("fallback_policy") or (handoff_payload or {}).get("fallback_policy"),
        "route_pool_mode": args.get("route_pool_mode") or (handoff_payload or {}).get("route_pool_mode"),
        "min_active_washes": args.get("min_active_washes", args.get("min_washes")),
        "max_active_washes": args.get("max_active_washes", args.get("max_washes")),
    }
    return {key: value for key, value in kwargs.items() if value is not None}, handoff_payload


def wrap_waste_management_pareto(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    kwargs, handoff_payload = _base_optimization_kwargs(step, ledger)
    feed_composition = step.tool_args_template.get("feed_composition_json") or (handoff_payload or {}).get("feed_composition")
    kwargs["feed_composition_json"] = _json_or_value(feed_composition)
    if kwargs.get("feed") is None or not kwargs.get("feed_composition_json"):
        return StepCallableResult(success=False, error="Pareto optimization is missing feed or feed_composition_json.")
    raw = run_waste_management_pareto(**kwargs)
    success, data, _display = _parse_tool_envelope(raw, tool_name="run_waste_management_pareto")
    if not success or data.get("analysis_type") != "pareto_front":
        return StepCallableResult(success=False, data=data, error=str(data.get("error") or "Pareto optimization failed."))
    output_paths = [str(data["pareto_payload_path"])] if data.get("pareto_payload_path") else []
    return StepCallableResult(
        success=True,
        artifacts=[
            _artifact(step, "optimization_pareto_front", payload=data, output_paths=output_paths),
            _artifact(step, "optimization_pareto_landscape", payload=data, output_paths=output_paths),
        ],
        data={"tool_name": "run_waste_management_pareto", "n_points": len(data.get("points") or [])},
    )


def wrap_waste_management_pareto_slices(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    kwargs, handoff_payload = _base_optimization_kwargs(step, ledger)
    slices = step.tool_args_template.get("composition_slices_json")
    kwargs["composition_slices_json"] = _json_or_value(slices)
    if kwargs.get("feed") is None or not kwargs.get("composition_slices_json"):
        return StepCallableResult(success=False, error="Pareto slice optimization is missing feed or composition_slices_json.")
    raw = run_waste_management_pareto_slices(**kwargs)
    success, data, _display = _parse_tool_envelope(raw, tool_name="run_waste_management_pareto_slices")
    if not success or data.get("analysis_type") != "pareto_slices":
        return StepCallableResult(success=False, data=data, error=str(data.get("error") or "Pareto slice optimization failed."))
    output_paths = [str(data["pareto_slices_payload_path"])] if data.get("pareto_slices_payload_path") else []
    return StepCallableResult(
        success=True,
        artifacts=[
            _artifact(step, "optimization_pareto_slices", payload=data, output_paths=output_paths),
            _artifact(step, "optimization_pareto_front", payload=data, output_paths=output_paths),
            _artifact(step, "optimization_pareto_landscape", payload=data, output_paths=output_paths),
            _artifact(step, "sidecar_file", payload=data, output_paths=output_paths),
        ],
        data={"tool_name": "run_waste_management_pareto_slices", "n_slices_solved": data.get("n_slices_solved")},
    )


def _source_payload_for_plot(step: PlanStep, ledger: ExecutionLedger, artifact_types: set[str]) -> dict[str, Any] | None:
    for contract in step.input_contracts:
        if contract.artifact_type not in artifact_types:
            continue
        payload = _artifact_payload(_latest_artifact(ledger, contract.artifact_type, source_step_id=contract.source_step_id))
        if payload:
            return payload
    for artifact_type in artifact_types:
        payload = _artifact_payload(_latest_artifact(ledger, artifact_type))
        if payload:
            return payload
    return None


def wrap_optimization_pareto_plot(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    payload = _source_payload_for_plot(step, ledger, {"optimization_pareto_front", "optimization_pareto_landscape"})
    if not payload:
        return StepCallableResult(success=False, error="No optimizer Pareto payload found in ledger.")
    output_stem = step.tool_args_template.get("output_stem")
    if not output_stem:
        output_stem = f"typed_runtime_{slugify_run_component(ledger.run_id)}_pareto"
    raw = plot_optimization_pareto_front(
        pareto_result_json=payload,
        plot_mode=str(step.tool_args_template.get("plot_mode") or "frontier_only"),
        plot_title=step.tool_args_template.get("plot_title"),
        output_dir=step.tool_args_template.get("output_dir"),
        output_path=step.tool_args_template.get("output_path"),
        output_stem=str(output_stem),
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="plot_optimization_pareto_front")
    paths = [str(path) for path in data.get("plot_paths") or []]
    if not success or not paths:
        return StepCallableResult(success=False, data=data, error=str(data.get("error") or "Pareto plot was not produced."))
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "optimization_pareto_plot", payload=data, output_paths=paths)],
        data={"tool_name": "plot_optimization_pareto_front", "plot_paths": paths},
    )


def wrap_optimization_pareto_slices_plot(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    payload = _source_payload_for_plot(step, ledger, {"optimization_pareto_slices"})
    if not payload:
        return StepCallableResult(success=False, error="No optimizer Pareto-slices payload found in ledger.")
    output_stem = step.tool_args_template.get("output_stem")
    if not output_stem:
        output_stem = f"typed_runtime_{slugify_run_component(ledger.run_id)}_pareto_slices"
    raw = plot_optimization_pareto_slices(
        pareto_slices_json=payload,
        plot_mode=str(step.tool_args_template.get("plot_mode") or "landscape"),
        output_dir=step.tool_args_template.get("output_dir"),
        output_path=step.tool_args_template.get("output_path"),
        output_stem=str(output_stem),
    )
    success, data, _display = _parse_tool_envelope(raw, tool_name="plot_optimization_pareto_slices")
    paths = [str(path) for path in data.get("plot_paths") or []]
    if not success or not paths:
        return StepCallableResult(success=False, data=data, error=str(data.get("error") or "Pareto slice plots were not produced."))
    return StepCallableResult(
        success=True,
        artifacts=[_artifact(step, "optimization_pareto_slices_plot", payload=data, output_paths=paths)],
        data={"tool_name": "plot_optimization_pareto_slices", "plot_paths": paths},
    )


def get_production_runtime_callable_registry(
    *,
    overrides: Mapping[str, StepCallable] | None = None,
) -> dict[str, StepCallable]:
    """Return production wrappers for the first selected typed workflows."""
    registry: dict[str, StepCallable] = {
        "plot_dynamic_programming_separation_options": wrap_dp_state_map_plot,
        "plan_multiple_separation_schemes": wrap_separation_topk,
        "get_solvent_safety_card": wrap_solvent_safety_card,
        "compare_solvent_safety_cards": wrap_solvent_safety_comparison,
        "predict_solubility_ml": wrap_hsp_single_pair,
        "screen_hsp_solubility_matrix": wrap_hsp_red_heatmap,
        "create_separation_tree_plot": wrap_separation_tree_plot,
        "create_selectivity_heatmap": wrap_separation_selectivity_heatmap,
        "run_biosteam_simulation": wrap_biosteam_simulation,
        "visualize_biosteam_results": wrap_biosteam_visualization,
        "build_handoff": wrap_optimization_handoff,
        "run_waste_management_pareto": wrap_waste_management_pareto,
        "run_waste_management_pareto_slices": wrap_waste_management_pareto_slices,
        "plot_optimization_pareto_front": wrap_optimization_pareto_plot,
        "plot_optimization_pareto_slices": wrap_optimization_pareto_slices_plot,
    }
    if overrides:
        registry.update(dict(overrides))
    return registry
