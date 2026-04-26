from __future__ import annotations

from pathlib import Path

from strap.planning.config import PlannerConfig
from strap.planning.executor import (
    StepCallableResult,
    VerificationResult,
    create_execution_ledger,
    next_runnable_step,
    record_step_result,
    run_plan,
    validate_execution_plan,
    verify_step_outputs,
)
from strap.planning.models import (
    ArtifactContract,
    ArtifactFrame,
    ExecutionLedger,
    InputContract,
    OutputContract,
    PlanStep,
    RequestPlan,
    RetryPolicy,
    StepExecutionRecord,
)


FIXED_TIME = "2026-04-26T00:00:00+00:00"


def _artifact(artifact_type: str, artifact_id: str, *, source_step_id: str | None = None, paths: list[str] | None = None) -> ArtifactFrame:
    return ArtifactFrame(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        source_step_id=source_step_id,
        output_paths=paths or [],
    )


def _output(artifact_type: str, *, path_policy: str = "optional", forbidden: list[str] | None = None) -> OutputContract:
    return OutputContract(
        contract_id=f"{artifact_type}_contract",
        artifact_contracts=[
            ArtifactContract(
                artifact_type=artifact_type,
                path_policy=path_policy,  # type: ignore[arg-type]
                forbidden_artifact_types=forbidden or [],
            )
        ],
    )


def _step(
    step_id: str,
    tool: str,
    artifact_type: str,
    *,
    depends_on: list[str] | None = None,
    input_contracts: list[InputContract] | None = None,
    retry_policy: RetryPolicy | None = None,
    path_policy: str = "optional",
    forbidden: list[str] | None = None,
    execution_kind: str = "tool",
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        label=step_id,
        role="direct_tool",
        execution_kind=execution_kind,  # type: ignore[arg-type]
        allowed_tools=[tool],
        depends_on=depends_on or [],
        input_contracts=input_contracts or [],
        output_contracts=[_output(artifact_type, path_policy=path_policy, forbidden=forbidden)],
        retry_policy=retry_policy or RetryPolicy(max_attempts=2),
    )


def _synthesis_step(*, depends_on: list[str], input_contracts: list[InputContract]) -> PlanStep:
    return PlanStep(
        step_id="synthesize",
        label="Synthesize",
        role="direct_tool",
        execution_kind="synthesis",
        allowed_tools=["synthesize"],
        depends_on=depends_on,
        input_contracts=input_contracts,
        output_contracts=[],
    )


def _plan(steps: list[PlanStep]) -> RequestPlan:
    return RequestPlan(
        plan_id="plan_executor_test",
        created_at=FIXED_TIME,
        compiler_version="test",
        capability_registry_version="test",
        user_query="executor test",
        mode="planned_workflow" if len(steps) > 1 else "single_tool_or_specialist",
        intent_family="mixed_workflow",
        complexity="complex",
        steps=steps,
    )


def _success_result(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
    artifact_type = step.output_contracts[0].artifact_contracts[0].artifact_type
    return StepCallableResult(
        artifacts=[_artifact(artifact_type, f"{step.step_id}_artifact")]
    )


def test_next_runnable_step_and_run_plan_execute_linear_workflow_in_order():
    calls: list[str] = []
    step_a = _step("a", "tool_a", "separation_topk_sequences")
    step_b = _step(
        "b",
        "tool_b",
        "optimization_stage_candidates",
        depends_on=["a"],
        input_contracts=[InputContract(artifact_type="separation_topk_sequences", source_step_id="a")],
    )
    plan = _plan([step_a, step_b])
    ledger = create_execution_ledger(plan, run_id="run_test", started_at=FIXED_TIME)

    assert next_runnable_step(plan, ledger).step_id == "a"  # type: ignore[union-attr]

    def tool_a(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        calls.append(step.step_id)
        return _success_result(step, ledger)

    def tool_b(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        calls.append(step.step_id)
        assert {artifact.artifact_type for artifact in ledger.artifacts} == {"separation_topk_sequences"}
        return _success_result(step, ledger)

    result = run_plan(
        plan,
        {"tool_a": tool_a, "tool_b": tool_b},
        PlannerConfig(mode="off"),
        ledger=ledger,
    )

    assert result.status == "succeeded"
    assert calls == ["a", "b"]
    assert [record.step_id for record in result.step_records if record.status == "succeeded"] == ["a", "b"]


def test_next_runnable_step_uses_latest_step_status():
    step_a = _step("a", "tool_a", "separation_topk_sequences")
    step_b = _step("b", "tool_b", "optimization_stage_candidates", depends_on=["a"])
    plan = _plan([step_a, step_b])
    ledger = create_execution_ledger(plan).model_copy(
        update={
            "step_records": [
                StepExecutionRecord(step_id="a", status="succeeded", attempt=1),
                StepExecutionRecord(step_id="a", status="failed", attempt=2),
            ]
        }
    )

    assert next_runnable_step(plan, ledger) is None


def test_record_step_result_returns_new_ledger_without_mutating_input():
    step = _step("a", "tool_a", "separation_topk_sequences")
    plan = _plan([step])
    ledger = create_execution_ledger(plan)
    result = _success_result(step, ledger)
    verification = verify_step_outputs(step, result, ledger)

    next_ledger = record_step_result(
        ledger,
        step,
        result,
        verification,
        attempt=1,
        callable_name="tool_a",
    )

    assert ledger.step_records == []
    assert ledger.artifacts == []
    assert len(next_ledger.step_records) == 1
    assert len(next_ledger.artifacts) == 1


def test_validate_execution_plan_detects_cycle_before_execution():
    step_a = PlanStep.model_construct(step_id="a", depends_on=["b"], allowed_tools=["tool_a"], output_contracts=[])
    step_b = PlanStep.model_construct(step_id="b", depends_on=["a"], allowed_tools=["tool_b"], output_contracts=[])
    plan = RequestPlan.model_construct(plan_id="bad", steps=[step_a, step_b])

    assert any("cycle detected" in error for error in validate_execution_plan(plan))
    result = run_plan(plan, {}, PlannerConfig(mode="off"))
    assert result.status == "failed"
    assert result.final_contract_status["failed_checks"] == ["execution_plan_invalid"]


def test_validate_execution_plan_detects_unknown_dependency_before_execution():
    step_a = PlanStep.model_construct(step_id="a", depends_on=["missing"], allowed_tools=["tool_a"], output_contracts=[])
    plan = RequestPlan.model_construct(plan_id="bad", steps=[step_a])

    assert "a: unknown dependency missing" in validate_execution_plan(plan)


def test_wrong_artifact_output_blocks_advancement():
    step = _step("a", "tool_a", "separation_topk_sequences")
    plan = _plan([step])

    def wrong(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        return StepCallableResult(artifacts=[_artifact("solubility_curve", "wrong")])

    result = run_plan(plan, {"tool_a": wrong}, PlannerConfig(mode="off"))

    assert result.status == "failed"
    assert "artifact_type_mismatch" in result.step_records[-1].failed_checks
    assert "required_artifact_missing" in result.step_records[-1].failed_checks


def test_callable_exception_is_distinct_from_verification_failure():
    step = _step("a", "tool_a", "separation_topk_sequences")
    plan = _plan([step])

    def raises(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        raise RuntimeError("boom")

    result = run_plan(plan, {"tool_a": raises}, PlannerConfig(mode="off"))

    assert result.status == "failed"
    assert "callable_exception" in result.step_records[-1].failed_checks
    assert "artifact_type_mismatch" not in result.step_records[-1].failed_checks


def test_required_output_path_is_validated(tmp_path: Path):
    output_path = tmp_path / "plot.png"
    output_path.write_text("png")
    step = _step("plot", "plot_tool", "optimization_pareto_plot", path_policy="required")
    plan = _plan([step])

    ok = verify_step_outputs(
        step,
        StepCallableResult(artifacts=[_artifact("optimization_pareto_plot", "plot", paths=[str(output_path)])]),
        create_execution_ledger(plan),
    )
    missing = verify_step_outputs(
        step,
        StepCallableResult(artifacts=[_artifact("optimization_pareto_plot", "plot_missing", paths=[str(tmp_path / "missing.png")])]),
        create_execution_ledger(plan),
    )

    assert ok.passed
    assert "output_path_not_found" in missing.failed_checks


def test_forbidden_output_path_is_validated(tmp_path: Path):
    output_path = tmp_path / "payload.json"
    output_path.write_text("{}")
    step = _step("payload", "payload_tool", "handoff_payload", path_policy="forbidden")
    verification = verify_step_outputs(
        step,
        StepCallableResult(artifacts=[_artifact("handoff_payload", "payload", paths=[str(output_path)])]),
        create_execution_ledger(_plan([step])),
    )

    assert verification.status == "failed"
    assert "output_path_forbidden" in verification.failed_checks


def test_forbidden_artifact_fails_verification():
    step = _step("a", "tool_a", "separation_topk_sequences", forbidden=["solubility_curve"])
    verification = verify_step_outputs(
        step,
        StepCallableResult(artifacts=[
            _artifact("separation_topk_sequences", "ok"),
            _artifact("solubility_curve", "bad"),
        ]),
        create_execution_ledger(_plan([step])),
    )

    assert verification.status == "failed"
    assert "forbidden_artifact_present" in verification.failed_checks


def test_source_step_id_mismatch_fails_dependency_verification():
    step = _step(
        "b",
        "tool_b",
        "optimization_stage_candidates",
        input_contracts=[InputContract(artifact_type="separation_topk_sequences", source_step_id="expected")],
    )
    ledger = create_execution_ledger(_plan([step])).model_copy(
        update={"artifacts": [_artifact("separation_topk_sequences", "wrong_source", source_step_id="other")]}
    )
    verification = verify_step_outputs(step, _success_result(step, ledger), ledger)

    assert verification.status == "failed"
    assert "source_step_id_mismatch" in verification.failed_checks


def test_retry_policy_uses_max_attempts_as_total_attempts():
    step = _step("a", "tool_a", "separation_topk_sequences", retry_policy=RetryPolicy(max_attempts=2))
    plan = _plan([step])
    attempts = 0

    def flaky(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return StepCallableResult(artifacts=[_artifact("solubility_curve", "wrong")])
        return _success_result(step, ledger)

    result = run_plan(plan, {"tool_a": flaky}, PlannerConfig(mode="off"))

    assert result.status == "succeeded"
    assert attempts == 2
    assert [record.attempt for record in result.step_records if record.status in {"failed", "succeeded"}] == [1, 2]
    assert result.repairs[0]["action"] == "retry_same_callable"


def test_recompile_request_records_not_implemented_when_attempts_exhaust():
    step = _step(
        "a",
        "tool_a",
        "separation_topk_sequences",
        retry_policy=RetryPolicy(max_attempts=1, allow_recompile=True),
    )
    plan = _plan([step])
    result = run_plan(
        plan,
        {"tool_a": lambda step, ledger: StepCallableResult(artifacts=[_artifact("solubility_curve", "wrong")])},
        PlannerConfig(mode="off"),
    )

    assert result.status == "failed"
    assert result.repairs[-1]["action"] == "recompile_not_implemented"


def test_multislice_plan_calls_only_slice_callable(tmp_path: Path):
    from strap.planning.compiler import compile_request

    compiled = compile_request(
        "For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year under scenario A, have the "
        "separation engineer propose the top 12 solvent candidates per polymer using the "
        "dynamic-programming planner with temperature recommendations. Then run cost-vs-circularity "
        "Pareto landscape optimization for five fixed feed compositions: 20/60/20, 34/33/33, "
        "60/20/20, 20/20/60, and 5/5/90. Require at least 1 STRAP wash and allow up to 2 washes. "
        "Save one PNG per composition and one combined comparison plot showing all feasible points.",
        created_at=FIXED_TIME,
    )
    assert compiled.plan is not None
    calls: list[str] = []
    plot_path = tmp_path / "slices.png"
    plot_path.write_text("png")

    def make_result(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        calls.append(step.allowed_tools[0])
        artifacts = [
            _artifact(artifact_contract.artifact_type, f"{step.step_id}_{contract_idx}_{artifact_idx}")
            for contract_idx, contract in enumerate(step.output_contracts)
            for artifact_idx, artifact_contract in enumerate(contract.artifact_contracts)
        ]
        # The optimization step has multiple artifact contracts in one contract.
        if step.step_id == "optimize_slices":
            artifacts = [
                _artifact("optimization_pareto_slices", "slices"),
                _artifact("optimization_pareto_front", "front"),
                _artifact("optimization_pareto_landscape", "landscape"),
                _artifact("sidecar_file", "sidecar"),
            ]
        return StepCallableResult(artifacts=artifacts)

    result = run_plan(
        compiled.plan,
        {
            "plan_multiple_separation_schemes": make_result,
            "build_handoff": make_result,
            "run_waste_management_pareto_slices": make_result,
            "plot_optimization_pareto_slices": lambda step, ledger: StepCallableResult(
                artifacts=[_artifact("optimization_pareto_slices_plot", "plot", paths=[str(plot_path)])]
            ),
        },
        PlannerConfig(mode="off"),
    )

    assert result.status == "succeeded"
    assert "run_waste_management_pareto_slices" in calls
    assert "run_waste_management_pareto" not in calls


def test_synthesis_step_runs_only_after_required_prior_contracts_pass():
    calls: list[str] = []
    step_a = _step("a", "tool_a", "optimization_pareto_slices")
    synth = _synthesis_step(
        depends_on=["a"],
        input_contracts=[InputContract(artifact_type="optimization_pareto_slices", source_step_id="a")],
    )
    plan = _plan([step_a, synth])

    def tool_a(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        calls.append(step.step_id)
        return _success_result(step, ledger)

    def synthesize(step: PlanStep, ledger: ExecutionLedger) -> StepCallableResult:
        calls.append(step.step_id)
        assert any(artifact.artifact_type == "optimization_pareto_slices" for artifact in ledger.artifacts)
        return StepCallableResult(data={"summary": "ok"})

    result = run_plan(plan, {"tool_a": tool_a, "synthesize": synthesize}, PlannerConfig(mode="off"))

    assert result.status == "succeeded"
    assert calls == ["a", "synthesize"]


def test_synthesis_step_does_not_run_when_dependency_contract_fails():
    calls: list[str] = []
    step_a = _step("a", "tool_a", "optimization_pareto_slices")
    synth = _synthesis_step(
        depends_on=["a"],
        input_contracts=[InputContract(artifact_type="optimization_pareto_slices", source_step_id="a")],
    )
    plan = _plan([step_a, synth])

    result = run_plan(
        plan,
        {
            "tool_a": lambda step, ledger: StepCallableResult(artifacts=[_artifact("solubility_curve", "wrong")]),
            "synthesize": lambda step, ledger: calls.append(step.step_id) or StepCallableResult(),
        },
        PlannerConfig(mode="off"),
    )

    assert result.status == "failed"
    assert calls == []
