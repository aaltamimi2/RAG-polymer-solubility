# v10 Typed Planning and Execution Harness Spec

Status: proposed for `v10-core`

Authoring context: current `v10-core` already contains structured handoffs, result extraction, direct fast paths, routing guards, verifier middleware, session artifacts, optimization payload sidecars, and query-bank examples in `docs/subagent_query_bank-v1.xlsx`. This spec defines the next architectural refactor: replace advisory/keyword-driven orchestration for complex tasks with an authoritative typed plan, enforced execution, and contract verification.

## 1. Problem Statement

The current DISSOLVE architecture has strong specialist tools, but the top-level orchestration is still too advisory:

- Routing often begins from keyword and prompt heuristics.
- Subagent prompts describe expected behavior, but do not create an enforceable execution contract.
- Handoffs are typed in some paths, but the orchestrator can still drift from the intended producer/consumer chain.
- Verification is partly prose-oriented and often happens after the wrong tool has already run.
- Stop conditions are based on message progress and structured results, not on explicit artifact/data contracts.
- Complex multi-step tasks depend on a good model following instructions rather than a harness that constrains, checks, repairs, and records execution.

The target behavior is closer to Claude Code / Codex-style planning and execution:

1. Compile the user request into a typed plan.
2. Validate and normalize that plan.
3. Execute only plan-approved steps/tools/subagents.
4. Verify each step output against an explicit contract.
5. Repair or retry failed steps within bounded policy.
6. Stop only when the plan contracts are satisfied or a typed failure is returned.

Keyword routing may remain as a weak signal into the compiler. Once a plan exists, it must govern execution.

## 2. Goals

- Add a first-class `RequestPlan` object for complex tasks.
- Make plan steps authoritative over subagent dispatch and tool selection.
- Introduce code-owned tool/subagent capability declarations.
- Add artifact/data contracts with deterministic validators.
- Persist the compiled plan, execution ledger, step outputs, repair attempts, and verification results.
- Make final synthesis read from verified contract outputs, not earlier prose.
- Use the query bank as an acceptance suite for plan compilation and execution validation.
- Preserve existing direct fast paths for simple deterministic queries.
- Preserve existing specialist tools and subagent prompts, but demote prompts from policy to implementation guidance.

## 3. Non-Goals

- Do not remove all keyword classifiers immediately. They become feature extractors and fallback routing for simple/direct tasks.
- Do not rewrite all tools in one PR.
- Do not require every historical tool result to satisfy the final schema in phase 1.
- Do not use the planner to invent scientific data. Plans choose tools and contracts; tools still produce domain data.
- Do not make an LLM-generated plan authoritative until it passes deterministic schema and capability validation.

## 4. Existing Assets to Reuse

- `src/strap/agent.py`: top-level agent assembly, middleware order, CLI loop.
- `src/strap/direct_fast_path.py`: deterministic simple-query bypass.
- `src/strap/orchestrator_runtime.py`: route decisions, artifact frames, run ledgers.
- `src/strap/session_state.py`: persisted structured context across turns.
- `src/strap/handoff_store.py` and `src/strap/handoffs.py`: append-only result/handoff store.
- `src/strap/handoff_adapters.py`: producer-to-consumer typed handoff adapters.
- `src/strap/result_extractor.py`: structured result capture.
- `src/strap/routing*.py`: current routing, progress, and guard utilities.
- `src/strap/guardrails.py` and `src/strap/guardrail_policy.py`: per-subagent tool-call mutation/blocking hooks.
- `src/strap/verifier.py`: final output verifier scaffold.
- `src/strap/tools/_helpers.py`: tool response helpers and output paths.
- `docs/subagent_query_bank-v1.xlsx`: acceptance suite seed.

## 5. Core Design

### 5.1 Execution Modes

The orchestrator chooses exactly one mode per user request:

- `direct_tool`: deterministic, no LLM planning, one tool, already supported by direct fast path.
- `single_agent`: one specialist subagent, no upstream handoff.
- `single_tool_or_specialist`: simple enough for direct execution, but may use a specialist if deterministic parsing is insufficient.
- `planned_workflow`: multi-step plan with explicit step contracts.
- `clarification_required`: missing required inputs and unsafe to assume.
- `unsupported`: typed refusal or typed infeasible response.

Every non-trivial request should compile to a `RequestPlan`, including `single_agent` and `single_tool_or_specialist`. A single-step request can still require strict artifact enforcement; for example, a request for a dynamic-programming state map must not call a solubility curve plotter just because both are visualization-related.

Only deterministic direct fast paths may bypass the planner. A direct fast path is allowed only when all required inputs, tool selection, and output format are unambiguous in code.

A request is non-trivial if any of these deterministic conditions are true:

- It requests a persisted artifact, file, plot, table, report, workbook, or case-study output.
- It requests any visualization whose artifact type matters, e.g. state map vs solubility curve vs Pareto landscape.
- It requires a handoff between agents, tools, or structured payloads.
- It names multiple subagents or asks one agent to pass results to another.
- It contains multiple entities where tool choice depends on entity type, e.g. multiple polymers, solvents, contaminants, scenarios, or compositions.
- It specifies forbidden outputs, provenance requirements, exact source usage, or “use only” constraints.
- It requests optimization, TEA/LCA, BioSTEAM simulation, safety assessment, citation-backed research, contaminant removal planning, or multi-step separation planning.
- It requests a sweep, batch, comparison, multi-slice calculation, Pareto frontier, Pareto landscape, or iterative workflow.
- It asks for final synthesis from prior structured results rather than a direct factual answer.

All other requests may still compile to a plan, but these conditions require planner involvement unless a direct fast path explicitly handles the full contract.

### 5.2 Request Lifecycle

1. `Context Assembly`
   - Load current user message.
   - Load compact session context.
   - Load relevant artifact frames and recent handoff summaries.
   - Load query-bank-derived examples only in tests or offline eval, not in normal execution.

2. `Complexity Gate`
   - Direct fast path first for unambiguous simple requests.
   - If request contains multi-step verbs, multiple subagents, handoff phrases, composition slices, plotting plus analysis, or “finally” chains, route to planning.

3. `Intent Compilation`
   - Deterministic extractors produce candidate entities and constraints.
   - A configured planner model proposes a `RequestPlan` through a provider-agnostic compiler interface.
   - The proposed plan is parsed into Pydantic models.
   - Deterministic validation normalizes aliases, fills safe defaults, rejects incompatible tool/intent combinations, and identifies missing inputs.

4. `Plan Authorization`
   - The capability registry checks whether each step can produce its required contract.
   - The orchestrator resolves required input edges from earlier steps, current session artifacts, or user inputs.
   - The plan is persisted before execution.

5. `Execution`
   - The executor runs as a deterministic state machine.
   - Once a plan is active, the model does not repeatedly decide what to do next.
   - The state machine authorizes the current step, calls the planned tool/subagent, verifies outputs, then advances, repairs, or stops.
   - A step may be a direct tool call or a subagent task.
   - Tool calls are guarded against the active plan step.
   - Outputs are captured as structured artifacts and handoffs.

6. `Step Verification`
   - Verify actual outputs against the step’s `OutputContract`.
   - If verification fails, run a bounded repair policy.
   - Do not advance to downstream steps until required outputs are valid.

7. `Final Synthesis`
   - The final answer is generated from verified plan outputs only.
   - It includes file paths, typed failures, unresolved caveats, and exact plan completion status.

8. `Persistence`
   - Save `RequestPlan`, `ExecutionLedger`, per-step verification results, artifacts, output paths, and repair attempts.

## 6. Typed Models

New package:

```text
src/strap/planning/
  __init__.py
  models.py
  compiler.py
  validators.py
  capability_registry.py
  executor.py
  middleware.py
  verification.py
  repair.py
  serialization.py
```

### 6.1 RequestPlan

```python
class RequestPlan(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: str
    created_at: str
    compiler_version: str
    capability_registry_version: str
    planner_model_id: str | None = None
    user_query: str
    mode: Literal[
        "direct_tool",
        "single_agent",
        "single_tool_or_specialist",
        "planned_workflow",
        "clarification_required",
        "unsupported",
    ]
    intent_family: Literal[
        "separation",
        "safety",
        "biosteam_tea_lca",
        "optimization",
        "visualization",
        "statistics_ml",
        "research",
        "contaminant_removal",
        "mixed_workflow",
    ]
    complexity: Literal["simple", "moderate", "complex"]
    assumptions: list[PlanAssumption] = []
    missing_inputs: list[MissingInput] = []
    global_constraints: dict[str, Any] = {}
    steps: list[PlanStep] = []
    final_response_contract: FinalResponseContract
    fallback_policy: Literal["clarify", "typed_failure", "best_effort_disclosed"] = "typed_failure"
```

Rules:

- `direct_tool` may have zero or one explicit step if handled by an existing deterministic fast path.
- `single_agent`, `single_tool_or_specialist`, and `planned_workflow` must have at least one step with an explicit output contract.
- `planned_workflow` should have at least two steps; if a single step has a non-trivial artifact contract, use `single_tool_or_specialist` or `single_agent` rather than bypassing enforcement.
- `clarification_required` must have `missing_inputs`.
- `unsupported` must have a typed reason.
- `steps[*].step_id` values must be unique.

### 6.2 PlanStep

```python
class PlanStep(BaseModel):
    step_id: str
    label: str
    role: Literal[
        "separation-engineer",
        "safety-analyst",
        "biosteam-analyst",
        "scholar-researcher",
        "patent-researcher",
        "rag-analyst",
        "visualization-specialist",
        "statistics-ml",
        "contaminant-removal-analyst",
        "optimization-engineer",
        "handoff_adapter",
        "direct_tool",
    ]
    execution_kind: Literal["tool", "subagent", "handoff_adapter", "synthesis"]
    allowed_tools: list[str]
    disallowed_tools: list[str] = []
    input_contracts: list[InputContract] = []
    output_contracts: list[OutputContract]
    depends_on: list[str] = []
    tool_args_template: dict[str, Any] = {}
    retry_policy: RetryPolicy = RetryPolicy()
    budget: StepBudget = StepBudget()
```

Rules:

- If `execution_kind == "tool"`, `allowed_tools` must contain exactly one tool unless the step is a deliberate tool-choice step.
- If `execution_kind == "subagent"`, `allowed_tools` means tools the subagent may call.
- `depends_on` must refer to earlier steps only.
- Every non-synthesis step must produce at least one `OutputContract`.

### 6.3 OutputContract

`OutputContract` is the step-level contract. It can require one or more concrete artifacts, structured data fields, and text/final-answer properties. `ArtifactContract` is nested inside `OutputContract` and describes files, payloads, plots, tables, or structured result objects.

```python
class OutputContract(BaseModel):
    contract_id: str
    required: bool = True
    artifact_contracts: list[ArtifactContract] = []
    data_requirements: list[DataRequirement] = []
    text_requirements: list[TextRequirement] = []
    forbidden_claims: list[str] = []
    validation_checks: list[str] = []
```

Rules:

- A non-synthesis `OutputContract` must include at least one `artifact_contract` or `data_requirement`.
- A synthesis `OutputContract` may include only `text_requirements`, but those text requirements must cite verified upstream artifacts by `artifact_id` or `source_step_id`.
- Final answer contracts must forbid substituting upstream prose for downstream structured outputs when optimization, TEA/LCA, safety, or visualization results are available.
- Examples in this spec may use shorthand where an item inside `output_contracts` looks like an `ArtifactContract`; implementation should serialize the expanded `OutputContract(artifact_contracts=[...])` form.

### 6.4 ArtifactContract

```python
class ArtifactContract(BaseModel):
    artifact_type: str
    required: bool = True
    count: CountConstraint = CountConstraint(min=1)
    entities: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    output_formats: list[Literal["json", "png", "csv", "xlsx", "markdown"]] = []
    path_policy: Literal["required", "optional", "forbidden"] = "optional"
    forbidden_artifact_types: list[str] = []
    validation_checks: list[str] = []
```

Examples of `artifact_type`:

- `solubility_curve`
- `solubility_table`
- `solvent_safety_card`
- `solvent_safety_comparison`
- `hsp_single_pair_summary`
- `hsp_red_heatmap`
- `separation_topk_sequences`
- `separation_dp_state_map`
- `separation_tree_plot`
- `optimization_point_result`
- `optimization_pareto_front`
- `optimization_pareto_landscape`
- `optimization_pareto_slices`
- `biosteam_tea_lca_result`
- `biosteam_tea_lca_plot`
- `research_citation_bundle`

### 6.5 Capability Declaration

Every tool and subagent gets code-owned capabilities:

```python
class CapabilitySpec(BaseModel):
    capability_id: str
    owner: str
    callable_name: str
    callable_kind: Literal["tool", "subagent", "adapter"]
    produces: list[str]
    consumes: list[str] = []
    required_inputs: list[str] = []
    optional_inputs: list[str] = []
    rejects: list[str] = []
    artifact_schema_versions: dict[str, str] = {}
    supports_batch: bool = False
    supports_multislice: bool = False
    deterministic: bool = False
```

Example:

```yaml
capability_id: optimization.pareto_slices
owner: optimization-engineer
callable_name: run_waste_management_pareto_slices
callable_kind: tool
produces:
  - optimization_pareto_slices
  - optimization_pareto_front
  - optimization_pareto_landscape
consumes:
  - optimization_stage_candidates
required_inputs:
  - feed_capacity_tpy
  - composition_slices_json
  - x_metric
  - y_metric
supports_multislice: true
```

Registry consistency is mandatory in CI:

- Every exported tool callable that can be reached by an agent must have a `CapabilitySpec`, even if it is marked `legacy_unplanned: true`.
- Every `CapabilitySpec.callable_name` must resolve to an actual exported tool, subagent, or adapter.
- Every artifact type referenced by a capability must exist in the artifact taxonomy.
- A capability may reject artifact types explicitly, but missing metadata must not silently imply compatibility.
- Query-bank P0 rows should fail compilation if they require a capability that is absent or stale.

### 6.6 ExecutionLedger

```python
class ExecutionLedger(BaseModel):
    plan_id: str
    run_id: str
    status: Literal["running", "succeeded", "failed", "partial"]
    started_at: str
    completed_at: str | None = None
    step_records: list[StepExecutionRecord]
    artifacts: list[ArtifactFrame]
    repairs: list[RepairAttempt]
    final_contract_status: ContractStatus
```

This ledger supersedes ad hoc progress text as the authoritative run record.

## 7. Artifact Metadata Standard

All tool results should eventually include:

```json
{
  "artifact_type": "optimization_pareto_landscape",
  "schema_version": "1.0",
  "entities": {
    "polymers": ["LDPE", "EVOH", "PET"],
    "feed_composition": {"PE": 0.2, "EVOH": 0.6, "PET": 0.2}
  },
  "inputs_used": {
    "feed": 8000,
    "scenario": "A",
    "n_points": 100
  },
  "output_paths": ["plots/optimization_pareto_circularity_...png"],
  "source_step_id": "optimize_pareto",
  "source_handoff_ids": ["h_abc123"],
  "validation_summary": {
    "status": "passed",
    "checks": ["frontier_points_present", "landscape_points_present"]
  }
}
```

Tool response helpers should support emitting this envelope without forcing every legacy tool to change immediately. Phase 1 can wrap legacy payloads into inferred artifact frames.

## 8. Plan Compiler

### 8.1 Inputs

- User query.
- Compact session context.
- Recent artifact frames.
- Recent structured results/handoffs.
- Capability registry.
- Optional application hints, e.g. current working directory or requested output directory.

### 8.2 Compiler Strategy

Use a two-pass compiler:

1. Deterministic pre-parser:
   - Extract polymers, solvents, temperatures, feed capacity, compositions, composition slices, top-k counts, metrics, required plots, and requested output folders.
   - Detect obvious direct fast path eligibility.
   - Detect multi-step structure from conjunctions, “then”, “finally”, “pass to”, “have X do Y”.

2. Model-assisted planner:
   - Use the configured planner model to propose `RequestPlan`.
   - Prompt with the capability registry and strict JSON schema.
   - Do not let model invent tool names; it must choose from registry IDs.
   - Parse into Pydantic.
   - Validate deterministically.

The model can propose. Code authorizes.

The planner model is configured by environment or app config, not hardcoded in the spec or runtime. The compiler interface should support swapping providers without changing execution semantics:

```text
DISSOLVE_PLANNER_MODEL=<provider/model-id>
```

### 8.3 Plan Normalization

Validators must:

- Normalize polymer aliases: `LDPE -> PE` for optimizer where appropriate, while retaining display aliases.
- Normalize solvent aliases: `GVL -> gamma-Valerolactone` where registry supports it.
- Normalize scenario strings.
- Convert `top N per polymer` into `top_k_per_polymer`.
- Convert composition expressions like `20/60/20` into explicit maps.
- Force `route_pool_mode` / shortlist semantics from user wording or plan defaults.
- Infer `plot_mode=landscape` when user requests “all feasible points” or “inner points”.
- For multi-slice requests, force `run_waste_management_pareto_slices` rather than repeated untracked one-off calls unless the user explicitly asks for stepwise manual runs.

### 8.4 Deterministic Executor State Machine

The executor must not be another model-driven routing loop. After compilation, execution is owned by code and proceeds through explicit states:

```text
COMPILED
  -> AUTHORIZING_STEP
  -> RUNNING_STEP
  -> VERIFYING_STEP
  -> STEP_SUCCEEDED -> AUTHORIZING_STEP(next)
  -> STEP_FAILED -> REPAIRING_STEP | PLAN_FAILED | PARTIAL_RESULT
  -> SYNTHESIZING
  -> COMPLETE
```

State transition rules:

- `AUTHORIZING_STEP`: select the next runnable step from `depends_on` and completed-step status; reject cycles and missing dependencies.
- `RUNNING_STEP`: call only the step’s planned tool/subagent/adapter; subagents receive a generated task from the step contract, not free-form orchestration advice.
- `VERIFYING_STEP`: validate artifacts, data fields, paths, provenance, and forbidden outputs before advancing.
- `REPAIRING_STEP`: run bounded repair from the step’s `RetryPolicy`; repair may change arguments within the same step contract but may not change the plan topology unless the repair policy explicitly permits a recompile.
- `PLAN_FAILED`: return typed failure with the ledger and failed contract.
- `PARTIAL_RESULT`: allowed only when the `RequestPlan.fallback_policy` permits partial results and required contracts are either satisfied or explicitly marked failed.
- `SYNTHESIZING`: generate final text from verified artifacts and ledger entries only.

This state machine is the core defense against LangGraph recursion failures: the planner decides once, code executes deterministically, and the model is only invoked inside bounded compile, subagent, repair, or synthesis slots.

## 9. Plan-Driven Tool Guard

New middleware: `PlanExecutionGuardMiddleware`.

Responsibilities:

- Read active `RequestPlan` and current `step_id`.
- Block any tool/subagent call not allowed by the active step.
- Validate tool args against the step’s input contract.
- Inject required handoff IDs or typed payloads when they are unambiguous.
- Reject malformed JSON strings where mappings are required.
- Emit a typed guard ToolMessage explaining the repair, not a vague prompt.

Existing optimization/separation guard logic should be migrated into this plan-aware guard over time.

The guard applies to every active plan step, regardless of request mode. It is not limited to `planned_workflow`; `single_agent` and `single_tool_or_specialist` can still have hard artifact contracts.

### 9.1 Tool Guard Decisions

Possible guard outcomes:

- `allow`
- `allow_with_arg_repair`
- `block_retry_same_step`
- `block_plan_failure`
- `clarification_required`

Example:

If active step requires `optimization_pareto_slices`, and model calls `run_waste_management_pareto`, block:

```text
Plan guard: active step `optimize_slices` requires artifact_type=optimization_pareto_slices.
Tool `run_waste_management_pareto` cannot satisfy this contract. Call
`run_waste_management_pareto_slices` with composition_slices_json from the plan.
```

## 10. Contract Verification

New module: `src/strap/planning/verification.py`.

Each `OutputContract` has deterministic validators:

- `required_fields_present`
- `artifact_type_matches`
- `schema_version_supported`
- `output_paths_exist`
- `expected_entities_match`
- `forbidden_artifact_absent`
- `n_points_min`
- `frontier_points_present`
- `landscape_points_present`
- `slice_count_matches`
- `source_handoff_consumed`
- `visualization_from_authoritative_payload`
- `no_upstream_prose_substitution`
- `temperature_recommendations_preserved`
- `residual_polymers_serialized`

The verifier should return structured status:

```json
{
  "status": "failed",
  "contract_id": "plot_pareto_landscape",
  "failed_checks": [
    {
      "check": "landscape_points_present",
      "message": "Expected all feasible landscape points; payload only contains frontier points."
    }
  ],
  "repair_hint": "Retry visualization with plot_mode='landscape' and source_handoff_id=h_..."
}
```

### 10.1 Verification Order

1. Structured contract verification.
2. Path/file existence verification.
3. Domain-specific validators.
4. Prose verifier only for final answer quality and visible caveats.

The prose verifier must not be the first or only gate for complex workflows.

## 11. Repair Policy

Each step has a bounded repair policy:

```python
class RetryPolicy(BaseModel):
    max_attempts: int = 2
    retry_on: list[str] = ["contract_failed", "tool_arg_invalid", "missing_artifact"]
    strategy: Literal["same_step_constrained", "alternate_capability", "typed_failure"] = "same_step_constrained"
```

Repair rules:

- If a tool call is incompatible with plan, retry same step with an explicit guard message.
- If output artifact type is wrong, retry same step with required tool and source handoff.
- If a subagent omits a structured result, retry once for structured result only.
- If a solver/tool returns typed infeasible, do not blindly retry unless policy allows relaxing constraints.
- If repeated failures are identical, stop with a visible typed caveat and attach failed checks.

## 12. Final Synthesis Contract

Final synthesis must be generated from `ExecutionLedger` and verified artifacts.

Required final fields for complex plans:

- `plan_id`
- completed step summary
- output paths
- key numeric results
- typed failures or partials
- caveats from contract verifier
- no invented plots or route descriptions

For optimization results:

- Read `points`, `route_reports`, `frontier_summary`, `candidate_telemetry`, and sidecar payloads.
- Do not describe upstream separation prose as optimized routes.
- If residual polymer metadata exists, report recovered and residual routing from optimizer payload.

For visualization:

- Report exact generated file paths.
- State which upstream payload/handoff was used.

## 13. Query Bank as Acceptance Suite

The workbook `docs/subagent_query_bank-v1.xlsx` has a common schema:

```text
query
status
priority
focus
expected_route_or_subagents
expected_tools_or_handoffs
required_inputs
expected_outputs_or_artifacts
validation_checks
last_run_or_artifact_path
notes
```

Add a test loader:

```text
tests/test_planning_query_bank.py
```

Initial acceptance rule:

- Load rows with `status=validated`.
- Compile each query into a `RequestPlan`.
- Assert expected subagents/tools/artifacts appear in plan.
- Do not run expensive tools by default.
- Mark expensive full execution cases as integration tests.

### 13.1 P0 Safety Examples

Example query:

```text
Show a safety card for THF at 60 C.
```

Expected plan:

- mode: `direct_tool` or `single_tool_or_specialist`
- intent_family: `safety`
- tool: `get_solvent_safety_card`
- contract: `solvent_safety_card`
- entities: `solvent=THF`, `operating_temp_c=60`
- checks: peroxide class, near-boiling/flash flags if applicable

### 13.2 P0 HSP Examples

Example query:

```text
Use HSP/RED to check whether PC is compatible with dichloromethane and generate the HSP radar and RED gauge visuals.
```

Expected plan:

- role: `statistics-ml`
- tool: `predict_solubility_ml(generate_visualizations=True)`
- contract: `hsp_single_pair_summary`
- artifacts: radar/RED PNG
- forbidden: temperature-dependent claims unless explicitly modeled

### 13.3 P0 Optimization Point Example

Example query:

```text
Optimize waste management for a mixed plastic feedstock of 8000 tonnes/year composed of 60% PE and 40% EVOH under scenario A. Restrict the candidate solvents to Toluene or Heptane for PE and Pyridazine or Ethylene Glycol for EVOH. Maximize profit, require at least 1 STRAP wash step and allow up to 2 wash steps. Report the selected solvents, total profit, total cost, and circularity.
```

Expected plan:

- mode: `single_agent` or `planned_workflow`
- role: `optimization-engineer`
- tool: `run_waste_management_optimization`
- contract: `optimization_point_result`
- checks:
  - feed composition explicit
  - solvent filters explicit
  - objective `max_profit`
  - min/max washes enforced
  - selected solvents from shortlist

### 13.4 P0 Routed Optimization Example

Example query:

```text
For a mixed plastic feedstock of 8000 tonnes/year composed of 60% PE and 40% EVOH under scenario A, have the separation engineer propose the top 3 solvent candidates per polymer using the dynamic-programming planner and include separation plots for the shortlisted PE and EVOH solvents. Then pass exactly those shortlisted candidates to the optimization engineer to maximize profit, requiring at least 1 STRAP wash step and allowing up to 2 wash steps. Finally, have the visualization specialist create a separation tree plot and an optimization figure summarizing the routed result. Report the selected solvents, total profit, total cost, circularity, and save all plots.
```

Expected steps:

1. `separation_topk`
   - role: `separation-engineer`
   - output: `separation_topk_sequences`, `optimization_stage_candidates`, separation plots

2. `build_optimization_handoff`
   - role: `handoff_adapter`
   - input: separation result
   - output: typed optimization-stage candidates

3. `optimize_point`
   - role: `optimization-engineer`
   - tool: `run_waste_management_optimization`
   - consumes: typed candidates
   - output: `optimization_point_result`

4. `plot_results`
   - role: `visualization-specialist`
   - tools: `create_separation_tree_plot`, `plot_optimization_point_result`
   - output: plot artifacts

Final answer must cite verified optimizer output and plot paths.

### 13.5 P0 Multi-Slice Optimization Example

Example query:

```text
For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year under scenario A, have the separation engineer propose the top 12 solvent candidates per polymer using the dynamic-programming planner with temperature recommendations. Then run cost-vs-circularity Pareto landscape optimization for five fixed feed compositions: 20/60/20, 34/33/33, 60/20/20, 20/20/60, and 5/5/90. Require at least 1 STRAP wash and allow up to 2 washes. Save one PNG per composition and one combined comparison plot showing all feasible points colored by composition, with each composition's frontier highlighted.
```

Expected steps:

1. `separation_candidates`
   - output: top-12 typed stage candidates with temperatures

2. `optimize_slices`
   - tool: `run_waste_management_pareto_slices`
   - output: `optimization_pareto_slices`
   - validator: 5 requested slices, all solved or typed failures recorded

3. `plot_slices`
   - tool: `plot_optimization_pareto_slices`
   - output: one combined PNG plus per-slice PNGs

4. `synthesize`
   - final answer from `pareto_slices_payload_path`

This query is the first full end-to-end P0 integration target.

## 14. Planner Examples

### 14.1 Safety Card Plan

```json
{
  "mode": "direct_tool",
  "intent_family": "safety",
  "steps": [
    {
      "step_id": "safety_card",
      "role": "direct_tool",
      "execution_kind": "tool",
      "allowed_tools": ["get_solvent_safety_card"],
      "tool_args_template": {
        "solvent_name": "Tetrahydrofuran",
        "operating_temp_c": 60
      },
      "output_contracts": [
        {
          "artifact_type": "solvent_safety_card",
          "entities": {"solvent": "Tetrahydrofuran", "operating_temp_c": 60},
          "validation_checks": ["hazard_flags_present", "peroxide_class_present"]
        }
      ]
    }
  ]
}
```

### 14.2 Routed Pareto Landscape Plan

```json
{
  "mode": "planned_workflow",
  "intent_family": "mixed_workflow",
  "steps": [
    {
      "step_id": "separation_candidates",
      "role": "separation-engineer",
      "execution_kind": "subagent",
      "allowed_tools": ["plan_separation_sequences"],
      "output_contracts": [
        {"artifact_type": "separation_topk_sequences", "count": {"min": 1}},
        {"artifact_type": "optimization_stage_candidates", "count": {"min": 1}}
      ]
    },
    {
      "step_id": "optimization_pareto",
      "role": "optimization-engineer",
      "execution_kind": "subagent",
      "depends_on": ["separation_candidates"],
      "allowed_tools": ["run_waste_management_pareto"],
      "input_contracts": [
        {"artifact_type": "optimization_stage_candidates", "source_step_id": "separation_candidates"}
      ],
      "output_contracts": [
        {
          "artifact_type": "optimization_pareto_landscape",
          "validation_checks": ["frontier_points_present", "landscape_points_present", "source_handoff_consumed"]
        }
      ]
    },
    {
      "step_id": "plot_pareto",
      "role": "visualization-specialist",
      "execution_kind": "subagent",
      "depends_on": ["optimization_pareto"],
      "allowed_tools": ["plot_optimization_pareto_front"],
      "output_contracts": [
        {
          "artifact_type": "optimization_pareto_landscape_plot",
          "path_policy": "required",
          "output_formats": ["png"]
        }
      ]
    }
  ]
}
```

## 15. UI and Progress Requirements

The user should see progress for long planned workflows:

```text
Compiled plan plan_abc123:
1. Separation candidates for LDPE/EVOH/PET
2. Pareto slices for 5 compositions
3. Combined and per-slice plots

Running step 1/3: separation_candidates...
Step 1 complete: 36 candidate pairs; 4 separation plots saved.
Running step 2/3: optimize_slices...
Slice 1/5 complete: 8 frontier points; payload saved at ...
...
```

Implementation:

- Extend `routing_progress.py` or add `planning/progress.py`.
- Progress messages derive from `ExecutionLedger`, not from free-form model text.
- CLI and future UI can consume the same ledger events.

## 16. Testing Strategy

### 16.1 Unit Tests

Add tests for:

- `RequestPlan` schema validation.
- deterministic entity extraction.
- plan normalization.
- capability registry lookup.
- tool guard allow/block/repair outcomes.
- contract verifier pass/fail cases.
- final synthesis source restrictions.

### 16.2 Query Bank Compilation Tests

```text
tests/test_planning_query_bank.py
```

Test rows where `status=validated`:

- P0 rows must compile exactly.
- P1 rows must compile with accepted alternative plan if contracts match.
- Empty rows are ignored.

Assertions:

- Expected subagents are present.
- Expected tools are present.
- Required inputs are represented.
- Expected artifact types are represented.
- Forbidden tools/artifacts are absent.

### 16.3 Integration Tests

Mark expensive tests:

```text
@pytest.mark.integration
@pytest.mark.expensive
```

Initial full execution targets:

- Safety direct card.
- HSP single-pair visual.
- Direct optimization point.
- Routed PE/EVOH point optimization.
- Routed LDPE/EVOH/PET Pareto landscape.
- Multi-slice LDPE/EVOH/PET Pareto.

### 16.4 Regression Tests From Known Failures

Add targeted tests for:

- Optimization final synthesis cannot use upstream separation prose as optimized route output.
- Visualization request for DP state map cannot call solubility plot tool.
- Multi-slice request cannot call single-slice Pareto tool only.
- `stage_candidates_json` must be object, not JSON plus prose.
- Plotting must use `source_handoff_id` or authoritative payload path.
- Pareto legend must include residual polymer routing when present.

## 17. Implementation Phases

### Phase 0: Baseline Lock

Deliverables:

- Keep `v10-core` branch green.
- Add this spec.
- Add query-bank loader utility for tests.
- Add no-op `RequestPlan` models without changing runtime.

Definition of done:

- Existing tests pass.
- Query bank can be read in tests.
- No runtime behavior change.

### Phase 1: Plan Models and Capability Registry

Deliverables:

- `src/strap/planning/models.py`
- `src/strap/planning/capability_registry.py`
- static capability specs for top tools:
  - safety card tools
  - HSP tools
  - separation planning tools
  - BioSTEAM TEA/LCA tools
  - optimization tools
  - visualization tools
- tests for registry consistency.

Definition of done:

- Every tool in P0 query bank rows has a capability declaration.
- Capability registry consistency tests compare declarations against actual exported tools.
- CI fails if a planned/exported tool lacks a capability or a capability points at a missing callable.
- Registry can answer “which callable can produce artifact X from input Y?”

### Phase 2: Plan Compiler

Deliverables:

- deterministic extractor.
- model-backed compiler abstraction.
- Pydantic validation and normalization.
- query-bank compile tests.

Definition of done:

- All validated P0 query-bank rows compile into expected plans.
- No runtime execution changes yet.

### Phase 3: Plan Persistence and Ledger

Deliverables:

- persist plan and execution ledger in session state and handoff scope.
- expose `get_current_plan`, `get_execution_ledger` debug tools.
- write plan/ledger JSON to case-study folders when requested.

Definition of done:

- Every planned workflow has an auditable plan and ledger.

### Phase 4: Plan Guard Middleware

Deliverables:

- `PlanExecutionGuardMiddleware`.
- active step state.
- allow/block/repair tool calls based on plan.
- subagent task descriptions generated from plan step, not raw advisory text.

Definition of done:

- A test proves wrong visualization tool is blocked.
- A test proves multi-slice query cannot use single-slice tool.
- A test proves optimization stage candidates are injected from handoff.
- Guard applies to non-trivial `single_tool_or_specialist` requests as well as `planned_workflow`.

### Phase 5: Contract Verifier

Deliverables:

- structured artifact contract validators.
- output path existence validation.
- source handoff validation.
- verifier status added to ledger.

Definition of done:

- Step cannot advance on wrong artifact type.
- Final synthesis cannot happen with unsatisfied required contracts unless response is typed partial/failure.

### Phase 6: Selected Production Runtime Bridge

Deliverables:

- opt-in production bridge for selected workflows only.
- `maybe_run_typed_runtime(query, context, legacy_runner)` integration point.
- `TypedRuntimeMiddleware` inserted before legacy advisory routing.
- production-only wrappers for:
  - DP state-map visualization.
  - separation -> optimization handoff.
  - routed Pareto optimization -> Pareto visualization.
  - routed multi-slice Pareto optimization -> slice visualization.
- selected typed failures return structured user-facing diagnostics instead of silently falling back.
- successful selected typed runs synthesize final text from verified ledger artifacts, not upstream prose.
- user output directories and filename hints propagate into selected visualization wrappers.
- unselected/off/shadow requests keep legacy behavior.

Definition of done:

- selected DP state-map requests run through typed runtime and cannot call solubility plotters.
- selected routed optimization requests run through typed runtime or fail with a diagnostic bundle.
- final selected optimization response cites optimizer structured artifacts.
- selected runtime attempts persist request, compile result, plan, ledger, artifacts, and manifest.
- existing direct fast path remains available.

### Phase 7: Expand Selected Cutover and Demote Legacy Routing for Complex Queries

This phase starts from the Phase 6 selected-runtime bridge. It does not assume
full-domain planned runtime yet.

Deliverables:

- 7A: define deterministic "complex request" gate using `is_non_trivial_request` plus compile-time facts.
- 7A: add the progress/status data model early, then enrich it as additional selected workflows are added.
- 7A: complex/non-trivial requests call the typed compiler before legacy routing.
- 7A: compile-first means "inspect and decide"; execution still requires:
  - `DISSOLVE_TYPED_PLANNER=enforce_selected`.
  - selected artifact or selected workflow match.
  - valid compiled plan.
  - registered production wrappers for every step.
- legacy keyword routing stays only for:
  - simple/direct fast paths.
  - unselected typed workflows.
  - compatibility fallback when mode is `off` or `shadow`.
- 7B: expand selected coverage to safety cards/comparisons and HSP/RED visualization.
- 7C: expand selected coverage to BioSTEAM TEA/LCA with fake-tool tests first; real subprocess smoke tests are optional/slow.
- 7D: expand selected coverage to separation tree and selectivity heatmap visualizations.
- the default `enforce_selected` artifact set includes the 7B/7C/7D artifacts once their production wrappers and fail-closed tests exist.
- 7E: expose ledger-backed progress/status summaries for selected workflows:
  - current step.
  - completed steps.
  - failed step, if any.
  - failed checks.
  - produced artifact paths.
  - diagnostic bundle path.
- add artifact-specific production wrappers for each newly selected workflow.
- wrappers must fail closed when required evidence is missing; generic legacy wrappers must not mint selected production artifacts.
- replace prompt-only route advice with typed plan authorization for selected complex workflows.
- keep `enforce` parsed but reserved until broader planned-runtime coverage is proven.

Definition of done:

- selected complex query behavior is plan-driven end to end.
- prompt-only route advice no longer controls selected workflow execution.
- selected workflows cannot execute tools outside declared capabilities.
- selected final answers are generated from or validated against ledger artifacts.
- unselected legacy behavior remains behavior-preserving.
- selected typed runtime failures include phase, step id when available, failed checks, and diagnostic bundle path.
- every new selected production wrapper has tests for:
  - success with real structured output shape.
  - missing required evidence fails closed.
  - wrong artifact/tool output fails verification.
  - output paths are copied into the diagnostic bundle when applicable.

### Phase 8: Enforce-Readiness and General Planned Runtime

Phase 8 starts with hardening. The selected runtime now owns the first
production workflows, but selected steps still need deeper scientific and
operational validation before broad `enforce` is safe.

#### Phase 8A: Typed-Runtime Hardening and Enforce Readiness

Implementation sub-spec:

- `architecture/phase8a_query_bank_multiturn_fix_spec.md` covers the first
  multi-turn query-bank smoke failure class: domain-general typed artifact
  follow-up handling, direct-fast-path domain conflict guards, ledger-backed
  summary follow-ups, and a reusable query-bank chat harness. The resolver
  applies to all selected typed artifact families, not only the HSP smoke case:
  HSP/RED, safety, BioSTEAM TEA/LCA, separation visualizations,
  optimization/Pareto, sidecar files, and future selected workflow artifacts.

Deliverables:

- Add domain validators for selected artifacts beyond type/path checks:
  - HSP artifacts require `analysis_type`, non-empty RED results, required RED/result fields, and real plot paths when a heatmap is requested.
  - Safety artifacts require solvent identity plus hazard, flash-point, boiling-point, and operating-temperature evidence when available.
  - BioSTEAM artifacts require finite numeric MSP, TCI, AOC, GWP, energy case, solvent, and target polymer.
  - Separation visualization artifacts require exact visualization type evidence, not just any PNG path.
  - Pareto artifacts require `analysis_type=pareto_front` or `pareto_slices`, non-empty points/slices, metric labels, feasible/infeasible status, and payload path when expected.
- Add real selected-runtime smoke tests with requested save paths and diagnostic bundles:
  - one HSP RED heatmap query.
  - one safety comparison query.
  - one BioSTEAM TEA/LCA plot query.
  - one separation tree or selectivity heatmap query.
  - one direct Pareto query.
- Tighten selected final-answer contracts:
  - selected responses cite ledger artifact types and produced paths.
  - optimization and BioSTEAM summaries use only verified structured payload values.
  - selected failure responses name failed validator, failed step, failed checks, and diagnostic bundle path.
- Add simple selected repair policy:
  - missing output path: retry once with normalized `output_dir`.
  - wrong plot artifact: fail closed; do not fall back to prose.
  - missing structured field: fail closed; do not infer metrics.
- Add an enforce-readiness report:
  - run query-bank P0/P1 rows in `shadow` and `enforce_selected`.
  - report compile success, selected execution success, legacy fallback, and typed failure reasons.
  - do not enable broad `enforce` from this phase.

Definition of done:

- Selected artifacts cannot pass on type/path evidence alone when domain-required fields are absent.
- Real selected smoke tests pass or produce typed diagnostic failures with bundle paths.
- Final selected answers are traceable to verified ledger artifacts.
- Readiness report identifies what would fail before broader enforce.
- `DISSOLVE_TYPED_PLANNER=enforce` remains parsed but not globally enabled.

Out of scope:

- No global `DISSOLVE_TYPED_PLANNER=enforce` cutover.
- No model replanning.
- No broad LangGraph replacement.
- No generic wrappers minting production artifacts.

#### Phase 8B: Broader Planned Runtime Coverage

Implementation sub-spec:

- `architecture/phase8b_typed_context_reuse_and_status_followups_spec.md`
  covers the optimization query-bank smoke failure class: ledger-backed
  status/provenance follow-ups, conservative typed-runtime context hydration
  for selected `same`/`again`/`rerun` requests, requested save-path propagation
  for hydrated reruns, and de-duplicated user-facing artifact paths.

Deliverables:

- First harden selected optimization multi-turn behavior:
  - status/provenance follow-ups answer from plan and ledger metadata.
  - selected rerun requests reuse prior typed-runtime context instead of
    falling to legacy prose.
  - hydrated reruns remain selected and execute through typed wrappers.
  - shared payload paths remain distinct artifact frames but are de-duplicated
    in progress and final user-facing output.
- Extend planned execution to additional domains after 8A passes:
  - research/RAG citation bundles.
  - contaminant-removal workflows.
  - multi-domain workflows that compose selected artifacts.
- Add domain-specific validators for the newly selected domains.
- Keep direct fast paths and unselected legacy fallback available.

Definition of done:

- The row-4 optimization multi-turn smoke has typed-runtime or
  typed-runtime-followup origins for all selected turns, including the
  `same Pareto landscape again with 8 points` rerun.
- P0 integration targets pass through planned runtime where selected.
- Existing direct fast path remains available.
- Complex multi-domain query behavior is plan-driven for selected workflows.

#### Phase 8C: Full Enforce Candidate

Deliverables:

- Evaluate enabling broader `enforce` only after readiness data shows selected workflows are stable.
- Add conservative fallback/partial-result semantics for unsupported domains.
- Document operational rollback knobs.

Definition of done:

- Broad enforce can be enabled in a controlled environment without losing diagnostics or legacy rollback.

## 18. File-Level Change Plan

Add:

```text
src/strap/planning/__init__.py
src/strap/planning/models.py
src/strap/planning/compiler.py
src/strap/planning/validators.py
src/strap/planning/capability_registry.py
src/strap/planning/executor.py
src/strap/planning/middleware.py
src/strap/planning/verification.py
src/strap/planning/repair.py
src/strap/planning/query_bank.py
tests/test_planning_models.py
tests/test_capability_registry.py
tests/test_plan_compiler.py
tests/test_plan_executor_state_machine.py
tests/test_plan_guard.py
tests/test_contract_verifier.py
tests/test_planning_query_bank.py
```

Modify:

```text
src/strap/agent.py
src/strap/orchestrator_runtime.py
src/strap/session_state.py
src/strap/result_extractor.py
src/strap/handoff_store.py
src/strap/routing.py
src/strap/routing_progress.py
src/strap/guardrails.py
src/strap/verifier.py
src/strap/tools/_helpers.py
src/strap/config/subagents/*.yaml
```

Tool modules should only be touched when adding artifact metadata, not during initial planner scaffolding unless necessary.

## 19. Compatibility and Migration

### 19.1 Backward Compatibility

- Existing tools continue returning current JSON envelopes.
- Artifact metadata can be inferred for legacy results.
- Existing subagent prompts remain but get shorter over time.
- Existing direct fast paths remain before planner.

### 19.2 Rollout Flag

Add env/config flag:

```text
DISSOLVE_TYPED_PLANNER=off|shadow|enforce_selected|enforce
```

Modes:

- `off`: current behavior.
- `shadow`: compile and log plan, but execute legacy routing.
- `enforce_selected`: enforce plan guard and contract verification only for selected artifact types or workflows.
- `enforce`: use plan guard and contract verifier.

Initial default should be `shadow` for development, then `enforce_selected` for the first high-value failures, then `enforce` after P0 tests pass.

`enforce_selected` must be configurable, not hardcoded:

```text
DISSOLVE_TYPED_PLANNER_ENFORCE_ARTIFACTS=separation_dp_state_map,optimization_pareto_slices,optimization_pareto_landscape
DISSOLVE_TYPED_PLANNER_ENFORCE_WORKFLOWS=routed_optimization,visualization_from_payload,final_synthesis_structured_only
```

Selection rules:

- If either list is empty, no selected enforcement is enabled for that dimension.
- Artifact matching uses `OutputContract.artifact_contracts[*].artifact_type`.
- Workflow matching uses stable workflow IDs assigned by the compiler, not free-text labels.
- Unknown artifact or workflow names in these env vars should fail startup in strict/dev mode and warn in production mode.
- Tests should configure these env vars explicitly instead of relying on defaults.

### 19.3 Shadow Evaluation

During `shadow`, compare:

- legacy selected route/tools
- compiled plan route/tools
- final artifacts
- contract satisfaction

Store mismatches for audit.

### 19.4 Selected Enforcement Targets

The first `enforce_selected` targets should be narrow and tied to known failures:

- `separation_dp_state_map` requests cannot call solubility curve/table plotters.
- `optimization_pareto_slices` requests cannot call single-slice Pareto tools unless wrapped by the slice executor.
- Visualization steps must consume authoritative payloads or handoff IDs rather than reconstructing from prose.
- Final synthesis for optimization must anchor on `route_reports`, `points`, `frontier_points`, `landscape_points`, and typed infeasibility records, not upstream separation prose.

## 20. Risks and Mitigations

Risk: Planner model emits plausible but invalid plan.

- Mitigation: Pydantic validation, capability registry, deterministic normalization, fail closed.

Risk: Over-constraining blocks legitimate exploratory tasks.

- Mitigation: use `enforce_selected` first for known high-value artifact contracts; keep deterministic direct fast paths outside the planner.

Risk: Tool capability registry becomes stale.

- Mitigation: registry consistency tests against actual exported tools.

Risk: Large plans increase latency.

- Mitigation: direct fast path first; plan compiler cached per user query/session; progress events visible.

Risk: Complex scientific workflows need partial success.

- Mitigation: typed partial results with per-contract status, not silent success.

Risk: Generated plans are hard to debug.

- Mitigation: persist plan JSON and execution ledger, expose debug tools, include plan summary in case studies.

## 21. Acceptance Criteria

Minimum for first enforceable release:

- `DISSOLVE_TYPED_PLANNER=shadow` compiles all validated P0 query-bank rows.
- P0 safety direct tool remains deterministic.
- P0 HSP visual plan compiles and verifies output artifact.
- P0 direct optimization point plan compiles and executes.
- P0 routed PE/EVOH point optimization compiles and executes with verified handoff.
- P0 LDPE/EVOH/PET Pareto landscape compiles and executes with verified plot.
- P0 multi-slice Pareto compiles to `run_waste_management_pareto_slices`, not repeated untracked single-slice calls.
- Wrong-tool calls for selected P0 artifact contracts are blocked in `enforce_selected` mode.
- Wrong-tool calls are blocked broadly in `enforce` mode.
- Final synthesis reads verified structured outputs only.
- Full existing test suite remains green.

## 22. Implemented PR Sequence and Next PRs

The initial roadmap has been split into smaller reviewable PRs. The current
sequence is:

### PR 1: Planning Data Model and Registry

Scope:

- Add Pydantic models.
- Add capability registry for query-bank P0 tools.
- Add query-bank loader.
- Add registry consistency tests against actual exported tools, subagents, and adapters.
- Add invalid-plan rejection tests for cycles, missing `OutputContract`, unknown tools, unknown artifact types, role/tool mismatch, stale capability names, and missing provenance fields.
- Add compile-only tests with hand-authored plans or deterministic stub.

No runtime behavior change.

### PR 2: Compiler and Shadow Diagnostics

Scope:

- Add compiler interface and deterministic P0 compiler paths.
- Add provider-agnostic planner backend behind abstraction.
- Add validation/normalization and compile diagnostics.
- Keep runtime behavior legacy.

### PR 3: Selected Plan Guard API

Scope:

- Add pure guard API for selected tool-call authorization.
- Block wrong visualization artifact/tool pairings.
- Block single Pareto tool for multi-slice requests.
- Add conservative final-synthesis source validator.

### PR 4: Deterministic Executor

Scope:

- Add deterministic executor with injected callables only.
- Add execution ledger, step records, artifact frames, retries, and contract verification.
- No live tool/model/LangGraph integration.

### PR 5: Runtime Bridge and Persistence

Scope:

- Add opt-in runtime bridge around compiler + executor.
- Add diagnostic persistence bundle.
- Keep production wrapper registry empty unless explicit wrappers are supplied.
- Keep normal orchestration unchanged.

### PR 6: Selected Production Runtime Bridge

Scope:

- Add production wrappers for selected workflows.
- Add `TypedRuntimeMiddleware` before legacy advisory routing.
- Generate selected success/failure responses from typed ledger artifacts.
- Persist diagnostics for selected compile/execution failures.

### PR 7: Expand Selected Cutover and Demote Legacy Routing for Complex Queries

Scope:

- Add compiler-first gating for complex/non-trivial requests without making complexity itself blocking.
- Route only selected/valid/wrapped workflows through typed runtime; preserve legacy fallback for unselected cases.
- Add progress/status summaries from typed ledgers.
- Add artifact-specific production wrappers for safety, HSP, BioSTEAM TEA/LCA, and separation visualizations.
- Keep legacy keyword routing as compatibility fallback for simple/unselected cases.

### PR 8: Enforce-Readiness and General Planned Runtime

Scope:

- PR 8A: typed-runtime hardening and enforce-readiness:
  - domain validators for selected artifacts.
  - real selected-runtime smoke tests.
  - stricter selected final-answer contracts.
  - simple selected repair policy.
  - query-bank readiness report for P0/P1 rows.
- PR 8B: broader planned runtime coverage for research/RAG, contaminant removal, and selected multi-domain workflows.
- PR 8C: full-enforce candidate only after readiness data shows selected workflows are stable.

## 23. Key Architectural Rule

The plan is the source of truth.

Prompts, keyword classifiers, and subagent descriptions may help produce or execute the plan, but they do not authorize tool calls, downstream handoffs, final claims, or stop conditions. Authorization comes from typed plan contracts plus capability validation. Execution success means contract satisfaction, not plausible prose.
