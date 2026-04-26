# Phase 8B Typed Context Reuse and Status Follow-Ups Spec

Status: proposed next implementation slice

Source smoke run:

- Seed row: `docs/subagent_query_bank-v1.xlsx`, `10 optimization-engineer`, row `4`
- Mode: `DISSOLVE_TYPED_PLANNER=enforce_selected`
- Transcript: `architecture/test_results/query_bank_phase8a_optimization_smoke/transcript.json`
- Summary: `architecture/test_results/query_bank_phase8a_optimization_smoke/summary.md`

## Problem

Phase 8A fixed artifact path and summary follow-ups for prior typed-runtime outputs. The optimization smoke showed that selected workflows still lose typed control in two important multi-turn cases:

1. A status/provenance follow-up was answered as a path lookup.
   - User: "What typed-runtime steps completed, and did the run use the routed Pareto plan rather than a direct solubility or safety path?"
   - Actual: `typed_runtime_followup`, but response listed artifact paths only.
   - Expected: answer from the prior plan and ledger: completed steps, workflow id, callables/tools used, and explicit statement that the run was `routed_optimization`, not direct solubility/safety.

2. A selected "same/again" rerun escaped typed runtime.
   - User: `Generate the same Pareto landscape again with 8 points and save it to ".../rerun".`
   - Actual: legacy `routing_single_specialist_prose`, solver ran, payload was created, but no typed runtime ledger, no selected verification, and no requested plot in the requested output directory.
   - Expected: hydrate the new selected request from the prior typed runtime plan/ledger, compile a selected typed plan, run through typed wrappers, produce a verified plot, and persist diagnostics.

3. Shared payload paths were duplicated in progress output.
   - `optimization_pareto_front` and `optimization_pareto_landscape` legitimately pointed to the same JSON payload.
   - Actual: `produced_artifact_paths` included the same payload path twice.
   - Expected: preserve both artifact frames, but de-duplicate user-facing path/progress lists.

## Goals

- Add a status/provenance follow-up mode backed by prior typed-runtime plan and ledger metadata.
- Add typed context reuse for new selected follow-up requests that explicitly ask to rerun/regenerate/recompute a prior typed result.
- Answer interpretive follow-ups such as "why", "explain", and "what were the frontier points" from prior typed optimizer artifacts without launching new selected workflows.
- Keep selected reruns inside typed runtime when wrappers exist; do not silently fall back to legacy prose for selected workflows.
- Honor new user overrides, especially `n_points`, metrics, and requested save/output directory.
- De-duplicate user-facing path lists while preserving distinct artifact contracts.
- Keep legacy behavior unchanged for unselected, off, and shadow paths; `off` and `shadow` must not short-circuit through typed follow-up handlers.

## Non-Goals

- Do not enable broad `DISSOLVE_TYPED_PLANNER=enforce`.
- Do not add model replanning.
- Do not infer missing context from arbitrary prose-only legacy turns.
- Do not allow generic wrappers to mint production artifacts.
- Do not make all follow-ups stateful; only hydrate from verified typed-runtime metadata.

## Target Behavior

### Status / Provenance Follow-Up

Given a prior typed runtime message with:

- `additional_kwargs["strap_origin"] == "typed_runtime"`
- `additional_kwargs["strap_run_ledger"]`
- `additional_kwargs["strap_manifest"]`
- optional persisted `plan.json` and `ledger.json`

Then follow-ups like:

- "What typed-runtime steps completed?"
- "Which tools ran?"
- "Was this routed Pareto or a direct optimization?"
- "Did this use direct solubility or safety?"
- "Was that typed runtime or legacy?"

must answer from the prior plan and ledger, not from path-only artifact formatting.

Expected answer fields:

- runtime status
- plan id
- workflow id
- completed step ids
- callable/tool names per step
- failed step/checks when present
- produced artifact types
- diagnostic bundle path
- direct answer to provenance question when determinable, such as:
  - "Workflow was `routed_optimization`."
  - "Completed tools were `plan_multiple_separation_schemes`, `build_handoff`, `run_waste_management_pareto`, and `plot_optimization_pareto_front`."
  - "No direct solubility or safety fast-path artifact was involved."

### Typed Context Reuse For Selected Reruns

Given a prior selected typed runtime optimization run and a follow-up like:

```text
Generate the same Pareto landscape again with 8 points and save it to ".../rerun".
```

the runtime must treat this as a new selected artifact request, not a prior-artifact summary and not legacy routing.

The compiler/runtime must hydrate missing facts from the most recent compatible typed run:

- feed capacity
- feed composition
- scenario
- objective/metrics
- min/max wash constraints
- candidate solvents or prior stage candidates
- prior workflow id and step structure
- prior requested artifact type

The current user turn overrides prior context:

- `n_points=8`
- requested save/output directory
- explicitly changed metrics/objective
- explicitly changed polymers/feed composition/scenario

Hydration is only valid when the current turn has explicit action intent such as:

- `rerun`
- `run that again`
- `generate the same ...`
- `regenerate`
- `recompute`
- `repeat`

Referential/explanatory language alone is not a rerun request. Examples that must not hydrate or execute a new selected workflow:

- "Why did that optimization choose landfill?"
- "Explain the frontier points."
- "What were the frontier points from that optimization?"

Those queries should be handled by the typed follow-up resolver from prior artifacts when possible. If the user says "same" or "again" without enough prior typed metadata, the selected workflow should return a typed diagnostic failure when selected, not silently fall back to legacy.

Negated action language such as "do not rerun", "don't run it again", or "without recomputing" must suppress hydration even if the word `rerun` or `run` appears.

### Optimizer Artifact Follow-Ups

Interpretive and data-summary optimizer follow-ups must answer from verified prior payload fields:

- frontier point count
- feasible/landscape point count
- metric labels
- compact frontier point table
- process stage selections, including landfill/`lf` if present in any recorded stage
- wash selections/routes when present
- optimizer payload path, plot path, and diagnostic bundle path when requested or relevant

The answer must explicitly state that no new run was started. It must not invent a causal explanation beyond the recorded objectives, constraints, selected designs, and metric values. If the artifact lacks enough evidence for a causal claim, it should say so.

### Rerun Execution Strategy

For a prior `routed_optimization` workflow, the first implementation should use one of these deterministic strategies:

1. Prefer reusing prior structured candidate/handoff artifacts if they contain enough evidence for optimizer tool args.
2. Otherwise rerun the full routed typed plan using the prior feedstock and top-k constraints.

In both cases:

- optimization must run through `run_waste_management_pareto` for single-slice Pareto.
- visualization must run through `plot_optimization_pareto_front`.
- plot step must consume the optimizer payload artifact, not upstream prose.
- requested `output_dir` from the new turn must propagate to the plot step.
- final response must cite the new ledger artifacts and save paths.

### De-Duplicated User-Facing Paths

If multiple artifact frames share the same output path:

- ledger artifacts remain distinct.
- `RuntimeProgressSummary.produced_artifact_paths` de-duplicates paths preserving first-seen order.
- follow-up `progress.produced_artifact_paths` de-duplicates paths.
- success/follow-up formatting should not list the same path redundantly unless it is intentionally grouped by artifact type.
- manifest copying should avoid creating multiple diagnostic copies of the same source file where possible.

## Proposed Design

### 1. Extend Typed Runtime Follow-Up Resolver

Update:

```text
src/strap/planning/typed_runtime_followups.py
```

Add a response mode:

```python
class FollowupResponseKind(Literal):
    path_status
    artifact_summary
    runtime_status
```

or equivalent internal classification.

Detection for `runtime_status` should include:

- `steps`, `completed`, `ran`, `tools`, `callables`
- `workflow`, `plan`, `routed`, `direct`
- `typed runtime`, `legacy`
- `solubility`, `safety`, `fast path`

The formatter should load plan/ledger from either message metadata or manifest files and return a concise status/provenance response.

### 2. Add Typed Runtime Context Snapshots

New module or extension:

```text
src/strap/planning/typed_runtime_context.py
```

Suggested models:

```python
class TypedRuntimeContextSnapshot(PlanningModel):
    plan_id: str | None
    workflow_id: str | None
    user_query: str | None
    plan: dict[str, Any] | None
    ledger: dict[str, Any] | None
    compile_result: dict[str, Any] | None
    artifacts: list[ArtifactFrame]
    manifest: dict[str, Any]


class HydratedRuntimeContext(PlanningModel):
    context: dict[str, Any]
    source_plan_id: str | None
    source_workflow_id: str | None
    source_run_dir: str | None
    hydration_notes: list[str]
```

Suggested APIs:

```python
def collect_recent_typed_runtime_snapshots(messages: list[Any], *, limit: int = 4) -> list[TypedRuntimeContextSnapshot]:
    ...


def maybe_hydrate_context_for_selected_followup(query: str, messages: list[Any]) -> HydratedRuntimeContext | None:
    ...
```

Hydration should be conservative:

- only use prior messages with `strap_origin="typed_runtime"`
- prefer the most recent compatible workflow/artifact family
- require explicit rerun/action intent, not just referential markers such as `that` or `previous`
- do not hydrate from `typed_runtime_followup` summaries alone
- do not hydrate from legacy prose

### 3. Feed Hydrated Context Into Compilation

Update:

```text
src/strap/planning/typed_runtime_integration.py
src/strap/planning/runtime.py
src/strap/planning/extractors.py
src/strap/planning/compiler.py
```

Middleware order:

1. Load planner config.
2. If mode is `off` or `shadow`, return to legacy behavior without typed follow-up answering or hydration.
3. Try prior-artifact follow-up resolver.
4. If not answered, try typed context hydration.
5. Call `maybe_run_typed_runtime(query, context=hydrated.context, ...)`.
6. If result is selected typed failure, return typed failure with diagnostics.
7. If unselected, continue legacy behavior.

The extractor/compiler should read hydrated context for missing facts, including:

- `feed_capacity_tpy`
- `feed_composition`
- `scenario`
- `metrics`
- `objective`
- `n_points`
- `min_washes`
- `max_washes`
- `candidate_solvents_by_polymer`
- `optimization_stage_candidates`
- `prior_workflow_id`
- `prior_artifact_types`
- `output_dir`

Current-turn facts always override hydrated facts.

### 4. Compile "Same Pareto Landscape Again" As Selected

The compiler should recognize selected rerun requests such as:

- "Generate the same Pareto landscape again with 8 points..."
- "Rerun that Pareto plot with 12 points..."
- "Make the previous cost-vs-emissions frontier again and save to..."

Expected plan shape for the row-4 optimization smoke:

- if prior optimizer-ready candidate/handoff payload is available:
  - `optimize_pareto`
  - `plot_optimization`
- otherwise:
  - `separation_candidates`
  - `build_optimization_handoff`
  - `optimize_pareto`
  - `plot_optimization`

The plan must be selected by `optimization_pareto_landscape` and/or `optimization_pareto_plot`.

### 5. De-Duplicate Paths

Update:

```text
src/strap/planning/typed_runtime_integration.py
src/strap/planning/typed_runtime_followups.py
src/strap/planning/runtime_persistence.py
```

Rules:

- de-duplicate `produced_artifact_paths` preserving order.
- de-duplicate diagnostic copies by normalized source path.
- keep artifact frames distinct so contracts remain auditable.
- format shared paths as "shared by ..." if useful, but do not repeat identical path lines in progress summaries.

## Required Tests

### Unit Tests

Add tests for status/provenance follow-ups:

- prior ledger with four completed routed optimization steps returns completed steps and callable names.
- "Was this routed Pareto or direct solubility/safety?" returns workflow/tool provenance, not path-only text.
- typed failure status follow-up includes failed step and failed checks.
- no prior typed runtime returns `should_answer=False`.

Add tests for typed context hydration:

- collects prior plan, compile result, ledger, artifacts, and manifest from message metadata.
- loads missing plan/ledger/artifact files from the diagnostic bundle when not present in message kwargs.
- hydrates `same Pareto landscape again with 8 points` with prior feedstock/composition/scenario/metrics and new `n_points=8`.
- does not hydrate interpretive follow-ups such as `Why did that optimization choose landfill?`.
- current-turn output directory overrides prior output directory.
- does not hydrate from legacy or typed-followup-only messages.
- no compatible prior typed run returns no hydration.

Add compiler/runtime tests:

- hydrated rerun compiles to selected Pareto plan.
- hydrated rerun with wrappers executes fake `run_waste_management_pareto` and fake `plot_optimization_pareto_front`.
- requested output directory appears in plot step args.
- selected hydration failure returns typed diagnostics rather than legacy fallback.
- unselected same/again request still falls through to legacy.

Add path de-duplication tests:

- two artifact frames sharing the same payload path produce one progress path.
- follow-up progress de-duplicates shared paths.
- manifest copies a shared source path once.

Add optimizer follow-up tests:

- frontier-point follow-up renders a compact table from payload `points`.
- interpretive/why follow-up answers from prior artifacts, includes no-rerun language, and does not call typed runtime.
- landfill checks inspect recorded process stages, not only stage 3.
- requested artifact/plot paths and diagnostic bundle path are included.
- `DISSOLVE_TYPED_PLANNER=off` and `shadow` do not answer typed follow-ups before legacy routing.

### Operational Smoke

Rerun:

```text
docs/subagent_query_bank-v1.xlsx
sheet: 10 optimization-engineer
row: 4
mode: DISSOLVE_TYPED_PLANNER=enforce_selected
```

Turns:

1. Seed workbook query.
2. "Where did the optimization Pareto landscape or plot get saved? Give exact plot paths and the diagnostic bundle path."
3. "Summarize the frontier points and metrics from the optimizer artifacts only. Include any plot or artifact paths you know about."
4. "Why did that optimization choose landfill? Answer from optimizer artifacts only; do not rerun."
5. "What typed-runtime steps completed, and did the run use the routed Pareto plan rather than a direct solubility or safety path?"
6. `Generate the same Pareto landscape again with 8 points and save it to ".../rerun".`

Expected:

- Turn 1: `typed_runtime / executed`
- Turn 2: `typed_runtime_followup / answered_from_prior_artifacts`
- Turn 3: `typed_runtime_followup / answered_from_prior_artifacts`
- Turn 4: `typed_runtime_followup / answered_from_prior_artifacts`, no new typed runtime execution, includes frontier table and path data when available
- Turn 5: `typed_runtime_followup / answered_from_prior_artifacts`, with completed step ids and callable/tool names
- Turn 6: `typed_runtime / executed`, not `routing_single_specialist_prose`
- Turn 6 creates a new plot under the requested `.../rerun` directory
- no direct solubility or safety path is used
- no prose-only claim that visualization is unavailable when a selected plot wrapper exists
- no duplicate payload path in user-facing progress

## Definition Of Done

- Optimization status/provenance follow-ups answer from ledger/plan metadata.
- "same/again/rerun" selected optimization requests hydrate from prior typed runtime metadata and execute through typed runtime.
- Interpretive optimization follow-ups answer from prior typed artifacts and do not hydrate/rerun.
- `off` and `shadow` remain behavior-neutral for typed follow-ups.
- Save-path overrides are honored for hydrated selected reruns.
- Shared artifact paths are de-duplicated in user-facing progress and diagnostics.
- The row-4 optimization multi-turn smoke passes with typed origins for all selected turns.
- Full test suite remains green.

## Follow-On Work

After this slice passes, reuse the same hydration/status machinery for:

- HSP reruns with changed polymer/solvent category.
- BioSTEAM TEA/LCA reruns with changed scenario or solvent.
- separation tree/selectivity reruns with changed target polymer.
- multi-slice Pareto reruns with changed composition grid or point count.
