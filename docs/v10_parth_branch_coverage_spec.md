# v10 Parth Branch Coverage Spec

Purpose: make `v10-core` cover the intent of Parth's `RAG-polymer-solubility_parth` branch without direct-porting its implementation. Parth's branch focused on optimization reliability, BioSTEAM/Excel coefficient handling, solver behavior, and prompt/routing failures. v10 already supersedes much of that with typed planning, richer optimizer support, residual checks, and structured runtime artifacts. This spec captures the remaining coverage gaps and the v10-native way to close them.

Reference branch:

- `origin/RAG-polymer-solubility_parth`
- latest audited commit: `cb2d20c Decouple optimization agent from upstream bioSTEAM dependencies`
- v10 baseline audited: `origin/v10-core` at `f4281c8 Remove transient Excel lock artifact`

Implementation status:

- Implemented v10-native negated-Pareto and single-objective extraction, typed compilation, and legacy routing preservation.
- Implemented audited PIW GWP fallback telemetry for unresolved pyrolysis/gasification workbook formulas, plus explicit model-reference CAPEX fallback for `gas_er`.
- Added optimizer payload exposure for available other-tech technologies and fallback/default telemetry.
- Added regression tests for typed planner, legacy routing, and other-tech data loading.
- Validation after implementation: `pytest -q` passed with `947 passed, 4 warnings`.

## Coverage Themes

| Parth theme | v10 status | Required v10 action |
| --- | --- | --- |
| Explicit "no Pareto" requests | Partially missing. Typed compiler currently treats negated Pareto text as Pareto because it matches `pareto` before negation semantics. Legacy routing can drop optimization entirely. | Add negated-Pareto extraction and route to single-objective optimization, default `objective=max_profit` when no objective is stated. |
| Single-objective aliases | Partially missing in typed compiler. Legacy routing sees some phrases such as `max circularity`, but typed compiler may return unsupported or default to `max_profit`. | Add deterministic objective extraction for `max_profit`, `min_emissions`, `min_total_cost`, `max_circularity`, plus misspellings/aliases where safe. |
| Optimization tool forcing / anti-hallucination | Mostly covered by typed runtime, guarded tool calls, final synthesis anchoring, and optimizer structured payload formatting. | Add regression tests ensuring selected optimization requests never synthesize from prose-only paths and always cite optimizer artifacts. |
| Optimization prompt synthesis failures | Superseded by v10 typed runtime and structured final synthesis guards. | No prompt-only port. Keep prompt concise; rely on contract execution and final-answer validation. |
| Pareto scaling with BioSTEAM capacity | Superseded conceptually by v10's baseline-backed simulation skips, materialized optimizer table, multi-polymer support, candidate telemetry, and dynamic feed-composition handling. | Add targeted tests proving feed-size changes propagate into compiled coefficients / payload telemetry for point, Pareto, and slices. |
| Other-tech data loading and gasification/Pyrolysis fallback | Missing important coverage. Current v10 constants include `py`, `gas_er`, `gas_h2`, `gas_h2cc`, but `load_othertech_data()` excludes them because workbook external formulas resolve to `None`. | Add explicit, documented paper-backed fallback values and telemetry for unresolved workbook metrics. Do not silently zero-fill scientific fields. |
| Solver warning / SCIP stdout flooding | Mostly covered by v10 solver log capture, `tee=False`, residual verification, and retry ladder. | Add logger suppression only if tests reproduce noisy output. Add a regression check that failed SCIP solves do not flood user-visible output. |
| Solver crashes / unbounded models | Superseded by v10 residual checks, fallback attempts, status diagnostics, and typed infeasible results. | Keep current solver approach. Add one regression linking Parth-style infeasible/original-problem failures to typed infeasible or retry behavior. |
| Optimization/BioSTEAM decoupling | v10 already decouples TEA/LCA subagent handoff from optimization execution. Optimizer still accepts separation/solvent handoffs by design. | Add tests proving optimization can run with feed/composition only, and separately with separation handoff. |

## Phase 1: Negated Pareto and Single-Objective Intent

### Problem

Queries such as:

```text
Optimize waste management for 8000 tonnes/year 60% PE 40% EVOH. I do not want Pareto.
```

should compile to:

- `intent_family="optimization"`
- tool: `run_waste_management_optimization`
- output: `optimization_point_result`
- objective: `max_profit` if no explicit objective is supplied

Current v10 behavior is wrong in two ways:

- typed compiler sees the word `pareto` and compiles `run_waste_management_pareto`
- legacy routing can interpret negated optimization/Pareto too broadly and drop the optimization route

### Implementation

Add deterministic facts in `src/strap/planning/extractors.py`:

- `pareto_requested: bool`
- `pareto_negated: bool`
- `single_objective_requested: bool`
- `objective: Literal["max_profit", "min_emissions", "min_total_cost", "max_circularity"] | None`

Rules:

- `do not want Pareto`, `no Pareto`, `without Pareto`, `not a Pareto`, `just optimize`, `single objective`, `single-point`, and `one optimum` force point optimization unless a stronger multi-slice/frontier request is present elsewhere.
- `maximize profit`, `max profit`, `profit objective` -> `max_profit`
- `minimize emissions`, `min emissions`, `lowest GWP` -> `min_emissions`
- `minimize cost`, `lowest cost`, `min total cost` -> `min_total_cost`
- `maximize circularity`, `max circularity`, `max_circularity` -> `max_circularity`
- If Pareto is negated and no explicit objective is present, default to `max_profit` and include a diagnostic explaining the default.

Compiler changes in `src/strap/planning/compiler.py`:

- `_extract_requested_artifacts()` should not add Pareto artifacts when Pareto is negated.
- `_compile_deterministic()` should prefer `_compile_direct_optimization()` for negated-Pareto or single-objective requests.
- routed optimization should do the same after separation/handoff when the user asks separation candidates plus "do not Pareto".

Routing changes in `src/strap/routing_classifier.py`:

- Negated Pareto must not mean negated optimization.
- Split `_NEGATED_OPTIMIZATION_RE` into:
  - negated whole optimization request
  - negated Pareto/frontier/tradeoff request
- `I do not want Pareto` should preserve `optimization.pathway`.

### Tests

Add tests in `tests/test_plan_compiler.py`:

- `test_compile_no_pareto_defaults_to_max_profit_point_optimization`
- `test_compile_no_pareto_with_min_emissions_uses_min_emissions`
- `test_compile_single_objective_max_circularity`
- `test_compile_minimize_cost_uses_min_total_cost`
- `test_compile_routed_no_pareto_uses_point_optimization_after_handoff`

Add tests in `tests/test_routing.py`:

- `test_negated_pareto_preserves_optimization_goal`
- `test_negated_optimization_still_blocks_optimization_goal`
- `test_max_circularity_routes_to_optimization`
- `test_common_optimization_misspelling_routes_when_feed_context_present`

Acceptance:

- No-Pareto queries compile to point optimization.
- Explicit objective aliases map to the right optimizer objective.
- Pareto requests still compile to Pareto.
- True negated optimization, e.g. `do not optimize`, still blocks optimization.

## Phase 2: Other-Tech Workbook Fallbacks and Telemetry

### Problem

Current v10 exposes these technology sets:

```python
OTHERTECH = ["lf", "we", "py", "gas_er", "gas_h2", "gas_h2cc"]
```

But the actual loaded available set is only:

```python
["lf", "we"]
```

because the workbook stores several pyrolysis/gasification values as formulas referencing external sheets/workbooks. `openpyxl` cannot resolve those references, so v10 correctly returns `None`, but then excludes those technologies. Parth's branch added exact paper GWP fallbacks, but also used broader zero-fill behavior. v10 should not silently zero-fill missing scientific/economic metrics.

### Implementation

Add explicit fallback metadata in `src/strap/waste_management/data_loader.py`:

```python
OTHERTECH_PAPER_GWP_FALLBACK = {
    "lf": 0.0864564,
    "we": 2.45971,
    "py": 0.682266,
    "gas_er": 1.07,
    "gas_h2": 5.42838,
    "gas_h2cc": 2.55583,
}
```

Fallback policy:

- Apply paper GWP fallback only when workbook GWP is missing or numerically zero because the source formula is unresolved.
- Do not overwrite nonzero workbook GWP by default.
- Do not invent CAPEX/OPEX fallbacks silently.
- Preserve known workbook values, including landfill and waste-to-energy values.
- Return telemetry describing every fallback:
  - `tech`
  - `metric`
  - `value`
  - `source="piw_paper_julia_reference"`
  - `reason="missing_external_workbook_formula"` or `reason="zero_or_missing_workbook_value"`

Add a structured loader API:

```python
load_othertech_data(..., return_telemetry: bool = False)
```

When `return_telemetry=False`, preserve the current return shape. When `True`, return `(other_data, telemetry)`.

Availability policy:

- `lf`: requires GWP and OPEX.
- `we`: requires GWP.
- `py`: requires GWP and OPEX.
- `gas_er`: requires GWP and either workbook CAPEX or model-level gasification CAPEX fallback explicitly documented in `model.py`.
- `gas_h2`, `gas_h2cc`: require GWP and any model-required economics.

Before marking gasification variants available, audit `model.py` to ensure missing CAPEX/OPEX is either not required for that tech or has an explicit model-level coefficient.

### Tests

Add tests in `tests/test_waste_optimization.py` or new `tests/test_waste_data_loader.py`:

- `test_load_othertech_applies_paper_gwp_fallback_for_unresolved_gasification`
- `test_load_othertech_does_not_overwrite_nonzero_workbook_gwp_by_default`
- `test_load_othertech_returns_fallback_telemetry`
- `test_available_othertechs_include_pyrolysis_and_gasification_only_when_required_metrics_present`
- `test_othertech_fallback_values_match_piw_reference`

Acceptance:

- `load_othertech_data()` no longer silently drops technologies solely because GWP external formulas are unresolved.
- Every fallback is auditable.
- No zero-fill is used to make technologies appear feasible.
- Optimizer payload can report which other-tech fallbacks were used.

## Phase 3: Coefficient Scaling and BioSTEAM Independence

### Problem

Parth fixed a stale Excel/BioSTEAM scaling issue. v10's implementation is structurally different and broader, but needs explicit tests proving the same theme is covered.

### Implementation

Add verification around the v10 optimizer path:

- Feed capacity must propagate into STRAP coefficient materialization.
- Feed composition must propagate into per-polymer capacities.
- Baseline-backed rows should skip BioSTEAM without being marked as simulation failures.
- Materialized rows should use BioSTEAM simulation results only when valid TEA/LCA metrics are present.
- Repeated calls should not reuse stale coefficients from a previous feed/composition.

Add diagnostic telemetry if missing:

- `feed_capacity_tpy`
- `feed_composition`
- `per_polymer_capacity_tpy`
- `simulation_skips`
- `simulation_failures`
- `coefficient_source_counts`

### Tests

Add tests in `tests/test_waste_optimization.py`:

- `test_point_optimization_coefficients_scale_with_feed_capacity`
- `test_pareto_coefficients_scale_with_feed_capacity`
- `test_pareto_slices_scale_each_composition_independently`
- `test_baseline_rows_are_skipped_not_failed`
- `test_invalid_biosteam_rows_do_not_replace_baseline_values`

Acceptance:

- Running the same solvent set at 8,000 t/y and 20,000 t/y produces distinct scaled coefficient telemetry.
- Pareto and point optimization use the same coefficient preparation path.
- Slices do not leak coefficients across compositions.

## Phase 4: Solver Output Hygiene and Failure Semantics

### Problem

Parth reduced solver stdout flooding and unbounded/infeasible crash behavior. v10 has stronger solver validation, but the behavior should be locked with tests.

### Implementation

Keep the current v10 model:

- solver log capture for SCIP
- `tee=False`
- `_load_verified_result()` gates solution loading behind residual checks
- retry ladder for numerical failures
- typed infeasible result when fail-closed semantics apply

Add light hygiene:

- Set Pyomo solver/core loggers to warning/error in the solver module if user-visible stdout flooding is reproducible.
- Add solver debug summaries to optimizer payloads whenever a result is rejected.

### Tests

Add tests:

- `test_scip_original_problem_infeasible_rejected_without_fake_solution`
- `test_solver_residual_violation_retries_or_fails_typed`
- `test_solver_warning_output_is_not_returned_as_final_answer`
- `test_solver_debug_payload_contains_rejection_reason`

Acceptance:

- Numerically invalid solver results never dominate Pareto filtering.
- Solver failures are visible as diagnostics, not stdout spam or fabricated successful results.

## Phase 5: Optimization Prompt and Typed Runtime Contract Coverage

### Problem

Parth used prompt edits to force tool usage and prevent "parrot trap" synthesis. v10 should encode this as executable contracts rather than prompt reliance.

### Implementation

Add/confirm typed runtime tests:

- selected optimization requests execute optimizer wrappers, not prose synthesis
- final answer cites optimizer artifact types and produced paths
- follow-up explanations use prior optimizer artifacts only
- upstream separation prose is never treated as optimized output
- no generic legacy wrapper can mint optimization artifacts

Prompt cleanup:

- Keep `10_optimization-engineer.yaml` direct and consistent with typed runtime.
- Mention `run_waste_management_optimization` for single objective and `run_waste_management_pareto` for tradeoffs.
- Mention explicit no-Pareto behavior.
- Do not encode large behavioral rules that are now enforced in code.

### Tests

Add tests in typed runtime suites:

- `test_selected_no_pareto_runtime_uses_point_optimizer_wrapper`
- `test_final_synthesis_requires_optimizer_artifact_for_optimization`
- `test_optimizer_followup_uses_prior_payload_not_separation_prose`
- `test_missing_optimizer_artifact_fails_closed`

Acceptance:

- Prompt drift cannot bypass typed runtime contracts.
- Selected optimization workflows fail closed when structured optimizer evidence is missing.

## Phase 6: Regression Query Set

Add a small Parth-coverage query set to the query bank:

1. `Optimize waste management for 8000 tonnes/year composed of 60% PE and 40% EVOH. I do not want Pareto.`
2. `Optimize waste management for 8000 tonnes/year composed of 60% PE and 40% EVOH. Maximize circularity.`
3. `Minimize emissions for 8000 tonnes/year composed of 60% PE and 40% EVOH under scenario A.`
4. `Run a cost-vs-emissions Pareto for 8000 tonnes/year composed of 60% PE and 40% EVOH under scenario A.`
5. `Run waste management optimization with gasification and pyrolysis allowed; report which Stage-3 technologies were considered.`
6. `Run routed separation-to-optimization for LDPE/EVOH/PET, but do not run Pareto; return the single max-profit route.`
7. `Run multi-composition Pareto slices for LDPE/EVOH/PET and show the diagnostic bundle.`

Expected coverage:

- point optimizer
- Pareto optimizer
- routed point optimizer
- typed runtime selected workflow
- other-tech fallback telemetry
- solver diagnostics

## Non-Goals

- Do not directly copy Parth's workbook into v10.
- Do not silently zero-fill unresolved scientific/economic values.
- Do not reintroduce prompt-only enforcement for optimization correctness.
- Do not remove separation handoff support from optimization; v10 intentionally uses separation-derived solvent candidates.
- Do not enable broad typed `enforce` mode as part of this coverage work.

## Definition of Done

- No-Pareto and single-objective optimization queries compile and route correctly in both typed planner and legacy fallback.
- Pyrolysis/gasification data availability is explicit, auditable, and tested.
- Feed scaling is regression-tested for point, Pareto, and slice workflows.
- Solver failures remain residual-clean and diagnostically visible.
- Optimization final answers are grounded only in verified optimizer artifacts.
- Focused tests pass.
- Full suite passes before commit.
