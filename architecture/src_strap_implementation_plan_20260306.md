# `src/strap` Implementation Plan (Re-Audit)

Date: 2026-03-06  
Supersedes: `architecture/src_strap_implementation_plan_20260305.md`

## Baseline

Current worktree state after audit-time repair and follow-on implementation:

- orchestration/handoff redesign is in place
- harness stores `full_answer`
- full local suite passes: `296 passed`
- one audit-found regression was fixed:
  - missing `json` import in `src/strap/services/biosteam_service.py`
- major decomposition completed beyond the original audit scope:
  - `advanced_separation.py` reduced from the multi-thousand-line monolith to `567` lines
  - dedicated tool modules now own:
    - precipitation / antisolvent tools
    - sequence planning tools
    - sequence analysis tools
    - separation visualization tools

This plan carries forward unfinished work from the prior plan and adds the issues exposed by the re-audit.

## Priority 0: close remaining runtime drift

### Goal

Reduce unnecessary route expansion and make live traces more stable without weakening the orchestration guards.

### Tasks

1. tighten classifier post-normalization in `src/strap/routing.py`
   - add more specific cleanup for separation + visualization query classes
   - avoid admitting `statistics-ml` on clearly process-design-oriented separation requests unless the query explicitly asks for HSP-only screening

2. add stronger plan compaction after first valid route step
   - once the orchestrator has committed to a sequential pair like `separation-engineer -> visualization-specialist`, drop unrelated advisory specialists from the active remaining plan

3. keep live route checks on:
   - `seq-sep-viz`
   - `seq-sep-tea`
   - one 3-agent route

### Acceptance criteria

- no extra specialist remains in `remaining=[]` for the standard `seq-sep-viz` workflow
- routed live traces stop showing unnecessary classifier-admitted tail steps

## Priority 1: finish contract discipline

### Goal

Make tool semantics uniformly machine-consumable and remove the last major fail-open path.

### Tasks

1. replace the raw unexpected-error formatter in `src/strap/tools/_helpers.py`
   - unexpected exceptions should also be wrapped in a structured envelope where possible
   - keep human-readable diagnostics, but stop returning plain `ERROR in ...` text to agent consumers

2. normalize remaining active tool families to the shared envelope:
   - `src/strap/tools/solvent_lookup.py`
   - `src/strap/tools/literature.py`
   - `src/strap/tools/rag_core.py`
   - `src/strap/tools/rag_diagnostics.py`
   - any residual separation helper entry points that still mix prose-only returns

3. add one contract-matrix regression test for the remaining non-normalized tools

### Acceptance criteria

- no active tool family relies on bespoke string-only returns for predictable user-facing outcomes
- unexpected exceptions no longer bypass the normalized tool contract in normal agent flows

## Priority 2: keep decomposing large active modules

### Goal

Reduce maintenance risk in the modules that now carry most orchestration and separation logic.

### Tasks

1. continue splitting `src/strap/tools/advanced_separation.py`
   - largely completed in this worktree
   - remaining work is now cleanup of the new split modules rather than more surgery on `advanced_separation.py` itself
   - prefer further reductions in:
     - `src/strap/tools/sequence_planning_tools.py`
     - `src/strap/tools/sequence_analysis_tools.py`

2. reduce policy density in:
   - `src/strap/routing_progress.py`
   - `src/strap/guardrails.py`
   - `src/strap/handoff_store.py`

3. prefer service/helper extraction with direct unit tests instead of large in-place rewrites

### Acceptance criteria

- `advanced_separation.py` is materially smaller and no longer the main tool-module hotspot
- new extracted units have direct test coverage
- routing/guardrail changes become easier to review in smaller pieces

## Priority 3: isolate vendor-heavy modules

### Goal

Contain the largest remaining structural debt without rewriting vendor logic wholesale.

### Tasks

1. define thin local adapter boundaries around:
   - `src/strap/vendor/rag.py`
   - `src/strap/vendor/_agent_sql_source.py`

2. move local orchestration assumptions and formatting decisions out of vendor files and into local wrappers/services

3. only after the boundary is clean, consider deeper internal cleanup

### Acceptance criteria

- orchestrator and tool layers no longer directly depend on large vendor-specific output shapes
- future contract cleanup can happen in local wrappers first

## Priority 4: warning hygiene

### Goal

Reduce CI/test noise so real regressions are more visible.

### Tasks

1. fix the pytest collection warning in `tests/test_hsp_screening.py`
   - rename `TestResult` or mark it non-collectable

2. silence the predictable overflow warning in `src/strap/solubility.py`
   - use a bounded `np.errstate(...)` or equivalent safe exponential path around `np.exp(ln_s)`

3. document any remaining third-party warnings that are intentionally tolerated

### Acceptance criteria

- the local suite stays green with fewer avoidable warnings

## Suggested Execution Order

1. Priority 1: contract cleanup
2. Priority 0: classifier/runtime drift cleanup
3. Priority 2: `advanced_separation` decomposition
4. Priority 4: warning hygiene
5. Priority 3: vendor isolation

## Why This Order

- contract cleanup still has the highest leverage for multi-agent reliability
- runtime drift is now mostly inefficiency, not architecture failure
- decomposition should continue, but after the semantics stop moving as much
- vendor isolation is important, but it is still the least urgent of the remaining work
