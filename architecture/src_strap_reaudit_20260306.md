# `src/strap` Re-Audit

Date: 2026-03-06  
Scope: Post-redesign audit of the `codex/orchestration-handoff-redesign` worktree at `/tmp/strap-orchestration-redesign`

## Validation Baseline

- Full local suite after audit-time repair: `264 passed, 4 warnings`
- Live harness verification:
  - `seq-sep-viz`
  - latest saved artifact: `architecture/test_results/test_results_20260306_194105.json`
  - harness now stores both `answer_preview` and `full_answer`

## Breakage Found During Audit

The audit found one real regression introduced by the refactor and fixed it immediately:

- `src/strap/services/biosteam_service.py`
  - `parse_json_array(...)` lost its `json` import
  - effect: invalid JSON inputs fell through the safe wrapper as raw `ERROR ...` text instead of the structured BioSTEAM envelope
  - status: fixed before the final suite run

This means the findings below are based on the repaired current worktree, not the broken intermediate state.

## Findings

### 1. Medium: tool contract discipline is still incomplete at the unexpected-error path

The main contract cleanup landed, but the shared fallback path still formats unexpected exceptions as raw text in `src/strap/tools/_helpers.py:65`. That means a single unhandled bug can still bypass the normalized JSON envelope even when the happy path and predictable validation errors were standardized.

Why this matters:

- this is exactly how the audit-time `biosteam_service.py` import regression surfaced before it was fixed
- downstream agent logic is now much more structured than before, so raw `ERROR in ...` strings are proportionally more disruptive

Impact:

- the contract cleanup is materially improved, but not yet fail-closed

### 2. Medium: classifier post-normalization still over-admits specialists on some mixed separation queries

The post-classification cleanup in `src/strap/routing.py:306` only strips `statistics-ml` when the query is clearly HSP-only (`routing.py:312-314`). In live reruns of `seq-sep-viz`, the classifier still sometimes admitted `statistics-ml` alongside `separation-engineer` and `visualization-specialist`, which expanded the routed plan and increased orchestration churn even though route guards prevented a full derailment.

Why this matters:

- runtime behavior is better than before, but the classifier still injects unnecessary plan complexity
- this is not a correctness failure in the current worktree, but it remains a source of wasted turns and longer traces

Impact:

- residual runtime inefficiency
- harder-to-read traces
- more pressure on router guardrails

### 3. Medium: active contract normalization still does not cover several tool families

The following active modules still return bespoke strings or ad hoc JSON instead of the shared `{"display": ..., "data": {...}}` envelope:

- `src/strap/tools/solvent_lookup.py:526`
- `src/strap/tools/literature.py:1038`
- `src/strap/tools/rag_core.py:80`

The core contract work already covers BioSTEAM, interpolation, ML prediction, safety, thermal prediction, listing, database query, and solvent properties, but these remaining tools still create uneven downstream semantics.

Why this matters:

- multi-agent behavior is now increasingly shaped around structured returns
- these modules are still more difficult to consume safely from the orchestrator or other specialists

Impact:

- inconsistent downstream handling
- continued reliance on prose parsing in some paths

### 4. Medium: `advanced_separation.py` remains the largest active tool bottleneck

The refactor reduced `src/strap/tools/advanced_separation.py`, but it is still `3495` lines and remains the biggest active tool module in the package. The active orchestration layer is also still concentrated in large policy-heavy modules:

- `src/strap/routing.py` (`2162` lines)
- `src/strap/guardrails.py` (`1238` lines)
- `src/strap/handoffs.py` (`1053` lines)

Why this matters:

- the redesign works, but a lot of runtime behavior now depends on a small set of dense policy modules
- future changes will remain riskier and slower until more of this logic is split into smaller, testable units

Impact:

- maintenance risk
- harder review surface
- slower follow-on refactors

### 5. Medium: vendor isolation is still largely untouched

The two largest modules in the package are still vendor-heavy implementation files:

- `src/strap/vendor/_agent_sql_source.py` (`12747` lines)
- `src/strap/vendor/rag.py` (`7884` lines)

These are still oversized and still expose legacy output patterns and logic styles that are inconsistent with the newer service/contract/orchestration layers.

Why this matters:

- current changes increasingly depend on local wrappers being clean
- vendor modules remain a large source of complexity and future integration drag

Impact:

- high structural debt
- harder future contract cleanup

### 6. Low: warning hygiene is still noisy

The full suite is green, but the audit baseline still reports warnings:

- `tests/test_hsp_screening.py:85`
  - `PytestCollectionWarning` because `TestResult` is a dataclass named like a test class
- `src/strap/solubility.py:165`
  - `RuntimeWarning: overflow encountered in exp`

These are not blocking, but they reduce signal quality in CI and hide real warnings more easily.

## What Is Now Solid

Compared to the earlier audit, the following areas are materially stronger:

- scoped append-only handoffs
- repeated same-subagent results without overwrite
- sequential handoff enforcement and auto-dispatch
- result/failure accounting for missing and invalid structured outputs
- stronger subagent stop conditions
- deterministic final-answer checks for key separation cases
- harness evidence quality via `full_answer`

## Overall Status

- Architecture: mostly in place
- Runtime behavior: improved, but still has classifier-side inefficiency
- Contract discipline: much stronger, but still incomplete at the remaining tool families and the raw unexpected-error path
- Codebase cleanup: meaningful progress on active modules, but `advanced_separation` and vendor isolation are still open

## Recommended Next Focus

1. finish contract cleanup for the remaining tool families and the raw fallback path
2. tighten classifier normalization for mixed separation workflows
3. continue decomposing `advanced_separation.py`
4. isolate vendor-heavy modules behind thinner local interfaces
