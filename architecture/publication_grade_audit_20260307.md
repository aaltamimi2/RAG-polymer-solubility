# Publication-Grade Audit — 2026-03-07

## Scope
Audit target: `/tmp/strap-orchestration-redesign/src/strap` plus the live-eval harness under `architecture/`.

Primary goals:
- correctness after the orchestration and BioSTEAM refactors
- benchmark integrity
- contract consistency
- remaining architectural bloat

## Validation Snapshot
- Full local suite run on this revision: `363 passed, 1 failed, 1 warning`
- Failing test:
  - `tests/test_tools.py::test_run_biosteam_batch_returns_partial_results_for_large_screen`
- Warning:
  - external `joblib` serial-mode warning, not project logic

## Priority Findings

### 1. High — large-screen BioSTEAM batch fallback no longer guarantees decision-quality partial output
Files:
- [src/strap/tools/biosteam_tea_lca.py](/tmp/strap-orchestration-redesign/src/strap/tools/biosteam_tea_lca.py#L234)
- [tests/test_tools.py](/tmp/strap-orchestration-redesign/tests/test_tools.py#L124)

The large-batch fallback stops on wall-clock budget and a coarse success counter, but it can now exit before it has accumulated the intended minimum of five successful ranked scenarios. The full suite already catches this: the current code can return only four successful results for `all_pe` screens while still treating the run as a valid partial batch.

Why this matters:
- this directly weakens the main publication-facing TEA screening workflow
- it is not theoretical; the regression is currently test-visible
- the current stopping rule optimizes elapsed time, not usefulness of the returned ranking

Required fix direction:
- make the large-batch stop condition enforce `>= 5` successful scenarios before returning a partial ranking, unless the batch is fully exhausted
- or explicitly degrade the response as `insufficient partial coverage` instead of presenting a top-list

### 2. High — timeout recovery can publish raw subagent prose without re-validating it
Files:
- [architecture/operational_eval_batch.py](/tmp/strap-orchestration-redesign/architecture/operational_eval_batch.py#L784)
- [architecture/operational_eval_batch.py](/tmp/strap-orchestration-redesign/architecture/operational_eval_batch.py#L811)

When a case times out and only `separation-engineer` ran, `_recover_timeout_result(...)` extracts the latest task output directly from the LangSmith trace, strips the structured block, and turns that prose into the recovered final answer. That bypasses the normal verifier/guardrail path and does not check whether the recovered prose is consistent with the latest validated payload.

Why this matters:
- benchmark artifacts can look successful even when the final answer came from an unverified intermediate specialist output
- this weakens auditability of long live campaigns, which is exactly where the harness is supposed to be strongest

Required fix direction:
- recover from validated payloads or recorded handoff state, not raw trace prose
- if only raw prose exists, mark the case as recovered-but-unverified rather than silently treating it as a normal answer

### 3. Medium — harness `routing_match` is still fail-open for extra subagents
Files:
- [architecture/test_harness.py](/tmp/strap-orchestration-redesign/architecture/test_harness.py#L531)

`run_query()` records `routing_match = expected_set.issubset(actual_set)`. That means a query can dispatch extra specialists and still be marked as `Routing OK: YES` in the saved harness result.

Why this matters:
- it overstates routing quality in single-case harness artifacts
- it can hide regressions that the operational batch evaluator would correctly flag

Required fix direction:
- change the harness-level definition to exact-set matching, or rename the field to `contains_expected_subagents` so it does not masquerade as a strict routing check

### 4. Medium — BioSTEAM solvent catalog is duplicated and already drifting
Files:
- [src/strap/services/biosteam_service.py](/tmp/strap-orchestration-redesign/src/strap/services/biosteam_service.py#L213)
- [src/strap/vendor/biosteam_runner.py](/tmp/strap-orchestration-redesign/src/strap/vendor/biosteam_runner.py#L993)

The service layer now derives solvent families from `60_common_solvents-TEA-LCA.csv`, but the runner still exposes hardcoded legacy solvent families through `get_supported_solvents()`. The tool layer mostly uses the service-derived lists, but the runner-level metadata remains stale and can diverge further over time.

Why this matters:
- two sources of truth for solvent families is exactly the kind of publication-grade drift that turns into non-reproducible results later
- future contributors can easily update one side and miss the other

Required fix direction:
- either remove runner-level solvent family catalogs entirely and import the service-derived catalog there
- or explicitly mark runner metadata as compatibility-only and stop exposing it as supported catalog output

### 5. Medium — `Database` singleton ignores later `data_dir` overrides
Files:
- [src/strap/database.py](/tmp/strap-orchestration-redesign/src/strap/database.py#L68)

`get_database(data_dir=...)` only honors the first `data_dir` ever passed. After that, every later call silently reuses the original singleton regardless of the requested directory.

Why this matters:
- alternate dataset runs are silently non-isolated
- test reproducibility and benchmark reproducibility are weaker than they appear
- this makes any future multi-dataset publication workflow fragile

Required fix direction:
- key the singleton by resolved `data_dir`, or make the helper explicitly reject a second conflicting `data_dir`

### 6. Medium — `statistics-ml` tools still do not reliably emit the standardized tool envelope metadata
Files:
- [src/strap/tools/statistical.py](/tmp/strap-orchestration-redesign/src/strap/tools/statistical.py#L137)
- [src/strap/tools/statistical.py](/tmp/strap-orchestration-redesign/src/strap/tools/statistical.py#L233)

This module still hand-rolls JSON envelopes instead of using `json_tool_response/json_tool_error`. The strings it returns rely on wrapper normalization and do not consistently declare `tool_name` themselves.

Why this matters:
- the codebase-wide “uniform contract” claim is still not strictly true
- this makes the module more fragile than the newer tool families
- future direct reuse of these helper outputs will be inconsistent

Required fix direction:
- convert the module to `tool_response_service` like the other refactored tool families

### 7. Low — public `classify_query()` still bypasses normalization that the middleware uses
Files:
- [src/strap/routing_classifier.py](/tmp/strap-orchestration-redesign/src/strap/routing_classifier.py#L186)
- [src/strap/routing_classifier.py](/tmp/strap-orchestration-redesign/src/strap/routing_classifier.py#L282)

The middleware calls `_normalize_matched_rules(...)`, but the public helper `classify_query()` just uses `classify_query_keywords()` and `_build_hint_from_matches()` directly. That means dry-run or diagnostic callers can still see routing hints that disagree with runtime behavior.

Why this matters:
- it does not break runtime routing, but it does weaken tooling/debugging consistency
- it makes publication-grade routing evidence harder to trust

Required fix direction:
- make `classify_query()` share the same normalization path as the middleware

### 8. Low — result extractor only captures the first `ToolMessage` from a `Command` update
Files:
- [src/strap/result_extractor.py](/tmp/strap-orchestration-redesign/src/strap/result_extractor.py#L395)

`_extract_text_from_result(...)` returns the first `ToolMessage` it sees inside a `Command.update["messages"]`. If a future `task()` path emits multiple tool messages and the structured-result-bearing message is not first, extraction will silently miss it.

Why this matters:
- not currently the dominant failure mode, but it is a brittle assumption in a central middleware
- it narrows future extensibility of task-return patterns

Required fix direction:
- scan all `ToolMessage` entries and prefer the last one containing a structured result block

## Bloat Map
Top remaining hotspots by line count in the active worktree:
- `src/strap/vendor/_agent_sql_source.py`: ~12.7k lines
- `src/strap/vendor/rag.py`: ~7.9k lines
- `src/strap/vendor/_langchain_tools_source.py`: ~2.1k lines
- `src/strap/vendor/biosteam_runner.py`: ~1.3k lines
- `src/strap/tools/biosteam_tea_lca.py`: ~1.3k lines
- `src/strap/routing_guards.py`: ~1.2k lines
- `src/strap/tools/visualization.py`: ~1.1k lines
- `src/strap/tools/literature.py`: ~1.0k lines
- `src/strap/engines/separation.py`: ~1.0k lines
- `src/strap/engines/precipitation.py`: ~0.9k lines
- `src/strap/verifier.py`: ~0.85k lines

Interpretation:
- the active tool shells are much better than earlier, but policy and vendor code are still disproportionately large
- publication-grade cleanup should now focus on policy correctness and vendor isolation, not more arbitrary splitting of already-smaller tool wrappers

## Broad Exception / Fail-Open Hotspots
Not every broad catch is wrong, but these remain the main concentration points:
- `src/strap/vendor/_agent_sql_source.py`
- `src/strap/vendor/rag.py`
- `src/strap/vendor/biosteam_worker.py`
- `src/strap/services/visualization_service.py`
- `src/strap/tools/statistical.py`

Publication-grade recommendation:
- explicitly classify these as `best-effort` boundaries or narrow the exception handling where practical
- avoid presenting them as equally hardened compared with the newer orchestration and tool-envelope layers

## Suggested Publication Gate
Do not call this branch publication-grade until these are true:
1. Full suite is green again.
2. Large-screen BioSTEAM partial batches guarantee a decision-quality minimum or explicitly degrade.
3. Timeout recovery no longer upgrades raw trace prose into normal final answers.
4. Harness strict-routing reporting matches the operational batch evaluator.
5. BioSTEAM solvent catalog exists in one canonical place only.
6. `statistics-ml` adopts the same response helper contract as the rest of the refactored tool surface.

## Recommended Next Fix Order
1. Fix the large-screen BioSTEAM regression.
2. Fix timeout recovery to use validated payloads only.
3. Fix harness `routing_match` semantics.
4. Remove BioSTEAM catalog duplication.
5. Fix `Database` singleton semantics.
6. Normalize `tools/statistical.py`.
7. Clean remaining low-risk classifier/extractor inconsistencies.
