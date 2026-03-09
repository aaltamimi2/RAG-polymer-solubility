# Operational Eval Remediation Plan — 2026-03-07

## Goal
Address the five failures surfaced by the live operational-eval campaign and then rerun the full 50-prompt benchmark with timeouts matched to category cost.

## Priority order
1. Fix BioSTEAM energy-case regression
2. Clamp top-level separation tool bypass
3. Force final synthesis when separation/BioSTEAM domain data already exists
4. Add synthesize-on-tool-budget-exhaustion repair for safety-analyst and biosteam-analyst
5. Re-run the 50-prompt benchmark with category-specific timeouts

Reason for this order:
- The BioSTEAM regression is a hard correctness bug and can poison downstream evaluation.
- Separation bypass weakens route discipline and can invalidate later synthesis fixes.
- Final-synthesis and budget-exhaustion fixes should be done after routing is deterministic enough to trust the traces.
- The 50-prompt rerun should be the validation gate, not the debugging tool.

---

## Track 1 — Fix `BaselineSTRAPProcess.B` regression

### Problem
`tea-pe-energy-cases` produced a real runtime failure:
- trace: `019cc91a-14af-7791-8cba-9eac00242110`
- symptom: `AttributeError: 'BaselineSTRAPProcess' object has no attribute 'B'`

This is not a harness issue. It is a BioSTEAM worker/runtime bug.

### Likely touch points
- [biosteam_worker.py](/tmp/strap-orchestration-redesign/src/strap/vendor/biosteam_worker.py)
- [biosteam_runner.py](/tmp/strap-orchestration-redesign/src/strap/vendor/biosteam_runner.py)
- [biosteam_tea_lca.py](/tmp/strap-orchestration-redesign/src/strap/tools/biosteam_tea_lca.py)
- [biosteam_service.py](/tmp/strap-orchestration-redesign/src/strap/services/biosteam_service.py)

### Implementation steps
1. Reproduce the failing path directly with the same energy-case comparison prompt or the underlying batch API.
2. Inspect where the worker assumes `.B` exists on `BaselineSTRAPProcess` for C2/C3 or scenario comparisons.
3. Replace raw attribute access with version-safe accessors or guarded scenario-specific logic.
4. Add a regression test covering multi-energy-case comparison for at least one solvent.
5. Re-run the direct BioSTEAM energy-case prompt after the patch.

### Acceptance criteria
- `tea-pe-energy-cases` no longer throws `BaselineSTRAPProcess.B`.
- The tool returns structured output for C1/C2/C3 comparisons.
- No new failures in existing BioSTEAM tests.

### Tests
- existing BioSTEAM test suite
- new targeted regression for multi-energy-case batch path
- live rerun of `tea-pe-energy-cases`

---

## Track 2 — Clamp top-level separation tool bypass

### Problem
Some separation queries still execute top-level separation tools directly instead of delegating to `separation-engineer`.
Example:
- `sep-ps-pvc-below-90`
- trace: `019cc910-f1da-7e33-885d-44fdc147d20e`
- tools: `rank_solvents_selectivity`, `rank_solvents_selectivity`
- no executed `separation-engineer` subagent

### Likely touch points
- [routing.py](/tmp/strap-orchestration-redesign/src/strap/routing.py)
- [routing_guards.py](/tmp/strap-orchestration-redesign/src/strap/routing_guards.py)
- [routing_progress.py](/tmp/strap-orchestration-redesign/src/strap/routing_progress.py)
- possibly [agent.py](/tmp/strap-orchestration-redesign/src/strap/agent.py) if free-tool policy needs to tighten further

### Implementation steps
1. Identify which top-level separation tools are still reachable in orchestrator state for separation-design queries.
2. Add a hard router guard: if the active route requires `separation-engineer`, block top-level separation-analysis tools and emit a correction message that requires `task(subagent_type="separation-engineer", ...)`.
3. Keep narrow exceptions only for explicitly single-tool statistics/HSP workflows.
4. Add tests for:
   - blocked top-level `rank_solvents_selectivity` on separation-design prompts
   - allowed HSP-only tool usage on statistics-ml prompts
5. Re-run the known bypass case live.

### Acceptance criteria
- Separation-design prompts delegate to `separation-engineer`.
- Top-level separation tool drift is blocked with router-guard messages.
- HSP-only prompts are not regressed.

### Tests
- routing guard unit tests
- live rerun: `sep-ps-pvc-below-90`
- live rerun: one HSP-only prompt to confirm no false block

---

## Track 3 — Force final synthesis once domain data exists

### Problem
Many cases execute the right route and tools but end with no top-level final answer before timeout.
Representative traces:
- `019cc910-2879-74c1-b4af-371d63343359`
- `019cc912-7dce-7540-8f65-4f9cb5b52235`
- `019cc917-bda4-7650-a95f-422208451b08`
- `019cc91d-2812-7781-9aa4-7305a9d48391`

Pattern:
- routed specialist executes
- domain tool data exists
- no final orchestrator synthesis is emitted in time

### Likely touch points
- [routing_progress.py](/tmp/strap-orchestration-redesign/src/strap/routing_progress.py)
- [routing_guards.py](/tmp/strap-orchestration-redesign/src/strap/routing_guards.py)
- [routing.py](/tmp/strap-orchestration-redesign/src/strap/routing.py)
- [result_extractor.py](/tmp/strap-orchestration-redesign/src/strap/result_extractor.py)

### Implementation steps
1. Define a generic `ready_to_finalize` condition from validated results:
   - separation: latest valid `separation-engineer` result exists
   - biosteam: latest valid `biosteam-analyst` result exists
   - sequential route: all expected routed specialists have valid or explicitly missing results
2. Add a router-level guard that, once `ready_to_finalize` is true, blocks further domain-tool exploration and injects a forced synthesis instruction.
3. If the model still tries more tool calls after `ready_to_finalize`, return error `ToolMessage`s until it synthesizes.
4. Add a fallback path that synthesizes from validated payloads when the route is complete but the model attempts to stop without emitting a user-facing answer.
5. Add focused tests for:
   - separation route completes -> final answer required
   - BioSTEAM route completes -> final answer required
   - completed sequential route cannot keep exploring

### Acceptance criteria
- Successful separation and BioSTEAM routes produce a non-empty final answer before timeout in smoke runs.
- Empty `full_answer` is no longer the dominant failure mode for successful tool traces.

### Tests
- routing/guard integration tests
- live reruns:
  - `sep-ldpe-hdpe-pp-atm`
  - `tea-pe-toluene-c1`
  - `septea-evoh-pe-film`

---

## Track 4 — Synthesize on tool-budget exhaustion for safety and BioSTEAM

### Problem
At least one mixed route hit tool-budget exhaustion and then failed to synthesize:
- `sepsafe-ps-over-pvc`
- earlier smoke: safety route exhausted budget and produced no usable final answer

This is a subagent closure problem, not just a routing problem.

### Likely touch points
- [guardrails.py](/tmp/strap-orchestration-redesign/src/strap/guardrails.py)
- [guardrail_policy.py](/tmp/strap-orchestration-redesign/src/strap/guardrail_policy.py)
- [02_safety-analyst.yaml](/tmp/strap-orchestration-redesign/src/strap/config/subagents/02_safety-analyst.yaml)
- [03_biosteam-analyst.yaml](/tmp/strap-orchestration-redesign/src/strap/config/subagents/03_biosteam-analyst.yaml)

### Implementation steps
1. Extend subagent guard middleware so that when `max_tool_calls` is reached for `safety-analyst` or `biosteam-analyst`, the model is given one final forced synthesis turn instead of just a hard stop.
2. Base that synthesis directive on the tool outputs already gathered in the subagent state.
3. Ensure the repair path still enforces `<STRUCTURED_RESULT>`.
4. Add tighter stop conditions where possible:
   - safety: once all requested solvents have GSK + PubChem coverage, synthesize immediately
   - biosteam: once required TEA/LCA metrics exist, synthesize immediately
5. Add tests for budget-exhaustion -> successful synthesis.

### Acceptance criteria
- Budget exhaustion yields a final synthesized answer instead of an empty result.
- Safety and BioSTEAM traces do not end in “promise to continue later” behavior after budget exhaustion.

### Tests
- guardrail middleware tests
- live reruns:
  - `safety-toluene-dmso-thf`
  - `sepsafe-ps-over-pvc`
  - one BioSTEAM batch comparison prompt

---

## Track 5 — Re-run the 50-prompt benchmark with category-specific timeouts

### Problem
The coarse 45s timeout worked as a stress tool but undercounts routes that were executing correctly and only lacked enough time to synthesize.

### Likely touch points
- [operational_eval_batch.py](/tmp/strap-orchestration-redesign/architecture/operational_eval_batch.py)
- [operational_eval_findings_20260307.md](/tmp/strap-orchestration-redesign/architecture/test_results/operational_eval_findings_20260307.md)

### Implementation steps
1. Add default timeout mapping by category:
   - `hsp`: 30-45s
   - `safety`: 30-45s
   - `separation`: 60s
   - `biosteam`: 60-90s
   - `sep-biosteam`: 90s
   - `sep-safety`: 90s
2. Allow explicit override via CLI, but use category defaults when not provided.
3. Re-run a smoke slice first:
   - one separation
   - one BioSTEAM
   - one HSP
   - one safety
   - one mixed route
4. If smoke improves, run the full 50 and generate a new findings markdown.

### Acceptance criteria
- The rerun yields a materially better completion rate than the 45s stress batch.
- Final artifact clearly separates:
   - route failures
n   - tool/runtime failures
   - synthesis failures
   - verifier failures

### Tests
- runner unit tests for timeout mapping
- live smoke slice
- full 50-prompt rerun

---

## Execution gate
Do not rerun the full 50 until Tracks 1-4 are complete.

Minimum gate before Track 5:
- `tea-pe-energy-cases` passes
- `sep-ps-pvc-below-90` delegates correctly
- one separation-only route ends with non-empty `full_answer`
- one BioSTEAM-only route ends with non-empty `full_answer`
- one mixed route ends with non-empty `full_answer`

## Success definition
The next full 50-prompt run should tell us something new.
That means:
- fewer empty-final-answer failures
- fewer direct-tool bypasses
- no known `BaselineSTRAPProcess.B` regression
- timeouts become edge cases, not the dominant outcome
