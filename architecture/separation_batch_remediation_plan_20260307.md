# Separation Batch Remediation Plan

## Scope
This plan addresses the failures exposed by the partial live 50-case run in:
- [operational_eval_20260307_132720.json](/tmp/strap-orchestration-redesign/architecture/test_results/operational_eval_20260307_132720.json)
- [operational_eval_20260307_132720_cases](/tmp/strap-orchestration-redesign/architecture/test_results/operational_eval_20260307_132720_cases)

Current checkpoint from the stopped batch:
- `sep-ldpe-hdpe-pp-atm`: 58.3%, timed out
- `sep-ps-pvc-below-90`: 100%
- `sep-evoh-ldpe-pet-film`: 91.7%, unexpected `visualization-specialist`
- `sep-ps-pet-pc-120`: 58.3%, timed out
- `sep-ps-pmma-pet-120`: 100%

## Root Causes
1. Final-answer provenance is still too weak for separation-only routes.
- The orchestrator can still emit unsupported numeric temperatures, selectivities, and step sequences that are not grounded in the separation payload.
- The verifier catches many of these, but repeated rewrite cycles can consume the full timeout and still end with no final answer.

2. Separation-only routes are not fully clamped to `separation-engineer`.
- `sep-evoh-ldpe-pet-film` routed into `visualization-specialist` even though the query requested only process design.

3. Separation synthesis still allows contradiction-heavy route construction.
- `sep-ps-pet-pc-120` invented a selective multi-step sequence that reused the same solvent/temperature conditions inconsistently.
- `sep-ldpe-hdpe-pp-atm` consumed multiple domain-tool passes and still timed out without producing a bounded final answer.

4. `write_todos` churn is reduced but not eliminated.
- It still appeared in otherwise successful separation cases and adds avoidable turns.

## Workstream A: Final-Answer Provenance Clamp
Goal: make separation final answers fail early unless every operational claim is traceable to the latest validated separation payload.

### Changes
- Add a deterministic provenance check for separation final synthesis.
- Reject final answers that introduce any of these unless present in the payload or quoted tool context:
  - new step temperatures
  - new selectivity numbers
  - new precipitation order / sequence order
  - claims that a polymer remains solid or dissolves when the payload does not support that level of certainty
- Add a rewrite path that forces the orchestrator to summarize only:
  - `best_sequence`
  - `steps`
  - `supported_polymers` / `unsupported_polymers`
  - explicit feasibility caveats
- If the answer still fails after one rewrite, return a constrained fallback summary from payload rather than timing out.

### Target files
- [src/strap/verifier.py](/tmp/strap-orchestration-redesign/src/strap/verifier.py)
- [src/strap/guardrail_checks.py](/tmp/strap-orchestration-redesign/src/strap/guardrail_checks.py)
- [src/strap/routing_progress.py](/tmp/strap-orchestration-redesign/src/strap/routing_progress.py)
- [src/strap/guardrail_messages.py](/tmp/strap-orchestration-redesign/src/strap/guardrail_messages.py)

### Tests
- Add deterministic tests modeled on:
  - `sep-ldpe-hdpe-pp-atm`
  - `sep-ps-pet-pc-120`
- Assert that unsupported temperatures/orderings are rejected before timeout.
- Assert that fallback final synthesis is non-empty even when the verifier blocks stronger claims.

## Workstream B: Separation-Only Route Purity
Goal: keep separation-only prompts on `separation-engineer` unless the user explicitly requests a plot or figure.

### Changes
- Add a route clamp: if the selected case/query category is separation-only and the root user query does not request visualization, block `visualization-specialist` dispatch.
- Prevent separation -> visualization handoff auto-build for separation-only workflows unless the user explicitly asked for a diagram, heatmap, chart, or plot.

### Target files
- [src/strap/routing_guards.py](/tmp/strap-orchestration-redesign/src/strap/routing_guards.py)
- [src/strap/routing_classifier.py](/tmp/strap-orchestration-redesign/src/strap/routing_classifier.py)
- [src/strap/handoff_adapters.py](/tmp/strap-orchestration-redesign/src/strap/handoff_adapters.py)
- [src/strap/routing_handoff_state.py](/tmp/strap-orchestration-redesign/src/strap/routing_handoff_state.py)

### Tests
- Add a regression for `sep-evoh-ldpe-pet-film` with exact expected route `['separation-engineer']`.
- Add one positive control where a separation query explicitly requests a plot and `visualization-specialist` remains allowed.

## Workstream C: Separation Subagent Stop Conditions
Goal: reduce timeout risk by making `separation-engineer` stop once it has decision-quality evidence.

### Changes
- Tighten `separation-engineer` tool budget behavior:
  - once a route-building tool has returned a plausible route, allow at most one targeted verification call
  - block repeated route-building with the same polymer set and same temperature bound
- Tighten `write_todos` handling for separation subagent conversations after the first real domain tool result.
- Prefer a bounded final answer over another verification round when the remaining uncertainty is already explicit in the payload.

### Target files
- [src/strap/guardrails.py](/tmp/strap-orchestration-redesign/src/strap/guardrails.py)
- [src/strap/guardrail_policy.py](/tmp/strap-orchestration-redesign/src/strap/guardrail_policy.py)
- [src/strap/config/subagents/01_separation-engineer.yaml](/tmp/strap-orchestration-redesign/src/strap/config/subagents/01_separation-engineer.yaml)

### Tests
- Recreate the `sep-ldpe-hdpe-pp-atm` shape and assert:
  - no more than one route-builder retry after a feasibility tool
  - no empty final answer when the tool budget is reached
- Add a `write_todos` suppression test after the first separation tool result.

## Workstream D: Rerun Gates
Goal: rerun only after the known blocker class is addressed.

### Gate cases
Run these before restarting the remaining 45 cases:
1. `sep-ldpe-hdpe-pp-atm`
2. `sep-evoh-ldpe-pet-film`
3. `sep-ps-pet-pc-120`
4. `sep-ps-pmma-pet-120`
5. `sep-ps-pvc-below-90`

### Acceptance criteria
- No timeout on the first three gate cases.
- No unexpected `visualization-specialist` on separation-only cases.
- Non-empty `full_answer` for all gate cases.
- No verifier revision-limit hit on the gate cases.
- `sep-ps-pvc-below-90` remains conservative and still passes.

### Resume command
After the gate cases pass, resume the full batch with the same logging setup:
```bash
PYTHONPATH=src python architecture/operational_eval_batch.py \
  --no-viz \
  --fresh-agent-per-case \
  --retry-on-fail 0 \
  --timeout-s 120 \
  --category-timeouts hsp=90,safety=120,separation=150,biosteam=150,sep-biosteam=210,sep-safety=210
```

## Recommended Execution Order
1. Workstream B: route purity
2. Workstream A: final-answer provenance clamp
3. Workstream C: separation stop conditions
4. Gate-case rerun
5. Resume remaining batch
