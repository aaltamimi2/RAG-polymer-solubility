# `src/strap` Implementation Plan

Date: 2026-03-05
Scope: Convert the `src/strap` review into an execution plan for the `codex/orchestration-handoff-redesign` worktree.

## Current Status

The worktree already contains the highest-priority orchestration redesign that is missing from the main repo:

- append-only scoped handoff storage
- derived downstream handoffs via `build_handoff(...)`
- routing guards for workflow enforcement
- auto-build and auto-dispatch of sequential downstream steps
- scoped/versioned sidecar artifacts

That means the next highest-value work is not more handoff plumbing. The next open items are:

1. module decomposition for large active tool files
2. tool contract consistency
3. separating orchestration assembly from prompt/config sprawl

## Priority Order

### P0: finish and harden orchestration

Status: mostly complete in this worktree

Remaining tasks:
- re-run TEA/LCA sequential routes after each major refactor
- keep missing/invalid handoffs first-class in tests
- avoid regressions in repeated subagent calls and parallel routes

Acceptance criteria:
- focused orchestration tests pass
- `seq-sep-tea` live route still completes end-to-end

### P1: break up the highest-impact active tool monoliths

Status: open

Targets:
1. `src/strap/tools/biosteam_tea_lca.py`
2. `src/strap/tools/visualization.py`
3. `src/strap/tools/advanced_separation.py`

Execution approach:
- move configuration, request-building, and result-shaping into `strap/services/`
- keep `tools/` modules as thin adapters
- preserve public tool names and output schema during refactor

Acceptance criteria:
- tool module gets materially smaller
- new service module has direct unit coverage
- existing tool tests remain green

### P2: standardize tool response contracts

Status: open

Targets:
1. BioSTEAM tools
2. interpolation / HSP tools
3. safety tools

Execution approach:
- every normal tool return should use one envelope shape:
  - `{"display": ..., "data": {...}}`
- predictable error cases should also use the same envelope with `success: false`
- reserve raw exception formatting for true unexpected failures caught by `safe_tool_wrapper`

Acceptance criteria:
- invalid user/tool inputs return structured JSON, not ad hoc `ERROR:` strings
- downstream orchestrator logic can inspect `data.success` reliably

### P3: reduce prompt/config sprawl

Status: open

Targets:
1. `src/strap/agent.py`
2. `src/strap/subagents.yaml`

Execution approach:
- move long prompt sections into dedicated prompt-builder modules
- separate routing graph config from subagent prompt text
- keep output contracts in code, not in YAML examples

Acceptance criteria:
- `agent.py` owns assembly, not large embedded workflow policy
- `subagents.yaml` is split by responsibility

## This Turn

The concrete work for this turn is:

1. capture this implementation plan in the worktree
2. execute P1 + P2 for the BioSTEAM tool family
3. run targeted tests and a live TEA route

## BioSTEAM Refactor Plan

### Changes

- add `src/strap/services/biosteam_service.py`
- move solvent catalog constants, shorthand expansion, config builders, and response helpers there
- update `src/strap/tools/biosteam_tea_lca.py` to call the service module
- normalize predictable error paths to the JSON envelope

### Why this target first

- it is one of the largest active tool files
- it is central to the live `separation-engineer -> biosteam-analyst` route
- it exposes expensive subprocess simulations, so better contracts matter

### Validation

- `tests/test_biosteam_sensitivity.py`
- `tests/test_result_extractor.py`
- `tests/test_routing.py`
- live `architecture/test_harness.py --query seq-sep-tea --no-viz`

## Next After This Turn

1. repeat the same pattern for `tools/visualization.py`
2. extract shared tool-response helpers for HSP/interpolation paths
3. split `agent.py` prompt assembly from runtime wiring
