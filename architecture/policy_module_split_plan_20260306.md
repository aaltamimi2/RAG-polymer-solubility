# Policy Module Split Plan — 2026-03-06

## Scope
Target the three remaining large policy modules in `src/strap`:
- `routing.py`
- `guardrails.py`
- `handoffs.py`

Goal: move each toward smaller, testable submodules without changing the public orchestration API.

## 1. routing.py
Current size: 319 lines

### Target split
1. `routing_classifier.py` (`done`, 275 lines)
- routing rule loading
- classifier prompt construction
- LLM classifier path
- keyword classifier path
- match normalization for ambiguous queries
- advisory hint construction
- routing table generation

2. `routing_progress.py` (`done`, 881 lines)
- task dispatch extraction
- task return / failure / handoff status inference
- ordered-plan construction
- progress directives
- completion synthesis anchor

3. `routing_guards.py` (`done`, 662 lines)
- write-todos guards
- filesystem guards
- workflow/tool validation
- ready-handoff and pending-handoff helpers

### Execution order
- Completed: `routing_classifier.py`
- Completed: `routing_progress.py`
- Completed: `routing_guards.py`
- Leave `RoutingMiddleware` orchestration shell in `routing.py`

## 2. guardrails.py
Current size: 585 lines

### Target split
1. `guardrail_utils.py` (`done`, 283 lines)
- content coercion
- structured-result parsing
- temperature limit parsing / temperature mention matching
- visualization tool directive parsing
- BioSTEAM batch normalization helpers
- tool-envelope parsing helpers

2. `guardrail_checks.py` (`done`, 358 lines)
- separation-feasibility checks
- temperature-bound checks
- unsupported-polymer scope checks
- support-coverage inference helpers

3. `guardrail_messages.py` (`done`, 122 lines)
- repair prompt fragments
- synthesis directives
- reusable user-facing guard messages

### Execution order
- Completed: `guardrail_utils.py`
- Completed: `guardrail_checks.py`
- Completed: `guardrail_messages.py`
- Keep `SubagentGuardMiddleware` class in `guardrails.py`

## 3. handoffs.py
Current size: 220 lines

### Target split
1. `handoff_adapters.py` (`done`, 322 lines)
- typed producer→consumer adapters
- visualization-intent inference
- supported/unsupported polymer partitioning

2. `handoff_models.py` (`done`, 114 lines)
- `HandoffScope`
- `HandoffRecord`
- artifact extraction helpers

3. `handoff_store.py` (`done`, 478 lines)
- scope binding / cleanup
- append-only record storage
- list/get latest result helpers

### Execution order
- Completed: `handoff_adapters.py`
- Completed: `handoff_models.py`
- Completed: `handoff_store.py`
- Keep compatibility entry points in `handoffs.py`

## Verification Gate Per Extraction
For every extraction:
1. `python3 -m compileall` for touched modules
2. focused pytest slice for the affected subsystem
3. one live route if the extraction touches active orchestration behavior

## Completed In This Worktree
1. `routing_classifier.py`
2. `guardrail_utils.py`
3. `guardrail_checks.py`
4. `handoff_adapters.py`
5. `handoff_models.py`

## Remaining Execution Order
- Policy-module split plan complete

## Current Verification Status
- Focused split regressions passed after each extraction
- Full local suite now passes after the follow-on tool/module splits as well: `296 passed, 1 warning`
- Follow-on module reductions completed on the same green baseline:
  - `advanced_separation.py` reduced to 567 lines
  - `separation_planning_tools.py` reduced to a 23-line compatibility re-export
  - new dedicated tool modules:
    - `sequence_planning_tools.py` (930)
    - `sequence_analysis_tools.py` (879)
    - `separation_visualization_tools.py` (247)
    - `precipitation_analysis.py` (621)
- Live harness validation is still intermittently blocked by external model/runtime stalls; the most recent successful artifact remains the authoritative live check, and local regression coverage remains green
