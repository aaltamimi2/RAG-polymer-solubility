# Final Answer Propagation Remediation Plan — 2026-03-07

## Scope
Address the intermittent failure mode where `sep-ps-pet-pc-120` logs a verifier fallback but still reaches the harness with an empty `full_answer` after timeout.

Targets:
1. Instrument middleware ordering and response propagation around [`src/strap/verifier.py`](../src/strap/verifier.py), [`src/strap/routing.py`](../src/strap/routing.py), and the final agent loop / harness boundary in [`architecture/test_harness.py`](../architecture/test_harness.py).
2. If propagation remains brittle, bypass the final orchestrator turn for completed single-specialist separation routes and emit the answer from the completed task result at the agent/harness boundary.
3. After the fix, rerun the full 50-case operational batch from the start with the existing per-case logging and category timeouts.

## Current Evidence

### Confirmed middleware order
From [`src/strap/agent.py`](../src/strap/agent.py) and upstream `deepagents.graph.create_deep_agent(...)`:
- Deepagents installs its default middleware first.
- Our custom middleware is appended afterward.
- LangChain middleware composition is `first in list = outermost`.
- Effective custom wrapper order is:
  1. `RoutingMiddleware`
  2. `OutputVerifierMiddleware`
  3. `StructuredResultExtractorMiddleware`
  4. `SubagentGuardMiddleware`

### Confirmed failing symptom
For failing `sep-ps-pet-pc-120` runs:
- route is correct: only `separation-engineer`
- domain tools ran successfully
- no child LangSmith tool/LLM errors
- verifier logs show deterministic fallback selection
- harness still records `full_answer=""` after timeout

Relevant artifact:
- [`architecture/test_results/operational_eval_20260307_160503_cases/sep-ps-pet-pc-120.md`](../architecture/test_results/operational_eval_20260307_160503_cases/sep-ps-pet-pc-120.md)

### Confirmed non-root causes
This does **not** currently look like:
- routing/classifier failure
- handoff-store loss
- LangSmith trace-capture failure
- harness parsing bug for normal completed runs

### Working hypothesis
The verifier fallback is being produced inside `wrap_model_call(...)`, but one of these is still true:
1. the fallback `ModelResponse` is returned on a non-terminal orchestrator turn and never becomes the final persisted AI message,
2. the loop continues after fallback and later stalls before completion,
3. the final state returned from `agent.invoke(...)` does not retain the middleware-generated fallback when the run times out,
4. the single-specialist routing short-circuit and verifier fallback overlap in a way that leaves the loop without a delivered terminal message.

## Workstream A — Instrumentation
Goal: make the propagation path observable without relying on inference from LangSmith alone.

### A1. Add per-run propagation IDs and terminal-stage logs
Files:
- [`src/strap/verifier.py`](../src/strap/verifier.py)
- [`src/strap/routing.py`](../src/strap/routing.py)
- [`architecture/test_harness.py`](../architecture/test_harness.py)

Changes:
- Tag verifier fallback responses with a unique local marker, for example:
  - `fallback_id`
  - `origin="verifier_fallback"`
  - `route_type="single_specialist_separation"`
- Log when verifier returns:
  - pass-through response
  - revised-model response
  - deterministic fallback response
- Log when routing returns:
  - single-specialist short-circuit response
  - normal model response
- Log the final `messages[-3:]` summary in the harness when a case times out or returns an empty answer.
- If possible, record whether the final AI message content matches the verifier fallback marker.

### A2. Persist propagation diagnostics into batch artifacts
Files:
- [`architecture/test_harness.py`](../architecture/test_harness.py)
- [`architecture/operational_eval_batch.py`](../architecture/operational_eval_batch.py)

Changes:
- Extend `QueryResult` / `CaseResult` with a small `final_answer_diagnostics` payload, such as:
  - `last_ai_message_excerpt`
  - `last_ai_had_tool_calls`
  - `verifier_fallback_seen`
  - `routing_short_circuit_seen`
  - `final_message_count`
- Store this in per-case JSON/markdown.

### A3. Reproduce with a tight focused loop
Run only:
- `sep-ps-pet-pc-120`
- one known clean separation control, e.g. `sep-ps-pvc-below-90`

Acceptance criteria for Workstream A:
- We can tell whether the verifier fallback reached the final returned message list.
- We can distinguish “fallback created but loop continued” from “fallback never persisted”.

## Workstream B — Hard Bypass Path
Goal: stop depending on one more orchestrator model cycle for completed single-specialist separation routes.

### B1. Move answer emission to the completed specialist boundary
Preferred implementation point:
- agent/runtime boundary rather than verifier text replacement alone

Candidate locations:
1. [`src/strap/routing.py`](../src/strap/routing.py) / [`src/strap/routing_guards.py`](../src/strap/routing_guards.py)
2. [`architecture/test_harness.py`](../architecture/test_harness.py) as a temporary evaluation-only safety net

Implementation recommendation:
- First choice: strengthen routing short-circuit so that once a single-specialist separation route is complete, the middleware returns a terminal response derived directly from the validated task result and no further orchestrator synthesis is attempted.
- Second choice, if runtime still proves brittle: add a harness-side fallback that derives `full_answer` directly from the final completed `separation-engineer` task result when:
  - expected route is exactly `['separation-engineer']`
  - no child tool/LLM errors occurred
  - no final AI answer was delivered
  - a validated structured result exists in the returned messages or trace-backed task output

### B2. Keep the bypass narrow
The bypass must apply only when all are true:
- single-specialist route
- specialist is `separation-engineer`
- route is complete
- downstream handoff not required
- there is a validated structured result or deterministic payload fallback

Do **not** generalize this to all routes yet.

### B3. Prefer deterministic answer shaping over another LLM turn
Reuse existing helpers where possible:
- [`src/strap/routing_guards.py`](../src/strap/routing_guards.py) `_build_separation_payload_fallback(...)`
- [`src/strap/verifier.py`](../src/strap/verifier.py) `_build_separation_verifier_fallback(...)`

Unify them if needed so one deterministic source of truth exists for single-specialist separation finalization.

Acceptance criteria for Workstream B:
- `sep-ps-pet-pc-120` cannot end with `full_answer=""` once the separation task completed successfully.
- Final answer delivery does not rely on an extra orchestrator synthesis turn.
- Clean separation-only routes still preserve the current quality checks.

## Workstream C — Regression Tests

### Unit/integration tests
Add tests for:
- verifier fallback marker creation
- routing short-circuit returning deterministic separation final answer
- harness-side emergency fallback only activating on empty final answer + completed single-specialist separation route
- non-separation routes not using the bypass

Candidate files:
- [`tests/test_verifier.py`](../tests/test_verifier.py)
- [`tests/test_routing.py`](../tests/test_routing.py)
- [`tests/test_test_harness.py`](../tests/test_test_harness.py)
- [`tests/test_operational_eval_batch.py`](../tests/test_operational_eval_batch.py)

### Focused live reruns before the full 50
Required reruns:
1. `sep-ps-pet-pc-120`
2. `sep-ps-pvc-below-90`
3. `sep-ldpe-hdpe-pp-atm`

Gate:
- all three must produce non-empty `full_answer`
- route must remain exact
- no new child trace errors

## Workstream D — Full 50 Rerun
Only after A-C pass.

Command:
```bash
PYTHONPATH=src python architecture/operational_eval_batch.py \
  --no-viz \
  --fresh-agent-per-case \
  --retry-on-fail 0 \
  --timeout-s 120 \
  --category-timeouts hsp=90,safety=120,separation=150,biosteam=150,sep-biosteam=210,sep-safety=210
```

Operational requirements:
- run with escalated network access
- keep per-case JSON + markdown artifacts
- keep aggregate JSON + markdown
- keep LangSmith `thread_id` / `run_id` / `trace_id`
- stop early only if a new repeated blocker appears in the first few cases and adds no new signal

## Implementation Order
1. Workstream A instrumentation
2. Focused rerun of `sep-ps-pet-pc-120`
3. Workstream B bypass if A shows propagation is still brittle
4. Local regression suite
5. Focused live reruns of the 3 separation controls
6. Full 50 rerun from the start

## Decision Rule
- If instrumentation shows the verifier fallback reliably reaches final returned messages, fix termination/loop control in the agent stack.
- If instrumentation shows fallback is produced but not reliably delivered, implement the single-specialist separation bypass immediately rather than spending more time on middleware-loop subtleties.

## Success Definition
The work is done when:
- `sep-ps-pet-pc-120` is stable across repeated live reruns
- single-specialist separation routes do not timeout with empty answers after successful tool execution
- the full 50-case batch runs from the start with the current logging and timeout policy
- no regression is introduced in clean HSP, safety, or BioSTEAM-only cases
