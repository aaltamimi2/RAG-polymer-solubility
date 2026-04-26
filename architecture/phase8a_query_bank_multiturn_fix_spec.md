# Phase 8A Query-Bank Multi-Turn Fix Spec

Status: proposed next implementation slice

Source smoke run:

- Seed row: `docs/subagent_query_bank-v1.xlsx`, `08 statistics-ml`, row `3`
- Mode: `DISSOLVE_TYPED_PLANNER=enforce_selected`
- Transcript: `architecture/test_results/query_bank_phase7_chat_20260426_112814/transcript.json`
- Summary: `architecture/test_results/query_bank_phase7_chat_20260426_112814/summary.md`

## Problem

The Phase 7 selected runtime works for the current selected artifact request, but follow-up turns can still leave the typed contract boundary.

The first query-bank chat smoke showed three failure modes:

1. A referential artifact question was compiled as a new selected HSP heatmap request.
   - User: "Where did that RED heatmap get saved?"
   - Actual: typed compile failure because no polymers or solvents were present.
   - Expected: answer from the prior typed runtime ledger or manifest.

2. The direct safety fast path intercepted an explicit HSP support query.
   - User: "Can the HSP model handle GVL for PET, or is GVL unsupported?"
   - Actual: `compare_solvent_safety_cards` on bogus solvent strings.
   - Expected: HSP/RED support check, typed HSP unsupported response, or a statistics/ML route. Never safety.

3. A final summary over recent typed artifacts fell into legacy routing and hallucinated unrelated separation/safety content.
   - User: "Summarize what we learned from the two heatmap requests and the GVL check. Include any file paths you know about."
   - Actual: legacy specialist dispatch, unsupported LDPE/EVOH/PP separation claims, invented paths.
   - Expected: artifact-led summary using the verified typed runtime ledgers and the actual HSP paths.

## Goals

- Preserve the Phase 7 selected runtime behavior for new selected artifact requests.
- Add a deterministic, domain-general follow-up path for questions about recent typed runtime artifacts.
- Prevent any direct fast path from claiming queries with explicit conflicting domain markers.
- Make artifact summary follow-ups read from verified ledger artifacts, not legacy prose.
- Add a reusable query-bank multi-turn harness so this failure class is regression-tested before expanding to more rows.

The implementation must not be HSP-specific. HSP/RED is the smoke-test case
that exposed the defect, but the same mechanism must apply to typed artifacts
from safety, BioSTEAM TEA/LCA, separation visualization, optimization/Pareto,
sidecar files, and future selected workflows such as contaminant-removal or
research artifacts when those domains are added to the typed runtime.

## Non-Goals

- Do not enable broad `DISSOLVE_TYPED_PLANNER=enforce`.
- Do not replace LangGraph or the whole routing stack.
- Do not add model replanning.
- Do not make generic wrappers mint selected production artifacts.
- Do not remove valid direct fast paths for unambiguous safety, solubility, and lookup requests.

## Target Behavior

### Artifact Path Follow-Up

Given a prior selected typed-runtime response with:

- `additional_kwargs["strap_origin"] == "typed_runtime"`
- `additional_kwargs["strap_runtime_progress"]["produced_artifact_paths"]`
- optional `additional_kwargs["strap_manifest"]["produced_file_copies"]`

Then a follow-up like:

- "Where did that RED heatmap get saved?"
- "Give me the exact path."
- "Was that typed runtime or legacy?"
- "Open/list the diagnostic bundle for that plot."

must return an answer from the prior typed runtime metadata without compiling a new HSP request.

If multiple recent artifacts match the noun phrase, prefer the most recent artifact whose type or path name matches the phrase. If ambiguous, list the recent matching artifacts rather than guessing.

This behavior is required for all typed artifact families, for example:

- "Where did that RED heatmap get saved?"
- "Where is the BioSTEAM TEA/LCA plot?"
- "Give me the path for the Pareto frontier plot."
- "What diagnostic bundle contains the separation tree?"
- "Where did the safety comparison get written?"

### Explicit Domain Override For Direct Fast Path

The direct fast path must not classify a query as safety when the query explicitly asks about HSP/RED/Hansen model support or compatibility.

Examples that must not route to safety:

- "Can the HSP model handle GVL for PET, or is GVL unsupported?"
- "Use HSP to predict LDPE compatibility with dodecane, and flag ambiguity."
- "Does the Hansen model support gamma-valerolactone for PET?"

This should be implemented as a domain-conflict guard before safety fast-path extraction, not by adding a wording-specific exception for only `GVL`.

The guard design should be reusable across fast paths. HSP-vs-safety is the
first production rule, but the API should support later conflicts such as:

- BioSTEAM TEA/LCA vs generic optimization wording
- separation visualization vs solubility plotting
- Pareto optimization vs single objective optimization
- contaminant-removal analysis vs generic solvent lookup

### Artifact-Led Summary Follow-Up

Given recent typed runtime artifacts, a follow-up like:

- "Summarize what we learned from the two heatmap requests."
- "Summarize the typed-runtime outputs and include paths."
- "What plots did we generate so far?"

must generate a summary from typed runtime ledgers/manifests first.

It must not dispatch unrelated specialists unless the user explicitly asks for a new domain analysis.

The summary should include:

- artifact type
- source step id when available
- original produced path
- diagnostic copy path when available
- concise structured payload facts when available, such as:
  - HSP category, solvent polarity, number of rows, or warning/error code
  - safety solvent names and operating temperature
  - BioSTEAM MSP, TCI, AOC, GWP, scenario, and target plastic
  - separation state-map/tree/selectivity artifact paths and source step ids
  - Pareto objective metrics, solved slices, and frontier point counts
- typed runtime status for each referenced turn

## Proposed Design

### 1. Add Typed Runtime Follow-Up Resolver

New module:

```text
src/strap/planning/typed_runtime_followups.py
```

Suggested API:

```python
class TypedRuntimeFollowupDecision(PlanningModel):
    should_answer: bool
    reason: str
    response_text: str | None = None
    matched_artifacts: list[ArtifactFrame] = []
    matched_manifests: list[dict[str, Any]] = []


def maybe_answer_typed_runtime_followup(
    query: str,
    messages: list[Any],
) -> TypedRuntimeFollowupDecision:
    ...
```

The resolver should:

- inspect recent `AIMessage.additional_kwargs`
- collect typed runtime progress, manifest, plan id, workflow id, ledger, and produced paths
- optionally load `artifacts.json`, `ledger.json`, and `manifest.json` from diagnostic bundles when present
- classify only follow-up/status/path/summary requests, not new domain requests
- return `should_answer=False` for genuinely new selected requests

Detection should be semantic enough to cover:

- path/status words: `where`, `saved`, `path`, `file`, `diagnostic`, `bundle`, `typed runtime`, `legacy`
- referential words: `that`, `this`, `those`, `previous`, `last`, `generated`
- summary words: `summarize`, `recap`, `what did we learn`, `plots generated`

The resolver should not answer a request with new required inputs, such as:

- "Generate another heatmap for nylons..."
- "Run BioSTEAM for..."
- "Create a new plot..."

Artifact matching must be capability/artifact-type aware, not hardcoded only to
HSP. Minimum initial aliases:

- `hsp_red_heatmap`, `hsp_single_pair_summary`: hsp, hansen, red, heatmap, compatibility
- `solvent_safety_card`, `solvent_safety_comparison`: safety, card, comparison, flash, boiling
- `biosteam_tea_lca_result`, `biosteam_tea_lca_plot`: biosteam, tea, lca, msp, tci, aoc, gwp
- `separation_dp_state_map`, `separation_tree_plot`, `separation_selectivity_heatmap`: dp, state map, tree, selectivity, separation
- `optimization_pareto_front`, `optimization_pareto_landscape`, `optimization_pareto_slices`, optimization plot artifacts: pareto, frontier, optimization, slices
- `sidecar_file`: sidecar, json, data file

### 2. Wire Resolver Before New Typed Compilation

Update:

```text
src/strap/planning/typed_runtime_integration.py
```

In `TypedRuntimeMiddleware.wrap_model_call()` and `awrap_model_call()`:

1. Extract query.
2. Call `maybe_answer_typed_runtime_followup(query, request.messages)`.
3. If it returns an answer, return a `ModelResponse` with:
   - `strap_origin="typed_runtime_followup"`
   - `strap_typed_runtime_status="answered_from_prior_artifacts"`
   - `strap_runtime_progress` copied or summarized from matched artifacts
4. Otherwise proceed to `maybe_run_typed_runtime(...)`.
5. Only if typed runtime returns `None`, call the legacy handler.

This avoids selected compile failures for path/status follow-ups.

### 3. Add Direct Fast-Path Domain Conflict Guards

Update:

```text
src/strap/direct_fast_path.py
```

Add a guard similar to:

```python
_HSP_DOMAIN_RE = re.compile(r"\b(HSP|RED|Hansen|Hansen model|compatib(?:le|ility))\b", re.I)

def _conflicts_with_safety_fast_path(user_request: str) -> bool:
    return bool(_HSP_DOMAIN_RE.search(user_request))
```

Before safety-card matching or safety solvent extraction, skip the safety fast path if the conflict guard is true.

The guard should be conservative: it should block direct safety only when the query explicitly asks about HSP/RED/Hansen compatibility/support. It should not block normal safety questions like "How should I safely heat toluene to 110 C?"

Use a shared registry-style helper so future fast-path conflicts can be added
without scattering one-off wording patches through individual routes.

### 4. Add Ledger-Backed Summary Formatter

The follow-up resolver should use a formatter for recent typed artifacts:

```python
def format_typed_artifact_summary(
    artifacts: list[ArtifactFrame],
    manifests: list[dict[str, Any]],
    *,
    include_payload_facts: bool = True,
) -> str:
    ...
```

For HSP heatmaps, include:

- `artifact_type=hsp_red_heatmap`
- plot path
- diagnostic copy path
- count of `results`
- `polymer_resolution` and `solvent_resolution` categories when available
- unsupported/warnings if present

For typed failures, include:

- failed phase
- missing inputs or failed checks
- diagnostic bundle path

No specialist subagent should be invoked for this path.

For non-HSP artifacts, include the same provenance fields and any concise
payload facts recognized by the formatter. If a payload schema is not yet
domain-specialized, fall back to artifact type, source step id, output paths,
diagnostic copies, and typed runtime status rather than routing to legacy prose.

### 5. Add Query-Bank Multi-Turn Harness

New optional harness:

```text
architecture/query_bank_chat_harness.py
```

Capabilities:

- load `docs/subagent_query_bank-v1.xlsx`
- select a row by sheet + row number, query text, or index
- run a seed turn plus scripted follow-ups in one persistent `thread_id`
- support env-controlled mode, defaulting to `DISSOLVE_TYPED_PLANNER=enforce_selected`
- save:
  - `transcript.json`
  - `summary.md`
  - per-turn origin/status/tool/subagent counts
  - typed runtime progress and manifest data
  - produced file paths and diagnostic copies

This harness is for local operational evaluation and should not require live model calls in CI.

## Required Tests

### Unit Tests

Add tests for `typed_runtime_followups.py`:

- path follow-up returns the prior HSP heatmap path from fake typed-runtime metadata
- diagnostic bundle follow-up returns manifest path and copied artifact path
- summary follow-up formats two recent HSP heatmaps without calling legacy handler
- new artifact request does not get swallowed by follow-up resolver
- no prior typed artifacts returns `should_answer=False`

Add tests for `direct_fast_path.py`:

- HSP/GVL support query does not route to safety direct fast path
- HSP compatibility query with `handle` does not route to safety
- normal safety query still routes to safety
- normal safety comparison still routes to safety

Add middleware tests:

- `TypedRuntimeMiddleware` answers a prior-artifact path follow-up without invoking `handler`
- `TypedRuntimeMiddleware` falls through for a new selected HSP heatmap request
- follow-up response metadata contains `strap_origin="typed_runtime_followup"`

### Operational Smoke

Rerun the same five-turn smoke:

1. Query-bank row: `Use the Hansen model to screen polyolefins against nonpolar solvents and show the RED heatmap.`
2. `Where did that RED heatmap get saved? Give me the exact path and tell me whether it came from typed runtime or legacy routing if you can see that.`
3. `Screen nylons against polar aprotic solvents using HSP and return the batch compatibility heatmap. Save it to ".../hsp_followup".`
4. `Can the HSP model handle GVL for PET, or is GVL unsupported? Do not substitute another solvent.`
5. `Summarize what we learned from the two heatmap requests and the GVL check. Include any file paths you know about.`

Expected:

- Turn 1: `typed_runtime / executed`
- Turn 2: `typed_runtime_followup / answered_from_prior_artifacts`
- Turn 3: `typed_runtime / executed`
- Turn 4: not `direct_tool_fast_path` safety; route to typed HSP/statistics path or produce an HSP unsupported diagnostic
- Turn 5: `typed_runtime_followup / answered_from_prior_artifacts`
- no separation/safety subagent dispatch in Turn 5
- no invented file paths

## Definition Of Done

- The five-turn HSP query-bank chat smoke passes with artifact-led follow-up behavior.
- The safety direct fast path no longer captures explicit HSP/RED/Hansen support queries.
- Follow-up path/status questions over typed runtime artifacts do not become selected compile failures.
- Summary follow-ups over recent typed artifacts do not dispatch unrelated specialists.
- Transcript and summary outputs are persisted by the new harness.
- Full test suite remains green.

## Follow-On Work

After this fix, expand the query-bank chat harness to:

- safety card/comparison rows
- BioSTEAM TEA/LCA rows
- separation tree/selectivity visualization rows
- direct and routed Pareto optimization rows
- multi-slice Pareto rows

Each expansion should include at least one artifact path follow-up and one summary follow-up.
