# Arbitrary N-Step Planning Across Subagents

## Goal

Replace fixed workflow enumeration with a planner that can:

- select any number of relevant subagents for one user request
- order them by actual data dependencies instead of hardcoded route tables
- run independent branches in parallel, capped by runtime policy
- bridge specialists with typed handoffs when available and generic context when not

## Current State

The runtime already has several pieces needed for a graph planner:

- Subagent manifests define names, descriptions, routing hints, and tool groups in `src/strap/config/subagents/*.yaml`.
- The orchestrator stores validated subagent outputs as append-only handoff records in `src/strap/handoffs.py` and `src/strap/handoff_store.py`.
- Pair-specific typed adapters exist in `src/strap/handoff_adapters.py`.
- Generic fallback handoffs already work for any producer -> consumer pair in `src/strap/handoffs.py`.

The limiting factor is that planning still assumes small fixed routes.

## Hardcoded Assumptions Blocking N-Step Planning

### 1. Execution topology is still encoded as pair tables

`src/strap/config/execution_pairs.yaml` models orchestration as:

- `parallel`
- `parallel_3way`
- `sequential`

That is enough for known routes but not for arbitrary workflow construction.

### 2. Query classification is not a planner

`src/strap/routing_classifier.py` can rank matching specialists and order them by inferred dependencies, but it still starts from a relevance match, not a goal-satisfaction search.

Important limits:

- the classifier prompt still says "Return 1-3 subagent names ordered by relevance"
- dependency ordering is based on a preferred-edge table, not a general workflow graph

### 3. Runtime progress logic still asks "what is the next sequential pair?"

These modules still read `SEQUENTIAL_PAIRS` directly:

- `src/strap/routing_progress.py`
- `src/strap/routing_handoff_state.py`
- `src/strap/routing_guards.py`

That means the runtime can only reason about downstream readiness in terms of a single next consumer.

### 4. Handoff typing is pair-specific

`src/strap/handoff_adapters.py` contains exact `(producer, consumer)` adapters.

That is fine for high-value handoffs, but it does not scale as the primary planning model. A graph planner needs contract-level reasoning, not only pair-level functions.

## Target Model

Use a capability graph, not a workflow list.

### Subagent planning metadata

Extend each subagent manifest with optional planning metadata such as:

```yaml
planning:
  goals:
    - separation.route
    - tea.economics
  produces:
    - separation.route.v1
    - solvent.shortlist.v1
  requires:
    - user.polymers
  consumes:
    - separation.route.v1
    - contaminant.screen.v1
    - generic.context.v1
  prefers:
    - typed_handoff
  parallel_group: process
  cost_hint: medium
  latency_hint: high
```

Suggested semantics:

- `goals`: user-facing objectives the subagent can satisfy
- `produces`: normalized output contracts or capability artifacts
- `requires`: data that must already exist before dispatch
- `consumes`: upstream contracts the subagent can use
- `parallel_group`: soft conflict domain for parallel execution
- `cost_hint` / `latency_hint`: planner weights

### Planner state

Represent planning as search over state:

- remaining user goals
- available inputs from the user query
- available upstream artifacts and contracts
- already executed steps
- branch concurrency budget

### Search strategy

Use deterministic best-first search over candidate actions:

1. Extract requested goals from the query.
2. Seed available inputs from the query itself.
3. Expand candidate subagents whose `requires` are satisfied.
4. Score candidates by:
   - number of newly satisfied goals
   - typed-handoff compatibility
   - lower latency / lower cost when equivalent
   - avoiding redundant domains
5. Build a DAG, not just a list.

This should prefer typed contracts, but allow generic context edges as a fallback with a worse score.

## Runtime Model

The planner output should be a workflow graph:

```json
{
  "steps": [
    {
      "step_id": "step_sep",
      "subagent": "separation-engineer",
      "depends_on": [],
      "satisfies": ["separation.route"],
      "produces": ["separation.route.v1"]
    },
    {
      "step_id": "step_contam",
      "subagent": "contaminant-removal-analyst",
      "depends_on": ["step_sep"],
      "consumes": ["separation.route.v1"],
      "produces": ["contaminant.screen.v1"]
    },
    {
      "step_id": "step_tea",
      "subagent": "biosteam-analyst",
      "depends_on": ["step_contam"],
      "consumes": ["contaminant.screen.v1"],
      "satisfies": ["tea.economics"]
    }
  ]
}
```

Runtime helpers should then derive:

- ready steps: dependencies complete
- required handoffs: which producer result must be adapted next
- dispatch ordering: sequential or parallel, up to concurrency cap
- completion: all required goals satisfied

## Typed vs Generic Handoffs

Keep the current typed adapters, but demote them from "workflow definition" to "edge optimization."

Recommended contract strategy:

- Typed adapters remain for high-value routes where the downstream tool expects a specific shape.
- Generic context remains available everywhere.
- The planner should know whether an edge is:
  - `typed_required`
  - `typed_preferred`
  - `generic_allowed`

This avoids the combinatorial explosion of defining every pair manually.

## Migration Path

### Phase 1. Add planning metadata without changing runtime behavior

- Extend subagent YAML schema with optional `planning`.
- Add a loader that builds a subagent capability graph.
- Add tests that validate metadata coverage and graph construction.

### Phase 2. Replace pair-table reads with graph dependency reads

- Replace direct `SEQUENTIAL_PAIRS` usage in progress and handoff state helpers.
- Compute predecessors from the active workflow DAG instead of from global pairs.

### Phase 3. Separate routing from planning

- Keep routing for "which domains are relevant?"
- Add a planner for "what workflow satisfies the full request?"
- Allow more than 3 total planned steps while keeping the existing max concurrent dispatch cap.

### Phase 4. Add plan search

- Translate query intents into normalized goals and constraints.
- Search the capability graph for the lowest-cost satisfying plan.
- Prefer typed edges and direct-goal steps.

### Phase 5. Improve plan quality

- Add edge weights from eval outcomes.
- Penalize unreliable or slow branches.
- Add cycle control and limited refinement loops.

## First Code Changes To Make

The smallest useful implementation slice is:

1. Create a planning-graph module that loads subagent planning metadata and exposes:
   - planner nodes
   - capability edges
   - fallback generic edges
2. Convert progress and handoff readiness logic to consume an explicit workflow DAG.
3. Keep the current router, but let it hand off a larger candidate specialist set to the planner.

This preserves most of the runtime while removing the need to enumerate workflows.

## Risks

- If planning metadata is too vague, the graph will overconnect and create noisy plans.
- If generic handoffs are treated as equal to typed handoffs, plan quality will drop.
- If query intent extraction stays tied to keyword routing only, the planner will still miss multi-goal requests.
- If the planner allows loops without strict scoring and visit limits, recursion failures will return.

## Recommendation

Implement the planner in two layers:

- `routing`: identify candidate domains from the user query
- `planning`: search a capability graph to build an executable DAG

That keeps current routing behavior intact while replacing the fixed-route assumption where it matters most.
