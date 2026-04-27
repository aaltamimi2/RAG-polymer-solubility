# v10 Claude Agent SDK Harness Migration Spec

Status: proposed on `v10-claudesdk`

Branch base: `v10-core` at `f4281c8 Remove transient Excel lock artifact`

Primary goal: evaluate Claude Agent SDK as an alternate DISSOLVE agent harness while preserving the v10 typed-runtime contract, direct fast paths, tool outputs, artifacts, and benchmarkability.

## 1. Context

DISSOLVE v10-core currently uses a LangChain/DeepAgents harness assembled in `src/strap/agent.py`. That harness wires:

- direct deterministic fast paths in `src/strap/direct_fast_path.py`
- opt-in typed runtime middleware in `src/strap/planning/typed_runtime_integration.py`
- legacy routing in `src/strap/routing.py`
- final output verification in `src/strap/verifier.py`
- structured result extraction in `src/strap/result_extractor.py`
- subagent definitions from `src/strap/config/subagents/*.yaml`
- persistent compact session state in `src/strap/session_state.py`

The current v10 typed planning runtime already defines the right control-plane target: compile a request into `RequestPlan`, execute authorized steps with explicit wrappers, verify artifacts, persist ledgers, and synthesize only from verified outputs. The Claude Agent SDK exploration should not discard that work. It should test whether Claude's production agent loop, sessions, hooks, subagents, permissions, MCP tool loading, and structured outputs can simplify or improve the harness around those contracts.

External SDK references used for this spec:

- Claude Agent SDK overview: https://code.claude.com/docs/en/agent-sdk/overview
- Agent loop: https://code.claude.com/docs/en/agent-sdk/agent-loop
- Sessions: https://code.claude.com/docs/en/agent-sdk/sessions
- Custom tools: https://code.claude.com/docs/en/agent-sdk/custom-tools
- MCP: https://code.claude.com/docs/en/agent-sdk/mcp
- Tool search: https://code.claude.com/docs/en/agent-sdk/tool-search
- Structured outputs: https://code.claude.com/docs/en/agent-sdk/structured-outputs
- Subagents: https://code.claude.com/docs/en/agent-sdk/subagents
- Hooks: https://code.claude.com/docs/en/agent-sdk/hooks
- Cost tracking: https://code.claude.com/docs/en/agent-sdk/cost-tracking

## 2. Why Explore This

The existing harness has grown a lot of compensating machinery around model behavior: routing hints, middleware rewrites, final verification retries, handoff reconstruction, session compaction, direct fast paths, and typed-runtime selected enforcement. That machinery is useful, but the LangChain/DeepAgents loop is still awkward in areas that Claude Agent SDK treats as first-class:

- multi-turn sessions are explicit SDK objects or resumable session IDs
- subagents have isolated context and per-agent tool scopes
- hooks can intercept tool use, lifecycle events, compaction, and final stop
- custom tools can be exposed via in-process MCP servers
- tool search can avoid loading every tool schema into every turn
- structured output schemas can validate the final result and retry on mismatch
- ResultMessage includes session ID, usage, cost, stop state, and result subtype

The hypothesis is not that Claude Agent SDK should autonomously replace typed planning. The hypothesis is that it may provide a cleaner loop substrate for DISSOLVE's existing typed control plane and specialist tool surface.

## 3. Goals

- Add a side-by-side Claude Agent SDK harness without removing the current LangChain harness.
- Preserve deterministic direct fast paths for simple lookups, tables, and plots.
- Preserve `RequestPlan`, `ExecutionLedger`, `ArtifactFrame`, and typed-runtime persistence as the authoritative benchmark record.
- Expose DISSOLVE tools to Claude SDK through explicit MCP wrappers with stable schemas.
- Convert YAML subagent specs into Claude SDK `AgentDefinition` objects.
- Use Claude SDK sessions for conversation continuity while keeping DISSOLVE's compact structured session state.
- Use SDK hooks to enforce tool permissions, capture artifacts, block unsafe/unplanned calls, and validate final outputs.
- Provide a benchmark harness that can compare `langchain` vs `claude_sdk` on the same query bank and 15-turn chats.

## 4. Non-Goals

- Do not default the production CLI to Claude SDK in the first implementation.
- Do not remove Gemini or the existing `/model` UI from the LangChain harness.
- Do not expose Bash/Edit/Write to normal DISSOLVE science workflows.
- Do not rely on Claude's free-form subagent prose as a replacement for artifact contracts.
- Do not preload every DISSOLVE tool into the Claude context.
- Do not rewrite scientific tools; wrap the existing functions.
- Do not use SDK session transcripts as the only source of application state.

## 5. Key SDK Capabilities Mapped To DISSOLVE

| SDK feature | DISSOLVE use |
| --- | --- |
| `ClaudeSDKClient` / `query()` | Alternate agent loop for CLI turns and benchmark runs |
| `ClaudeAgentOptions.allowed_tools` | Strict tool allowlist per mode, role, and plan step |
| `permission_mode` | Default to deny-by-default or preapproved MCP-only tools |
| `max_turns`, `max_budget_usd`, `effort` | Hard loop and cost controls per query |
| `ResultMessage` | Capture result subtype, session ID, token usage, cost, stop reason |
| `include_partial_messages` | Render CLI streaming and tool progress without waiting for final synthesis |
| schema-validated final-result flow | Validate final envelopes or compiled plan proposals when that flow is enabled |
| in-process MCP servers | Wrap existing Python DISSOLVE tools without a separate process |
| MCP tool naming | Stable names like `mcp__dissolve_solubility__predict_solubility_range` |
| `ToolSearch` | Avoid dumping 80+ tool schemas into the prompt |
| `AgentDefinition` | Programmatic conversion of DISSOLVE YAML subagents |
| subagent context isolation | Prevent main chat context pollution from exploratory domain calls |
| hooks | Enforce typed-plan authorization, ledger capture, output validation, compaction archival |
| session resume/fork | Real multi-turn continuity plus branchable case-study explorations |

## 6. Proposed Architecture

Add a new package:

```text
src/strap/claude_sdk_harness/
  __init__.py
  runner.py
  options.py
  messages.py
  sessions.py
  tool_catalog.py
  mcp_server.py
  agents.py
  hooks.py
  structured_output.py
  cli_adapter.py
  benchmarks.py
```

### 6.1 Runtime Selection

Add an explicit harness switch:

```bash
DISSOLVE_AGENT_HARNESS=langchain
DISSOLVE_AGENT_HARNESS=claude_sdk
```

Default remains `langchain`. The CLI should also accept:

```bash
dissolve --harness claude_sdk
```

The switch should happen at the outer CLI boundary only. Domain tools and typed-runtime models remain shared. The first implementation should not switch harnesses in the middle of a running chat session; `/harness claude_sdk` and `/harness langchain` are deferred until a context-migration policy is explicitly implemented.

### 6.2 Request Lifecycle

Claude SDK mode should use this order:

1. Parse slash commands and CLI local commands.
2. Run deterministic direct fast path first.
3. If direct fast path answers, update DISSOLVE session state and return without a Claude model call.
4. If typed-runtime selected enforcement applies, compile and optionally execute through existing typed runtime.
5. If typed runtime returns an executed or typed-failure result, emit any artifact frames through the typed runtime and return without a Claude synthesis turn.
6. Otherwise call Claude Agent SDK with constrained tools, compact DISSOLVE session context, and hooks.
7. Capture SDK messages, costs, session ID, subagent events, tool calls, and final result.
8. Update DISSOLVE session state and artifact frames from tool envelopes, hook observations, and structured outputs.

This preserves fast-path latency and prevents Claude SDK from becoming a new broad router for simple one-tool tasks.

### 6.3 Tool Exposure

Do not expose Python functions directly as arbitrary model-side tools. Wrap them as MCP tools in one or more in-process SDK MCP servers.

Recommended servers:

```text
dissolve_solubility
  list_available_solvents
  list_available_polymers
  predict_solubility
  predict_solubility_range
  rank_solvents_selectivity
  plot_solubility_vs_temperature

dissolve_safety
  get_solvent_safety_card
  compare_solvent_safety_cards
  get_solvent_gscore
  get_pubchem_safety_info

dissolve_separation
  plan_multiple_separation_schemes
  plan_sequential_separation
  find_optimal_separation_sequence
  plot_dynamic_programming_separation_options
  create_separation_tree_plot
  create_selectivity_heatmap

dissolve_biosteam
  run_biosteam_simulation
  run_biosteam_batch
  visualize_biosteam_results

dissolve_optimization
  run_waste_management_pareto
  run_waste_management_pareto_slices
  plot_optimization_pareto_front
  plot_optimization_pareto_slices

dissolve_research
  ask_literature
  search_literature_rag
  search_google_scholar
```

Tool wrappers must:

- preserve existing tool JSON envelopes
- return compact text plus machine-readable JSON payloads
- normalize output paths through `src/strap/planning/runtime_paths.py`
- capture `ArtifactFrame` candidates in `PostToolUse` hooks for Claude-driven MCP tool calls
- mark read-only tools where possible so SDK parallelism and tool search can use them safely
- keep plotting and simulation tools non-read-only
- keep Python dict schemas minimal: every listed key is required by the SDK, so optional-heavy tools should omit optional keys from the dict schema, document them in the description, and read them with `args.get()`
- use full JSON Schema only when enums, nested objects, or explicit optional fields are needed
- catch wrapper exceptions and return standard error envelopes with SDK `is_error` set so the agent loop can recover

The first spike should expose only `dissolve_solubility` and `dissolve_safety`. Expand after benchmark evidence.

### 6.4 Tool Name Map

Keep one central `ToolNameMap` in `src/strap/claude_sdk_harness/tool_catalog.py`. `RequestPlan.allowed_tools` and existing typed-runtime contracts use legacy callable names such as `predict_solubility_range`; Claude SDK allowlists and tool calls use fully qualified MCP names such as `mcp__dissolve_solubility__predict_solubility_range`.

The map is authoritative for:

- plan-step allowlist construction
- `PreToolUse` guard checks
- subagent `AgentDefinition.tools`
- benchmark tool-call normalization
- diagnostic records that need both MCP names and legacy callable names

Suggested interface:

```python
class ToolNameMap:
    def mcp_name(self, legacy_name: str) -> str: ...
    def legacy_name(self, mcp_name: str) -> str: ...
    def allowed_for_legacy(self, legacy_name: str) -> list[str]: ...
    def allowed_for_intent(self, intent: str, plan_step: object | None = None) -> list[str]: ...
```

The implementation should fail closed when a plan references a callable with no registered MCP mapping. Do not compare legacy callable names directly against Claude SDK tool names in guards, subagent scopes, hooks, or benchmarks.

Example mappings for the first spike:

```text
list_available_solvents -> mcp__dissolve_solubility__list_available_solvents
predict_solubility_range -> mcp__dissolve_solubility__predict_solubility_range
plot_solubility_vs_temperature -> mcp__dissolve_solubility__plot_solubility_vs_temperature
get_solvent_safety_card -> mcp__dissolve_safety__get_solvent_safety_card
```

### 6.5 Tool Search Policy

Claude docs note that large tool sets consume context and degrade tool selection. Tool search should be enabled for broad agent mode, but disabled for small direct experiments.

Recommended policy:

```bash
ENABLE_TOOL_SEARCH=auto:5
```

Use explicit intent-scoped `allowed_tools` by default. Avoid a global science allowlist that includes every DISSOLVE server; it can recreate the separation-detour behavior this migration is meant to prevent.

```python
allowed_tools = tool_name_map.allowed_for_intent(intent, plan_step=active_plan_step)
```

For plan-step execution, allow only the single required tool or exact server wildcard needed for that step. If `RequestPlan` says the current callable is `predict_solubility_range`, the allowed set should contain `mcp__dissolve_solubility__predict_solubility_range` and should not include separation or optimization tools.

Typed-runtime plan execution should bypass tool search when the active plan step already names an exact callable. If tool search is used for a broader Claude-driven turn and the selected subset omits the tool required by the active plan, `PreToolUse` should block the call, record `tool_search_miss`, and retry once with the exact mapped MCP tool explicitly allowed. If the retry still fails, surface a typed failure rather than letting Claude explore unrelated tools.

### 6.6 Subagent Mapping

Convert each YAML subagent spec into `AgentDefinition`:

```python
AgentDefinition(
    description=spec["description"],
    prompt=spec["system_prompt"] + FILE_IO_DIRECTIVE + THINK_DIRECTIVE,
    tools=[...mapped MCP tool names...],
    model="inherit",
)
```

Rules:

- Include `Agent` only in the top-level allowed tools, not inside subagents.
- Subagents cannot spawn nested subagents, so workflows requiring sequential specialists must be coordinated by the top-level runner or typed plan executor.
- Descriptions must remain concise because Claude uses them for automatic delegation.
- Explicit plan-driven delegation should mention the target agent name in the top-level prompt.
- Subagent final responses must include a structured result block or a schema-constrained result where relevant.
- Parent synthesis must not treat subagent prose as authoritative unless an artifact or structured result supports the claim.
- `AgentDefinition` must only use documented fields such as `description`, `prompt`, `tools`, and `model`. Per-call controls such as `effort`, `max_turns`, and cost limits belong on `ClaudeAgentOptions` or the parent runner, not on each agent definition.

### 6.7 Session Model

Claude SDK sessions are useful but not enough by themselves.

Use two session layers:

1. Claude SDK transcript session:
   - SDK session ID
   - stored by Claude under its project/session directory
   - useful for natural follow-ups and subagent continuity

2. DISSOLVE structured session state:
   - existing `src/strap/session_state.py`
   - stores compact feedstock, solvent candidates, run basis, artifact paths, typed-runtime snapshots
   - robust across harnesses and less sensitive to SDK transcript compaction

Persist a bridge file:

```python
session_paths(thread_id)["dir"] / "claude_sdk_session.json"
```

This preserves the current `src/strap/session_state.py` storage contract: sessions live under `DISSOLVE_SESSION_DIR` when set, otherwise under `~/.dissolve/sessions/<thread_id>/`.

Suggested shape:

```json
{
  "schema_version": "1.0",
  "thread_id": "abc123",
  "claude_session_id": "...",
  "cwd": "/home/aaltamimi2/langchain-STRAP-v10-core",
  "harness": "claude_sdk",
  "harness_profile": "science",
  "model_alias": "claude-sonnet",
  "model_id": "anthropic:<resolved-model-id>",
  "permission_mode": "dontAsk",
  "allowed_tools_fingerprint": "sha256:...",
  "tool_name_map_version": "1",
  "created_at": "...",
  "updated_at": "...",
  "last_result_subtype": "success",
  "last_cost_usd": 0.0,
  "last_usage": {}
}
```

Important constraint: SDK session resume depends on consistent `cwd`; DISSOLVE should validate `cwd` before resuming and fall back to structured session state if the SDK transcript cannot be resumed.

### 6.8 Prompt And Memory

Add a Claude SDK-specific harness prompt, but do not rely on arbitrary package paths being auto-loaded:

```text
src/strap/claude_sdk_harness/prompts/dissolve_sdk.md
```

Load that text explicitly through SDK options:

```python
system_prompt={
    "type": "preset",
    "preset": "claude_code",
    "append": dissolve_sdk_prompt,
}
```

Alternatively, place a project-level `.claude/CLAUDE.md` at repo root and ensure `setting_sources` includes `"project"` whenever settings sources are set explicitly. A `CLAUDE.md` under `src/strap/claude_sdk_harness/` is package data only; it is not SDK project memory unless the runner reads and appends it.

It should be concise and include:

- DISSOLVE role and domain boundaries
- tool-first scientific answering rule
- no unsupported process claims
- exact path and artifact reporting rules
- compaction preservation instructions
- final answer source discipline

Do not load the full v10 typed-planning spec into every SDK session. Use compact memory plus tool descriptions.

### 6.9 Hooks

Implement hooks as the enforcement and observability surface.

Required hooks must be limited to documented SDK hook events:

- `UserPromptSubmit`: inject compact DISSOLVE session context and active typed-plan context.
- `PreToolUse`: block unapproved tools, validate arguments, enforce active plan step, block Bash/Edit/Write in science CLI mode.
- `PostToolUse`: parse tool envelopes, capture artifact frames, update ledger candidates, copy produced files into diagnostic bundles.
- `PreToolUse` with `tool_name="Agent"`: record the selected subagent, parent tool-use ID, and delegated task payload before subagent execution.
- `SubagentStop`: parse subagent structured result, bind result to handoff scope.
- `Stop`: validate final answer against sources, run selected final-contract checks, save session bridge.
- `PreCompact`: archive transcript metadata and ensure compacted memory preserves feedstock, selected solvents, assumptions, artifacts, and current plan.
- `Notification`: record SDK warnings or permission notices as diagnostics when emitted.

Hooks should not silently rewrite scientific outputs. They should either block, attach typed diagnostics, or force a typed failure.

The compact-session injection requirement is not deferred to the full Phase 5 hook implementation. By Phase 3, every non-fast-path SDK turn must receive the current DISSOLVE compact context, either through a minimal `UserPromptSubmit` hook or an equivalent prompt-building adapter. Phase 5 expands hook coverage for enforcement and observability.

Artifact ownership must be explicit:

- Typed-runtime short-circuit paths emit `ArtifactFrame` records directly from the typed runtime ledger, manifest, and runtime wrappers.
- `PostToolUse` hooks emit `ArtifactFrame` records only for Claude-driven MCP tool calls.
- A typed-runtime result should not depend on SDK hooks firing, because no Claude turn or SDK tool call may occur.

### 6.10 Structured Outputs

Use SDK schema-validated result flows for two bounded tasks:

1. Plan proposal:
   - output schema mirrors `RequestPlan`
   - deterministic validators still decide whether the plan is accepted
   - model output is advisory until validated

2. Final answer envelope:
   - human-facing Markdown
   - cited artifact IDs
   - assumptions/defaults disclosed
   - produced file paths
   - unresolved checks
   - compact session facts to persist

Do not require structured outputs for every ordinary conversational response. For simple chat, use direct fast paths or normal result text.

Do not assume a generic free-form `output_format` option. The implementation should use the documented SDK structured-result mechanism available in the pinned SDK version, such as a schema-validated final-result tool or equivalent final-result schema flow, and cover that integration with version-pinned tests.

### 6.11 Permissions

Default normal science mode should remove built-in file/code tools from context and preapprove only the tools required for the detected intent:

```python
tools = []
allowed_tools = tool_name_map.allowed_for_intent(intent, plan_step=active_plan_step)
disallowed_tools = ["Bash", "Edit", "Write"]  # defense in depth if built-ins are later added
permission_mode = "dontAsk"
```

Intent-scoped examples:

```text
solubility_lookup: exact solubility tool names only
safety_lookup: exact safety tool names only
typed_plan_step: exact mapped tool for the active RequestPlan step
complex_workflow: selected domain servers plus Agent, only when explicit workflow complexity requires delegation
```

Benchmark mode should avoid user approval prompts and rely on narrow allowlists. Developer/codebase mode can be a separate harness profile with `Read`, `Grep`, and `Glob`; it should not be mixed with normal DISSOLVE science operation.

Do not use `bypassPermissions` outside isolated tests.

### 6.12 Failure Modes

The harness should expose failure states explicitly instead of falling back to broad agent exploration.

- SDK transport errors such as HTTP 5xx and rate limits should use bounded exponential backoff only before tool execution or after read-only tool calls whose result can be safely retried. After a non-read-only tool call, return an interrupted typed failure unless idempotency is proven.
- Partial stream interruption or missing tool-result frames should mark the turn `interrupted`, persist partial transcript/session metadata, and avoid claiming that files or calculations were produced.
- User cancellation mid-turn should stop streaming, save the DISSOLVE session bridge, and return a `cancelled` status without final synthesis.
- Stop-hook validation failure should surface failed checks and relevant artifact IDs in the CLI. Allow at most one bounded final-synthesis retry when the failure is formatting/source-disclosure only; scientific contradictions or unsupported artifacts should become typed failures.
- Permission or allowlist violations should include the blocked MCP tool name, normalized legacy name when available, active intent, and active plan step.

## 7. CLI UX

The Claude SDK harness should preserve existing DISSOLVE CLI features:

- `/model`
- `/mode`
- energy-case/defaults review screen
- multiline paste handling
- artifact path reporting
- session resume by DISSOLVE thread ID

New commands:

```text
/harness
/claude-session
/claude-fork
/cost
```

`/harness` should show the current harness and explain that switching mid-session is not supported in the first implementation:

```text
Harness: claude_sdk
Claude session: <id>
DISSOLVE thread: <id>
SDK model: claude-sonnet (anthropic:<resolved-model-id>)
Tool search: auto:5
Last cost: $...
Switching: restart with dissolve --harness langchain|claude_sdk
```

`/model` in Claude SDK mode should only show Claude-compatible models. Gemini model aliases remain available in `langchain` mode and must not be passed through the Claude SDK harness.

### 7.1 Claude Model Registry

Add a Claude-specific model registry analogous to the existing CLI model alias registry:

```text
claude-sonnet -> anthropic:<default sonnet model id>
```

The registry should live in `src/strap/claude_sdk_harness/models.py` and resolve model IDs from environment overrides before defaults. Do not hardcode one retired-prone model string throughout the harness.

Required behavior:

- Starting `--harness claude_sdk` with a stale or unavailable Claude alias should fail clearly in noninteractive benchmark mode and offer the default Claude alias in interactive CLI mode.
- Starting `--harness claude_sdk` while the persisted `/model` selection is a Gemini alias should automatically select the default Claude alias, print a short notice, and record both the previous alias and resolved Claude alias in the bridge file.
- Switching back to `langchain` should restore the prior provider-agnostic model selection if available, but this is a new process/session action for v1 rather than an in-chat `/harness` mutation.
- Benchmark records and session bridge files should store both `model_alias` and resolved `model_id`.

## 8. Observability

Capture for every SDK call:

- Claude SDK session ID
- result subtype
- stop reason
- total cost
- usage
- number of turns
- tool calls by MCP name and normalized legacy callable name
- subagent invocations
- parent tool-use IDs for subagent messages
- artifact frames
- diagnostic bundle path

Persist these in the existing runtime manifest shape where possible. If OpenTelemetry is configured, wire SDK events to OTel. Keep LangSmith for current harness comparisons if still available.

## 9. Benchmark Plan

Use the same benchmark set for both harnesses:

- existing `docs/subagent_query_bank-v1.xlsx`
- focused 15-turn solvent/plot follow-up chat
- TEA/LCA energy-case clarification prompts
- separation-to-optimization handoff prompts
- Pareto plot artifact prompts
- simple direct solubility and solvent lookup prompts

Each benchmark record should include:

```json
{
  "query_id": "...",
  "harness": "claude_sdk",
  "model_alias": "claude-sonnet",
  "model_id": "anthropic:<resolved-model-id>",
  "success": true,
  "result_subtype": "success",
  "turns": 3,
  "tool_calls_mcp": ["mcp__dissolve_solubility__list_available_solvents"],
  "tool_calls_legacy": ["list_available_solvents"],
  "subagents": [],
  "artifacts": [],
  "cost_usd": 0.0,
  "latency_s": 0.0,
  "failure_codes": []
}
```

Acceptance thresholds for the first spike:

- `fast_path_preserved`: with normal production dispatch enabled, a direct-fast-path-eligible EVOH solvent lookup should complete with zero Claude model calls and zero Claude MCP tool calls.
- `sdk_tool_invocation`: with direct fast path disabled only for this smoke test, or with a prompt fixture that is not direct-fast-path eligible, the simple EVOH solvent lookup should complete with one DISSOLVE MCP tool call and no subagent.
- Follow-up plot of "these solvents" should preserve context and not route to separation planning.
- Explicit plot request should produce a path and artifact frame.
- Direct fast-path queries should still make zero Claude model calls.
- No Bash/Edit/Write calls in science CLI mode.
- No runaway loop: every query has `max_turns` and `max_budget_usd`.
- Session follow-ups should resolve prior feedstock, solvent candidates, and artifacts.
- Typed-runtime selected workflows should still produce `RequestPlan`, `ExecutionLedger`, and manifest files.

## 10. Implementation Phases

### Implementation Invariants

- Direct fast path remains authoritative and can produce zero-Claude-call answers.
- Claude SDK receives compact DISSOLVE session context on every non-fast-path turn, including turns after earlier fast-path answers.
- Hook events are limited to documented SDK events: `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `PreCompact`, and `Notification`. Subagent-start observation goes through `PreToolUse(tool_name="Agent")`.
- Typed-runtime short-circuit paths emit `ArtifactFrame` records directly; SDK hooks emit artifacts only for Claude-driven MCP tool calls.
- Legacy callable names are translated through one central `ToolNameMap`; guards and benchmarks never compare legacy names directly with MCP names.
- The typed runtime stays MCP-agnostic. Translation to Claude MCP tool names happens only in `src/strap/claude_sdk_harness/tool_catalog.py`.
- `AgentDefinition` objects use only documented fields; `effort`, `max_turns`, and budget controls live on `ClaudeAgentOptions` or parent runner settings.
- Default SDK allowlists are intent-scoped, not global science-domain wildcards.
- Session bridge files live under existing DISSOLVE session paths from `session_paths(thread_id)["dir"]`.
- Harness package prompts are loaded explicitly with `system_prompt` append, or repo-root `.claude/CLAUDE.md` is used with project setting sources enabled.
- Claude SDK mode uses a Claude model registry. A persisted Gemini `/model` selection is not passed to Claude SDK; it is replaced by the default Claude alias with an explicit session notice.
- Mid-session `/harness` switching is disallowed in v1; harness changes happen only at process/session start.

### Phase 0: Branch And Spec

- Status: complete locally; commit/push remains a separate release step.
- `v10-claudesdk` is branched from clean `v10-core`.
- This migration spec is present for review.
- No runtime behavior change.

### Phase 1: Minimal SDK Smoke Harness

Files:

- `src/strap/claude_sdk_harness/runner.py`
- `src/strap/claude_sdk_harness/options.py`
- `tests/test_claude_sdk_harness_options.py`

Changes:

- Add optional dependency `claude-agent-sdk`.
- Put the dependency behind an optional extra such as `project.optional-dependencies.claude`, not the default install path.
- Add `DISSOLVE_AGENT_HARNESS`.
- Implement a no-tool SDK query runner behind an import guard.
- Return clear error if SDK is not installed.

Acceptance:

- Unit tests pass without requiring Anthropic credentials.
- Live smoke test is skipped unless `RUN_LIVE_CLAUDE_SDK=1`.

### Phase 2: In-Process MCP Tool Server

Files:

- `src/strap/claude_sdk_harness/mcp_server.py`
- `src/strap/claude_sdk_harness/tool_catalog.py`
- `tests/test_claude_sdk_tool_catalog.py`

Scope:

- Wrap `list_available_solvents`
- Wrap `predict_solubility_range`
- Wrap `plot_solubility_vs_temperature`
- Wrap `get_solvent_safety_card`
- Implement `ToolNameMap` for legacy-to-MCP and MCP-to-legacy names.

Acceptance:

- Tool schemas are stable and narrow.
- Optional-heavy wrappers do not accidentally make optional parameters required.
- Tool outputs preserve existing JSON envelopes.
- Tool names follow `mcp__dissolve_*__*`.
- `ToolNameMap` fails closed for unmapped callables.
- No tool needs to know about Claude SDK internals.

### Phase 3: Session Bridge

Files:

- `src/strap/claude_sdk_harness/sessions.py`
- `tests/test_claude_sdk_sessions.py`

Scope:

- Store Claude session ID by DISSOLVE thread ID.
- Store bridge metadata at `session_paths(thread_id)["dir"] / "claude_sdk_session.json"`.
- Resume only when `cwd` matches.
- Keep existing DISSOLVE compact session state as the authoritative cross-harness context.
- Inject compact DISSOLVE session context into every non-fast-path SDK turn, including after a previous turn returned through direct fast path.

Acceptance:

- Start, resume, and invalid-cwd fallback are covered by tests.
- A follow-up turn after a fast-path answer can resolve prior feedstock, selected solvents, and artifact paths without relying on Claude transcript memory.

### Phase 4: CLI Adapter

Files:

- `src/strap/claude_sdk_harness/cli_adapter.py`
- `src/strap/claude_sdk_harness/models.py`
- minimal changes to `src/strap/agent.py`

Scope:

- Add `--harness`.
- Add Claude model registry and stale `/model` fallback behavior.
- Preserve `/model`, `/mode`, and review screens.
- Make `/harness` informational only in v1; do not switch harnesses mid-session.
- Stream final text and progress safely.

Acceptance:

- Current CLI still defaults to LangChain.
- `--harness claude_sdk` fails gracefully when dependency or key is missing.
- A persisted Gemini model selection is not used by the Claude SDK harness; the CLI selects the default Claude alias with a notice.

### Phase 5: Hooks And Artifact Capture

Files:

- `src/strap/claude_sdk_harness/hooks.py`
- `src/strap/claude_sdk_harness/messages.py`
- `tests/test_claude_sdk_hooks.py`

Scope:

- Preserve the Phase 3 compact-session injection path as a required hook or adapter behavior.
- PreToolUse allowlist enforcement.
- PostToolUse envelope parsing.
- Stop hook final-answer validation.
- Cost/session capture from `ResultMessage`.

Acceptance:

- Attempted Bash/Edit/Write is blocked.
- Claude-driven MCP tool outputs produce artifact frames when paths are present.
- Typed-runtime selected workflows emit artifact frames through the typed runtime ledger/manifest path, not through SDK hooks.
- Result metadata updates session bridge.

### Phase 6: YAML Subagent Conversion

Files:

- `src/strap/claude_sdk_harness/agents.py`
- `tests/test_claude_sdk_agents.py`

Scope:

- Convert current YAML specs into `AgentDefinition`.
- Map tool groups to MCP tool names.
- Preserve guardrail intent through top-level `ClaudeAgentOptions`, tool scopes, and prompt text; do not add undocumented per-agent fields.

Acceptance:

- All configured subagents load.
- No subagent has `Agent` in its tools.
- Tool names resolve to a registered MCP tool or are explicitly deferred.
- Generated `AgentDefinition` objects contain only documented fields.

### Phase 7: Typed Runtime Integration

Files:

- `src/strap/claude_sdk_harness/structured_output.py`
- selected changes in `src/strap/planning/*`

Scope:

- Option A: keep current typed runtime before Claude SDK and let it short-circuit selected workflows.
- Option B: ask Claude SDK for a structured plan proposal, then run current deterministic validators/executor.

Recommendation: implement Option A first. Add Option B only after side-by-side benchmarks prove the SDK plan proposal is useful.

Acceptance:

- Selected typed workflows still bypass free-form agent synthesis.
- Manifest and ledger formats are unchanged.
- Typed-runtime artifact capture works even when no Claude turn or SDK hook fires.

### Phase 8: Benchmark Harness

Files:

- `src/strap/claude_sdk_harness/benchmarks.py`
- `architecture/claude_sdk_harness_benchmark.py`
- `tests/test_claude_sdk_benchmark_records.py`

Scope:

- Run query-bank examples against either harness.
- Run 15-turn chat transcript against either harness.
- Compare route, MCP tool names, normalized legacy tool names, artifacts, latency, turns, cost, and failures.

Acceptance:

- Benchmark output is JSONL/JSON and suitable for case-study appendices.
- Harness comparison does not require changing prompt text between runs.

## 11. Risks And Mitigations

Risk: Claude SDK autonomous loop recreates the same runaway exploration issues.

Mitigation: preserve direct fast path first, use `max_turns`, `max_budget_usd`, narrow allowed tools, and PreToolUse plan enforcement.

Risk: Tool schema context gets too large.

Mitigation: use MCP server partitioning and tool search; start with only solubility and safety tools.

Risk: Session transcripts become a hidden source of truth.

Mitigation: keep DISSOLVE structured session state and artifact manifests authoritative; use Claude sessions for natural continuity only.

Risk: Subagent final prose is plausible but unsupported.

Mitigation: require structured result blocks, parse artifacts in hooks, and validate final synthesis against artifacts.

Risk: Claude SDK is Claude-only and conflicts with `/model`.

Mitigation: make `claude_sdk` a harness mode with Claude-compatible model choices; keep Gemini in the existing LangChain harness.

Risk: Python SDK session persistence writes under Claude's session directories.

Mitigation: store bridge metadata, validate `cwd`, and provide cleanup/listing commands.

Risk: in-process MCP wrappers hide exceptions or malformed outputs.

Mitigation: every wrapper returns a standard error envelope and tests malformed tool outputs.

Risk: structured-output retry adds latency.

Mitigation: use structured output only for plan proposals and final envelopes in benchmark/typed workflows, not for every chat turn.

Risk: SDK transport errors or stream interruptions leave partial tool results.

Mitigation: persist interrupted session state, retry only when no non-idempotent tool has executed, and surface `interrupted` or typed-failure status instead of synthesizing unsupported results.

Risk: Tool search selects a subset that excludes the required planned tool.

Mitigation: exact typed-plan steps bypass tool search; broader Claude-driven turns retry once with the exact mapped MCP tool and then fail closed with `tool_search_miss`.

Risk: mid-session harness switching corrupts context or mixes provider-specific transcripts.

Mitigation: disallow `/harness` mutation in v1. Harness changes require a new process/session start and use DISSOLVE structured session state as the migration boundary.

Risk: a stale or non-Claude `/model` alias is active when starting Claude SDK mode.

Mitigation: resolve through the Claude model registry, replace Gemini aliases with the default Claude alias with a notice, and fail clearly in noninteractive mode if no Claude model is configured.

## 12. Open Decisions

- Whether to keep only Python SDK or evaluate TypeScript V2 for richer stream/session ergonomics later.
- Whether to expose `AskUserQuestion` directly or continue using DISSOLVE's existing prompt-toolkit review screens for scientific defaults.
- Whether Claude SDK hooks can fully replace parts of `OutputVerifierMiddleware`, or whether verifier remains a shared post-processing step.
- Whether to encode DISSOLVE tools as one MCP server with tool search or multiple smaller MCP servers with explicit allowlists.
- Whether typed plan proposal should be Claude SDK structured output or remain the current deterministic compiler plus optional provider-agnostic planner backend.

## 13. First Spike Recommendation

Implement the smallest meaningful slice:

1. Add optional dependency and import guard.
2. Add `DISSOLVE_AGENT_HARNESS=claude_sdk`.
3. Add in-process MCP wrappers for:
   - `list_available_solvents`
   - `predict_solubility_range`
   - `plot_solubility_vs_temperature`
   - `get_solvent_safety_card`
4. Add session bridge.
5. Run split smoke tests:
   - `fast_path_preserved`: normal dispatch, direct-fast-path-eligible EVOH lookup, zero Claude calls.
   - `sdk_tool_invocation`: test-only direct-fast-path bypass or non-fast-path fixture, one solubility MCP call, no subagent.
   - `context_resume_chat`: normal dispatch with compact DISSOLVE context injected before each SDK turn.
6. Use this 4-turn `context_resume_chat`:
   - `i have an LDPE/EVOH/PET feedstock. what are good solvents for dissolving EVOH`
   - `what is the solubility of EVOH in DMF from room temp to 80C`
   - `plot it up to 90C and save to docs`
   - `where was that plot saved?`

Success means the SDK harness preserves fast paths, can call the right tools when fast path is bypassed, maintains context across fast-path and SDK turns, avoids separation detours, reports paths, and keeps costs/turns bounded. If this slice fails, do not migrate further.
