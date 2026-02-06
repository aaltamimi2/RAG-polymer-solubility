# DISSOLVE v4 — Architecture Enhancement Summary

Five features adopted from the [langchain-ai/deepagents examples](https://github.com/langchain-ai/deepagents/tree/master/examples) and integrated into the DISSOLVE multi-agent system.

## Features Implemented

### 1. Think Tool for Subagent Reflection
**Source**: `deep_research` example
**Files**: `tools/reflection.py` (new), `tools/__init__.py`, `agent.py`, `guardrails.py`

Zero-side-effect `think(reflection)` tool added to all 8 subagents. Forces the LLM to pause after each domain tool call and assess: Is this finding grounded in tool output? What's missing? Should I stop?

**Guardrail integration**: Think calls are excluded from `max_tool_calls` via `free_tools={"think"}` in `SubagentGuardMiddleware`. Safety-analyst prompt hardened with explicit "NEVER answer from general knowledge alone" directive.

### 2. Prompt-Embedded Stop Conditions
**Source**: `deep_research` example
**Files**: `agent.py` (now in `subagents.yaml`)

Domain-specific `## STOP CONDITIONS` sections in all 7 non-separation subagent prompts (separation-engineer already had `## HARD RULES`). These are soft, LLM-driven heuristics that complement middleware hard caps.

Examples:
- safety-analyst: "G-score AND PubChem data for each solvent"
- tea-lca-analyst: "TEA results for each scenario; don't re-run with tweaked params"
- scholar-researcher: "3+ papers found, or duplicate results — stop"

### 3. Progressive Skill Loading
**Source**: `text-to-sql-agent` and `content-builder-agent` examples
**Files**: `AGENTS.md` (new), `skills/*/SKILL.md` (3 new), `agent.py`

Split the monolithic system prompt into three tiers:
- **Memory** (`AGENTS.md`) — always loaded: agent identity, data overview, guidelines
- **System prompt** — always loaded: dynamic routing table + delegation policy
- **Skills** (`skills/*/SKILL.md`) — loaded on-demand: multi-agent workflow, data-lookup, separation-design

Requires `FilesystemBackend(root_dir=str(_PACKAGE_DIR))` for path resolution.

### 4. Externalize Subagent Configs to YAML
**Source**: `content-builder-agent` example
**Files**: `subagents.yaml` (new), `agent.py`, `pyproject.toml`

Replaced the ~250-line hardcoded `_build_subagents()` with:
- `subagents.yaml` — all 8 subagent definitions (name, description, system_prompt, tool_groups, guardrails)
- `_TOOL_GROUP_REGISTRY` — maps YAML string names to Python getter functions
- `_resolve_tools()` and `_resolve_guardrails()` — parse YAML config into runtime objects

Domain experts can now modify agent behavior by editing YAML, not Python.

### 5. File-Based Inter-Agent State Passing
**Source**: `deep_research` example
**Files**: `routing.py`, `agent.py`

Sequential multi-agent chains now use `/chain_state/step_N_<agent>.md` files for intermediate results instead of concatenating full results into task description strings.

- Routing hints instruct the orchestrator to tell subagents to `write_file` their findings
- Between steps, orchestrator uses `read_file` and passes file paths + brief summaries
- `write_file` and `read_file` are always free tools (don't count toward tool-call budget)

## Files Changed (Complete List)

| File | Status | Features |
|------|--------|----------|
| `src/strap/tools/reflection.py` | NEW | F1 |
| `src/strap/tools/__init__.py` | Modified | F1 |
| `src/strap/guardrails.py` | Modified | F1 |
| `src/strap/AGENTS.md` | NEW | F3 |
| `src/strap/skills/multi-agent-workflow/SKILL.md` | NEW | F3 |
| `src/strap/skills/data-lookup/SKILL.md` | NEW | F3 |
| `src/strap/skills/separation-design/SKILL.md` | NEW | F3 |
| `src/strap/subagents.yaml` | NEW | F2, F4 |
| `src/strap/agent.py` | Modified | F1-F5 |
| `src/strap/routing.py` | Modified | F5 |
| `pyproject.toml` | Modified | F4 |
| `architecture/features/*.md` | NEW | Docs |

## Architecture Before vs After

```
BEFORE (v3):
┌─ agent.py (monolithic) ─────────────────────────────┐
│ SYSTEM_PROMPT (3K chars, always loaded)              │
│ _build_subagents() (~250 lines, 8 hardcoded defs)    │
│   - No reflection mechanism                          │
│   - No stop conditions (except separation-engineer)  │
│   - Context passed via string concatenation          │
│   - All subagent descriptions in every LLM call      │
└──────────────────────────────────────────────────────┘

AFTER (v4):
┌─ AGENTS.md (memory, always loaded) ─────────────────┐
│ Agent identity, available data, guidelines           │
├─ skills/ (on-demand) ───────────────────────────────┤
│ multi-agent-workflow/ | data-lookup/ | sep-design/   │
├─ subagents.yaml (config) ───────────────────────────┤
│ 8 subagent defs with tool_groups + guardrails        │
├─ agent.py (thin orchestration) ─────────────────────┤
│ SYSTEM_PROMPT (routing table + delegation + file IO) │
│ _build_subagents() → YAML loader (~40 lines)         │
│ _THINK_DIRECTIVE + _FILE_OUTPUT_DIRECTIVE (auto-append)│
├─ routing.py (file-state aware) ─────────────────────┤
│ Sequential hints include /chain_state/ file paths    │
├─ guardrails.py (free_tools support) ────────────────┤
│ think, write_file, read_file excluded from budgets   │
└──────────────────────────────────────────────────────┘
```

## Expected Impact

| Metric | v3 Baseline | v4 Expected | Mechanism |
|--------|-------------|-------------|-----------|
| Ungrounded responses | safety-analyst 0 tools (Trace 2) | Reduced — think_tool + "NEVER answer from general knowledge" | F1 |
| Tool call waste | 10/10 budget hit (Trace 2, 5) | Fewer — soft stop conditions trigger before hard cap | F2 |
| System prompt tokens | ~3K always loaded | ~1.5K always + skills on-demand | F3 |
| Config change effort | Edit Python code | Edit YAML | F4 |
| Context truncation (3+ agents) | String concat in task descriptions | File paths + brief summaries | F5 |

## Testing

Each feature was verified independently:
- F1: `think` tool imports, executes, and appears in all 8 subagent tool lists
- F2: All subagent prompts contain `STOP CONDITIONS` or `HARD RULES`
- F3: Agent creates with `memory=`, `skills=`, `FilesystemBackend`
- F4: All 8 subagents load from YAML with correct tool counts and guardrail configs
- F5: Sequential routing hints include `/chain_state/` file paths; `write_file`/`read_file` in all free_tools sets

Full integration test: `create_dissolve_agent()` succeeds with all features active simultaneously.
