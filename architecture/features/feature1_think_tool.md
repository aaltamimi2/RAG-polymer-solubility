# Feature 1: Think Tool for Subagent Reflection

## What It Does
Adds a zero-side-effect `think` tool to all 8 subagents that forces the LLM to pause after domain tool calls and assess whether its findings are grounded in tool output or general knowledge.

## Problem Solved
In Trace 2, safety-analyst answered entirely from LLM parametric knowledge (0 domain tool calls), producing ungrounded claims. The think tool creates a mandatory reflection checkpoint.

## Files Changed
- `src/strap/tools/reflection.py` — **NEW**: `think(reflection: str)` function
- `src/strap/tools/__init__.py` — Added `get_reflection_tools()` getter
- `src/strap/agent.py` — Added `_THINK_DIRECTIVE` prompt snippet, wired `get_reflection_tools()` into all 8 subagents, added `free_tools={"think"}` to guardrails
- `src/strap/guardrails.py` — Added `free_tools` parameter to `SubagentGuardMiddleware`; think calls don't count toward `max_tool_calls`

## How to Cherry-Pick
```bash
# From v4 worktree, copy these files to your target:
cp src/strap/tools/reflection.py <target>/src/strap/tools/
# Then apply the diffs to __init__.py, agent.py, guardrails.py
```

## Key Design Decisions
1. **Free tools**: `think` calls are excluded from `max_tool_calls` budget via `free_tools={"think"}` in guardrails, so reflection doesn't eat into the analysis budget
2. **Domain-aware docstring**: The tool's docstring coaches the LLM to cite specific numbers and check grounding
3. **Shared directive**: `_THINK_DIRECTIVE` is appended to all subagent system prompts for consistency
4. **Safety-analyst hardening**: Extra prompt text explicitly forbids answering from general knowledge alone
