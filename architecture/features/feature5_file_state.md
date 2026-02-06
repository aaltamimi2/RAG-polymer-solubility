# Feature 5: File-Based Inter-Agent State Passing

## What It Does
In sequential multi-agent chains, subagents now write their findings to files in `/chain_state/` and subsequent subagents read those files via `read_file`, instead of the orchestrator concatenating full results into task description strings.

## Problem Solved
In 3+ agent sequential chains (Trace 5: sep → tea → safety, 311s), the orchestrator had to paraphrase and concatenate prior results into each task description. This was fragile — long results got truncated, context was lost, and token usage inflated. File-based state enables clean state passing with structured markdown output.

## Files Modified
- `src/strap/routing.py` — Updated 2-agent and 3+ agent sequential routing hints to include `/chain_state/step_N_<agent>.md` file path instructions
- `src/strap/agent.py`:
  - Added `_FILE_IO_DIRECTIVE` prompt snippet (appended to all subagent prompts) — enforces READ FIRST / WRITE LAST ordering
  - Added `_ALWAYS_FREE_TOOLS = {"write_file", "read_file"}` — always excluded from tool-call budgets
  - Updated `_resolve_guardrails()` to merge `_ALWAYS_FREE_TOOLS` into every subagent's `free_tools`
  - Added "Inter-agent file state" section to `SYSTEM_PROMPT` delegation policy

## How It Works

**2-agent sequential chain** (e.g., separation → TEA):
```
Orchestrator → task(sep-eng, "... write findings to /chain_state/step_1_separation-engineer.md")
               sep-eng writes to file
Orchestrator → read_file("/chain_state/step_1_separation-engineer.md")
Orchestrator → task(tea-lca, "Summary: ... Full details at /chain_state/step_1_... write to step_2_...")
               tea-lca reads prior file, writes its own
Orchestrator → read_file both files → synthesize final answer
```

**3+ agent sequential chain** (e.g., separation → TEA → safety):
Same pattern, each step writes to `step_N_<agent>.md`. Each subagent gets file paths to prior steps.

## How to Cherry-Pick
```bash
# Apply routing.py diff (2-agent and 3+ agent sequential hint changes)
# Apply agent.py diff (_FILE_OUTPUT_DIRECTIVE, _ALWAYS_FREE_TOOLS, SYSTEM_PROMPT update)
```

## Key Design Decisions
1. **READ FIRST / WRITE LAST**: The `_FILE_IO_DIRECTIVE` enforces ordering — the subagent's VERY FIRST action must be `read_file` on the prior step's output, and its FINAL action must be `write_file`. This prevents the subagent from working off the orchestrator's brief summary alone and skipping the full prior-step data.
2. **Augment, don't replace**: File paths are passed alongside brief summaries in task descriptions. If file reading fails, the summary provides fallback context.
3. **Deterministic paths**: `/chain_state/step_N_<agent>.md` — predictable, debuggable
4. **Free tools**: `write_file` and `read_file` don't count against `max_tool_calls` budget
5. **Auto-appended directive**: `_FILE_IO_DIRECTIVE` is appended to all subagent prompts in the YAML loader (same mechanism as `_THINK_DIRECTIVE`)
