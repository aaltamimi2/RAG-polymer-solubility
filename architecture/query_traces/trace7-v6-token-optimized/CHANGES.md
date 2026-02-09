# Trace 7: v6 Token-Optimized — What Changed

## Results

| Metric | Trace 6 (before) | Trace 7 (after) | Improvement |
|--------|------------------|-----------------|-------------|
| Duration | 274.9s | 50.0s | 5.5x faster |
| Tokens | ~2M (est.) | ~50K (est.) | ~40x fewer |
| Messages | 30 | 4 | 7.5x fewer |
| Tool Calls | 53 LLM / 360 tool | 1 | 53x fewer |
| Subagent Invocations | 3x sep-eng + 1x TEA | 1x sep-eng | 4x fewer |
| Peak RSS | 1565 MB | 898 MB | 42% lower |

## 6 Changes Made

### 1. Multi-Scheme Tool (`plan_multiple_separation_schemes`)

**Biggest win.** Replaced 3 separate subagent invocations with a single tool call that generates 3 diverse schemes in one shot.

- **Before**: Orchestrator called `separation-engineer` 3 times. First 2 hit token budget (216K, 220K each). Third hit tool-call limit.
- **After**: Single tool call returns 3 schemes (selectivity, safety/GSK, energy/BP) in 2493 chars, 0.23s.
- **How it works**: Pre-loads solvent properties (BP, LogP, G-score) once. Runs a generalized greedy algorithm with 3 different ranking functions. Each ranker picks the best (polymer, solvent) pair at each step.

File: `src/strap/tools/advanced_separation.py`

### 2. Orchestrator-Level Guardrails

Added `SubagentGuardMiddleware` to the orchestrator itself (not just subagents).

- `max_iterations=50`: Caps total orchestrator LLM calls
- `token_budget=500_000`: Hard stop at 500K total input tokens
- `max_tool_calls=30`: Prevents runaway tool-calling loops
- `truncate_tool_results_after=3000`: Limits context growth

File: `src/strap/agent.py`

### 3. Subagent Guard Tightening

Separation-engineer limits reduced to match the multi-scheme tool's efficiency:

- `max_tool_calls`: 8 -> 5
- `token_budget`: 200K -> 100K
- `truncate_tool_results_after`: 2000 -> 800 chars

File: `src/strap/subagents.yaml`

### 4. Aggressive Context Truncation

- Truncation triggers after iteration 1 (was iteration 3)
- Keeps only last 4 messages untruncated (was 6)
- Prevents quadratic context growth from tool results

File: `src/strap/guardrails.py`

### 5. Trimmed System Prompts

- `_THINK_DIRECTIVE`: 7 lines -> 2 lines
- `_FILE_IO_DIRECTIVE`: 7 lines -> 1 line
- Separation-engineer prompt: 25 lines -> 10 lines
- Added multi-scheme delegation hint to orchestrator

Files: `src/strap/agent.py`, `src/strap/subagents.yaml`

### 6. Interpolation Model (No SQL)

Solubility queries use `ln(S) = A + B/T + C/T^2` polynomial fit instead of SQL database queries. 352 polymer-solvent pairs pre-fitted. Eliminates DB round-trips and verbose SQL result formatting.

File: `src/strap/solubility.py`
