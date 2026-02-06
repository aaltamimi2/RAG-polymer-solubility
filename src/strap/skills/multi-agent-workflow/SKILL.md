# Multi-Agent Workflow

Use this skill when a query requires delegating to TWO OR MORE subagents.

## Parallel Execution (2 independent agents)

When the routing hint says to launch two task() calls in parallel:
1. Include BOTH task() calls in a single response
2. Each task description should be self-contained (don't reference the other)
3. After both return, synthesize their results — look for contradictions

## Sequential Execution (2+ dependent agents)

When the routing hint says to delegate in sequence:
1. Call the FIRST subagent with a clear task description
2. Wait for its result, then extract key findings
3. Call the NEXT subagent, including relevant prior results in the task description
4. Repeat until all subagents have returned
5. Synthesize all results into a coherent final answer

## Synthesis Guidelines

- Cross-reference findings between subagents (e.g., cheapest solvent vs safest)
- Flag any contradictions explicitly
- Present a recommendation with trade-offs, not just raw data
- Cite which subagent produced which finding
