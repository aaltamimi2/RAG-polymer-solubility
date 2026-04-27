# Query Trace Log

## Trace 1: Single-Agent Separation (with guardrails v2)

- **LangSmith Trace ID**: `019c3002-e4eb-7102-85d7-03d65e540bb7`
- **LangSmith Project**: `strap-agent`
- **Date**: 2026-02-05
- **PNG**: `chain_trace.png`
- **Script**: `visualize_chain_trace.py`
- **Query**: "Find the optimal separation sequence for LDPE, EVOH, and PET. Use selective dissolution at temperatures up to 120C."
- **Duration**: 41s
- **Tokens**: 63K in, 4.3K out (67K total)
- **Subagents**: 1 (separation-engineer)
- **Guardrails active**: tool-call budget (8), synthesis injection, tool-result truncation (2000 chars), token budget (200K), iteration limit (25)
- **Notes**: First trace after implementing guardrails v2. Separation-engineer completed in 4 LLM calls / 8 tool calls (6 executed). Synthesis injection fired after `plan_sequential_separation`. Tool-call limit stopped over-exploration. Previous run without guardrails: 13 LLM calls, 22 tool calls, 224K tokens, 83s.

## Trace 6: P0-P3 Validation — 3-Scheme + Safety + TEA

- **LangSmith Trace ID**: `019c3149-e973-77b0-9d84-9ed33f07cd68`
- **LangSmith Project**: `strap-agent`
- **Date**: 2026-02-06
- **Directory**: `trace6-3agent-3scheme-safety-tea-p0p3/`
- **PNG**: `019c3149-e973-77b0-9d84-9ed33f07cd68.png`
- **Script**: `visualize_trace6.py`
- **Query**: "Find the optimal separation sequence for PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, and PET. Propose THREE different dissolution schemes. Then run a safety assessment and TEA on each scheme."
- **Duration**: 274.9s
- **Tokens**: 745K in, 25K out (770K total)
- **Messages**: 30
- **LLM Calls**: 53
- **Subagents invoked**: separation-engineer (x3), tea-lca-analyst (x1). Safety-analyst was NOT invoked.
- **Guardrails active**: token budget (200K), tool-call budget (sep-eng: 20, tea-lca: 15, safety: 20), synthesis injection, tool-result truncation
- **Notes**: First run with P0-P3 safety-analyst improvements active (rewritten system prompt, directive "not found" messages, guardrail config, `family_override`). Key findings:
  - Separation-engineer was called 3 times — first 2 hit token budget (~216K, ~220K), 3rd hit tool-call budget. Produced 3 differentiated schemes.
  - Tea-lca-analyst hit tool-call budget after 9 `analyze_solvent_recovery_tea` calls.
  - **Safety-analyst subagent was NOT invoked** — the orchestrator self-served safety assessment by querying the GSK dataset directly via `query_database`.
  - Orchestrator cited **absolute G-scores** (THF 4.79, NMP 5.49, Toluene 5.96, Ethyl Acetate 6.66, Cyclohexanone 7.24, Benzyl Alcohol 7.68).
  - Scheme 2 was proactively designed as "Green/Alternative" with safer solvents — a behavioral shift not seen in Trace 5.
  - No fabricated G-scores observed.
  - Compared to Trace 5 (311s, 628K tokens, 3 subagents): 13% faster, 22% more tokens, but significantly better safety analysis quality.

## Trace 7: v6 Token-Optimized — 3-Scheme Multi-Criteria

- **LangSmith Trace ID**: (local run — no LangSmith trace)
- **Date**: 2026-02-06
- **Directory**: `trace7-v6-token-optimized/`
- **PNG**: `trace7-v6-token-optimized.png`
- **Script**: `visualize_trace7.py`
- **Changes**: `CHANGES.md`
- **Query**: "Find the optimal separation sequence for a mixed polymer waste stream containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, and PET. Use selective dissolution at atmospheric pressure. Propose THREE different sets of solvents and conditions for this 9-polymer dissolution scheme."
- **Duration**: 50.0s
- **Messages**: 4 (1 human, 2 AI, 1 tool)
- **Tool Calls**: 1 (plan_multiple_separation_schemes)
- **HTTP Requests**: 7
- **Peak RSS**: 898 MB
- **Subagents invoked**: separation-engineer (x1)
- **Notes**: First run with v6 token-optimized architecture. Key changes:
  1. **Multi-scheme tool** (`plan_multiple_separation_schemes`): 3 schemes in 1 call (2493 chars, 0.23s) vs 3 separate subagent invocations. Biggest win.
  2. **Orchestrator guardrails**: SubagentGuardMiddleware on orchestrator (500K token budget, 30 tool-call cap, 3000 char truncation).
  3. **Subagent tightening**: sep-eng max_tool_calls 8->5, token_budget 100K, truncate 800 chars.
  4. **Aggressive truncation**: kicks in after iter 1 (was 3), keeps last 4 messages (was 6).
  5. **Trimmed prompts**: _THINK_DIRECTIVE 7->2 lines, _FILE_IO_DIRECTIVE 7->1 line, sep-eng prompt 25->10 lines.
  6. **Interpolation model**: `ln(S) = A + B/T + C/T^2` replaces all SQL solubility queries (352 pairs pre-fitted).
  - Compared to Trace 6 (275s, ~2M tokens, 30 msgs, 53 LLM calls): **5.5x faster, ~40x fewer tokens, 7.5x fewer messages**.
