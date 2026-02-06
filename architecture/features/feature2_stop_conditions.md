# Feature 2: Prompt-Embedded Stop Conditions

## What It Does
Adds domain-specific "STOP CONDITIONS" sections to each subagent's system prompt, telling the LLM WHEN to stop calling tools and synthesize. These are soft, LLM-driven heuristics that complement the hard middleware guardrails.

## Problem Solved
Middleware guardrails fire as hard caps (e.g., "tool budget exhausted" at call #10). By then the subagent may have wasted calls on redundant queries. Prompt-embedded conditions let the LLM self-regulate earlier — e.g., "stop when you have G-score AND PubChem data for each solvent."

## Files Changed
- `src/strap/agent.py` — Added `## STOP CONDITIONS` blocks to 7 subagent system prompts (separation-engineer already had equivalent "HARD RULES")

## Stop Conditions by Subagent

| Subagent | Key Stop Heuristics |
|----------|-------------------|
| safety-analyst | G-score AND PubChem data for each solvent; both sources queried |
| tea-lca-analyst | TEA results for each scenario; LCA if asked; no re-running with tweaked params |
| scholar-researcher | 3+ papers found or duplicate results; max 2 broad searches |
| patent-researcher | 3+ patents or <3 results from search; max 2 broad searches |
| rag-analyst | Q&A answered, ingestion confirmed, or index empty — stop immediately |
| visualization-specialist | Plot created and file path confirmed; no unsolicited variations |
| statistics-ml | Requested test run, CI/p-values reported; no exploratory side-analyses |

## How to Cherry-Pick
The changes are entirely in `src/strap/agent.py` system_prompt strings. Search for `## STOP CONDITIONS` and copy the relevant blocks into your target's subagent definitions.

## Interaction with Other Features
- **Feature 1 (think_tool)**: Stop conditions tell the LLM *what* to check; think_tool provides the *mechanism* to check it. The `_THINK_DIRECTIVE` is appended after stop conditions.
- **Guardrails**: Stop conditions are soft limits (LLM-decided). Middleware remains as the hard backstop.
