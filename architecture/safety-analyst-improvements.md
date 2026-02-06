# Safety-Analyst Subagent Improvements (P0-P3)

**Date**: 2026-02-05
**Motivation**: Trace 5 (`019c30ab`) revealed that the safety-analyst subagent hallucinated G-scores for solvents not in the GSK dataset, failed to use PubChem fallback tools, and provided only relative rankings without absolute safety context.

## Failure Modes Addressed

1. **Hallucinated G-scores**: Agent claimed Nitrobenzene has "GSK G-score 1" -- solvent is not in the GSK dataset (154 solvents). Score was fabricated.
2. **Incomplete coverage**: 3 of ~15 solvents (m-Cresol, Nitrobenzene, Phenol) absent from GSK. Agent didn't flag gaps or use PubChem tools.
3. **Relative-only framing**: Scheme 1 called "safest" but 5/7 solvents are Problematic (G-score 4-6). No absolute context provided.
4. **No safer alternatives suggested**: Agent didn't use `get_family_alternatives` to propose replacements for Problematic/Hazardous solvents.

## P0: Rewrite Safety-Analyst System Prompt

**File**: `src/strap/agent.py` (lines 153-193)

**Before** (3 lines):
```
You are a chemical safety analyst. You have tools for GSK G-score
lookups, PubChem safety/toxicity data retrieval, and safety
visualizations. Provide thorough safety assessments.
```

**After** (~23 lines): Structured prompt with 2 sections modeled on the separation-engineer's prompt:
- **WORKFLOW**: 5-step sequence: LOOKUP -> FALLBACK -> CONTEXTUALIZE -> ALTERNATIVES -> SYNTHESIZE. Includes G-score thresholds inline.
- **HARD RULES**: Never fabricate scores, always provide absolute ratings, note average G-scores per scheme, budget tool calls

The TOOLS section was intentionally omitted -- the LLM already sees each tool's docstring and `WHEN TO USE` section via the function definitions. Only behavioral instructions belong in the system prompt.

**Rationale**: The separation-engineer has a 35-line prompt with explicit WORKFLOW, HARD RULES, KEY TOOLS, and CONSTRAINT sections. All other subagents had 3-line prompts. The safety-analyst's minimal prompt was the primary root cause of all 4 failure modes. Subagents don't have access to skills -- the system prompt is the only lever for behavioral instructions.

## P1: Improve "Not Found" Messages in `get_solvent_gscore`

**File**: `src/strap/tools/safety_gsk.py` (lines 161, 167)

**Before**:
```
No G-score data found for '{solvent_name}'. The GSK dataset contains
153 solvents. Try `list_tables()` to see available solvents.
```

**After**:
```
**NOT FOUND**: '{solvent_name}' is not in the GSK dataset (154 solvents).
Do NOT estimate or fabricate a G-score. Instead, call
get_pubchem_safety_info('{solvent_name}') to retrieve GHS hazard
classification from PubChem as a fallback. Report this solvent as
'Not in GSK database' in your assessment.
```

**Rationale**: The old message was passive and suggested `list_tables()` (a tool the safety-analyst doesn't have). The new message is:
- **Unambiguous**: "NOT FOUND" prefix in bold
- **Directive**: Explicitly says "Do NOT estimate or fabricate"
- **Actionable**: Points to the specific PubChem fallback tool by name
- **Self-documenting**: Tells the agent how to report the gap

## P2: Configure Safety-Analyst Guardrail Middleware

**File**: `src/strap/agent.py` (line 198)

**Before**:
```python
middleware=[SubagentGuardMiddleware()],
```

**After**:
```python
middleware=[SubagentGuardMiddleware(
    max_tool_calls=20,
    synthesis_tools={
        "compare_pubchem_safety",
        "visualize_gscores",
    },
    truncate_tool_results_after=2000,
)],
```

**Rationale**:
- `max_tool_calls=20`: Sufficient for ~15 G-score lookups + ~3 PubChem fallbacks + ~2 family alternatives. Default of 10 is too tight (Trace 5 hit 15 limit).
- `synthesis_tools`: After calling `compare_pubchem_safety` or `visualize_gscores`, inject synthesis directive to prevent over-exploration. These are "summary" tools typically called near the end of analysis.
- `truncate_tool_results_after=2000`: PubChem tools return verbose GHS data. Truncation prevents context inflation.

## P3: Add `family_override` to `get_family_alternatives`

**File**: `src/strap/tools/safety_gsk.py` (lines 215-270)

**Before**: Function only accepts `solvent_name` -- if the solvent is not in the GSK dataset, it returns "Could not find solvent" with no next step.

**After**: Added `family_override: Optional[str] = None` parameter:
- If `family_override` is provided, skips the solvent name lookup and queries that family directly
- Enables workflow: "Nitrobenzene is not in GSK, but it's an Aromatic -- show me safer Aromatics"
- Updated "not found" message to list all valid family names and suggest using `family_override`
- Valid families: Alcohols, Aromatics, Carbonates, Dipolar Aprotics, Esters, Ethers, Halogenated, Hydrocarbons, Ketones, Other, water and acids

**Rationale**: Without this, the agent couldn't suggest alternatives for solvents not in the GSK database (like Nitrobenzene, m-Cresol, Phenol). With `family_override`, the workflow step "ALTERNATIVES: call get_family_alternatives for worst offenders" works even for missing solvents.

## Verification

Both modified files pass Python syntax checking:
```
python3 -c "import ast; ast.parse(open('src/strap/agent.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('src/strap/tools/safety_gsk.py').read()); print('OK')"
```

## Files Modified

| File | Changes |
|---|---|
| `src/strap/agent.py` | P0: safety-analyst system prompt rewrite, P2: guardrail middleware config |
| `src/strap/tools/safety_gsk.py` | P1: "not found" messages, P3: `family_override` parameter |

## Skills System (reverted)

Two approaches to enabling skills were tested and both caused hangs:
1. `backend=FilesystemBackend` + `skills=` on `create_deep_agent` — gives ALL subagents filesystem tools, causing infinite loops
2. `SkillsMiddleware` added only to orchestrator `middleware=` list — still hung (state schema registration issue)

Skills code has been fully reverted. The orchestrator uses only `routing_middleware`.

## Validation (Trace 6)

Reran the 3-agent sequential query with P0-P3 active (skills off). Duration: 274.9s, 30 messages, 14 tool calls. Key improvements:
- Safety-analyst now cites **absolute G-scores** for each solvent (THF 4.79, Toluene 5.96, Ethyl Acetate 6.66, Benzyl Alcohol 7.68, etc.)
- Scheme 2 explicitly designed as **"Green/Alternative"** with highest-scoring solvents
- No fabricated scores observed
- The structured WORKFLOW prompt influenced scheme generation: the agent proactively designed a safety-optimized scheme

## Not Implemented (P4, deferred)

**Expand GSK dataset** with m-Cresol, Nitrobenzene, Phenol. Deferred because:
- Requires validated published G-scores (Henderson et al. 2011, Byrne et al. 2016)
- P0-P3 address the root causes regardless of dataset completeness
- PubChem fallback (enabled by P0+P1) covers missing solvents at query time
