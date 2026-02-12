# DISSOLVE Agent — Hardcoded Features Audit

Audit date: 2026-02-10. Covers `src/strap/` in v6 worktree.

---

## RED — LLM Bypassed Entirely

| # | Feature | File:Line | What's Hardcoded | Verdict |
|---|---------|-----------|-----------------|---------|
| 1 | **Deterministic selectivity** | routing.py:618-700 | N≥2 polymers + no multi-agent keywords → pre-computes selectivity tables, returns AIMessage directly. LLM never called. | Biggest flexibility loss. Works great for pure selectivity but silently drops any follow-up intent the user appended (e.g., Query 2 "safe" bug). |
| 2 | **Duplicate task rewriting** | routing.py:706-819 | If LLM calls wrong/duplicate task() → middleware rewrites to correct agent. If all steps done + LLM calls task() → strips it, returns results. If LLM doesn't call task() when steps remain → injects forced task(). | Necessary for reliability (Gemini loops otherwise), but LLM has zero agency over execution order. |
| 3 | **Selectivity + safety combined path** | routing.py:680-700 | When deterministic selectivity fires AND safety keywords detected → pre-computes selectivity, injects into system prompt, forces LLM to route only to safety-analyst. Separation-engineer is completely blocked. | Correct optimization but tightly coupled to keyword detection (fragile). |

## ORANGE — Magic Numbers Repeated Across Codebase

| # | Feature | Locations | Value | Issue |
|---|---------|-----------|-------|-------|
| 4 | **Default temperature 120°C** | routing.py:650, engines/separation.py (5×), visualization.py:765, advanced_separation.py, tools/interpolation.py | `120.0` | Repeated 10+ times. No single constant. If the dataset range changes, must find-and-replace everywhere. |
| 5 | **Parallel pairs whitelist** | routing.py:155-157 | Only `{separation-engineer, safety-analyst}` | Adding a new parallel pair (e.g., scholar+patent) requires code edit. Not in YAML. |
| 6 | **Context keep count** | guardrails.py:~269 | `keep_recent = 4` messages | Hardcoded inside truncation logic, not a middleware parameter. Some subagents may need more context. |
| 7 | **Agent result extraction cap** | routing.py:396-397 | `4000` chars | Hardcoded truncation on subagent results returned to orchestrator. Not configurable. |
| 8 | **Model name** | agent.py:181 | `"google_genai:gemini-3-flash-preview"` | Hardcoded as function default. Should be env var. |

## YELLOW — Limits Not Exposed as Parameters

| # | Feature | File:Line | Value | Impact |
|---|---------|-----------|-------|--------|
| 9 | **Selectivity top-N** | routing.py:587, 669 | Top 10 (display), top 5 (injection) | User/LLM cannot request more or fewer solvents in deterministic path. |
| 10 | **API search result cap** | literature.py:223,382,564,630 | `min(max_results, 20)` | Hard ceiling of 20; parameter `max_results` is clamped. |
| 11 | **Fuzzy match threshold** | safety_gsk.py:160,282 | `threshold=80` | Hardcoded in function call, not tool parameter. |
| 12 | **Temperature tolerance** | solubility.py:626,675,733 | `temp_tolerance=10.0` | Default ±10°C in SQL fallback. Not exposed to LLM. |
| 13 | **Heatmap annotation cutoffs** | visualization.py:812-814 | 150/50/100 cell thresholds | Controls font size and whether annotations appear. Not configurable. |
| 14 | **"154 solvents" in error msg** | safety_gsk.py:177,186 | Literal string | Will be wrong if GSK dataset grows. Should query `COUNT(*)`. |

## GREEN — Justified & Well-Placed

| # | Feature | File | Why It's Fine |
|---|---------|------|---------------|
| 15 | Routing keyword scores (3/2/1/-1) | routing.py:35-149 | Language-semantic scoring; fixed values appropriate for classifier stability. |
| 16 | G-score rating thresholds (8/6/4) | safety_gsk.py:203-214 | GSK domain standard. Not arbitrary. |
| 17 | Precipitation thresholds (1/10/50/20%) | engines/precipitation.py:30-33 | Physical chemistry conventions. Named constants. |
| 18 | Temperature range 25–160°C | engines/optimization.py:178 | Matches interpolation model fitted data range. |
| 19 | Per-subagent budgets in YAML | subagents.yaml | Iteration caps, token budgets, tool-call budgets — all YAML-configurable. |
| 20 | Synthesis injection directives | guardrails.py:243-252 | Intentional guardrail. Forces synthesis after key tools. Necessary because Gemini ignores soft hints. |
| 21 | Sequential pair ordering | routing.py:159-164 | Domain-dependency logic (e.g., separation before TEA). Correct. |
| 22 | Solvent/polymer alias dicts | solubility.py:66-113, 540-564 | Static reference data. Appropriate in code. |
| 23 | `_ALWAYS_FREE_TOOLS` | agent.py:130-133 | Filesystem tools free from budget. Necessary to prevent budget exhaustion on non-domain calls. |

---

## Where Determinism Helps vs. Hurts

### Helps (keep these)
- **Guardrail budgets** — Without them, Gemini explores the filesystem for 20+ tool calls before doing any real work. Token budgets and tool-call caps are essential.
- **Synthesis injection** — Gemini frequently ignores "synthesize now" in system prompts. Middleware injection after key tools is the only reliable way to stop infinite exploration loops.
- **Sequential enforcement** — Without `_rewrite_duplicate_task_calls`, Gemini calls the same subagent repeatedly or skips agents in the plan. Hard rewriting is necessary.
- **Deterministic selectivity (pure)** — For simple "compare polymer X and Y" queries, pre-computing selectivity saves ~200K tokens and 2 minutes. Clear win.

### Hurts (reconsider these)
- **Deterministic selectivity (combined)** — The keyword-gated combined path is fragile. "Safe" vs "safety" bug shows substring matching is brittle. Consider: let the LLM see the pre-computed selectivity AND decide routing, rather than forcing safety-analyst via keyword match.
- **Parallel pair whitelist** — Only 1 pair is allowed concurrent. New research agents (scholar+patent) can't run in parallel without code change. Move to YAML.
- **120°C everywhere** — Not wrong, but should be `DEFAULT_TEMP` constant. One change point instead of 10.
- **Context keep = 4 fixed** — Safety-analyst with 2 solvents needs 4+ tool results in context. Truncating to 4 messages may drop earlier solvent data mid-analysis.

---

## Suggested Priority Fixes

1. **Extract `DEFAULT_SEPARATION_TEMP = 120.0`** into a constants module → replace all 10+ occurrences
2. **Move `PARALLEL_PAIRS` to subagents.yaml** → allow new pairs without code edit
3. **Make `keep_recent` a guardrails parameter** → per-subagent in YAML
4. **Add `"safe"` variants to keyword lists** → (already done in this session)
5. **Model name from env var** → `os.getenv("STRAP_MODEL", "google_genai:gemini-3-flash-preview")`
6. **Query GSK count at runtime** → replace "154 solvents" literal
