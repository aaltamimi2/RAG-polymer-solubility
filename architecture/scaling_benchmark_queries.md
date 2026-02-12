# Scaling Benchmark Queries

Queries used for the DISSOLVE agent scaling benchmark (2-9 polymers).
Each query uses the same template with an increasing polymer list.

**Template**: `From the available solvents, which best dissolve each of {polymer_list} at 120C while rejecting the others? Rank by solubility ratio.`

**Polymer pool** (added incrementally): PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, PET

## Queries

| N | Polymers | Query |
|---|----------|-------|
| 2 | PS, PVC | From the available solvents, which best dissolve each of PS, PVC at 120C while rejecting the others? Rank by solubility ratio. |
| 3 | PS, PVC, LDPE | From the available solvents, which best dissolve each of PS, PVC, LDPE at 120C while rejecting the others? Rank by solubility ratio. |
| 4 | PS, PVC, LDPE, HDPE | From the available solvents, which best dissolve each of PS, PVC, LDPE, HDPE at 120C while rejecting the others? Rank by solubility ratio. |
| 5 | PS, PVC, LDPE, HDPE, PP | From the available solvents, which best dissolve each of PS, PVC, LDPE, HDPE, PP at 120C while rejecting the others? Rank by solubility ratio. |
| 6 | PS, PVC, LDPE, HDPE, PP, EVOH | From the available solvents, which best dissolve each of PS, PVC, LDPE, HDPE, PP, EVOH at 120C while rejecting the others? Rank by solubility ratio. |
| 7 | PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6 | From the available solvents, which best dissolve each of PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6 at 120C while rejecting the others? Rank by solubility ratio. |
| 8 | PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66 | From the available solvents, which best dissolve each of PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66 at 120C while rejecting the others? Rank by solubility ratio. |
| 9 | PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, PET | From the available solvents, which best dissolve each of PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, PET at 120C while rejecting the others? Rank by solubility ratio. |

## Results Summary (2 runs, current agent)

| N | Run 1 (s) | Run 2 (s) | Mean (s) | Run 1 tokens | Run 2 tokens | Mean tokens | Run 1 tools | Run 2 tools |
|---|-----------|-----------|----------|--------------|--------------|-------------|-------------|-------------|
| 2 | 11.4 | 39.8 | 25.6 | 22,610 | 24,045 | 23,328 | 2 | 2 |
| 3 | 31.2 | 16.0 | 23.6 | 24,410 | 23,146 | 23,778 | 3 | 3 |
| 4 | 29.9 | 18.2 | 24.1 | 24,451 | 23,805 | 24,128 | 4 | 4 |
| 5 | 56.2 | 31.8 | 44.0 | 26,660 | 23,732 | 25,196 | 5 | 5 |
| 6 | 25.1 | 5.9* | 15.5 | 23,701 | 10,311* | 17,006 | 6 | 0* |
| 7 | 22.4 | 35.6 | 29.0 | 24,451 | 40,711 | 32,581 | 7 | 8 |
| 8 | 24.1 | 25.5 | 24.8 | 24,876 | 24,686 | 24,781 | 8 | 8 |
| 9 | 44.1 | 47.8 | 46.0 | 25,126 | 26,335 | 25,731 | 9 | 9 |

*N=6 Run 2 hit the deterministic selectivity bypass (0 tool calls, 0 output tokens).

**Notes**:
- Model: google_genai:gemini-2.5-pro (default STRAP_MODEL)
- Orchestrator guardrails: max_iterations=50, token_budget=500K, max_tool_calls=12
- Output verifier: enabled (Gemini 2.0 Flash)
- Routing middleware: enabled (keyword classifier)
- Date: 2026-02-10
