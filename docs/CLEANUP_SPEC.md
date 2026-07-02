# STRAP v10 Cleanup & Hardening Spec

**Status:** proposal — no code changed yet.
**Purpose:** consolidate redundancy, remove dead code, and fix latent risks
*before* the next build-out phase, so new features land on a clean base.
**Method:** three read-only audit sweeps (duplication/monkey-patches, dead-code/routing
remnants, latent-issues/hygiene) plus direct verification. Every item below
cites file:line evidence.

Legend — **Effort**: S (<1h) · M (half day) · L (1–2 days).
**Risk**: how likely a change is to break behavior (Low/Med/High).

---

## 0. Executive summary

The codebase is functionally healthy (full suite green: 1006 + 84 waste-opt),
but three sessions of routing rewrites, a v9→v10 port, and a BioSTEAM
integration have left predictable scar tissue:

- **~13k lines of dead vendor code** and two removable routing functions.
- **8 parallel polymer-canonicalization implementations** and the last
  straggler solvent-alias dict (the solvent side was already consolidated into
  `solvent_registry.py`; polymers never were).
- **Small utility helpers copy-pasted 3–7×** (message extraction, dedupe,
  float coercion).
- **Two deploy-blocking dependency risks** (BioSTEAM/Python-version mismatch,
  unpinned pandas/duckdb) that make a *fresh* build crash even though the
  current dev env works.
- **One real engine defect** (the Pareto phantom-anchor point) already
  characterized in case study 02.

The keyword routing layer is **not** dead — it is the intentional offline
fallback for the planner. Only two functions in it are actually removable.

**Recommended order:** P0 (deploy-blockers) → P1 (registries + engine defect) →
P2 (helper consolidation + dead code) → P3 (refactors + hygiene). Registries
must land before their call-site migrations.

---

## P0 — Deploy-blockers (do first; small, high-value)

### P0.1 Pin BioSTEAM / fix Python floor — **Effort S · Risk Low**
`pyproject.toml:48-50` declares `biosteam>=2.50`, `thermosteam>=0.50` with
`requires-python>=3.10` (line 9), but current BioSTEAM/thermosteam releases
require Python ≥3.12 while `Dockerfile:9` is `python:3.11-slim`. A fresh
`pip install strap-agent[biosteam]` or Docker build pulls an incompatible
release and crashes on import (`SyntaxError` in thermosteam under 3.11).
- **Fix:** pin to the last 3.11-compatible pair (e.g. `biosteam>=2.50,<2.51`,
  `thermosteam>=0.50,<0.52`) **or** bump the Docker base to `python:3.12-slim`
  and set `requires-python>=3.12`. Choose one deliberately — the whole app
  currently runs on 3.11.
- **Verify:** clean venv install of the `biosteam` extra imports successfully.

### P0.2 Upper-bound pandas / duckdb — **Effort S · Risk Low**
`pyproject.toml:16-17` leaves `pandas` and `duckdb` unpinned. pandas 3.x broke
`duckdb.register()` in this env (observed directly this session). A fresh
install can pull pandas 3.x and break all DuckDB-backed tools.
- **Fix:** `pandas>=2.0,<3.0`, `duckdb<2.0` until 3.x compatibility is verified.

### P0.3 Remove macOS-hardcoded SCIP path — **Effort S · Risk Low**
`waste_management/solver.py:26`:
`_SCIP_EXECUTABLE = shutil.which("scip") or "/opt/homebrew/bin/scip"`.
The fallback is a macOS Homebrew path that doesn't exist on the Linux
Docker/prod target; when SCIP is absent the error points at a nonexistent
file instead of saying "install SCIP".
- **Fix:** drop the hardcoded fallback; if `shutil.which("scip")` is None,
  raise a clear "SCIP not found on PATH; install it or set SCIP_PATH" error.

---

## P1 — Registries & the engine defect (foundational; unblocks P2 migrations)

### P1.1 Create `polymer_registry.py` — **Effort M · Risk Med**
Polymer canonicalization is implemented **8 ways** with subtly different
outputs — the biggest correctness hazard in the audit:

| Site | Symbol | Divergence |
|---|---|---|
| `solubility.py:128` | `POLYMER_ALIASES` | `POLYETHYLENE→HDPE`, uses `NYLON6/NYLON66` |
| `planning/extractors.py:26` | `_POLYMER_ALIASES` | `→PE`, uses `PA6/PA66` |
| `tools/waste_optimization.py:43` | `_OPTIMIZATION_POLYMER_ALIASES` | optimization canonical |
| `handoff_adapters.py:171` | `_canonical_optimization_polymer()` | inline if/elif copy of the above |
| `direct_fast_path.py:217` | `_resolve_polymer_name()` | fast-path resolver |
| `ml_assets.py:118` | `resolve_polymer_entry()` | ML catalog |
| `tools/ml_prediction.py:587` | `_resolve_polymer_inputs()` | ML tool |
| `services/contaminant_screening_service.py:70` | `_resolve_polymer_or_none()` | contaminant |

The `HDPE`-vs-`PE` and `NYLON6`-vs-`PA6` splits are real: the solubility engine
keys on `HDPE/NYLON6`, the optimizer on `PE/PA6`. Any centralization **must
preserve both target vocabularies**, not force one.
- **Fix:** mirror the proven `solvent_registry.py` design. One
  `POLYMER_REGISTRY` with per-polymer `{aliases, solubility_key,
  optimization_key, canonical}`, exposing
  `resolve_polymer_for_solubility()`, `resolve_polymer_for_optimization()`,
  `resolve_polymer_canonical()`, and `build_polymer_regex(...)`.
- **Migrate call sites incrementally**, each behind the existing test suite;
  keep the old names as thin shims for one release to bound risk.

### P1.2 Migrate the last solvent-alias orphan — **Effort S · Risk Low**
PLAN.md's 10 solvent dicts are 9/10 consolidated into `solvent_registry.py`.
The straggler is `planning/extractors.py:40` `_SOLVENT_ALIASES` (23 entries),
used by `_extract_solvents()`.
- **Fix:** replace with `solvent_registry.resolve_to_*`. Retire PLAN.md once done.

### P1.3 Fix the Pareto phantom-anchor defect — **Effort M · Risk Med**
Characterized in `case-studies/02-cost-emissions-pareto/README.md`: the
engine's native `points[]` frontier is built from separate min-cost/min-objective
anchor solves that can return a `capital_cost=0` design costing ~$92k while
every real (capacity-building) design costs $2.5M+. That phantom dominates the
space and collapses the reported frontier to a single point.
`tools/waste_optimization.py:1229-1236` `_row_has_usable_economics` only rejects
a row when **≥2** of {CAPEX, OPEX, GWP} are zero — so `CAPEX=0, OPEX>0` slips
through.
- **Fix (two layers):** (a) tighten the guard to require `CAPEX>0 AND (OPEX>0
  OR GWP>0)`; (b) make the reported frontier a dominance-filter over
  `landscape_points` so `points[] ⊆ landscape_points` by construction (the
  case study already does this post-hoc — move it into the tool). This makes
  every future agent Pareto run return the rich frontier without a post-hoc fix.
- **Verify:** re-run case study 02 `--live`; Scenario B emissions should report
  3 frontier points natively (currently 2 incl. phantom); the 84 waste-opt
  tests stay green.

---

## P2 — Helper consolidation & dead-code removal

### P2.1 `message_utils.py` — collapse message helpers — **Effort M · Risk Low**
- `_get_last_human_message` ×3: `routing_message_state.py:438` (returns
  `None`, uses `isinstance(HumanMessage)`), `direct_fast_path.py:146` &
  `typed_runtime_integration.py:68` (return `""`, duck-typed, handle
  dict-role). Unify as `get_last_human_message(messages, *, default="")`.
- `_extract_text`/`_message_text` ×6: `agent.py:943`, `verifier.py:174`,
  `route_planner.py:341`, `direct_fast_path.py:132`,
  `typed_runtime_integration.py:54`, `typed_runtime_followups.py:138`. Two real
  variants (filter-empties vs keep-all) → one
  `extract_message_text(content, *, filter_empty=True)`.
- **Risk note:** preserve each caller's current default (None vs "") — the
  return-type divergence is load-bearing in a couple of spots.

### P2.2 `collections_utils.py` — dedupe helpers — **Effort S · Risk Low**
Seven dedupe helpers (`paths.py:15`, `waste_management/data_loader.py:35`,
`direct_fast_path.py:276`, `typed_runtime_integration.py:78`,
`planning/extractors.py:148`, `waste_optimization.py:1866` & `:2011`). Provide
`dedupe_strings(items, *, case_sensitive=True, strip=True)` and
`dedupe_paths(paths)`. The two `_stable_unique` in `waste_optimization.py` are
**exact duplicates 145 lines apart** — collapse first.

### P2.3 Float coercion — **Effort S · Risk Low**
`_f` is defined **twice in the same file** (`waste_optimization.py:1219,1242`,
identical); plus `services/biosteam_service.py:143` `_safe_float`,
`guardrails.py:716` `_coerce_float`. One module-level `coerce_float(value,
default=0.0)`.

### P2.4 Delete dead code — **Effort S · Risk Low**
- `vendor/_agent_sql_source.py` — **12,747 lines, zero importers**, PLAN.md
  labels it a legacy archive. Single biggest win.
- `routing_classifier.py:1323` `classify_query()` — never called in production
  (only defined). Remove (and its 16 test references, which test a dead path).
- `routing_classifier.py` `explain_routing_decision()` — test-only (24 test
  calls, 0 production). Either delete with its tests or relocate to a test
  helper if you want the diagnostic. Recommend **keep as a documented
  diagnostic** (`explain_routing_decision` is genuinely useful for debugging
  routing) but move its tests to assert current planner-first behavior.
- Dead "kept for API compat" args in `tools/visualization.py:265,645`
  (`table_name`, `*_column` on the plot tools) — the interpolation path ignores
  them. Deprecate: default to `None`, drop from the docstring's front.

### P2.5 Do **not** delete the keyword-fallback layer
For the record, because it looks dead: every module-level regex in
`routing_classifier.py` (lines ~49-296) and every `routing:` block in
`config/subagents/*.yaml` feed **only** the offline keyword fallback
(`classify_query_keywords` → `select_workflow_rules` → `plan_workflow_rules`,
consumed by `route_planner.fallback_route_plan`). They are dormant when the
planner backend is healthy but **must stay** for no-API/degraded operation.
Same for `_EXPLICIT_BIOSTEAM_ANALYSIS_RE`/`_NEGATED_BIOSTEAM_RE`/
`_VISUALIZATION_REQUEST_RE` — they double as plan-aware guards in
`routing_guards.py`. Consolidate their shared keyword strings (P3.3) rather
than removing them.

---

## P3 — Refactors, logging hygiene, repo hygiene

### P3.1 Split `waste_optimization.py` (5,581 lines, 123 defs) — **Effort L · Risk Med**
Natural seams verified: workbook parse/materialize (129-507), BioSTEAM sim
orchestration (1255-1466), Pareto solver ladder (2293-3052), frontier/landscape
processing (4049-4437). Split into `workbook.py`, `biosteam_update.py`,
`pareto_solver.py`, `frontier.py` behind the existing public tool functions.
Do this **after** P1.3 so the phantom-anchor fix lands in a small file.

### P3.2 Extract `_pareto_sweep()` template — **Effort M · Risk Med**
`waste_management/solver.py:520-720` has **5 epsilon-constraint sweeps**
(`pareto_profit_vs_emissions`, `_vs_ce`, `emissions_vs_ce`, `cost_vs_emissions`,
`cost_vs_ce`) that are ~95% identical (only objective sense, constraint var, and
`<=`/`>=` differ). One parameterized `_pareto_sweep(...)` + five 5-line
wrappers removes ~150 duplicated lines. Cover with a solver test first (SCIP is
now installed, so these run in CI).

### P3.3 De-drift the intent regexes — **Effort S · Risk Low**
Four BioSTEAM regexes in `routing_classifier.py:73-98` share the keyword set
`tea|lca|biosteam|capex|opex|msp|gwp` copy-pasted with variations. Define
`_BIOSTEAM_CORE_KEYWORDS` / `_BIOSTEAM_EXPANDED_KEYWORDS` constants and build the
regexes from them so they can't drift apart. Same for the triplicated polymer
regex (`routing_classifier.py:73,95` = `direct_fast_path.py:95`; `session_state.py:29`
is a reduced variant) — source it from `build_polymer_regex()` (P1.1).

### P3.4 logger over print in library code — **Effort M · Risk Low**
`waste_management/solver.py:333-627` has 25+ `print()` calls; also present in
`tools/waste_optimization.py`. These cannot be silenced in the server and leak
solver chatter into stdout (already seen interleaved with tool output). Add
`logger = logging.getLogger(__name__)` and downgrade to `logger.debug/info`.
(Vendor modules like `serpapi_*`, `wos_client` also print but are lower
priority.)

### P3.5 Instrument silent excepts — **Effort M · Risk Low**
~40 `except Exception: pass`/silent-fallback sites (e.g.
`services/visualization_service.py:297,340,352,362,373,385`,
`result_extractor.py:267-272`). Add `logger.warning(..., exc_info=True)` — don't
change control flow, just stop swallowing. Prioritize the routing/handoff/plot
paths where a silent failure becomes a mystery empty result.

### P3.6 BioSTEAM failure spam — **Effort S · Risk Low**
Live BioSTEAM subprocess failures print `No module named 'plastics'` once per
(solvent, polymer) — dozens of lines per Pareto run in this env. Bound it: log
the first failure at warning, subsequent at debug, and cache the "runner
unavailable" verdict per process so it isn't retried 200×.

### P3.7 Repo hygiene — **Effort S · Risk Low**
- `architecture/test_results/` — **137 tracked files (~5.4 MB)** of generated
  run artifacts. Add to `.gitignore` and `git rm --cached`; keep only curated
  summaries.
- Add `**/__pycache__/` to `.gitignore` (currently untracked but littering the
  working tree, e.g. `case-studies/_shared/__pycache__/`).
- Pin the langchain family (`langchain-anthropic`, `langchain-google-genai`,
  `langsmith`) to major versions to prevent silent breaking upgrades.
- `architecture/record_route_planner_goldens.py` hardcodes `/home/aaltamimi2`
  absolute paths in its case-study query strings (self-inflicted this session).
  Make them relative to the repo root.

---

## Monkey patches — reviewed, both justified (no action)

- `agent.py:920,935`: `deepagents_graph.SubAgentMiddleware =
  TracedSubAgentMiddleware` inside a try/finally that restores the original.
  Correct pattern; the swap is the only way to inject subagent tracing into
  deepagents. **Keep**, but add a one-line comment explaining why the global
  swap is necessary and that the finally-restore is load-bearing.
- `vendor/rag.py:155`: `sys.modules.setdefault("rag_module", ...)` for
  backward-compatible pickle deserialization. **Keep.**

---

## Suggested execution sequence

1. **P0.1–P0.3** — deploy-blockers (one PR, small, ship immediately).
2. **P1.1 + P1.2** — registries land (polymers + last solvent orphan), with
   shims; migrate call sites file-by-file behind tests.
3. **P1.3** — Pareto phantom fix in the current file, re-validate case study 02.
4. **P2.1–P2.4** — helper consolidation + dead-code deletion (the 13k-line
   vendor drop is a satisfying, zero-risk start).
5. **P3.1 + P3.2** — the big waste-optimization split and the sweep template,
   now that the phantom fix and helper moves are in.
6. **P3.3–P3.7** — regex de-drift, logging, hygiene.

Each tier is independently shippable and leaves the suite green. Nothing here
changes agent-facing behavior except P1.3 (which makes frontiers richer) and
the P0 pins (which only affect fresh installs).
```
```

---

## Appendix — quick-reference counts

| Category | Count | Worst offender |
|---|---|---|
| Dead vendor lines | 12,747 | `vendor/_agent_sql_source.py` |
| Polymer canonicalizers | 8 impls + 4 alias dicts | see P1.1 |
| `_extract_text`/`_message_text` | 6 defs | P2.1 |
| `_get_last_human_message` | 3 defs | P2.1 |
| dedupe helpers | 7 defs (2 exact dupes) | `waste_optimization.py` |
| `_f` float coercion | 2 exact dupes (same file) | `waste_optimization.py:1219,1242` |
| Pareto sweep funcs | 5 (~95% identical) | `waste_management/solver.py` |
| BioSTEAM intent regexes | 4 (shared keywords) | `routing_classifier.py:73-98` |
| Largest module | 5,581 lines | `tools/waste_optimization.py` |
| Deploy-blocking dep risks | 2 | biosteam/py-version, pandas/duckdb |
