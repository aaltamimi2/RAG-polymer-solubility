# DISSOLVE Routing Logic — Deep Dive

> How a user query like *"separate LDPE, PET, EVOH at 120 C"* becomes the right
> subagent call with the right polymer names, solvent names, and numeric parameters.

---

## 1. Three-Layer Classification

Every user message passes through up to three layers before a subagent is invoked.

```
User query
  │
  ▼
Layer 1 — LLM semantic classifier  (Gemini Flash)
  │  returns JSON: {"subagents":["separation-engineer"],"confidence":"HIGH"}
  │  on failure ↓
Layer 2 — Keyword/regex fallback
  │  scores each of 8 ROUTING_RULES, returns sorted matches
  │
  ▼
Layer 3 — Advisory hint injection
  │  appends [ADVISORY] text to the orchestrator's system prompt
  │  (the orchestrator LLM decides whether to actually delegate)
  │
  ▼
Orchestrator LLM decides → task(subagent_type="separation-engineer", ...)
```

**Key design choice:** routing is _advisory_, not forced. The hint is appended
to the system prompt; the orchestrator can override it.

---

## 2. ROUTING_RULES — Single Source of Truth

`src/strap/routing.py:35-170` defines 8 rules. Each rule has:

| Field | Purpose |
|-------|---------|
| `phrases` | Regex patterns → score **3** (strongest) |
| `high_stems` | Single-keyword signals → score **2** |
| `low_stems` | Weak signals → score **1** only if **2+ match** |
| `negatives` | Cancel the match entirely (score **-1**) |
| `priority` | Tiebreaker (lower = higher priority) |

### Scoring function (`_match_rule`, line 300)

```
Check negatives first → if any match, return -1 (blocked)
Check phrases         → if any regex matches, return 3
Check high_stems      → if any regex matches, return 2
Count low_stem hits   → if ≥ 2 match, return 1; else return 0
```

All matching uses `re.search()` on the lowercased query, so even "plain
string" phrases like `"life cycle"` work as regex.

### The 8 Rules

**separation-engineer** (priority 1)
```
phrases:   "separation sequence", "optimal separation", "selective solvent",
           "polymer separation", r"dissolve.*but not", r"\d+\s*scheme",
           r"separate\s+\w+\s+from", ...
high:      "precipitat", "antisolvent", "selectiv", "greedy", r"branch.and.bound"
low:       "separat", "dissolution", "dissolve"
negatives: "sql", "database", "list polymers", "list solvents", "describe table"
```
The negatives prevent queries like *"list all polymers in the database"* from
triggering the separation engineer.

**safety-analyst** (priority 2)
```
phrases:   r"g.score", "ghs hazard", "pubchem safety", "solvent safety"
high:      "pubchem", "gscore", "ld50", "lc50", "biodegradation", "safe"
low:       "hazard", "toxic"
```

**biosteam-analyst** (priority 3)
```
phrases:   r"techno.economic", "life cycle", "operating cost", "biosteam",
           r"\bpet\b.*(?:dissolution|simulation|solvent|tea|lca|msp|biosteam)",
           r"\bldpe\b.*(?:...)", r"\bevoh\b.*(?:...)"  # polymer+TEA combos
high:      "msp", "ghg", "payback", "biosteam", "capex", "opex"
low:       "tea", "lca", "emission", "cost", "gwp", "rigorous"
```
Note the bidirectional regex patterns: `\bpet\b.*(?:tea|lca|...)` AND
`(?:tea|lca|...).*\bpet\b` — catches *"PET TEA analysis"* and *"TEA for PET"*.

**scholar-researcher** (priority 4), **patent-researcher** (priority 5),
**rag-analyst** (priority 6), **visualization-specialist** (priority 7),
**statistics-ml** (priority 8) — follow the same pattern with domain-specific
keywords.

### Multi-agent execution patterns (lines 176-186)

```python
PARALLEL_PAIRS = {
    {"separation-engineer", "safety-analyst"},
    {"biosteam-analyst",    "safety-analyst"},
}

SEQUENTIAL_PAIRS = {
    ("separation-engineer", "biosteam-analyst"),        # sep first, then TEA
    ("separation-engineer", "visualization-specialist"),
    ("statistics-ml",       "visualization-specialist"),
    ("scholar-researcher",  "rag-analyst"),
}
```

When 2 rules match, the hint builder checks these sets to advise parallel vs
sequential execution order.

---

## 3. LLM Semantic Classifier

`classify_query_llm()` (line 223) sends the user query to Gemini Flash with
a system prompt built dynamically from ROUTING_RULES:

```
You are a query router for a polymer dissolution analysis system.
Given a user query, identify which specialist(s) should handle it.

Available specialists:
- separation-engineer: Separation sequences, selectivity, ...
- safety-analyst: GSK G-scores, PubChem hazard/GHS, ...
- biosteam-analyst: TEA, LCA, BioSTEAM process simulation, ...
  [... all 8 listed ...]

Respond with JSON only:
{"subagents": ["name1"], "confidence": "HIGH"|"MEDIUM"|"LOW"}

Rules:
- Return 1-3 subagent names ordered by relevance
- Return {"subagents": []} if the orchestrator can handle it directly
- "separation-engineer" handles dissolution, purification, sequences
- When BOTH separation AND safety → return both specialists
```

On failure (timeout, parse error), falls back to keyword matching silently.

---

## 4. How Polymer Names Flow (They Are NOT Extracted by Routing)

**Routing does not parse polymer names.** It only decides *which subagent*.
The polymer names travel through a different path:

```
User: "separate LDPE, PET, EVOH at 120C"
       │
       ├─ Routing: regex r"separate\s+\w+\s+from" or stem "separat"
       │  → matches separation-engineer (score 3 or 2)
       │  → advisory hint injected into system prompt
       │
       ├─ Orchestrator LLM reads the hint + user query
       │  → decides to call task(subagent_type="separation-engineer")
       │  → passes user's polymer names in the task description (natural language)
       │
       └─ Subagent's tools parse the polymer string
          → split on "," → strip → .upper()
          → resolve via POLYMER_ALIASES + substring matching
```

### Polymer normalization (`src/strap/solubility.py:133-161`)

```python
POLYMER_ALIASES = {
    "POLYETHYLENE": "HDPE",       # NOTE: maps to HDPE, not LDPE
    "NYLON 6": "NYLON6",
    "PA6": "NYLON6",
    "PA66": "NYLON66",
    "POLYCARBONATE": "PC",
    "POLYSTYRENE": "PS",
    "POLYVINYLCHLORIDE": "PVC",
    "POLYVINYL CHLORIDE": "PVC",
    "POLYPROPYLENE": "PP",
    # ...12 total aliases
}

def resolve_polymer(name, known_polymers):
    norm = name.strip().upper()
    if norm in known_polymers:           return norm        # exact
    alias = POLYMER_ALIASES.get(norm)
    if alias and alias in known_polymers: return alias      # alias
    for kp in known_polymers:                               # substring
        if norm in kp or kp in norm:     return kp
    return None
```

**Why it doesn't confuse polymers with solvents:**
- There is **no code-level disambiguation** between polymer and solvent strings.
  The system relies entirely on **parameter naming** — tools have separate
  parameters like `target_polymer` vs `solvent`, and the LLM maps user intent
  to the correct one.
- The `.upper()` / `.lower()` calls are **data normalization** to match storage
  conventions (polymers stored uppercase in the DB because they're abbreviations;
  solvents stored lowercase in the coefficient JSON). They are not a
  disambiguation mechanism.

### In separation tools (`src/strap/tools/advanced_separation.py:72`)

```python
def parse_polymer_list(polymers: str) -> list[str]:
    return [p.strip().upper() for p in polymers.split(',') if p.strip()]
```

### In the SQL database

```
polymer column:  queried with UPPER(polymer) = UPPER('LDPE')
solvent column:  queried with LOWER(solvent) = LOWER('toluene')
```

The case convention is the disambiguation mechanism — no semantic parsing needed.

---

## 5. How Solvent Names Flow — Unified Solvent Registry

All solvent alias resolution is consolidated into a single source of truth:
**`src/strap/solvent_registry.py`**.

### The registry (`solvent_registry.py`)

Each canonical solvent has one entry keyed by its **interp-key** (the lowercase
key used in the solubility coefficient JSON). Each entry carries the canonical
form for every subsystem:

```python
SOLVENT_REGISTRY = {
    "thf": {
        "interp_key":  "thf",                       # coefficient JSON key
        "property_db": "Tetrahydrofuran (THF)",      # solvent_data SQL table
        "gsk_db":      "THF",                        # gsk_dataset SQL table
        "biosteam":    "Tetrahydrofuran",             # BioSTEAM thermosteam name
        "bp_db_key":   "tetrahydrofuran (thf)",       # lowercase BP/LogP cache key
        "aliases":     ["tetrahydrofuran"],           # user-facing spellings
    },
    "dimethylformamide": {
        "interp_key":  "dimethylformamide",
        "property_db": "N,N-Dimethylformamide",
        "gsk_db":      "DMF",
        "biosteam":    "N,N-Dimethylformamide",
        "bp_db_key":   "dimethyl formamide (dmf)",
        "aliases":     ["dmf", "n,n-dimethylformamide", "dimethyl formamide"],
    },
    # ... 32 solvents total
}
```

A flat index `_ALIAS_TO_INTERP` is built at import time from all aliases,
enabling O(1) lookup from any user-facing name to the interp-key.

### Resolver functions

Each subsystem imports the resolver it needs — no local alias dicts:

| Subsystem | Import | Returns |
|-----------|--------|---------|
| Solubility coefficients | `resolve_to_interp_key(name)` | `"thf"`, `"ch2cl2"` |
| BP / LogP cache | `resolve_to_bp_db_key(name)` | `"tetrahydrofuran (thf)"` |
| Property DB / GSK DB | `resolve_for_databases(name, target)` | `"Tetrahydrofuran (THF)"` or `"THF"` |
| BioSTEAM | `resolve_to_biosteam(name)` | `"Tetrahydrofuran"` |
| SQL fuzzy match | `ABBREVIATION_MAP` (shared constant) | `"tetrahydrofuran"` for LIKE queries |
| Property search | `get_search_terms(name)` | `["tetrahydrofuran"]` for SQL LIKE |

### What was consolidated (8 dicts from 6 files)

| File | Old dict | Now replaced by |
|------|----------|----------------|
| `solubility.py` | `SOLVENT_ALIASES` (30 entries) | `resolve_to_interp_key()` |
| `solubility.py` | `_SOLVENT_ALIASES` (22 entries) | `resolve_to_bp_db_key()` |
| `tools/_helpers.py` | `SOLVENT_NAME_MAP` (33 entries) | `resolve_for_databases()` |
| `tools/biosteam_tea_lca.py` | `_SOLVENT_ALIASES` (13 entries) | `resolve_to_biosteam()` |
| `tools/advanced_separation.py` | `_ABBREVIATION_MAP` (26 entries) | `import ABBREVIATION_MAP` |
| `tools/solvent_properties.py` | `ABBREVIATION_MAP` (26 entries, local) | `import ABBREVIATION_MAP` |
| `tools/solvent_properties.py` | `SOLVENT_ALIASES` (11 entries, local) | `get_search_terms()` |
| `tools/visualization.py` | `ABBREVIATION_MAP` (26 entries, local) | `import ABBREVIATION_MAP` |

### What remains local (by design)

| File | Dict | Why separate |
|------|------|-------------|
| `tools/solvent_lookup.py` | `_SOLVENT_DB` per-entry aliases | Contains price/GWP data, not just names |
| `tools/visualization.py` | `SOLVENT_NAME_MAPPING` | One-to-many expansion (e.g., `"xylene"→["1,2-dimethylbenzene","1,4-dimethylbenzene"]`) |
| `tools/ml_prediction.py` | `common_to_iupac` | Maps to IUPAC names for ML model lookup |
| `tools/safety_pubchem.py` | `name_mapping` | PubChem CID lookup (small, scoped) |

Adding a new solvent abbreviation now requires editing **one file**
(`solvent_registry.py`) instead of five.

---

## 6. How Numbers Are Disambiguated

The system **does not parse numbers from the query itself**. Numbers flow
through tool parameters with explicit names — the LLM maps user intent
to the correct parameter.

### Temperature vs threshold — different parameter names

| User says... | LLM maps to... | Tool parameter |
|---|---|---|
| "at 120C" | `temperature=120.0` | `temperature_c`, `temperature`, `start_temperature` |
| "selectivity above 30" | `min_selectivity=30.0` | `min_selectivity`, `initial_selectivity`, `selectivity_threshold` |
| "solubility below 1%" | `precipitation_threshold=1.0` | `precipitation_threshold` |
| "20,000 tons/year" | `processing_capacity=20000` | `processing_capacity` |
| "60% plastic content" | `target_plastic_percent=60` | `target_plastic_percent` |

### Default values per context

**Temperature defaults** (°C):

| Tool | Default | Context |
|---|---|---|
| `rank_solvents_selectivity` | 120.0 | Selectivity ranking |
| `find_optimal_separation_sequence` | 120.0 | Sequence optimization |
| `plan_sequential_separation` | 120.0 | Multi-step planning |
| `calculate_selectivity_detailed` | 100.0 | Pairwise comparison |
| `rank_solvents_for_separation` | 100.0 | Solvent ranking |
| `build_compatibility_matrix` | 100.0 | Matrix building |
| `predict_solubility_ml` | 25.0 | ML prediction (room temp) |
| BioSTEAM `precipitation_temp_c` | 25.0 | Cooling precipitation |
| BioSTEAM `dissolution_temp_c` | per-solvent | From `_SOLVENT_DEFAULTS` |

**Selectivity/threshold defaults:**

| Parameter | Default | Meaning |
|---|---|---|
| `min_selectivity` | 5.0 | Min gap (percentage points) between target and other |
| `initial_selectivity` | 30.0 | Starting threshold for adaptive search |
| `selectivity_threshold` | 10.0 | For flagging "challenging" pairs |
| `precipitation_threshold` | 1.0 | Solubility % below which = precipitated |

**Adaptive threshold cascade** (`_helpers.py:430`):
```python
SELECTIVITY_THRESHOLDS = [50, 30, 20, 15, 10, 5, 2, 1, 0.5, 0.1]
SOLUBILITY_THRESHOLDS  = [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01]
```
The adaptive analyzer starts at the most stringent threshold and relaxes
until it finds results.

### Temperature range parsing

`analyze_selective_solubility_enhanced` accepts `temperature_range` as a string:
- `"25-120"` → splits on `"-"` → `temp_min=25, temp_max=120`
- `"100"` → `temp=100` with automatic ±5°C window (95–105)

### Extrapolation safety

The interpolation model flags extrapolation when temperature falls outside
the fitted range (`t_min_c` to `t_max_c` per polymer-solvent pair). The
R² threshold for reliability is 0.98 — below this, results carry a warning.

SQL fallback uses a 10°C tolerance window:
`temperature___c_ BETWEEN {T - 10} AND {T + 10}`.

---

## 7. Advisory Hint Injection

`RoutingMiddleware._inject_hint()` (line 576) runs on **every** model call:

**First call** (no completed subagents):
```
[ADVISORY: This query is well-suited for the "separation-engineer"
specialist (Separation sequences, selectivity, dissolution, precipitation).
Consider delegating via task(subagent_type="separation-engineer").
For simple queries you can also answer directly using your own tools.]
```

**Two agents — parallel:**
```
[ADVISORY: This query may benefit from multiple specialists:
"separation-engineer" (...) and "safety-analyst" (...).
You may delegate to them in parallel or sequentially as appropriate.]
```

**Two agents — sequential:**
```
[ADVISORY: This query may benefit from two specialists in sequence:
first "separation-engineer" (...), then "biosteam-analyst" (...).]
```

**After subagent calls** — progress tracking kicks in:
```
[PROGRESS: Completed subagents: separation-engineer.
Suggested next: task(subagent_type="biosteam-analyst") for TEA/LCA...
Remaining steps: biosteam-analyst.]
```

**All done:**
```
[PROGRESS: All subagent steps are complete.
Consider synthesizing findings from all subagents into a final answer.]
```

---

## 8. Middleware Execution Order

```python
middleware = [routing, output_verifier, orchestrator_guard]
```

Wrapped inside-out, so per model call:

```
1. routing._inject_hint()           ← adds [ADVISORY] or [PROGRESS]
   2. output_verifier._maybe_verify()  ← may re-invoke LLM once
      3. orchestrator_guard.check()     ← enforces token/tool budgets
         4. → actual LLM call →
      3. orchestrator_guard.post()      ← strips tool_calls if over budget
   2. output_verifier.post()           ← verifies final synthesis
1. routing._log_decision()           ← logs what tools the LLM chose
```

Subagents get their own `SubagentGuardMiddleware` only — no routing or
verifier middleware. The orchestrator is the only agent that receives
routing hints.

---

## 9. End-to-End Example

**User:** *"What's the safest way to separate LDPE from PET and EVOH at 120C?"*

**Step 1 — LLM classifier:**
Gemini Flash sees the query → returns
`{"subagents": ["separation-engineer", "safety-analyst"], "confidence": "HIGH"}`

**Step 2 — Hint builder:**
Checks `PARALLEL_PAIRS` → `{"separation-engineer", "safety-analyst"}` is a
parallel pair → builds parallel advisory hint.

**Step 3 — System prompt injection:**
```
[ADVISORY: This query may benefit from multiple specialists:
"separation-engineer" (Separation sequences...) and
"safety-analyst" (GSK G-scores...).
You may delegate to them in parallel or sequentially.]
```

**Step 4 — Orchestrator decides:**
Reads the hint + user query → calls `task(subagent_type="separation-engineer")`
with description *"Plan sequential separation for LDPE, PET, EVOH at 120C"*.

**Step 5 — Subagent tool call:**
`plan_sequential_separation(polymers="LDPE,PET,EVOH", temperature=120.0)`
- `parse_polymer_list` → `["LDPE", "PET", "EVOH"]` (uppercase)
- For each pair, queries DB: `UPPER(polymer) = 'LDPE'` + solvent results
- Solvents resolved lowercase: `resolve_solvent("toluene", known_solvents)`
- Temperature 120.0 used for interpolation; BP check ensures solvent BP > 120°C

**Step 6 — Progress tracking:**
Next model call: `_extract_completed_subagents()` finds `["separation-engineer"]`
→ injects `[PROGRESS: Completed: separation-engineer. Next: safety-analyst.]`

**Step 7 — Safety analysis:**
Orchestrator calls `task(subagent_type="safety-analyst")` with the solvents
from step 5. Safety tools resolve names via the unified `solvent_registry`
or their own small PubChem-specific lookup.

**Step 8 — Synthesis + verification:**
Orchestrator produces final answer → `OutputVerifierMiddleware` sends it to
Gemini Flash for scientific accuracy check → if `pass: true`, returns as-is.

---

## 10. Known Edge Cases

1. **"polyethylene" → HDPE** (not LDPE): `POLYMER_ALIASES` maps "POLYETHYLENE"
   to "HDPE". A user saying "polyethylene" when they mean LDPE will get HDPE.

2. **Missing long-form aliases**: "polyethylene terephthalate" has no alias to
   "PET". "ethylene vinyl alcohol" has no alias to "EVOH".

3. **Three local alias dicts remain outside the registry**: `visualization.py`
   (`SOLVENT_NAME_MAPPING`, one-to-many), `ml_prediction.py` (`common_to_iupac`,
   IUPAC names), and `safety_pubchem.py` (`name_mapping`, PubChem CIDs). These
   serve different purposes and won't auto-update when the registry grows.

4. **Negative temperatures in range strings**: `temperature_range="-10-120"`
   would mis-parse (splits on first `"-"`).

5. **Single low-stem doesn't trigger**: A query with just "cost" (one low-stem
   for biosteam-analyst) scores 0. Needs "cost" + "emission" (two low-stems)
   to reach score 1.

6. **"PET" ambiguity in biosteam routing**: The regex
   `r"\bpet\b.*(?:tea|lca|...)"` uses word boundaries, so "PET" the polymer
   matches but "carpet" does not. However, "pet" (the animal) in a stray
   query would also match.

7. **BP/LogP unavailable for some solvents**: `methanol`, `ethanol`, `toluene`,
   `benzene`, `hexane` have `bp_db_key: None` in the registry. If their
   interp-key doesn't directly match the `solvent_data` table's lowercase
   `solvent_name`, BP/LogP alias lookup will silently return `None`.
