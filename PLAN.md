# Solvent Alias Consolidation Plan

## Problem

10 disconnected solvent alias dicts across the codebase. Adding a new abbreviation requires edits in up to 5+ files.

## Inventory (actual count: 10)

| # | File | Variable | Entries | Canonical form |
|---|------|----------|---------|---------------|
| 1 | `solubility.py:100` | `SOLVENT_ALIASES` | 30 | lowercase interp-key (`"thf"`, `"ch2cl2"`) |
| 2 | `solubility.py:609` | `_SOLVENT_ALIASES` | 22 | lowercase solvent_data table key (`"tetrahydrofuran (thf)"`) |
| 3 | `tools/_helpers.py:161` | `SOLVENT_NAME_MAP` | 33 | `(property_db_name, gsk_db_name)` tuples |
| 4 | `tools/solvent_properties.py:212` | `ABBREVIATION_MAP` (local) | 26 | full lowercase for SQL LIKE |
| 5 | `tools/solvent_properties.py:384` | `SOLVENT_ALIASES` (local) | 11 | search term lists |
| 6 | `tools/biosteam_tea_lca.py:88` | `_SOLVENT_ALIASES` | 13 | title-case BioSTEAM name |
| 7 | `tools/advanced_separation.py:2004` | `_ABBREVIATION_MAP` | 26 | identical copy of #4 |
| 8 | `tools/visualization.py:89` | `SOLVENT_NAME_MAPPING` | 43 | one-to-many interp-key lists |
| 9 | `tools/visualization.py:380` | `ABBREVIATION_MAP` (local) | 26 | identical copy of #4 |
| 10 | `tools/solvent_lookup.py` | `_SOLVENT_DB` per-entry aliases | ~26 | price/GWP canonical key |

`vendor/_agent_sql_source.py` has a duplicate of #3 but is a legacy archive — not touched.

## Design: `src/strap/solvent_registry.py`

### Data structure

```python
SOLVENT_REGISTRY: dict[str, dict] = {
    "thf": {
        "interp_key":  "thf",                       # solubility coefficient key
        "property_db": "Tetrahydrofuran (THF)",      # solvent_data table name
        "gsk_db":      "THF",                        # gsk_dataset table name
        "biosteam":    "Tetrahydrofuran",             # BioSTEAM thermosteam name
        "bp_db_key":   "tetrahydrofuran (thf)",       # lowercase solvent_data key
        "aliases":     ["tetrahydrofuran", "thf", "oxolane"],
    },
    "dimethylformamide": {
        "interp_key":  "dimethylformamide",
        "property_db": "N,N-Dimethylformamide",
        "gsk_db":      "DMF",
        "biosteam":    "N,N-Dimethylformamide",
        "bp_db_key":   "dimethyl formamide (dmf)",
        "aliases":     ["dmf", "n,n-dimethylformamide", "dimethyl formamide"],
    },
    # ... one entry per canonical solvent (~40 entries)
}
```

### Derived indexes (built at import time)

```python
_ALIAS_TO_INTERP: dict[str, str] = {}   # flat alias→interp_key map
```

### Exported resolvers

```python
resolve_to_interp_key(name)   → str|None   # replaces SOLVENT_ALIASES in solubility.py
resolve_to_property_db(name)  → str|None   # replaces SOLVENT_NAME_MAP["property"]
resolve_to_gsk_db(name)       → str|None   # replaces SOLVENT_NAME_MAP["gsk"]
resolve_to_biosteam(name)     → str|None   # replaces _SOLVENT_ALIASES in biosteam
resolve_to_bp_db_key(name)    → str|None   # replaces _SOLVENT_ALIASES in solubility.py:609
resolve_for_databases(name, target)         # drop-in for normalize_solvent_name()
```

### Exported constants

```python
ABBREVIATION_MAP: dict[str, str]   # the shared 26-entry abbrev→full-name dict (for SQL LIKE)
```

## What changes per file

### `src/strap/solubility.py`
- **Delete** `SOLVENT_ALIASES` (line 100-131)
- **Delete** `_SOLVENT_ALIASES` (line 609-633)
- `resolve_solvent()`: replace `SOLVENT_ALIASES.get(norm)` → `resolve_to_interp_key(norm)`
- `get_boiling_point()`: replace `_SOLVENT_ALIASES.get(key)` → `resolve_to_bp_db_key(solvent)`
- `get_logp()`: same change as `get_boiling_point()`

### `src/strap/tools/_helpers.py`
- **Delete** `SOLVENT_NAME_MAP` (line 161-194)
- `normalize_solvent_name()`: replace body with `resolve_for_databases(name, target)`
- All downstream callers (visualization, advanced_separation, adaptive_separation) unchanged

### `src/strap/tools/biosteam_tea_lca.py`
- **Delete** `_SOLVENT_ALIASES` (line 88-102)
- `_expand_solvents()`: replace `_SOLVENT_ALIASES.get(s)` → `resolve_to_biosteam(s)`

### `src/strap/tools/advanced_separation.py`
- **Delete** `_ABBREVIATION_MAP` (line 2004-2033)
- Add `from strap.solvent_registry import ABBREVIATION_MAP as _ABBREVIATION_MAP`

### `src/strap/tools/solvent_properties.py`
- **Delete** local `ABBREVIATION_MAP` (line 212-241) inside `lookup_solvent_properties()`
- **Delete** local `SOLVENT_ALIASES` (line 384) inside `get_solvent_properties()`
- Import `ABBREVIATION_MAP` and `get_search_terms()` from registry

### `src/strap/tools/visualization.py`
- **Delete** local `ABBREVIATION_MAP` (line 380) inside `_lookup_solvent_properties_for_viz()`
- Import `ABBREVIATION_MAP` from registry
- **Keep** `SOLVENT_NAME_MAPPING` (line 89) — it has one-to-many expansion semantics not supported by the registry's single-result resolvers

### `src/strap/tools/solvent_lookup.py`
- **No changes** — `_SOLVENT_DB` contains price/GWP data, its aliases serve a different resolution domain

## Implementation order

1. Create `src/strap/solvent_registry.py` (no imports from any tool/model file)
2. Update `solubility.py` (most critical — everything chains from here)
3. Update `tools/_helpers.py` (propagates to 3+ downstream callers)
4. Update `tools/biosteam_tea_lca.py`
5. Update `tools/advanced_separation.py`
6. Update `tools/solvent_properties.py`
7. Update `tools/visualization.py`
8. Run test suite to verify no regressions

## Risks

- **Missing entries**: `bp_db_key` must match `solvent_data` table exactly — copy verbatim from current `_SOLVENT_ALIASES` at line 609
- **None values**: 6 entries in `SOLVENT_NAME_MAP` have `None` for property_db or gsk_db — preserve `None` in registry, `resolve_for_databases()` must return `None` (not fallback)
- **Circular import**: `solvent_registry.py` must import only from stdlib
- **`visualization.py` one-to-many**: `SOLVENT_NAME_MAPPING` stays local (follow-up task)
