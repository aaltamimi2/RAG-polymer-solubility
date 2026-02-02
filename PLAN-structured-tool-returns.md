# Implementation Plan: Structured Tool Returns

## Executive Summary

Replace fragile regex-based extraction with structured tool returns where tools return:

```python
{
    "display": formatted_string,  # For LLM/user display
    "data": {                      # For programmatic access
        "cost_per_kg": 0.79,
        "capex_usd": 40_350_000,
        ...
    }
}
```

## Phase 1: Schema and Helper Functions

### 1.1 Add to `agent_schemas.py`

```python
class StructuredToolReturn(BaseModel):
    """Standard return format for tools needing both display and data."""
    display: str = Field(description="Formatted string for LLM/user display")
    data: Dict[str, Any] = Field(description="Structured data for programmatic access")

class STRAPToolOutput(ToolOutputBase):
    """Validated output from STRAP analysis tools."""
    tool_name: str = "strap_tool"
    polymers: List[str] = Field(default_factory=list)
    feedstock_composition: Dict[str, float] = Field(default_factory=dict)
    capacity_mt_yr: float = 10000.0
    tci_millions: Optional[float] = None
    unit_operating_cost: Optional[float] = None
    simple_payback_years: Optional[float] = None
    roi_pct: Optional[float] = None
    msp_by_polymer: Dict[str, float] = Field(default_factory=dict)
    gwp_by_polymer: Dict[str, float] = Field(default_factory=dict)
    recovery_steps: List[Dict[str, str]] = Field(default_factory=list)
```

## Phase 2: Modify Tool Implementations

### High Priority Tools

| Tool Name | File Line | Current Return |
|-----------|-----------|----------------|
| `plan_sequential_separation` | 3650 | `str` |
| `analyze_selective_solubility_enhanced` | 1932 | `str` |
| `analyze_solvent_recovery_tea` | 7561 | `str` |
| `analyze_strap_process` | 7896 | `str` |
| `calculate_strap_msp` | 8032 | `str` |

### Modification Pattern

**Before:**
```python
@tool
async def analyze_strap_process(...) -> str:
    results = tea_lca.run_full_strap_analysis(...)
    return format_output(results)  # Plain string
```

**After:**
```python
@tool
async def analyze_strap_process(...) -> str:
    results = tea_lca.run_full_strap_analysis(...)
    display = format_output(results)

    data = STRAPToolOutput(
        polymers=polymers,
        unit_operating_cost=results['tea']['economics']['unit_operating_cost_usd_kg'],
        simple_payback_years=results['tea']['economics']['simple_payback_years'],
        roi_pct=results['tea']['economics']['return_on_investment_pct'],
        msp_by_polymer=results['msp']['msp_by_polymer_usd_kg'],
        ...
    )

    return json.dumps({"display": display, "data": data.model_dump()})
```

## Phase 3: Update Extraction in workflow_engine.py

Replace `_extract_separation_from_messages()` and `_extract_tea_from_messages()`:

```python
def _extract_tea_from_messages(self, messages: List) -> Optional[Dict]:
    for msg in reversed(messages):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Try structured extraction first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "data" in parsed:
                data = parsed["data"]
                return {
                    "cost_per_kg": data.get("unit_operating_cost"),
                    "total_capex": data.get("tci_millions", 0) * 1e6 if data.get("tci_millions") else None,
                    "payback_years": data.get("simple_payback_years"),
                    "roi_pct": data.get("roi_pct"),
                    "msp_values": data.get("msp_by_polymer", {}),
                    "tool_output": parsed.get("display", content),
                }
        except json.JSONDecodeError:
            pass

        # Fallback to regex (remove after full migration)
        # ... existing regex code ...
```

## Phase 4: Migration Strategy

1. **Step 1**: Add schemas and helper (agent_schemas.py)
2. **Step 2**: Modify extraction methods to try JSON first, fall back to regex
3. **Step 3**: Migrate high-priority tools one by one
4. **Step 4**: Test each tool migration
5. **Step 5**: Remove regex fallback after all tools migrated

## Files to Modify

| File | Changes |
|------|---------|
| `agent_schemas.py` | Add `StructuredToolReturn`, `STRAPToolOutput` |
| `agent_sql_final_1212_patched.py` | Modify ~10 key tools |
| `workflow_engine.py` | Update extraction methods (lines 650-843) |
| `multi_agent_system.py` | Simplify smart_aggregator_node |

## Benefits

1. **Generalizable**: Any new tool follows same pattern
2. **Type-safe**: Pydantic validation ensures correct data types
3. **No regex**: Eliminates fragile pattern matching
4. **Backward compatible**: JSON string still works with existing code
5. **Self-documenting**: Schema defines expected output structure
