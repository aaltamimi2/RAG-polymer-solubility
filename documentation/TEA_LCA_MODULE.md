# TEA/LCA Module Documentation

## Overview

The TEA/LCA (Techno-Economic Analysis / Life Cycle Assessment) module provides cost and environmental impact calculations for polymer-solvent separation processes. It is designed as a **standalone file** (`tea_lca_module.py`) that TEA/LCA specialists can easily modify without understanding the agent architecture.

---

## Quick Start

### Agent Queries

Ask the agent natural language questions:

```
"Run TEA for toluene recovery at 100 kg/hr"
"What's the carbon footprint of using DMF?"
"Compare toluene, acetone, and ethanol on cost and emissions"
```

### Direct Python Usage

```python
from tea_lca_module import (
    run_full_tea_analysis,
    run_full_lca_analysis,
    compare_solvents_tea_lca
)

# TEA Analysis
tea = run_full_tea_analysis(
    solvent='toluene',
    polymer_throughput_kg_hr=100,
    solvent_to_polymer_ratio=10,
    recovery_fraction=0.95,
    process_temp_c=80
)
print(f"Capital cost: ${tea['capital_costs']['fixed_capital_investment_usd']:,.0f}")
print(f"Cost per kg: ${tea['economics']['cost_per_kg_polymer_usd']:.4f}/kg")

# LCA Analysis
lca = run_full_lca_analysis(
    solvent='toluene',
    polymer_throughput_kg_hr=100
)
print(f"CO2 emissions: {lca['emissions']['total_tonnes_co2eq_yr']:.2f} tonnes/year")
```

---

## Architecture

```
tea_lca_module.py (standalone - edit this file!)
│
├── Configuration Classes
│   ├── TEAConfig      → Financial parameters, operating hours, labor costs
│   ├── LCAConfig      → Emission factors for electricity, steam, solvents
│   └── SolventProperties → Boiling points, heat capacities, prices
│
├── Core Calculation Functions
│   ├── calculate_distillation_energy()
│   ├── estimate_equipment_cost()
│   ├── calculate_operating_costs()
│   └── calculate_carbon_footprint()
│
├── High-Level Analysis Functions (called by agent)
│   ├── run_full_tea_analysis()
│   ├── run_full_lca_analysis()
│   └── compare_solvents_tea_lca()
│
└── Formatting Functions (for agent output)
    ├── format_tea_results()
    ├── format_lca_results()
    └── format_comparison_results()
```

---

## Configuration Guide

### TEAConfig - Economic Parameters

```python
@dataclass
class TEAConfig:
    # Project timeline
    project_duration: int = 20          # years
    construction_years: int = 2
    start_year: int = 2026

    # Financial parameters
    IRR: float = 0.15                   # Internal Rate of Return (15%)
    income_tax: float = 0.21            # Corporate tax (21%)
    depreciation: str = 'MACRS7'        # Depreciation schedule

    # Operating parameters
    operating_days: int = 330           # Days per year
    operating_hours: int = 7920         # Hours per year

    # Cost factors (fraction of Fixed Capital Investment)
    lang_factor: float = 3.0            # Total capital multiplier
    maintenance: float = 0.03           # 3% of FCI per year
    property_tax: float = 0.01          # 1% of FCI per year
    property_insurance: float = 0.007   # 0.7% of FCI per year

    # Labor costs
    labor_cost_per_operator: float = 75000  # USD/year
    operators_per_shift: int = 2
    shifts_per_day: int = 3
    fringe_benefits: float = 0.40       # 40% of labor

    # Working capital
    working_capital_fraction: float = 0.05  # 5% of FCI
```

**To modify:** Edit the default values directly in `tea_lca_module.py`.

### LCAConfig - Emission Factors

```python
@dataclass
class LCAConfig:
    # Electricity emission factor (kg CO2eq / kWh)
    electricity_emission: float = 0.42  # US grid average

    # Steam/heat emission factor (kg CO2eq / MJ)
    steam_emission: float = 0.07        # Natural gas boiler

    # Solvent production emissions (kg CO2eq / kg solvent)
    solvent_emissions: Dict[str, float] = {
        'toluene': 1.2,
        'xylene': 1.3,
        'acetone': 2.1,
        'methanol': 0.8,
        'ethanol': 1.5,
        'dmf': 3.5,
        'dmso': 2.8,
        'thf': 4.2,
        'dcm': 1.8,
        'chloroform': 2.0,
        'hexane': 0.9,
        'cyclohexane': 1.1,
        'water': 0.001,
        'nmp': 3.8,
        'default': 2.0  # Unknown solvents
    }
```

**Sources:** EPA, GREET, ecoinvent databases.

### SolventProperties - Physical Data

```python
@dataclass
class SolventProperties:
    boiling_points: Dict[str, float]     # °C
    heat_vaporization: Dict[str, float]  # kJ/kg
    specific_heat: Dict[str, float]      # kJ/kg·K
    prices: Dict[str, float]             # USD/kg
```

**To add a new solvent:** Add entries to each dictionary in `tea_lca_module.py`.

---

## Agent Tools

### 1. `analyze_solvent_recovery_tea()`

**Purpose:** Full techno-economic analysis for solvent recovery.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| solvent | str | required | Solvent name |
| polymer_throughput_kg_hr | float | 100.0 | Polymer processing rate |
| solvent_to_polymer_ratio | float | 10.0 | Mass ratio solvent:polymer |
| recovery_fraction | float | 0.95 | Recovery efficiency (0-1) |
| process_temp_c | float | 80.0 | Process temperature |

**Output:**
```
# Techno-Economic Analysis Results

## Process Summary
- Solvent: toluene
- Polymer throughput: 100 kg/hr
- Solvent flow: 1000 kg/hr
- Recovery efficiency: 95.0%

## Capital Costs
| Equipment | Cost (USD) |
|-----------|------------|
| Distillation column | $1,352,175 |
| Heat exchanger | $606,866 |
| Pump | $121,661 |
| Storage tank | $203,791 |
| **Fixed Capital Investment** | **$2,741,392** |

## Economic Metrics
- Cost per kg polymer: $1.4251/kg
- Simple payback: 0.52 years
```

### 2. `analyze_solvent_recovery_lca()`

**Purpose:** Life cycle assessment for environmental impact.

**Output:**
```
# Life Cycle Assessment Results

## Annual Greenhouse Gas Emissions
| Source | Emissions (kg CO2eq/yr) |
|--------|-------------------------|
| Electricity | 107,027 |
| Steam | 149,838 |
| Solvent makeup | 475,200 |
| **Total** | **732,065** |

## Comparison to No-Recovery Baseline
- Emission reduction: 8,771,935 kg CO2eq/yr
- Reduction percentage: 92.3%
```

### 3. `compare_solvents_tea_lca()`

**Purpose:** Compare multiple solvents on cost AND environmental metrics.

**Output:**
```
# Solvent Comparison: TEA & LCA

| Solvent | FCI ($) | Annual Cost ($) | $/kg | CO2 (t/yr) | Rank |
|---------|---------|-----------------|------|------------|------|
| toluene | 2,741,392 | 1,128,674 | 1.43 | 732 | 1 |
| ethanol | 2,741,392 | 1,124,680 | 1.42 | 1126 | 2 |
| acetone | 2,741,392 | 1,174,201 | 1.48 | 1134 | 3 |

## Rankings
- Best Overall: toluene
- Lowest Cost: ethanol
- Lowest Emissions: toluene
```

---

## Calculation Methods

### Energy Calculation

```
Q_heating = m_dot × Cp × (T_boil - T_feed)
Q_vaporization = m_dot × H_vap × recovery_fraction
Q_total = Q_heating + Q_vaporization
```

### Equipment Cost Scaling

Based on Turton et al. correlations:

```
C = C_base × (Capacity / Capacity_base)^n × Material_factor × CEPCI_ratio
```

Where:
- n = 0.6 for distillation columns (six-tenths rule)
- Material factors: Carbon steel (1.0), Stainless steel (1.8), Hastelloy (3.5)
- CEPCI adjustment: 2020 to 2026

### Carbon Footprint

```
CO2_total = CO2_electricity + CO2_steam + CO2_solvent_makeup

CO2_electricity = kWh × 0.42 kg CO2eq/kWh
CO2_steam = MJ × 0.07 kg CO2eq/MJ
CO2_solvent = kg_makeup × emission_factor
```

---

## Example Queries

### Simple TEA
```
"What's the capital cost for recovering toluene at 100 kg/hr?"
"Calculate payback period for acetone solvent recovery"
```

### Simple LCA
```
"What's the carbon footprint of using DMF for polymer separation?"
"How much CO2 does ethanol recovery emit per year?"
```

### Comparison
```
"Compare toluene, acetone, and ethanol on cost and emissions"
"Which solvent is cheapest and greenest for LDPE separation?"
```

### Detailed Parameters
```
"Run TEA for toluene at 200 kg/hr with 98% recovery at 120°C"
"LCA analysis for DMF with 15:1 solvent ratio"
```

---

## Extending the Module

### Adding a New Solvent

1. Open `tea_lca_module.py`
2. Add to `SolventProperties`:
```python
boiling_points = {'new_solvent': 150.0, ...}
heat_vaporization = {'new_solvent': 400, ...}
specific_heat = {'new_solvent': 2.0, ...}
prices = {'new_solvent': 2.50, ...}
```
3. Add to `LCAConfig.solvent_emissions`:
```python
solvent_emissions = {'new_solvent': 2.5, ...}
```

### Modifying Cost Assumptions

Edit the defaults in `TEAConfig` or pass a custom config:

```python
from tea_lca_module import TEAConfig, run_full_tea_analysis

custom_config = TEAConfig(
    IRR=0.12,                    # Lower IRR
    operating_days=300,          # Fewer operating days
    labor_cost_per_operator=90000  # Higher labor cost
)

results = run_full_tea_analysis(
    solvent='toluene',
    polymer_throughput_kg_hr=100,
    config=custom_config
)
```

### Adding New Metrics

Add functions to `tea_lca_module.py`:

```python
def calculate_water_footprint(solvent: str, ...) -> Dict:
    """Calculate water usage for solvent recovery."""
    # Your calculation here
    return {...}
```

Then create an agent tool wrapper in `agent_sql_final_1212_patched.py`.

---

## Integration with Agent

The TEA/LCA tools are integrated into the DISSOLVE Agent:

1. **Import:** `tea_lca_module.py` is imported at agent startup
2. **Tools:** Three wrapper tools call the module functions
3. **System prompt:** Includes TEA/LCA tool descriptions for agent guidance
4. **Frontend:** "TEA/LCA Analysis" button with example queries

### Agent Reasoning Example

```
User: "Compare DMF vs DMSO for cost and carbon footprint"

Agent ITERATION 1:
  THINK: "User wants TEA/LCA comparison. I'll use compare_solvents_tea_lca."
  ACT:   compare_solvents_tea_lca(['DMF', 'DMSO'])
  OBSERVE: Comparison table with rankings

Agent ITERATION 2:
  THINK: "I have the data. Summarize for user."
  ACT:   Return formatted comparison with recommendation
```

---

## File Locations

| File | Purpose |
|------|---------|
| `tea_lca_module.py` | **EDIT THIS** - Standalone calculations |
| `agent_sql_final_1212_patched.py` | Agent tool wrappers (lines ~7100-7220) |
| `frontend/src/App.js` | TEA/LCA button (lines ~1304-1321) |
| `documentation/TEA_LCA_MODULE.md` | This documentation |

---

## References

- Turton, R., et al. "Analysis, Synthesis, and Design of Chemical Processes" (equipment cost correlations)
- GREET Model (Argonne National Laboratory) - emission factors
- ecoinvent database - solvent production emissions
- EPA eGRID - electricity emission factors
