# Agent Architecture Guide

A guide to understanding the DISSOLVE multi-agent system architecture through its key files.

## File Hierarchy (by importance)

```
multiagent-v2-tree2/
├── multi_agent_system.py      # [1] Core architecture - START HERE
├── agent_sql_final_1212_patched.py  # [2] Base agent and tool execution
├── tools/
│   ├── langchain_tools.py     # [3] All 76+ tools
│   └── precipitation.py       # [4] Precipitation analysis module
├── workflow_engine.py         # [5] Hybrid workflow patterns
└── docs/
    ├── ARCHITECTURE_GUIDE.md  # This file
    └── METHODS_SECTION.md     # Publication methods section
```

---

## 1. multi_agent_system.py (Core Architecture)

**This is the most important file.** It defines the routing logic, specialist agents, and graph construction.

### Key Sections

| Line | Component | Purpose |
|------|-----------|---------|
| ~448 | `LLM_ROUTER_PROMPT` | Prompt that teaches the LLM how to classify queries |
| ~520 | `class LLMRouter` | Routes queries via single LLM call (no orchestrator) |
| ~739 | `enhanced_complexity_router()` | Entry point for routing decisions |
| ~1492 | `SEPARATION_PLANNER_PROMPT` | Instructions for separation specialist |
| ~1545 | `TEA_LCA_ANALYST_PROMPT` | Instructions for TEA/LCA specialist |
| ~1600 | `LITERATURE_RESEARCHER_PROMPT` | Instructions for literature specialist |
| ~4407 | `multi_agent_router_node()` | Graph node that invokes routing |
| ~4505 | `build_multi_agent_graph()` | Constructs the LangGraph state machine |
| ~4527 | `separation_agent_node()` | Separation specialist execution |

### Architecture Flow

```
User Query
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  LLM_ROUTER_PROMPT (line ~448)                          │
│  - Defines 4 paths: fast, standard, specialist, integrated │
│  - Lists entity extraction rules                        │
│  - Specifies specialist triggers                        │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  LLMRouter.route() (line ~601)                          │
│  - Single LLM call (Gemini 2.0 Flash)                   │
│  - Returns: path, complexity, specialist, entities      │
│  - Cached for 5 minutes (RouterCache)                   │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  multi_agent_router_node() (line ~4407)                 │
│  - Extracts routing decision                            │
│  - Sets state: path, specialist, categories             │
│  - NO orchestrator agent - direct dispatch              │
└─────────────────────────────────────────────────────────┘
     │
     ├── Fast/Standard ──► sql_agent_node (direct tool execution)
     │
     └── Specialist/Integrated ──► Specialist Agent Nodes
                                   ├── separation_agent_node
                                   ├── tea_agent_node
                                   └── literature_agent_node
```

### Reading Order

1. **Start with `LLM_ROUTER_PROMPT`** (~line 448) - understand how queries are classified
2. **Read `LLMRouter` class** (~line 520) - see the routing mechanism
3. **Check specialist prompts** (~lines 1492-1650) - understand each specialist's capabilities
4. **Examine `build_multi_agent_graph()`** (~line 4505) - see how nodes connect

---

## 2. agent_sql_final_1212_patched.py (Base Agent)

The foundational agent that executes tools. All specialists ultimately use this for tool execution.

### Key Sections

| Component | Purpose |
|-----------|---------|
| `sql_agent_node()` | Core function that binds tools and executes LLM calls |
| `all_tools` | Combined list of all available tools |
| `agent_graph` | Base LangGraph agent (single-agent mode) |
| Tool binding | How tools are made available to the LLM |

### How It Connects

```
multi_agent_system.py                 agent_sql_final_1212_patched.py
        │                                        │
        │  imports                               │
        ├───────────────────────────────────────►│
        │                                        │
separation_agent_node() ──► sql_agent_node() ──► Tool Execution
        │                         │
        │  passes:                │  binds:
        │  - specialist_prompt    │  - selected tools
        │  - selected_categories  │  - LLM (Gemini)
        │                         │
        ▼                         ▼
   State with               Tool Results
   specialist context       returned to state
```

---

## 3. tools/langchain_tools.py (Tool Definitions)

All 76+ tools organized by category. Each tool is a `@tool` decorated function.

### Tool Categories

| Category | Count | Key Tools |
|----------|-------|-----------|
| `database` | 6 | `list_tables`, `describe_table`, `query_database` |
| `dissolution` | 7 | `analyze_polymer_dissolution`, `find_optimal_separation_conditions` |
| `separation` | 4 | `plan_sequential_separation`, `analyze_integrated_separation` |
| `advanced_separation` | 22 | Precipitation, atmospheric, antisolvent tools |
| `solvent_properties` | 4 | `get_solvent_properties`, `rank_solvents_by_property` |
| `visualization` | 5 | `plot_selectivity_heatmap`, `plot_atmospheric_feasibility` |
| `safety` | 4 | `get_pubchem_safety_info`, `compare_solvent_safety` |
| `economics` | 8 | `analyze_solvent_recovery_tea`, `compare_solvents_tea_lca` |
| `strap` | 5 | `analyze_strap_process`, `calculate_strap_msp` |
| `literature` | 4 | `search_google_scholar`, `search_web_of_science` |
| `rag` | 20 | `search_literature_rag`, `ask_literature` |
| `ml_prediction` | 1 | `predict_polymer_solubility_ml` |

### Key Tool Lists

```python
# Line ~1594 - Advanced separation tools including precipitation
ADVANCED_SEPARATION_TOOLS = [
    find_optimal_separation_sequence,
    compare_separation_algorithms,
    # ... separation tools ...
    find_differential_precipitation_solvents,
    analyze_multi_polymer_precipitation,
    check_atmospheric_feasibility,
    check_multi_polymer_atmospheric_feasibility,
    plot_atmospheric_feasibility,
    find_antisolvents,
    find_antisolvent_pairs,
    analyze_selective_antisolvent_precipitation,
]
```

### Tool Anatomy

```python
@tool                           # LangChain tool decorator
@safe_tool_wrapper              # Error handling wrapper
def check_atmospheric_feasibility(
    polymer1: str,              # Typed parameters
    polymer2: str,
    min_temperature_gap: float = 20.0,  # Defaults
) -> str:                       # Returns markdown string
    """Docstring becomes tool description for LLM."""

    conn = get_db_connection()  # Get database
    analyzer = PrecipitationAnalyzer(conn)  # Use analysis class
    results = analyzer.check_atmospheric_feasibility(...)
    return format_atmospheric_feasibility_results(results)
```

---

## 4. tools/precipitation.py (Precipitation Analysis)

Dedicated module for temperature-dependent solubility analysis.

### Key Classes

| Class | Purpose |
|-------|---------|
| `PrecipitationAnalyzer` | Main analysis class with database queries |
| `PrecipitationPoint` | Single polymer-solvent precipitation data |
| `DifferentialPrecipitationResult` | Two-polymer comparison result |
| `MultiPolymerAtmosphericResult` | N-polymer atmospheric feasibility |
| `AtmosphericFeasibilityResult` | 2-polymer atmospheric feasibility |

### Key Methods

```python
class PrecipitationAnalyzer:
    def get_solubility_curve(polymer, solvent)
        # Returns temperature vs solubility DataFrame

    def find_precipitation_temperature(polymer, solvent, threshold=1.0)
        # Returns temp where solubility < threshold

    def find_differential_precipitation_solvents(polymer_A, polymer_B, min_gap=20)
        # Finds solvents where A and B precipitate at different temps

    def check_atmospheric_feasibility(polymer1, polymer2)
        # Validates process works below solvent boiling point

    def check_multi_polymer_atmospheric_feasibility(polymers)
        # N-polymer sequential precipitation at 1 atm
```

### Data Flow

```
Database (DuckDB)
     │
     │  SQL query
     ▼
┌─────────────────────────────────────────┐
│  common_solvents_database               │
│  Columns: Polymer, Solvent, Temp, Sol%  │
└─────────────────────────────────────────┘
     │
     │  PrecipitationAnalyzer queries
     ▼
┌─────────────────────────────────────────┐
│  Analysis Results                       │
│  - Precipitation temperatures           │
│  - Temperature gaps                     │
│  - Atmospheric feasibility              │
└─────────────────────────────────────────┘
     │
     │  Formatting functions
     ▼
   Markdown output for LLM/user
```

---

## 5. workflow_engine.py (Hybrid Workflows)

Defines predefined workflow patterns for integrated queries.

### Workflow Patterns

| Workflow | Trigger | Flow |
|----------|---------|------|
| `tea_first` | >3 polymers | Profitability → Separation → TEA |
| `parallel_sep_lit` | separation + literature | Separation ∥ Literature → Aggregate |
| `standard_sep_tea` | separation + economics | Separation → Review → TEA |
| `literature_only` | literature without separation | Literature → Aggregate |

### Feedback Loop

```
Separation Agent
       │
       ▼
 Feedback Condition
 (cost > threshold?)
       │
       ├── No ──► Smart Aggregator ──► Response
       │
       └── Yes ──► Loop back (max 2x)
                   with new constraints
```

---

## Quick Start: Understanding the Architecture

### Step 1: Trace a Query

Follow a query like "Find antisolvents for LDPE" through the system:

1. **Routing** (`multi_agent_system.py:448`): `LLM_ROUTER_PROMPT` classifies as `specialist:separation`
2. **Dispatch** (`multi_agent_system.py:4407`): `multi_agent_router_node()` sets path
3. **Specialist** (`multi_agent_system.py:4527`): `separation_agent_node()` invoked
4. **Tool Selection**: Agent sees `SEPARATION_PLANNER_PROMPT` with antisolvent tools listed
5. **Execution** (`tools/langchain_tools.py:1595`): `find_antisolvents()` called
6. **Analysis** (`tools/precipitation.py`): Database queried for low-solubility solvents
7. **Response**: Formatted markdown returned to user

### Step 2: Key Code Patterns

```python
# Routing Decision (multi_agent_system.py)
decision = LLMRouter.route(query)  # Single LLM call
# Returns: path="specialist", specialist="separation", entities={...}

# Specialist Dispatch (multi_agent_system.py)
async def separation_agent_node(state):
    state_copy["specialist_prompt"] = SEPARATION_PLANNER_PROMPT
    state_copy["selected_categories"] = ["separation", "advanced_separation", ...]
    return await sql_agent_node(state_copy)

# Tool Execution (agent_sql_final_1212_patched.py)
async def sql_agent_node(state):
    tools = get_tools_for_categories(state["selected_categories"])
    response = await llm.bind_tools(tools).ainvoke(messages)
    return {"messages": [response]}
```

### Step 3: Adding New Capabilities

To add a new tool:

1. **Define tool** in `tools/langchain_tools.py`:
   ```python
   @tool
   @safe_tool_wrapper
   def my_new_tool(param: str) -> str:
       """Description for LLM."""
       # Implementation
       return "result"
   ```

2. **Add to category list** (e.g., `ADVANCED_SEPARATION_TOOLS`)

3. **Update specialist prompt** in `multi_agent_system.py` to mention the tool

4. **Update router prompt** if new keywords needed for routing

---

## Architecture Diagram

```
                              User Query
                                  │
                                  ▼
                         ┌───────────────┐
                         │  LLM Router   │  (Gemini 2.0 Flash)
                         │  Single Call  │
                         └───────┬───────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
         Fast Path         Standard Path      Specialist Path
         (simple)          (moderate)         (complex)
              │                  │                  │
              │                  │                  ▼
              │                  │         ┌───────────────────┐
              │                  │         │ Hybrid Workflow   │
              │                  │         │ Engine            │
              │                  │         └─────────┬─────────┘
              │                  │                   │
              │                  │     ┌─────────────┼─────────────┐
              │                  │     │             │             │
              │                  │     ▼             ▼             ▼
              │                  │ Separation    TEA/LCA      Literature
              │                  │   Agent        Agent         Agent
              │                  │     │             │             │
              │                  │     └─────────────┼─────────────┘
              │                  │                   │
              │                  │                   ▼
              │                  │            Feedback Loop
              │                  │            (if needed)
              │                  │                   │
              └──────────────────┴───────────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    Smart      │
                         │  Aggregator   │
                         └───────┬───────┘
                                 │
                                 ▼
                           Final Response
```

---

## File Size Reference

| File | Lines | Primary Responsibility |
|------|-------|----------------------|
| `multi_agent_system.py` | ~4800 | Architecture, routing, specialists |
| `agent_sql_final_1212_patched.py` | ~12500 | Base agent, tool execution |
| `tools/langchain_tools.py` | ~1700 | Tool definitions |
| `tools/precipitation.py` | ~1000 | Precipitation analysis |
| `workflow_engine.py` | ~800 | Workflow patterns |
