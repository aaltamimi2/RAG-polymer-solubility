# DISSOLVE Agent - Agentic Architecture

## What Makes This an "Agent"?

The **DISSOLVE Agent** (Data-Integrated Solubility Solver via LLM Evaluation) is not a simple chatbot or query system—it's an **autonomous reasoning agent** that thinks, acts, observes, and iterates to solve complex polymer-solvent separation problems.

### The ReAct Pattern: How the Agent Thinks

The agent operates using the **ReAct (Reasoning + Acting)** pattern, a fundamental concept in modern AI agents:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ReAct LOOP                                        │
│                                                                         │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐    │
│    │  THINK   │────▶│   ACT    │────▶│ OBSERVE  │────▶│  THINK   │    │
│    │          │     │          │     │          │     │  (again) │    │
│    │ "I need  │     │ Call     │     │ Process  │     │ "Now I   │    │
│    │  to find │     │ tools    │     │ tool     │     │  should  │    │
│    │  data"   │     │          │     │ results  │     │  ..."    │    │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘    │
│         │                                                    │         │
│         └────────────────────────────────────────────────────┘         │
│                            (iterate until done)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Example ReAct sequence:**

```
User: "Find the cheapest solvent to separate LDPE from PET at 25°C"

ITERATION 1:
  THINK: "I need to find solvents that selectively dissolve LDPE but not PET.
          I'll use the adaptive separation tool."
  ACT:   [Calls find_optimal_separation_conditions(target='LDPE', compare='PET')]
  OBSERVE: "Found 5 selective solvents: toluene (42%), xylene (38%), ..."

ITERATION 2:
  THINK: "Now I have selective solvents. I need to rank them by cost (energy).
          I'll call the solvent properties tool."
  ACT:   [Calls get_solvent_properties('toluene,xylene,chloroform,...')]
  OBSERVE: "Toluene: 450 J/g, Xylene: 524 J/g, ..."

ITERATION 3:
  THINK: "I have selectivity AND cost data. I can now provide a complete answer
          with recommendations. No more tools needed."
  ACT:   [Returns final answer to user with integrated analysis]
```

### Iterations vs Tool Calls

Understanding the difference is crucial:

- **Tool Call**: A single invocation of one tool (e.g., `query_database()`)
- **Iteration**: One complete think-act-observe cycle (one LLM API call)

**Multiple tools can execute in a single iteration** when the agent calls them in parallel:

```
ITERATION 1:
  THINK: "I need separation data AND safety data. These are independent queries."
  ACT:   [Calls 3 tools IN PARALLEL via asyncio.gather()]
         - find_optimal_separation_conditions()
         - get_solvent_gscore()
         - get_solvent_properties()
  OBSERVE: [All 3 results return simultaneously]

ITERATION 2:
  THINK: "I have all the data I need. Synthesize and respond."
  ACT:   [Returns comprehensive answer]
```

**Performance gain**: 3 tools in 1 iteration = ~3x faster than 3 sequential iterations

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        React Frontend                                    │
│  - Dark mode chat interface                                             │
│  - Inline plot rendering (PNG + interactive HTML)                       │
│  - Session management with localStorage persistence                      │
│  - Quick action buttons with cycling examples                           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP/JSON
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                     FastAPI Server (app_server.py)                       │
│  - RESTful API endpoints                                                │
│  - Static file serving (frontend + plots)                               │
│  - Thread-safe session management                                        │
│  - Export management (CSV with TTL cleanup)                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│              LangGraph Agent (agent_sql_final_1212_patched)              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    LLM (Google Gemini)                             │ │
│  │  - gemini-2.5-flash-lite (default: fast + cheap)                   │ │
│  │  - gemini-2.5-flash (balanced)                                     │ │
│  │  - gemini-2.5-pro (most capable)                                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              ▲ ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              Tool Execution Layer (38 tools)                       │ │
│  │                                                                    │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │
│  │  │  Database   │ │  Adaptive   │ │   Solvent   │ │ Statistical │ │ │
│  │  │   (6)       │ │ Analysis(7) │ │ Props (4)   │ │    (4)      │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │
│  │  │Visualization│ │ GSK Safety  │ │  Listing    │ │ ML Predict  │ │ │
│  │  │    (5)      │ │    (4)      │ │    (2)      │ │    (1)      │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │              PubChem External API (4 tools)                 │ │ │
│  │  │  - GHS hazard lookup, toxicity, safety comparison           │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                    │ │
│  │  Execution: asyncio.gather() for parallel tool calls               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│                          Data Layer                                      │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              DuckDB (In-Memory SQL Database)                      │   │
│  │  - 4 tables, ~12,000+ rows loaded from CSV at startup            │   │
│  │  - Indexed columns for fast queries                               │   │
│  │  - Column-oriented storage for analytics                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              ML Models                                            │   │
│  │  - Random Forest classifier (99.998% accuracy)                    │   │
│  │  - Trained on 84 MB of RED values                                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              External APIs (with timeouts)                        │   │
│  │  - PubChem: GHS hazards, toxicity, biodegradation data           │   │
│  │  - 10-20 second timeouts to prevent hanging                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Framework: LangGraph

### What is LangGraph?

LangGraph is a framework for building **stateful, agentic workflows**. Unlike simple prompt-response systems, LangGraph enables:

- **Cyclic execution**: Agent can call tools, process results, and decide next actions
- **State management**: Maintains conversation context and iteration tracking
- **Parallel tool orchestration**: Execute independent tools concurrently
- **Error recovery**: Graceful handling of failures with retry logic

### The Agent Graph

```
                    ┌─────────────────┐
                    │     START       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  User Message   │
                    │ (HumanMessage)  │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │       Agent Node            │
              │    (sql_agent_node)         │
              │                             │
              │  1. Inject system prompt    │
              │  2. Call LLM with history   │
              │  3. Get reasoning + tools   │
              │  4. Increment iteration     │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   Tool calls?   │
                    └───┬─────────┬───┘
                        │         │
                   Yes  │         │  No
                        │         │
                        ▼         ▼
          ┌─────────────────┐   ┌─────────────────┐
          │   Tool Node     │   │      END        │
          │                 │   │ (Return to user)│
          │ Execute tools   │   └─────────────────┘
          │ IN PARALLEL     │
          │ via asyncio     │
          └────────┬────────┘
                   │
                   │ Tool results
                   │ (ToolMessages)
                   │
                   └────────┐
                            │
              ┌─────────────▼─────────────┐
              │  Check iteration limit    │
              │  (max: 50 iterations)     │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Continue or END          │
              └───────────────────────────┘
                            │
                            │ (loop back to Agent Node)
                            │
                            └─────────────────────────┐
                                                      │
              ┌───────────────────────────────────────▼
              │        Agent Node (next iteration)
              └───────────────────────────────────────
```

### Parallel Tool Execution: The Key to Performance

When the agent needs multiple independent pieces of information, it calls tools **in parallel**:

```python
# Inside AsyncToolNode
async def execute_parallel_tools(tool_calls):
    # All tools run CONCURRENTLY, not sequentially
    results = await asyncio.gather(*[
        execute_single_tool(tc) for tc in tool_calls
    ])
    return results
```

**Real-world example:**

```
User: "Analyze LDPE separation with cost and safety data"

Agent ITERATION 1:
  THINK: "I need 3 types of data that don't depend on each other"

  PARALLEL TOOL CALLS (all execute simultaneously):
  ├── find_optimal_separation_conditions(target='LDPE', ...)
  ├── get_solvent_properties('toluene,xylene,chloroform')
  └── get_solvent_gscore('toluene,xylene,chloroform')

  Total time: ~2 seconds (not 6 seconds if sequential)

  OBSERVE: All 3 results available

Agent ITERATION 2:
  THINK: "I have all data. Synthesize comprehensive response."
  ACT: Return final answer with integrated analysis
```

**Performance improvement**: 4-6x faster with parallel execution

---

## Tool Categories (38 Total)

| Category | Count | Purpose |
|----------|-------|---------|
| Core Database | 6 | Query, validate, explore SQL data |
| Adaptive Analysis | 7 | Separation with intelligent threshold adaptation |
| Solvent Properties | 4 | Cost, toxicity, boiling point analysis |
| Statistical | 4 | Correlation, regression, hypothesis testing |
| Visualization | 5 | Static PNG and interactive HTML plots |
| GSK Safety (G-Score) | 4 | Industrial safety scoring |
| PubChem External | 4 | GHS hazards, LD50, biodegradation |
| Listing | 2 | Enumerate polymers and solvents |
| ML Prediction | 1 | Hansen-based solubility prediction |
| **Total** | **38** | |

### NEW: PubChem External API Tools

The agent can now query **live external data** from PubChem:

1. **get_pubchem_safety_info()** - GHS hazard pictograms, statements
2. **compare_pubchem_safety()** - Side-by-side safety comparison (max 5 solvents)
3. **visualize_pubchem_safety()** - Visual safety chart
4. **get_pubchem_toxicity()** - LD50, LC50, biodegradation, aquatic toxicity

All external API calls have **10-20 second timeouts** to prevent hanging.

---

## Adaptive Analysis: Intelligent Threshold Relaxation

### The Problem with Fixed Thresholds

Traditional systems fail when:
- Data is sparse for certain polymer combinations
- User needs "best available" even if not ideal
- Fixed thresholds return nothing or too much

### The Agent's Solution

The agent **starts stringent and relaxes intelligently**:

```
SELECTIVITY_THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]

Agent behavior:
  1. Try 50% selectivity threshold
  2. If no results → try 40%
  3. Continue until results found
  4. Return results with confidence metrics
  5. Explain why threshold was relaxed (if it was)
```

This is a form of **agent reasoning**—the tool itself adapts based on data availability.

---

## Data Layer: DuckDB + External APIs

### Internal Databases

| Table | Rows | Description |
|-------|------|-------------|
| common_solvents_database | 10,612 | Polymer-solvent solubility data |
| solvent_data | 1,007 | Solvent physical/chemical properties |
| polymer_hsps_final | 466 | Hansen Solubility Parameters |
| gsk_dataset | 154 | GSK safety G-scores |

### External Data Sources

| Source | Data Type | Timeout |
|--------|-----------|---------|
| PubChem REST API | GHS hazards, toxicity | 10-20s |

---

## Error Handling and Validation

### Input Validation (Prevent Hallucination)

Before executing queries, the agent validates:

```
User: "Find solubility of LDPE in water"

Agent validation:
  ✅ Table 'common_solvents_database' exists
  ✅ Column 'polymer' exists
  ✅ Value 'LDPE' found (896 rows)
  ✅ Column 'solvent' exists
  ✅ Value 'water' found (128 rows)
  → Execute query
```

### Safe Tool Wrapper

All tools use `@safe_tool_wrapper` decorator:

```python
@safe_tool_wrapper
def my_tool(...):
    # If error occurs, wrapper catches it
    # Returns user-friendly error message
    # Never crashes the agent
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | React, Tailwind CSS | Modern chat UI |
| Backend | FastAPI, Uvicorn | Web server + API |
| Agent Framework | LangGraph | Agentic workflow orchestration |
| LLM Provider | Google Gemini | Natural language understanding |
| Database | DuckDB | In-memory SQL analytics |
| ML Framework | scikit-learn | Random Forest classifier |
| Visualization | Matplotlib, Plotly, Seaborn | Static + interactive plots |
| External APIs | PubChem REST | Safety and toxicity data |
| Deployment | Render | Cloud hosting |

---

## Session Management

### Multi-User Support

```python
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.lock = threading.Lock()  # Thread-safe
```

- Each user gets unique session ID (UUID)
- Conversation history isolated per session
- Frontend persists button states in localStorage

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Typical query | 2-4 iterations |
| Simple lookup | 1-2 iterations |
| Complex analysis | 3-6 iterations |
| Parallel tools/iteration | Up to 5+ tools |
| Tool execution speedup | 4-6x with parallel |
| External API timeout | 10-20 seconds |

---

## Summary: Why This is an Agent

1. **Autonomous reasoning**: Decides which tools to call based on user query
2. **Multi-step iteration**: Thinks → Acts → Observes → Repeats
3. **Parallel execution**: Optimizes performance by running independent tools concurrently
4. **Adaptive behavior**: Relaxes thresholds when data is sparse
5. **External integration**: Fetches live data from PubChem APIs
6. **Error recovery**: Gracefully handles failures and provides useful feedback

The DISSOLVE Agent doesn't just answer questions—it **reasons about how to answer them**.
