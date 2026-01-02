# DISSOLVE Agent - Core Architecture

## Overview

The **DISSOLVE Agent** (Data-Integrated Solubility Solver via LLM Evaluation) is an AI-powered system for polymer-solvent solubility analysis. It combines large language models (LLMs) with structured data analysis to provide intelligent, adaptive responses to complex separation engineering queries.

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│  - Modern UI with dark mode                             │
│  - Real-time chat interface                             │
│  - Inline plot rendering                                │
│  - Session management                                    │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/JSON
                   │
┌──────────────────▼──────────────────────────────────────┐
│              FastAPI Server (app_server.py)             │
│  - RESTful API endpoints                                │
│  - WebSocket support                                    │
│  - Static file serving (frontend + plots)              │
│  - Session management with thread-safe locks           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │
┌──────────────────▼──────────────────────────────────────┐
│       LangGraph Agent (agent_sql_final_1212_patched)    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  LLM (Google Gemini)                            │   │
│  │  - gemini-2.5-flash-lite (default, fast+cheap)  │   │
│  │  - gemini-2.5-flash (balanced)                  │   │
│  │  - gemini-2.5-pro (most capable)                │   │
│  └─────────────────────────────────────────────────┘   │
│                      ▲                                   │
│                      │                                   │
│  ┌──────────────────┴──────────────────────────────┐   │
│  │      Tool Execution Layer (34 tools)            │   │
│  │  - Database tools                               │   │
│  │  - Adaptive analysis tools                      │   │
│  │  - Solvent property tools                       │   │
│  │  - Statistical analysis tools                   │   │
│  │  - Visualization tools                          │   │
│  │  - ML prediction tools                          │   │
│  └──────────────────┬──────────────────────────────┘   │
└───────────────────┬─┴──────────────────────────────────┘
                    │
                    │
┌───────────────────▼────────────────────────────────────┐
│               Data Layer                                │
│  ┌────────────────────────────────────────────────┐    │
│  │ DuckDB (In-Memory SQL Database)                │    │
│  │ - Loaded from CSV files at startup             │    │
│  │ - 4 tables with ~10,600+ rows                  │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
│  CSV Data Sources:                                      │
│  - COMMON-SOLVENTS-DATABASE.csv (10,613 rows)          │
│  - Solvent_Data.csv (1,007 rows)                       │
│  - GSK_Dataset.csv (154 rows)                          │
│  - Polymer_HSPs_Final.csv (466 rows)                   │
│                                                         │
│  ML Models:                                             │
│  - Random Forest classifier (99.998% accuracy)         │
│  - Decision Tree for visualization                     │
│  - RED_values_complete_CORRECTED.csv (84 MB training)  │
└─────────────────────────────────────────────────────────┘
```

---

## Agent Framework: LangGraph

### What is LangGraph?

LangGraph is a framework for building **stateful, agentic workflows** using LLMs. Unlike simple prompt-response systems, LangGraph enables:

- **Cyclic execution**: Agent can call tools, process results, and decide next actions
- **State management**: Maintains conversation context and iteration tracking
- **Tool orchestration**: Parallel and sequential tool execution
- **Error recovery**: Graceful handling of failures

### Agent Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│                    START                                 │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   User Message        │
        │  (HumanMessage)       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Agent Node          │
        │  (sql_agent_node)     │
        │                       │
        │  1. Add system prompt │
        │  2. Call LLM          │
        │  3. Get response      │
        └───────────┬───────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Decision:     │
            │ Tool calls?   │
            └───┬───────┬───┘
                │       │
          Yes   │       │  No
                │       │
                ▼       ▼
    ┌───────────────┐  ┌──────────────┐
    │  Tool Node    │  │     END      │
    │               │  │  (Return to  │
    │ Execute tools │  │    user)     │
    │ in parallel   │  └──────────────┘
    └───────┬───────┘
            │
            ▼
    ┌───────────────────┐
    │  Tool Messages    │
    │  (ToolMessage)    │
    └───────┬───────────┘
            │
            │ (Loop back to Agent)
            │
            └──────────┐
                       │
                       ▼
            (Check iteration limit)
                       │
                       ▼
            (Continue or END)
```

### Key Components

#### 1. **Agent State** (`AgentState`)
Tracks conversation context across iterations:
```python
class AgentState(MessagesState):
    iteration_count: int      # Current iteration
    max_iterations: int       # Max allowed iterations (default: 50)
```

#### 2. **Agent Node** (`sql_agent_node`)
The "brain" of the system:
- Adds system prompt with tool descriptions
- Calls LLM with full message history
- Decides whether to call tools or respond to user
- Handles iteration tracking and limits

#### 3. **Tool Node** (`AsyncToolNode`)
Executes tool calls in **parallel** for performance:
- Extracts tool calls from LLM response
- Runs multiple tools concurrently via `asyncio.gather()`
- Handles both sync and async tools
- Returns `ToolMessage` results to agent

#### 4. **Message Types**
- `HumanMessage`: User input
- `SystemMessage`: Agent instructions and context
- `AIMessage`: LLM responses
- `ToolMessage`: Tool execution results

---

## Data Layer: DuckDB Integration

### Why DuckDB?

DuckDB is an **in-memory analytical database** that provides:
- Fast SQL queries (column-oriented storage)
- Zero external dependencies
- Pandas integration
- Low memory footprint

### Database Schema

#### Table 1: `common_solvents_database` (10,613 rows)
Primary solubility data from experiments.

| Column               | Type    | Description                     |
|----------------------|---------|---------------------------------|
| `solvent`            | VARCHAR | Solvent name                    |
| `temperature___c_`   | DOUBLE  | Temperature in Celsius          |
| `solubility____`     | DOUBLE  | Solubility percentage (0-100)   |
| `polymer`            | VARCHAR | Polymer name                    |

**Polymers available**: EVOH, HDPE, LDPE, LLDPE, Nylon6, Nylon66, PC, PES, PET, PMMA, PP, PS, PTFE, PVC, PVDF

**Example query**:
```sql
SELECT solvent, temperature___c_, solubility____
FROM common_solvents_database
WHERE polymer = 'LDPE' AND temperature___c_ = 25
ORDER BY solubility____ DESC
LIMIT 10
```

#### Table 2: `solvent_data` (1,007 rows)
Physical and chemical properties of solvents.

| Column         | Type    | Description                          |
|----------------|---------|--------------------------------------|
| `s_n`          | BIGINT  | Serial number                        |
| `solvent_name` | VARCHAR | Solvent name                         |
| `cas_number`   | VARCHAR | CAS registry number                  |
| `bp__oc_`      | DOUBLE  | Boiling point (°C)                   |
| `logp`         | DOUBLE  | LogP (lipophilicity/toxicity proxy)  |
| `cp__j_g_k_`   | DOUBLE  | Heat capacity (J/g·K)                |
| `energy__j_g_` | DOUBLE  | Energy cost (J/g)                    |

**Example query**:
```sql
SELECT solvent_name, bp__oc_, logp, energy__j_g_
FROM solvent_data
WHERE logp < 0  -- Less toxic solvents
ORDER BY energy__j_g_ ASC  -- Cheapest first
LIMIT 10
```

#### Table 3: `gsk_dataset` (154 rows)
GSK safety G-scores for solvents.

| Column                 | Type    | Description                   |
|------------------------|---------|-------------------------------|
| `classification`       | VARCHAR | Safety class (Recommended, etc.) |
| `solvent_common_name`  | VARCHAR | Solvent name                  |
| `cas_number`           | VARCHAR | CAS registry number           |
| `g_score`              | BIGINT  | Safety score (1-10, lower=safer) |

**Example query**:
```sql
SELECT solvent_common_name, g_score, classification
FROM gsk_dataset
WHERE g_score <= 3  -- Safest solvents
ORDER BY g_score ASC
```

#### Table 4: `polymer_hsps_final` (466 rows)
Hansen Solubility Parameters for ML predictions.

| Column               | Type    | Description                          |
|----------------------|---------|--------------------------------------|
| `number`             | BIGINT  | Index                                |
| `type`               | VARCHAR | Entry type                           |
| `polymer`            | VARCHAR | Polymer name                         |
| `dispersion`         | DOUBLE  | Dispersion parameter (δD)            |
| `polar`              | DOUBLE  | Polar parameter (δP)                 |
| `hydrogen_bonding`   | DOUBLE  | Hydrogen bonding parameter (δH)      |
| `interaction_radius` | DOUBLE  | Interaction radius (R₀)              |

---

## Tool-Based Architecture

The agent has access to **34 specialized tools** organized into 7 categories:

### 1. Core Database Tools (6 tools)
- Query databases
- Validate inputs
- Check column values
- Verify data accuracy

### 2. Adaptive Analysis Tools (5 tools)
- Find optimal separation conditions
- Adaptive threshold search
- Enhanced selectivity analysis
- **Sequential separation planning** (enumerates all permutations)
- View alternative sequences

### 3. Solvent Property Tools (4 tools)
- List solvent properties
- Get properties for specific solvents
- Rank by cost, toxicity, or boiling point
- Integrated separation + properties analysis

### 4. Statistical Analysis Tools (4 tools)
- Statistical summaries with confidence intervals
- Correlation analysis
- Hypothesis testing
- Regression analysis

### 5. Visualization Tools (6 tools)
- Temperature vs solubility curves (static + interactive)
- Selectivity heatmaps
- Multi-panel analysis
- Comparison dashboards
- Solvent property plots

### 6. GSK Safety Tools (3 tools)
- Get G-scores
- Find family alternatives
- Visualize safety rankings

### 7. Listing Tools (2 tools)
- List all available solvents
- List all available polymers

### 8. ML Prediction Tools (1 tool)
- Hansen-based solubility prediction with 5 visualizations

---

## Adaptive Analysis System

### Problem: Fixed Thresholds Don't Work

Traditional approaches use fixed thresholds (e.g., "selectivity must be > 30%"). This fails when:
- Data is sparse
- Polymers are chemically similar
- User needs "best available" even if not ideal

### Solution: Adaptive Thresholds

The agent **starts stringent and relaxes intelligently**:

```
SELECTIVITY_THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]
```

**Workflow**:
1. Try threshold 0.5 (50% selectivity)
2. If no results, try 0.4 (40%)
3. Continue until results found or all thresholds exhausted
4. Return best available with clear confidence metrics

**Example**:
```
User: "Find solvents to separate PP from PET"
Agent:
  - Try 50% selectivity -> No results
  - Try 40% selectivity -> No results
  - Try 30% selectivity -> Found 3 solvents  (OK)
  - Return: "Found 3 solvents at 30% selectivity threshold"
```

---

## Machine Learning Integration

### Hansen Solubility Parameters (HSP)

HSPs quantify molecular interactions:
- **δD (Dispersion)**: London dispersion forces
- **δP (Polar)**: Dipole-dipole interactions
- **δH (Hydrogen Bonding)**: Hydrogen bond strength

**Rule**: Similar HSPs -> High solubility

### ML Model

**Algorithm**: Random Forest Classifier
**Training Data**: 84 MB of RED (Relative Energy Difference) values
**Accuracy**: 99.998% on validation set
**Input**: Polymer HSPs + Solvent HSPs
**Output**: Solubility prediction + confidence

### Visualizations Generated

1. **3D Interactive Sphere** (HTML): User favorite! Rotatable 3D plot
2. **Radar Plot**: HSP parameter overlap
3. **RED Gauge**: Solubility likelihood meter
4. **HSP Comparison Bars**: Side-by-side parameters
5. **Text Summary**: Detailed prediction report

---

## Session Management

### Multi-Session Support

The FastAPI server supports **multiple concurrent users**:

```python
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.lock = threading.Lock()  # Thread-safe
```

**Session Isolation**:
- Each user gets a unique session ID (UUID)
- Messages and state isolated per session
- No cross-contamination between users

**Session Lifecycle**:
1. First chat message -> Create session
2. Subsequent messages -> Reuse session
3. Inactive 1 hour -> Auto-cleanup (future)

---

## Error Handling and Validation

### Input Validation

Before executing queries, the agent validates:
1. **Table exists** in database
2. **Columns exist** in table
3. **Values exist** in column (fuzzy matching for typos)
4. **Data types match** expected format

**Example**:
```
User: "Find solubility of LDPE in water at 25C"

Agent validates:
   (OK) Table 'common_solvents_database' exists
   (OK) Column 'polymer' exists
   (OK) Value 'LDPE' found in 'polymer' (896 rows)
   (OK) Column 'solvent' exists
   (OK) Value 'water' found in 'solvent' (128 rows)
  -> Execute query
```

### Error Recovery

The agent uses **safe_tool_wrapper** to catch and report errors gracefully:

```python
@safe_tool_wrapper
def my_tool(...):
    # If error occurs, wrapper catches it and returns user-friendly message
```

**User sees**:
```
 Tool Error (query_database): Column 'Polymer' not found.
 Did you mean 'polymer'? (case-sensitive)
```

---

## Performance Optimizations

### 1. Parallel Tool Execution
Multiple tool calls run concurrently:
```python
# Execute 3 tools in parallel (not sequential)
tool_messages = await asyncio.gather(*[
    execute_tool_call(tc1),
    execute_tool_call(tc2),
    execute_tool_call(tc3)
])
```

### 2. Memory Management
- Tool outputs truncated to 30,000 chars
- Periodic garbage collection
- Matplotlib figures closed after saving
- Pandas DataFrames deleted after use

### 3. Caching
- DuckDB loads CSV once at startup
- Gemini API responses not cached (dynamic)
- Frontend caches static assets

---

## Deployment Architecture

### Local Development
```
User -> http://localhost:8000 -> FastAPI -> Agent -> DuckDB
```

### Production (Render)
```
User -> https://app.onrender.com -> FastAPI -> Agent -> DuckDB
                                   ↓
                            Environment Variable:
                            GOOGLE_API_KEY (secret)
```

**Key Differences**:
- Production uses PORT from environment
- API key stored as Render secret
- Frontend uses `window.location.origin` for API base URL
- Auto-deploys from GitHub on push to `production` branch

---

## System Prompt Engineering

The agent's behavior is controlled by a **comprehensive system prompt** that includes:

1. **Mission Statement**: Be thorough and accurate
2. **Database Schema**: Exact table/column names (prevents hallucination)
3. **Behavioral Principles**:
   - Verify before reporting
   - Use adaptive thresholds
   - Action over explanation
4. **Mandatory Workflow**: Step-by-step guide
5. **Special Cases**: Tool-specific instructions
6. **Interpretation Guides**: How to read LogP, energy costs, etc.

**Result**: Agent behaves consistently and predictably while remaining flexible for edge cases.

---

## Technology Stack

| Component           | Technology                  | Purpose                        |
|---------------------|-----------------------------|--------------------------------|
| **Frontend**        | React, Tailwind CSS         | User interface                 |
| **Backend**         | FastAPI, Uvicorn            | Web server and API             |
| **Agent Framework** | LangGraph                   | Agentic workflow orchestration |
| **LLM Provider**    | Google Gemini               | Natural language understanding |
| **Database**        | DuckDB                      | In-memory SQL analytics        |
| **ML Framework**    | scikit-learn                | Random Forest classifier       |
| **Visualization**   | Matplotlib, Plotly, Seaborn | Static and interactive plots   |
| **Data Processing** | Pandas, NumPy               | Data manipulation              |
| **Deployment**      | Render                      | Cloud hosting (free tier)      |

---

## Future Enhancements

### Planned Features
1. **Persistent Storage**: Save user queries and results
2. **User Authentication**: Multi-user with private workspaces
3. **Batch Processing**: Upload custom CSV, run batch queries
4. **API Rate Limiting**: Prevent abuse on public deployment
5. **Advanced Caching**: Cache common queries (e.g., "list polymers")
6. **Real-time Collaboration**: Multiple users on same analysis
7. **Export Formats**: PDF reports, Excel exports
8. **Custom ML Models**: Train user-specific models

### Architectural Improvements
1. **Database Migration**: DuckDB -> PostgreSQL for persistence
2. **Async Database**: Use `duckdb` async mode
3. **Horizontal Scaling**: Multi-instance deployment with load balancer
4. **Monitoring**: Add Prometheus/Grafana for metrics
5. **Logging**: Structured logging with ELK stack

---

## Conclusion

The DISSOLVE Agent combines:
- **Intelligent LLM reasoning** (Google Gemini)
- **Structured data analysis** (DuckDB + SQL)
- **Adaptive algorithms** (threshold relaxation)
- **Machine learning** (Hansen parameter predictions)
- **User-friendly interface** (React + FastAPI)

This architecture enables natural language queries to complex polymer-solvent separation problems while maintaining accuracy, performance, and scalability.
