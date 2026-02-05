# DISSOLVE Agent Test Queries

Run these from either worktree:
- **v2 (with routing/guardrails):** `/home/aaltamimi2/langchain-STRAP/`
- **v1 (original architecture):** `/home/aaltamimi2/langchain-STRAP-v1/`

Each command is self-contained with LangSmith tracing and token output limits.

---

## Category 1: Orchestrator-Only (no subagent delegation)

### 1.1 List available polymers

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'What polymers are available in the database?'}]},
    {'recursion_limit': 50},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:3000]); break
"
```

**Expected:** Orchestrator calls `list_available_polymers`, returns list. No subagent.

### 1.2 Solvent property lookup

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'What are the boiling point and LogP of toluene and xylene?'}]},
    {'recursion_limit': 50},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:3000]); break
"
```

**Expected:** Orchestrator calls `get_solvent_properties`. No subagent.

### 1.3 Direct SQL query

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'Query the database for the top 5 solvents with the highest solubility for LDPE at 120 degrees.'}]},
    {'recursion_limit': 50},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:3000]); break
"
```

**Expected:** Orchestrator calls `describe_table` then `query_database`. No subagent.

---

## Category 2: Single Subagent

### 2.1 Separation engineer — optimal sequence

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'Find the optimal separation sequence for LDPE, HDPE, PP, and PET at 120C.'}]},
    {'recursion_limit': 150},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:5000]); break
"
```

**Expected:** Routes to `separation-engineer`. Calls `find_optimal_separation_sequence`.

### 2.2 Safety analyst — GSK G-scores

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'What is the GSK G-score for toluene, and what safer alternatives exist in the same solvent family?'}]},
    {'recursion_limit': 150},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:3000]); break
"
```

**Expected:** Routes to `safety-analyst`. Calls `get_solvent_gscore` + `get_family_alternatives`.

### 2.3 TEA/LCA analyst — cost analysis

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'What is the techno-economic cost of recovering toluene at 100 kg/hr polymer throughput with 95% recovery?'}]},
    {'recursion_limit': 150},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:3000]); break
"
```

**Expected:** Routes to `tea-lca-analyst`. Calls `analyze_solvent_recovery_tea`.

### 2.4 Statistics/ML — statistical summary

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'Give me a statistical summary of solubility values in the common_solvents_database grouped by polymer, with confidence intervals.'}]},
    {'recursion_limit': 150},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:5000]); break
"
```

**Expected:** Routes to `statistics-ml`. Calls `statistical_summary`.

---

## Category 3: Multi-Subagent Sequential

### 3.1 Separation + Safety

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'Find the optimal separation sequence for LDPE, PP, and PET at 120C, then check the safety G-scores of the recommended solvents.'}]},
    {'recursion_limit': 250},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:5000]); break
"
```

**Expected:** Sequential delegation: `separation-engineer` first, then `safety-analyst`.

### 3.2 Separation + TEA

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ.setdefault('LANGSMITH_PROJECT', 'strap-agent')
from strap.agent import create_dissolve_agent
agent = create_dissolve_agent()
result = agent.invoke(
    {'messages': [{'role': 'user', 'content': 'Plan a separation sequence for HDPE and PS at 100C, then estimate the operating cost and payback period for the solvents used.'}]},
    {'recursion_limit': 250},
)
for msg in reversed(result['messages']):
    if hasattr(msg, 'content') and msg.type == 'ai' and msg.content:
        print(msg.content[:5000]); break
"
```

**Expected:** Sequential delegation: `separation-engineer` first, then `tea-lca-analyst`.

---

## Notes

- **recursion_limit** prevents runaway loops at the LangGraph level (50/150/250 by category)
- **SubagentGuardMiddleware** (v2 only) adds a second safety net: 25 iterations + 200K token cap per subagent
- **Output truncation**: each command truncates printed output to 3000-5000 chars to keep terminal clean
- **LangSmith**: all traces go to the `strap-agent` project — check at https://smith.langchain.com
- v1 does NOT have routing middleware or guardrails, so subagent queries may take longer or spiral — the recursion_limit is your only safety net there
