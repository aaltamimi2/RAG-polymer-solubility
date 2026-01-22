# Contributing to Polymer Solubility App

Welcome to the team! This guide will help you get started adding new tools and workflows to the polymer solubility analysis agent.

## Quick Overview

- **Ali** handles the frontend and server (`app_server.py`, `frontend/*`)
- **You** can add new analysis tools, databases, and workflows
- All tools live in one file: `agent_sql_final_1212_patched.py`

---

## Getting Started

### 1. Clone the Repository

```bash
# Clone the repo to your local machine
git clone https://github.com/aaltamimi2/RAG-polymer-solubility.git
cd RAG-polymer-solubility

# Check you're on main branch
git branch
```

### 2. Create Your Own Branch

Always work on your own branch - never commit directly to `main`.

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Examples:
git checkout -b feature/logd-contaminant-analysis
git checkout -b feature/new-solvent-database
git checkout -b fix/temperature-calculation-bug
```

### 3. Set Up Your Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy the environment file and add your API keys
cp .env.example .env
# Edit .env with your keys (ask Ali if you need access)
```

### 4. Verify Everything Works

```bash
# Start the server
source .env
python -m uvicorn app_server:app --host 0.0.0.0 --port 8000

# You should see:
# - 54 tools loaded
# - 4 tables loaded
# - Server running at http://localhost:8000
```

---

## Adding New Tools

### Where Tools Live

All agent tools are defined in `agent_sql_final_1212_patched.py`. Scroll to around line 8430 to see the `SQL_AGENT_TOOLS` list.

### Tool Template

Here's the pattern to follow when adding a new tool:

```python
@tool
def analyze_contaminant_by_logd(
    contaminant: str,
    target_logd_range: Optional[tuple] = None,
    temperature: float = 25.0
) -> str:
    """
    Analyze contaminant removal potential based on LogD partitioning data.

    **When to use**: For evaluating separation strategies based on
    lipophilicity/hydrophilicity of contaminants.

    Args:
        contaminant: Name of the contaminant to analyze
        target_logd_range: Optional (min, max) LogD range to filter results
        temperature: Temperature in Celsius (default: 25.0)

    Returns:
        Formatted analysis with recommended separation conditions

    Examples:
        - "Analyze styrene contamination using LogD data"
        - "What solvents can remove PVC based on LogD?"
        - "Show LogD profile for common plastic additives"
    """
    try:
        # Your analysis logic here
        # Query databases, run calculations, format results

        results = []
        # ... do work ...

        # Always return a formatted markdown string
        output = f"""# LogD Analysis: {contaminant}

**Temperature:** {temperature}°C

## Results

| Solvent | LogD | Selectivity |
|---------|------|-------------|
| Example | 2.3  | High        |

## Recommendations

Based on the LogD profile, consider...
"""
        return output

    except Exception as e:
        logger.error(f"LogD analysis failed: {e}")
        return f"Analysis failed: {str(e)}. Please check the contaminant name and try again."
```

### Register Your Tool

After defining your tool, add it to the `SQL_AGENT_TOOLS` list:

```python
SQL_AGENT_TOOLS = [
    # ... existing tools ...

    # === YOUR NEW TOOLS (add at the end) ===
    analyze_contaminant_by_logd,
    calculate_logd_selectivity,
    # ... etc ...
]
```

### Key Conventions

| Do                                           | Don't                                 |
| Do: Add new tools at the end of the file     | Don't: Modify existing working tools  |
| Do: Return formatted markdown strings        | Don't: Return raw dicts/lists         |
| Do: Include helpful docstrings with examples | Don't: Leave sparse documentation     |
| Do: Handle errors gracefully with try/except | Don't: Let exceptions crash the agent |
| Do: Test your tool before committing         | Don't: Push untested code             |

---

## Adding New CSV Databases

### Step 1: Prepare Your CSV

- Place your CSV file in the `data/` directory
- Use clear column names (no spaces, use underscores)
- Ensure consistent data types in each column

Example structure for `logd_contaminants.csv`:
```csv
compound,logd_ph2,logd_ph7,logd_ph10,molecular_weight,category
styrene,-0.5,1.2,1.1,104.15,monomer
bpa,2.1,3.4,2.8,228.29,additive
```

### Step 2: Register the CSV for Auto-Loading

The system auto-loads CSVs from `data/`. To ensure your CSV is loaded with proper indexing, add it to the loading logic in `agent_sql_final_1212_patched.py`.

Find the `load_csv_files()` method in the `SQLDatabase` class and add your table:

```python
def load_csv_files(self):
    """Load CSV files into DuckDB."""
    csv_files = {
        "solvent_data": "Solvent_Data.csv",
        "polymer_hsps_final": "POLYMER-HSPs-FINAL.csv",
        "gsk_dataset": "GSK-Solvent-Sustainability-Guide.csv",
        "common_solvents_database": "COMMON-SOLVENTS-DATABASE.csv",
        # Add your new CSV here:
        "logd_contaminants": "logd_contaminants.csv",
    }
```

### Step 3: Add Indexes (Optional but Recommended)

For faster queries, add indexes on frequently-searched columns:

```python
# In the load_csv_files method, after loading your table:
if table_name == "logd_contaminants":
    self.conn.execute("CREATE INDEX idx_logd_compound ON logd_contaminants(compound)")
    self.conn.execute("CREATE INDEX idx_logd_category ON logd_contaminants(category)")
```

### Step 4: Query Your New Table

```python
@tool
def get_logd_data(compound: str) -> str:
    """Get LogD data for a compound."""
    try:
        result = _sql_db.execute_query("""
            SELECT * FROM logd_contaminants
            WHERE LOWER(compound) LIKE LOWER(?)
        """, [f"%{compound}%"])

        if not result:
            return f"No LogD data found for '{compound}'"

        # Format and return results
        # ...
    except Exception as e:
        return f"Query failed: {e}"
```

---

## Database Reference

### Existing Tables

| Table | Rows | Key Columns | Use For |
|-------|------|-------------|---------|
| `solvent_data` | 1007 | polymer, solvent, temperature, solubility | Main solubility lookups |
| `polymer_hsps_final` | 466 | Polymer, dD, dP, dH, Type | Hansen parameters |
| `gsk_dataset` | 154 | Solvent, Safety, Health, Env, Overall | GSK safety scores |
| `common_solvents_database` | 10612 | solvent, property, value, unit | Solvent properties (BP, density, etc.) |

### Query Examples

```python
# Simple query
result = _sql_db.execute_query(
    "SELECT * FROM solvent_data WHERE polymer = ?",
    ["PVDF"]
)

# Join tables
result = _sql_db.execute_query("""
    SELECT s.*, g.Overall as safety_score
    FROM solvent_data s
    LEFT JOIN gsk_dataset g ON LOWER(s.solvent) = LOWER(g.Solvent)
    WHERE s.polymer = ? AND s.temperature = ?
""", ["PS", 120])

# Aggregation
result = _sql_db.execute_query("""
    SELECT polymer, AVG(solubility) as avg_sol, COUNT(*) as n
    FROM solvent_data
    GROUP BY polymer
    ORDER BY avg_sol DESC
""")
```

---

## Testing Your Changes

### Quick Tool Test (No Server)

```bash
source .env
python3 -c "
from agent_sql_final_1212_patched import your_new_tool
result = your_new_tool.invoke({'compound': 'styrene'})
print(result)
"
```

### Full Integration Test

```bash
# Start the server
source .env
python -m uvicorn app_server:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000 and test your tool via the chat interface
# Try queries like: "Analyze styrene using LogD data"
```

### Verify Tool Count

When the server starts, check that your tool was loaded:
```
INFO - 🔧 Agent Tools: 55  # Should increase by the number of tools you added
```

---

## Submitting Your Changes

### 1. Commit Your Work

```bash
# Check what you've changed
git status

# Stage your changes
git add agent_sql_final_1212_patched.py
git add data/your_new_file.csv

# Commit with a descriptive message
git commit -m "Add LogD-based contaminant analysis tools

- Added analyze_contaminant_by_logd tool
- Added logd_contaminants.csv database
- Added indexes for compound and category columns"
```

### 2. Push to GitHub

```bash
# Push your branch to GitHub
git push -u origin feature/your-feature-name
```

### 3. Create a Pull Request

1. Go to https://github.com/aaltamimi2/RAG-polymer-solubility
2. You'll see a banner suggesting to create a PR for your branch - click it
3. Or go to "Pull Requests" > "New Pull Request"
4. Set:
   - **base:** `main`
   - **compare:** `feature/your-feature-name`
5. Fill in the PR template:
   - What does this add/change?
   - How did you test it?
   - Any new dependencies?
6. Submit and tag Ali for review

### 4. Address Review Feedback

Ali will review and may request changes. To update your PR:

```bash
# Make the requested changes
git add .
git commit -m "Address review feedback: fix edge case handling"
git push
```

The PR will automatically update.

---

## Files You Should NOT Edit

These are managed by Ali - reach out if you think changes are needed:

- `app_server.py` - FastAPI server and endpoints
- `frontend/*` - React frontend
- `services/*` - Issue reporting system
- `.env` - API keys (never commit this!)

---

## Tips for Success

1. **Start small** - Get one tool working before building a whole suite
2. **Test locally first** - Don't push broken code
3. **Write good docstrings** - The agent uses these to decide when to use your tool
4. **Return markdown** - The frontend renders markdown, so format your output nicely
5. **Handle edge cases** - What if the compound isn't found? What if the CSV is empty?

