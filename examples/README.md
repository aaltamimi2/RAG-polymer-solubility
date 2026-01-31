# Reproducibility Examples

This directory contains example scripts that demonstrate key analytical capabilities of the Polymer Solubility Analysis system.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Ensure data files are present:**
   - `./data/COMMON-SOLVENTS-DATABASE.csv`
   - `./data/Solvent_Data.csv`

3. **Server must be running** (optional for standalone scripts):
   ```bash
   python app_server.py
   ```

## Available Examples

### 1. Separation Analysis: PVDF/PET

**File:** `separation_pvdf_pet.py`

Demonstrates optimal solvent selection for separating PVDF from PET contamination.

**What it does:**
- Searches for solvents with high selectivity (≥30)
- Analyzes temperature range: 25-160°C
- Exports results to CSV

**Run:**
```bash
python examples/separation_pvdf_pet.py
```

**Expected output:**
- Optimal solvent recommendations
- Temperature-solubility profiles
- CSV export with ranked alternatives

---

### 2. Temperature Optimization

**File:** `temperature_optimization.py`

Analyzes how temperature affects polymer solubility in different solvents.

**What it does:**
- Temperature sweep for PVDF in NMP
- Compares multiple solvents at fixed temperature (80°C)
- Exports data for external plotting

**Run:**
```bash
python examples/temperature_optimization.py
```

**Expected output:**
- Solubility vs temperature data
- Ranked solvent comparison at 80°C
- CSV exports for both analyses

---

### 3. Statistical Comparison

**File:** `statistical_comparison.py`

Performs statistical tests to compare solubility profiles between polymer types.

**What it does:**
- Independent t-tests comparing polymer groups
- Effect size calculation (Cohen's d)
- Statistical significance testing (p-values)

**Run:**
```bash
python examples/statistical_comparison.py
```

**Expected output:**
- Test statistics and p-values
- Effect sizes and confidence intervals
- Interpretation guidance

---

## Customization

All scripts can be modified to analyze different:
- **Polymers:** Change `target_polymer` or `group1`/`group2` parameters
- **Solvents:** Modify SQL queries to filter specific solvents
- **Temperature ranges:** Adjust `start_temperature` and range limits
- **Export options:** Enable/disable CSV export with `export_csv=True/False`

## Output

### CSV Exports

When `export_csv=True` is set, results are saved to `./exports/` directory with:
- **Filename format:** `{tool_name}_{export_id}_{timestamp}.csv`
- **TTL:** 30 minutes (automatic cleanup)
- **Access:** Available via `/api/export/{export_id}` endpoint

### Console Output

All examples print:
- Formatted results with markdown tables
- Statistical summaries
- Interpretation guidance

## Extending Examples

To create your own reproducibility script:

1. **Import required tools:**
   ```python
   from agent_sql_final_1212_patched import query_database, find_optimal_separation_conditions
   ```

2. **Use async execution:**
   ```python
   async def main():
       result = await query_database(sql_query="...", export_csv=True)
       print(result)

   if __name__ == "__main__":
       asyncio.run(main())
   ```

3. **Enable CSV export for data reproducibility:**
   ```python
   result = await tool_function(..., export_csv=True)
   ```

## Citation

If you use these examples in your research, please cite:

```bibtex
@software{polymer_solubility_examples,
  title = {Polymer Solubility Analysis Examples},
  year = {2025},
  url = {https://github.com/yourusername/polymer-solubility-app}
}
```

## Troubleshooting

**Import errors:**
- Ensure you're running from the repository root
- Check that `agent_sql_final_1212_patched.py` exists

**Database errors:**
- Verify CSV files are in `./data/` directory
- Check that table names match (case-sensitive)

**Export not found:**
- Exports expire after 30 minutes
- Check `./exports/` directory for recent files

## Support

For issues or questions, please open an issue on GitHub.
