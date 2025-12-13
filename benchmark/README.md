# STRAP Solubility Search - Benchmark Suite

Automated benchmark suite for testing and validating the STRAP Solubility Search tool.

## Quick Start

```bash
# Install dependencies
pip install -r benchmark/requirements.txt

# Start the server (in another terminal)
export GOOGLE_API_KEY="your-key"
python app_server.py

# Run the benchmark
python benchmark/run_benchmark.py
```

## Output

Results are saved to `benchmark/results_TIMESTAMP/`:
- `benchmark_results.json` - Raw results data
- `benchmark_report.pdf` - Comprehensive PDF report
- `images/` - All generated plots and visualizations

## Options

```bash
# Custom server URL
python benchmark/run_benchmark.py --server-url http://localhost:8001

# Custom output directory
python benchmark/run_benchmark.py --output-dir my_results

# Longer delay between API calls (for rate limiting)
python benchmark/run_benchmark.py --delay 5.0

# Run only specific category
python benchmark/run_benchmark.py --subset "Multilayer Film"
```

## Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| Data Exploration | 2 | Database schema and data queries |
| Basic Queries | 2 | Single polymer solubility lookups |
| Solvent Properties | 3 | Property ranking and lookup |
| Polymer Separation | 2 | Two-polymer separation analysis |
| Multilayer Film | 2 | Sequential separation strategies |
| Temperature Analysis | 2 | Temperature curves and windows |
| Statistical Analysis | 2 | Statistics and correlation |
| Visualization | 2 | Heatmaps and dashboards |
| Advanced | 3 | Complex multi-factor queries |

## Benchmark Prompts

The suite tests 20 prompts covering:
1. Table exploration and polymer listing
2. LDPE and PET solubility queries
3. Solvent ranking by cost and toxicity
4. LDPE/PET and LDPE/EVOH separation
5. 80% LDPE / 12% PET / 8% EVOH multilayer film analysis
6. Temperature curves and separation windows
7. Statistical summaries and correlations
8. Heatmaps and comparison dashboards
9. Multi-factor optimization queries
