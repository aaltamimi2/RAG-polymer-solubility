"""
Complexity Assessment Script for DISSOLVE Agent
Runs through all test queries and captures results, complexity scores, and visualizations.
"""

import os
import sys
import json
import time
import shutil
import requests
from datetime import datetime
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
OUTPUT_DIR = Path("./documentation/complexity_assessment/outputs")
VIZ_DIR = Path("./documentation/complexity_assessment/visualizations")
PLOTS_DIR = Path("./plots")

# Test queries organized by level
TEST_QUERIES = {
    "1.1": {
        "level": 1,
        "name": "Basic Solvent Screening",
        "query": "What solvents dissolve LDPE at 110°C? Show the top 5 ranked by solubility percentage."
    },
    "1.2": {
        "level": 1,
        "name": "Basic Safety Check",
        "query": "What is the G-score safety rating for dodecane, heptane, and toluene?"
    },
    "1.3": {
        "level": 1,
        "name": "Basic TEA",
        "query": "Run a TEA analysis for heptane solvent recovery at 100 kg/hr throughput."
    },
    "2.1": {
        "level": 2,
        "name": "Solvent + Safety",
        "query": "Find solvents that dissolve LDPE at 110°C with G-score above 5. Rank by solubility."
    },
    "2.2": {
        "level": 2,
        "name": "Dissolution + Properties",
        "query": "What solvents dissolve PE at 110°C? For the top 3, show their boiling points, cost, and LogP toxicity values."
    },
    "2.3": {
        "level": 2,
        "name": "TEA + Visualization",
        "query": "Run TEA for dodecane recovery at 500 kg/hr and generate a cost breakdown visualization."
    },
    "3.1": {
        "level": 3,
        "name": "Multilayer Separation Strategy",
        "query": "Design a separation strategy for a PE/EVOH multilayer film. Find selective solvents for each layer at 110°C, prioritizing safety (G-score > 5)."
    },
    "3.2": {
        "level": 3,
        "name": "TEA + LCA Comparison",
        "query": "Compare TEA and LCA for heptane vs dodecane solvent recovery at 1000 kg/hr. Which has lower operating cost and carbon footprint?"
    },
    "3.3": {
        "level": 3,
        "name": "Solvent Screening with Full Analysis",
        "query": "For LDPE dissolution at 110°C, find the best solvent considering: (1) solubility > 80%, (2) G-score safety, (3) boiling point > 100°C, and (4) cost per kg. Show comparison table."
    },
    "4.1": {
        "level": 4,
        "name": "Full STRAP Process Analysis",
        "query": "Run a full STRAP analysis for PE recovery from biocontainer film waste at 5000 kg/hr using heptane as solvent. Include TEA, LCA, and generate visualizations."
    },
    "4.2": {
        "level": 4,
        "name": "Two-Stage Separation with Economics",
        "query": "Design a two-stage STRAP process for PE/EVOH separation: Stage 1 dissolves PE selectively, Stage 2 recovers EVOH. Run TEA at 1000 kg/hr for each stage and compare total costs. Show which solvents are optimal for each stage."
    },
    "4.3": {
        "level": 4,
        "name": "Scenario Comparison",
        "query": """Compare three recycling scenarios for PE/EVOH film at 5000 kg/hr:
- Scenario 1: PE recovery only using heptane
- Scenario 2: PE recovery with EVOH sold as residue
- Scenario 3: Sequential PE and EVOH recovery with two solvents

Run TEA and LCA for each scenario and generate comparison visualizations."""
    },
    "5.1": {
        "level": 5,
        "name": "Complete Process Design with RAG",
        "query": "I need to design a solvent-based recycling process for multilayer biocontainer films (PE/EVOH). First, search the RAG knowledge base for PE dissolution temperatures and optimal process conditions from published STRAP studies. Then find optimal solvents for selective PE dissolution at 110°C ranked by safety. Run TEA and LCA at 5000 kg/hr scale, and generate visualizations comparing virgin vs recycled polymer environmental impact."
    },
    "5.2": {
        "level": 5,
        "name": "Minimum Selling Price Analysis",
        "query": """For a STRAP process recovering PE from multilayer film using dodecane at 110°C:
1. Find the solvent properties and safety data
2. Run TEA at three scales: 1000, 5000, and 10000 kg/hr
3. Calculate minimum selling price at each scale
4. Run LCA to determine GHG reduction vs virgin PE
5. Generate scale economics visualization"""
    },
    "6.1": {
        "level": 6,
        "name": "Comprehensive Solvent Selection Pipeline",
        "query": """I'm designing a PE recycling process and need a comprehensive solvent analysis:

1. List all solvents that dissolve LDPE above 80% at temperatures between 100-120°C
2. For each candidate, retrieve: G-score, LogP, boiling point, cost per kg
3. Get PubChem GHS hazard data for the top 5 safest options
4. Run TEA comparison at 2000 kg/hr for these 5 solvents
5. Generate a radar chart comparing safety, cost, and efficiency
6. Recommend the optimal solvent with full justification

Show your reasoning at each step."""
    },
}


def evaluate_complexity(query: str) -> dict:
    """Evaluate query complexity using the LLM judge."""
    try:
        response = requests.post(
            f"{API_BASE}/api/evaluate-complexity",
            json={"query": query},
            timeout=30
        )
        if response.ok:
            return response.json()
    except Exception as e:
        print(f"  Complexity evaluation failed: {e}")
    return {"score": 0, "label": "Unknown", "reasoning": "Failed to evaluate", "estimated_tools": 0}


def run_query(query: str, session_id: str = None) -> dict:
    """Run a query through the agent."""
    try:
        payload = {
            "message": query,
            "model": "gemini-2.5-flash"  # Use flash for better reasoning
        }
        if session_id:
            payload["session_id"] = session_id

        response = requests.post(
            f"{API_BASE}/api/chat",
            json=payload,
            timeout=300  # 5 minute timeout for complex queries
        )
        if response.ok:
            return response.json()
        else:
            return {"error": response.text, "response": f"Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e), "response": f"Error: {e}"}


def copy_new_plots(before_plots: set, query_id: str) -> list:
    """Copy any new plots to the visualization directory."""
    copied = []
    current_plots = set(PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else set()
    new_plots = current_plots - before_plots

    for plot in new_plots:
        dest = VIZ_DIR / f"{query_id}_{plot.name}"
        shutil.copy(plot, dest)
        copied.append(str(dest.name))

    return copied


def run_assessment():
    """Run the full complexity assessment."""
    print("=" * 70)
    print("DISSOLVE AGENT COMPLEXITY ASSESSMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"API: {API_BASE}")
    print()

    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "assessment_date": datetime.now().isoformat(),
        "api_base": API_BASE,
        "queries": {}
    }

    # Sort queries by level
    sorted_queries = sorted(TEST_QUERIES.items(), key=lambda x: (x[1]["level"], x[0]))

    for query_id, query_info in sorted_queries:
        print(f"\n{'='*70}")
        print(f"Query {query_id}: {query_info['name']} (Level {query_info['level']})")
        print("=" * 70)
        print(f"Query: {query_info['query'][:100]}...")

        # Get plots before query
        before_plots = set(PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else set()

        # Evaluate complexity
        print("\n1. Evaluating complexity...")
        complexity = evaluate_complexity(query_info['query'])
        print(f"   Score: {complexity['score']}/5 ({complexity['label']})")
        print(f"   Reasoning: {complexity['reasoning']}")
        print(f"   Estimated tools: {complexity['estimated_tools']}")

        # Run query
        print("\n2. Running query...")
        start_time = time.time()
        result = run_query(query_info['query'])
        elapsed = time.time() - start_time

        iterations = result.get("iterations", 0)
        print(f"   Elapsed: {elapsed:.1f}s")
        print(f"   Iterations: {iterations}")

        # Copy new visualizations
        new_plots = copy_new_plots(before_plots, query_id)
        if new_plots:
            print(f"   Visualizations: {len(new_plots)} generated")

        # Check for errors
        has_error = "error" in result or "Error" in result.get("response", "")[:50]
        status = "FAIL" if has_error else "PASS"
        print(f"   Status: {status}")

        # Store result
        results["queries"][query_id] = {
            "level": query_info["level"],
            "name": query_info["name"],
            "query": query_info["query"],
            "complexity": complexity,
            "elapsed_seconds": elapsed,
            "iterations": iterations,
            "response": result.get("response", ""),
            "images": result.get("images", []),
            "visualizations": new_plots,
            "status": status,
            "session_id": result.get("session_id", "")
        }

        # Save individual result
        output_file = OUTPUT_DIR / f"query_{query_id.replace('.', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(results["queries"][query_id], f, indent=2)
        print(f"   Saved: {output_file.name}")

        # Brief pause between queries
        time.sleep(2)

    # Save full results
    full_results_file = OUTPUT_DIR / "full_assessment_results.json"
    with open(full_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nFull results saved to: {full_results_file}")

    # Generate summary markdown
    generate_summary_markdown(results)

    return results


def generate_summary_markdown(results: dict):
    """Generate a summary markdown file."""
    md_path = Path("./documentation/complexity_assessment/ASSESSMENT_RESULTS.md")

    lines = [
        "# DISSOLVE Agent Complexity Assessment Results",
        "",
        f"**Assessment Date:** {results['assessment_date']}",
        "",
        "## Summary",
        "",
        "| Query | Level | Complexity | Status | Iterations | Time (s) | Visualizations |",
        "|-------|-------|------------|--------|------------|----------|----------------|",
    ]

    # Summary table
    for query_id, data in sorted(results["queries"].items(), key=lambda x: (x[1]["level"], x[0])):
        complexity = data["complexity"]
        lines.append(
            f"| {query_id} | {data['level']} | {complexity['score']}/5 ({complexity['label']}) | "
            f"{data['status']} | {data['iterations']} | {data['elapsed_seconds']:.1f} | "
            f"{len(data['visualizations'])} |"
        )

    lines.extend([
        "",
        "## Detailed Results",
        ""
    ])

    # Detailed results by level
    current_level = 0
    for query_id, data in sorted(results["queries"].items(), key=lambda x: (x[1]["level"], x[0])):
        if data["level"] != current_level:
            current_level = data["level"]
            lines.append(f"\n### Level {current_level}")
            lines.append("")

        lines.extend([
            f"#### Query {query_id}: {data['name']}",
            "",
            "**Query:**",
            "```",
            data["query"],
            "```",
            "",
            f"**Complexity:** {data['complexity']['score']}/5 ({data['complexity']['label']})",
            f"- Reasoning: {data['complexity']['reasoning']}",
            f"- Estimated tools: {data['complexity']['estimated_tools']}",
            "",
            f"**Execution:**",
            f"- Status: {data['status']}",
            f"- Iterations: {data['iterations']}",
            f"- Time: {data['elapsed_seconds']:.1f}s",
            f"- Visualizations: {len(data['visualizations'])}",
            "",
            "**Response Preview:**",
            "```",
            data["response"][:500] + "..." if len(data["response"]) > 500 else data["response"],
            "```",
            ""
        ])

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Summary saved to: {md_path}")


if __name__ == "__main__":
    run_assessment()
