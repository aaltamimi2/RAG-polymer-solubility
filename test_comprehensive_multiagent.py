"""
Comprehensive Multi-Agent System Tests

Tests complex queries across all capabilities:
1. Separation + TEA (integrated)
2. Memory recall
3. RAG literature search
4. Safety criteria (GSK)
5. LCA analysis
6. Complex multi-step queries
7. Edge cases
"""

import asyncio
import time
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('agent_sql_final_1212_patched').setLevel(logging.WARNING)


# ============================================================
# TEST QUERIES - Organized by complexity and feature
# ============================================================

TEST_QUERIES = [
    # ========== SEPARATION + TEA TESTS ==========
    {
        "name": "Simple 2-polymer separation",
        "query": "Find the cheapest way to separate PE and PP at 70°C",
        "expected_path": "integrated",
        "checks": ["solvents", "cost_per_kg"],
        "category": "separation_tea",
    },
    {
        "name": "4-polymer with throughput constraint",
        "query": "Plan the most cost-effective separation for PET, LDPE, HDPE, PS at 90°C with 500 kg/hr throughput",
        "expected_path": "integrated",
        "checks": ["solvents", "cost_per_kg", "sequence"],
        "category": "separation_tea",
    },
    {
        "name": "Separation with specific solvent preference",
        "query": "Can we separate PVC from PET using environmentally friendly solvents? Include economic analysis.",
        "expected_path": "integrated",
        "checks": ["solvents", "cost_per_kg"],
        "category": "separation_tea",
    },

    # ========== SAFETY/GSK TESTS ==========
    {
        "name": "GSK safety scoring",
        "query": "What is the GSK safety score for cyclohexane, toluene, and DMF?",
        "expected_path": "specialist",
        "expected_specialist": "separation",
        "checks": ["response_contains:gsk", "response_contains:score"],
        "category": "safety",
    },
    {
        "name": "Safe solvent alternatives",
        "query": "Find safer alternatives to chloroform for dissolving polystyrene",
        "expected_path": "specialist",
        "checks": ["solvents"],
        "category": "safety",
    },

    # ========== RAG/LITERATURE TESTS ==========
    {
        "name": "Literature search - deinking",
        "query": "What does the literature say about surfactants for deinking printed plastics?",
        "expected_path": "specialist",
        "expected_specialist": "literature",
        "checks": ["response_length:500"],
        "category": "literature",
    },
    {
        "name": "Research papers on polymer recycling",
        "query": "Search for recent research on PE film recycling challenges",
        "expected_path": "specialist",
        "expected_specialist": "literature",
        "checks": ["response_length:300"],
        "category": "literature",
    },

    # ========== SOLVENT PROPERTIES TESTS ==========
    {
        "name": "Solvent property comparison",
        "query": "Compare the boiling points and Hansen parameters of hexane, cyclohexane, and toluene",
        "expected_path": "standard",
        "checks": ["response_contains:boiling", "response_contains:hansen"],
        "category": "solvent_properties",
    },
    {
        "name": "Solubility prediction",
        "query": "Will polypropylene dissolve in xylene at 100°C?",
        "expected_path": "standard",
        "checks": ["response_length:100"],
        "category": "solvent_properties",
    },

    # ========== COMPLEX MULTI-STEP QUERIES ==========
    {
        "name": "Full separation pipeline",
        "query": "I need to separate a 5-layer packaging film containing LDPE, EVOH, PA6, PET, and PP. Find the optimal separation sequence, estimate costs at 200 kg/hr, and identify any safety concerns with the recommended solvents.",
        "expected_path": "integrated",
        "checks": ["solvents", "cost_per_kg", "sequence"],
        "category": "complex",
    },
    {
        "name": "Environmental + economic tradeoff",
        "query": "For separating PS and ABS, compare the most economical solvent vs the greenest option. Include LCA if possible.",
        "expected_path": "integrated",
        "checks": ["solvents"],
        "category": "complex",
    },

    # ========== FAST PATH TESTS ==========
    {
        "name": "Simple listing query",
        "query": "List all polymers in the database",
        "expected_path": "fast",
        "checks": ["response_length:100"],
        "category": "fast",
    },
    {
        "name": "Database info",
        "query": "What tables are available?",
        "expected_path": "fast",
        "checks": ["response_contains:table"],
        "category": "fast",
    },

    # ========== EDGE CASES ==========
    {
        "name": "Unknown polymer handling",
        "query": "How do I separate PTFE from Kevlar?",
        "expected_path": "standard",
        "checks": ["response_length:50"],
        "category": "edge_case",
    },
    {
        "name": "Ambiguous query",
        "query": "Best solvent?",
        "expected_path": "fast",
        "checks": ["response_length:50"],
        "category": "edge_case",
    },
]


async def run_single_test(query_config: dict, graph, test_num: int) -> dict:
    """Run a single test query and validate results."""
    from langchain_core.messages import HumanMessage

    print(f"\n{'='*70}")
    print(f"TEST {test_num}: {query_config['name']} [{query_config['category']}]")
    print(f"{'='*70}")
    print(f"Query: {query_config['query'][:80]}...")

    start_time = time.time()
    thread_id = f"comprehensive_test_{test_num}_{int(time.time())}"

    try:
        state = {"messages": [HumanMessage(content=query_config["query"])]}
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 35}

        result = await asyncio.wait_for(
            graph.ainvoke(state, config),
            timeout=120.0  # 2 minute timeout per query
        )

        elapsed = time.time() - start_time

        # Extract results
        messages = result.get("messages", [])
        separation_results = result.get("separation_results", {})
        tea_results = result.get("tea_results", {})
        path = result.get("path", "unknown")
        handoff_metrics = result.get("handoff_metrics", [])

        # Get final response text
        final_response = ""
        if messages:
            final_msg = messages[-1]
            final_response = final_msg.content if hasattr(final_msg, 'content') else str(final_msg)

        # Validate checks
        issues = []
        passed_checks = []

        for check in query_config.get("checks", []):
            if check == "solvents":
                solvents = separation_results.get("solvents", [])
                if solvents and len(solvents) > 0:
                    passed_checks.append(f"solvents: {len(solvents)} found")
                else:
                    issues.append("No solvents found")

            elif check == "cost_per_kg":
                cost = tea_results.get("cost_per_kg")
                if cost is not None:
                    passed_checks.append(f"cost_per_kg: ${cost}")
                else:
                    issues.append("No cost_per_kg extracted")

            elif check == "sequence":
                seq = separation_results.get("best_sequence", [])
                if seq and len(seq) > 0:
                    passed_checks.append(f"sequence: {len(seq)} steps")
                else:
                    issues.append("No separation sequence found")

            elif check.startswith("response_contains:"):
                keyword = check.split(":")[1].lower()
                if keyword in final_response.lower():
                    passed_checks.append(f"contains '{keyword}'")
                else:
                    issues.append(f"Response missing '{keyword}'")

            elif check.startswith("response_length:"):
                min_len = int(check.split(":")[1])
                if len(final_response) >= min_len:
                    passed_checks.append(f"response length: {len(final_response)}")
                else:
                    issues.append(f"Response too short ({len(final_response)} < {min_len})")

        # Check path expectation
        expected_path = query_config.get("expected_path")
        if expected_path and path != expected_path:
            issues.append(f"Path mismatch: got {path}, expected {expected_path}")

        success = len(issues) == 0

        # Print results
        print(f"\n  Elapsed: {elapsed:.2f}s | Path: {path} | Handoffs: {len(handoff_metrics)}")

        if separation_results.get("solvents"):
            print(f"  Solvents: {separation_results['solvents'][:5]}")
        if tea_results.get("cost_per_kg"):
            print(f"  TEA Cost: ${tea_results['cost_per_kg']}/kg")

        print(f"\n  Checks passed: {len(passed_checks)}/{len(query_config.get('checks', []))}")
        for check in passed_checks:
            print(f"    ✓ {check}")

        if issues:
            print(f"\n  Issues:")
            for issue in issues:
                print(f"    ✗ {issue}")

        status = "PASS" if success else "FAIL"
        print(f"\n  Result: {status}")

        return {
            "name": query_config["name"],
            "category": query_config["category"],
            "success": success,
            "elapsed": elapsed,
            "path": path,
            "issues": issues,
            "checks_passed": len(passed_checks),
            "total_checks": len(query_config.get("checks", [])),
            "solvents_count": len(separation_results.get("solvents", [])),
            "cost_per_kg": tea_results.get("cost_per_kg"),
            "handoffs": len(handoff_metrics),
        }

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"\n  TIMEOUT after {elapsed:.2f}s")
        return {
            "name": query_config["name"],
            "category": query_config["category"],
            "success": False,
            "elapsed": elapsed,
            "path": "timeout",
            "issues": ["Query timed out"],
            "checks_passed": 0,
            "total_checks": len(query_config.get("checks", [])),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERROR: {type(e).__name__}: {str(e)[:100]}")
        return {
            "name": query_config["name"],
            "category": query_config["category"],
            "success": False,
            "elapsed": elapsed,
            "path": "error",
            "issues": [str(e)[:200]],
            "checks_passed": 0,
            "total_checks": len(query_config.get("checks", [])),
        }


async def run_comprehensive_tests(categories: list = None):
    """Run comprehensive test suite."""
    print("=" * 70)
    print("COMPREHENSIVE MULTI-AGENT SYSTEM TEST SUITE")
    print("=" * 70)
    print(f"Total queries: {len(TEST_QUERIES)}")
    print("Categories: separation_tea, safety, literature, solvent_properties, complex, fast, edge_case")
    print()

    # Import the graph
    from agent_sql_final_1212_patched import multi_agent_graph

    # Filter by category if specified
    queries = TEST_QUERIES
    if categories:
        queries = [q for q in TEST_QUERIES if q["category"] in categories]
        print(f"Running {len(queries)} tests for categories: {categories}")

    results = []
    for i, query_config in enumerate(queries, 1):
        try:
            result = await run_single_test(query_config, multi_agent_graph, i)
            results.append(result)
        except Exception as e:
            logger.error(f"Test {i} failed catastrophically: {e}")
            results.append({
                "name": query_config["name"],
                "category": query_config["category"],
                "success": False,
                "elapsed": 0,
                "issues": [str(e)],
            })

        # Brief pause between tests
        await asyncio.sleep(0.5)

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    # Group by category
    by_category = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    total_passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\nOverall: {total_passed}/{total} passed ({100*total_passed/total:.1f}%)")
    print(f"Total time: {sum(r.get('elapsed', 0) for r in results):.1f}s")

    print("\nBy Category:")
    for cat, cat_results in sorted(by_category.items()):
        passed = sum(1 for r in cat_results if r["success"])
        total_cat = len(cat_results)
        print(f"  {cat}: {passed}/{total_cat}")

    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r["success"] else "✗"
        cost_str = f"${r.get('cost_per_kg')}" if r.get('cost_per_kg') else "-"
        print(f"  [{status}] {r['name'][:40]:<40} | {r.get('elapsed', 0):>6.1f}s | {r.get('path', '-'):<12} | cost={cost_str}")
        if r.get("issues"):
            for issue in r["issues"][:2]:
                print(f"       └─ {issue[:60]}")

    # Performance metrics
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    sep_tea_results = [r for r in results if r.get("category") == "separation_tea"]
    if sep_tea_results:
        avg_time = sum(r.get("elapsed", 0) for r in sep_tea_results) / len(sep_tea_results)
        costs = [r.get("cost_per_kg") for r in sep_tea_results if r.get("cost_per_kg")]
        print(f"\nSeparation+TEA queries:")
        print(f"  Average time: {avg_time:.1f}s")
        print(f"  Cost extraction rate: {len(costs)}/{len(sep_tea_results)}")
        if costs:
            print(f"  Cost range: ${min(costs):.2f} - ${max(costs):.2f}/kg")

    return results


if __name__ == "__main__":
    import sys

    # Allow filtering by category from command line
    categories = None
    if len(sys.argv) > 1:
        categories = sys.argv[1].split(",")
        print(f"Filtering to categories: {categories}")

    asyncio.run(run_comprehensive_tests(categories))
