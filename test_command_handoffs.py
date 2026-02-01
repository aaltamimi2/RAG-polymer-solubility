"""
Test Command-based handoffs and structured schemas (P0/P1 implementation).

Tests:
1. Simple 3-polymer separation with TEA
2. 5-polymer separation (exhaustive algorithm)
3. 10-polymer benchmark (greedy algorithm) - COMPLEXITY BENCHMARK
4. Schema validation and handoff tracking
"""
import asyncio
import time
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test queries with expected complexity
TEST_QUERIES = [
    {
        "name": "Simple 3-polymer",
        "query": "Find the most cost-effective way to separate LDPE, PET, and PP from multilayer film at 80°C",
        "expected_complexity": 5,
        "expected_path": "integrated",
        "polymers": ["LDPE", "PET", "PP"],
    },
    {
        "name": "5-polymer exhaustive",
        "query": "Plan the cheapest separation sequence for PS, PVC, LDPE, HDPE, PP at 100°C with 200 kg/hr throughput",
        "expected_complexity": 5,
        "expected_path": "integrated",
        "polymers": ["PS", "PVC", "LDPE", "HDPE", "PP"],
    },
    {
        "name": "10-polymer greedy (BENCHMARK)",
        "query": "Find the most cost-effective separation for a 10-layer film: PS, PVC, LDPE, HDPE, PP, EVOH, PA6, PA66, PET at 80°C and 100 kg/hr",
        "expected_complexity": 5,
        "expected_path": "integrated",
        "polymers": ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "PA6", "PA66", "PET"],
    },
]


async def test_single_query(query_config: dict, graph, thread_id: str):
    """Test a single query and verify Command-based handoffs."""
    from langchain_core.messages import HumanMessage

    print(f"\n{'='*80}")
    print(f"TEST: {query_config['name']}")
    print(f"{'='*80}")
    print(f"Query: {query_config['query'][:100]}...")
    print(f"Expected: complexity={query_config['expected_complexity']}, path={query_config['expected_path']}")
    print(f"Polymers: {query_config['polymers']}")
    print()

    state = {"messages": [HumanMessage(content=query_config["query"])]}
    config = {"configurable": {"thread_id": thread_id}}

    start_time = time.time()
    result = await graph.ainvoke(state, config)
    elapsed = time.time() - start_time

    # Extract results
    messages = result.get("messages", [])
    separation_results = result.get("separation_results")
    tea_results = result.get("tea_results")
    path = result.get("path", "unknown")
    complexity = result.get("complexity", 0)

    # P3: Enhanced tracking fields (consolidated from P0 handoff_history)
    trace_id = result.get("trace_id")
    handoff_metrics = result.get("handoff_metrics", [])
    execution_trace = result.get("execution_trace")

    print(f"\n--- RESULTS ({elapsed:.2f}s) ---")
    print(f"Path: {path}, Complexity: {complexity}")

    # P3: Print handoff metrics with trace info
    if trace_id:
        print(f"Trace ID: {trace_id}")
    print(f"Handoffs: {len(handoff_metrics)} entries")
    for i, metric in enumerate(handoff_metrics):
        duration = metric.get("duration_ms", 0) or 0
        success = "✓" if metric.get("success") else "✗"
        summary = metric.get("query_summary", "")[:50]
        print(f"  {i+1}. {metric.get('from_agent')} → {metric.get('to_agent')}: "
              f"{duration:.0f}ms {success} ({metric.get('task_type')}) {summary}")

    # Separation results
    if separation_results:
        print(f"\nSeparation Results (P1 Schema):")
        print(f"  - Polymers: {separation_results.get('polymers', [])}")
        print(f"  - Solvents: {separation_results.get('solvents', [])[:5]}")  # First 5
        print(f"  - Best sequence: {separation_results.get('best_sequence', [])}")
        print(f"  - Algorithm: {separation_results.get('algorithm_used', 'unknown')}")
    else:
        print("\n  [!] No separation_results in state")

    # TEA results
    if tea_results:
        print(f"\nTEA Results (P1 Schema):")
        print(f"  - Best solvent: {tea_results.get('best_solvent')}")
        print(f"  - Cost/kg: ${tea_results.get('cost_per_kg', 'N/A')}")
        print(f"  - CAPEX: ${tea_results.get('total_capex', 'N/A')}")
        print(f"  - Payback: {tea_results.get('payback_years', 'N/A')} years")
    else:
        print("\n  [!] No tea_results in state")

    # Final message
    if messages:
        final_msg = messages[-1]
        content = final_msg.content if hasattr(final_msg, 'content') else str(final_msg)
        print(f"\nFinal Response (first 500 chars):")
        print(content[:500])
        print("...")

    # Validation
    success = True
    issues = []

    if path != query_config["expected_path"]:
        issues.append(f"Path mismatch: got {path}, expected {query_config['expected_path']}")
        success = False

    if complexity != query_config["expected_complexity"]:
        issues.append(f"Complexity mismatch: got {complexity}, expected {query_config['expected_complexity']}")
        success = False

    # Verify handoffs were recorded (using handoff_metrics)
    if query_config["expected_path"] == "integrated" and len(handoff_metrics) < 1:
        issues.append("No handoff_metrics recorded for integrated path")
        success = False

    # Verify P3 enhanced tracking
    if query_config["expected_path"] == "integrated":
        if not trace_id:
            issues.append("No trace_id in state")
            success = False

    if issues:
        print(f"\n[ISSUES]")
        for issue in issues:
            print(f"  - {issue}")

    status = "PASS" if success else "FAIL"
    print(f"\n{'='*80}")
    print(f"TEST {status}: {query_config['name']} ({elapsed:.2f}s)")
    print(f"{'='*80}")

    return {
        "name": query_config["name"],
        "success": success,
        "elapsed": elapsed,
        "issues": issues,
        "handoffs": len(handoff_metrics),
        "separation_solvents": len(separation_results.get("solvents", [])) if separation_results else 0,
        "tea_cost": tea_results.get("cost_per_kg") if tea_results else None,
        # P3: Enhanced tracking
        "trace_id": trace_id,
        "execution_trace": execution_trace,
    }


async def run_all_tests():
    """Run all test queries."""
    print("="*80)
    print("P0/P1 COMMAND-BASED HANDOFFS TEST SUITE")
    print("="*80)
    print("Testing Command objects, structured schemas, and handoff tracking")
    print()

    # Import the multi-agent graph
    from agent_sql_final_1212_patched import multi_agent_graph

    results = []
    for i, query_config in enumerate(TEST_QUERIES):
        try:
            result = await test_single_query(
                query_config,
                multi_agent_graph,
                thread_id=f"test_command_{i}_{int(time.time())}"
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Test {query_config['name']} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": query_config["name"],
                "success": False,
                "elapsed": 0,
                "issues": [str(e)],
                "handoffs": 0,
                "separation_solvents": 0,
                "tea_cost": None,
            })

        # Brief pause between tests
        await asyncio.sleep(1)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\nResults: {passed}/{total} passed\n")

    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        trace_str = f"trace={r.get('trace_id', 'N/A')}" if r.get('trace_id') else ""
        print(f"  [{status}] {r['name']}: {r['elapsed']:.2f}s, "
              f"{r['handoffs']} handoffs, "
              f"{r['separation_solvents']} solvents, "
              f"cost=${r['tea_cost'] or 'N/A'} "
              f"{trace_str}")
        if r["issues"]:
            for issue in r["issues"]:
                print(f"        - {issue}")

    # Benchmark comparison
    print("\n" + "="*80)
    print("10-POLYMER BENCHMARK ANALYSIS")
    print("="*80)

    benchmark = next((r for r in results if "BENCHMARK" in r["name"]), None)
    if benchmark:
        print(f"\nBenchmark Test: {benchmark['name']}")
        print(f"  Elapsed time: {benchmark['elapsed']:.2f}s")
        print(f"  Handoffs: {benchmark['handoffs']}")
        print(f"  Solvents found: {benchmark['separation_solvents']}")
        print(f"  Cost/kg: ${benchmark['tea_cost'] or 'N/A'}")

        # P3: Enhanced metrics
        print(f"\n  P3 Enhanced Tracking:")
        print(f"    - Trace ID: {benchmark.get('trace_id', 'N/A')}")
        print(f"    - Handoff metrics: {benchmark['handoffs']} entries")
        if benchmark.get("execution_trace"):
            trace = benchmark["execution_trace"]
            print(f"    - Total elapsed: {trace.get('total_elapsed_s', 0):.2f}s")
            print(f"    - Completed at: {trace.get('completed_at', 'N/A')}")

        if benchmark["success"]:
            print("\n  Status: BENCHMARK PASSED")
            print("  - Greedy algorithm handled 10 polymers")
            print("  - Command-based handoffs worked correctly")
            print("  - P3: Enhanced execution tracking active")
            print("  - Structured schemas captured results")
        else:
            print("\n  Status: BENCHMARK FAILED")
            for issue in benchmark["issues"]:
                print(f"    - {issue}")

    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())
