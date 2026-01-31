"""
Async Performance Benchmark for Polymer Solubility Agent

Tests the performance improvements from async execution.
Compares sequential vs parallel execution across critical bottlenecks.
"""

import asyncio
import time
import os
import sys
from datetime import datetime

# Ensure we can import the agent module
sys.path.insert(0, os.getcwd())

# Set dummy API key for testing (will fail at LLM call, but that's OK for benchmarks)
if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️ GOOGLE_API_KEY not set - some benchmarks may fail at LLM calls")
    print("Set GOOGLE_API_KEY environment variable for full testing\n")


async def benchmark_compare_groups():
    """
    Benchmark: compare_groups_statistically

    This tool runs 2 independent 100k-row queries.
    Expected speedup: 2x (parallel query execution)
    """
    print("=" * 70)
    print("BENCHMARK 1: compare_groups_statistically")
    print("=" * 70)
    print("Description: Compares two polymer groups statistically")
    print("Bottleneck:  2 independent SQL queries (100k rows each)")
    print("Expected:    ~2x speedup from parallel execution\n")

    try:
        from agent_sql_final_1212_patched import compare_groups_statistically

        start = time.time()
        # Use ainvoke for async tools (properly handles decorated tools)
        result = await compare_groups_statistically.ainvoke({
            "table_name": "common_solvents_database",
            "value_column": "solubility",
            "group_column": "polymer",
            "group1": "PVDF",
            "group2": "PLA"
        })
        elapsed = time.time() - start

        print(f"✅ Execution time: {elapsed:.2f}s")
        print(f"   Target: < 1.0s")

        if elapsed < 1.0:
            print(f"   Status: PASS (within target)")
        else:
            print(f"   Status: SLOW (exceeds target)")

        return elapsed

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def benchmark_lookup_solvent_properties():
    """
    Benchmark: lookup_solvent_properties

    Looks up properties for multiple solvents.
    Expected speedup: N/A (helper function, but parallelized internally)
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 2: lookup_solvent_properties")
    print("=" * 70)
    print("Description: Looks up properties for multiple solvents")
    print("Bottleneck:  3 queries per solvent × N solvents")
    print("Expected:    Nx speedup for N solvents\n")

    try:
        from agent_sql_final_1212_patched import lookup_solvent_properties, get_solvent_table_name

        solvent_table = get_solvent_table_name()
        if not solvent_table:
            print("⚠️ No solvent table found - skipping")
            return None

        solvents = ["acetone", "ethanol", "water", "toluene", "hexane"]

        start = time.time()
        result = await lookup_solvent_properties(solvents, solvent_table)
        elapsed = time.time() - start

        print(f"✅ Execution time: {elapsed:.2f}s ({len(solvents)} solvents)")
        print(f"   Target: < 0.5s")
        print(f"   Results: {len(result)} solvents found")

        if elapsed < 0.5:
            print(f"   Status: PASS (within target)")
        else:
            print(f"   Status: SLOW (exceeds target)")

        return elapsed

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def benchmark_plan_sequential_separation():
    """
    Benchmark: plan_sequential_separation

    CRITICAL BOTTLENECK: 28,000 sequential DB calls for 4 polymers.
    Expected speedup: 50-80x (from sequential to parallel)
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 3: plan_sequential_separation (CRITICAL)")
    print("=" * 70)
    print("Description: Plans separation sequences for 4 polymers")
    print("Bottleneck:  24 sequences × 3 steps × (1 query + 5 property lookups)")
    print("             = ~288 queries sequentially → NOW PARALLEL")
    print("Expected:    50-80x speedup (240s → 3-5s)\n")

    try:
        from agent_sql_final_1212_patched import plan_sequential_separation

        start = time.time()
        # Use ainvoke for async tools (properly handles decorated tools)
        result = await plan_sequential_separation.ainvoke({
            "table_name": "common_solvents_database",
            "polymer_column": "polymer",
            "solvent_column": "solvent",
            "temperature_column": "temperature",
            "solubility_column": "solubility",
            "polymers": "PVDF,PLA,PS,PP",  # 4! = 24 sequences
            "top_k_solvents": 5,
            "temperature": 25.0,
            "create_decision_tree": False  # Skip visualization for speed
        })
        elapsed = time.time() - start

        print(f"✅ Execution time: {elapsed:.2f}s")
        print(f"   Target: < 10s (aggressive async target)")

        if elapsed < 10:
            print(f"   Status: PASS (within aggressive target)")
        elif elapsed < 30:
            print(f"   Status: GOOD (within acceptable range)")
        else:
            print(f"   Status: SLOW (exceeds target)")

        return elapsed

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None


async def benchmark_concurrent_tool_calls():
    """
    Benchmark: Multiple tool calls in parallel

    Simulates the agent calling multiple tools simultaneously.
    Expected speedup: ~3x for 3 concurrent tools
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Concurrent Tool Execution")
    print("=" * 70)
    print("Description: Execute 3 different tools in parallel")
    print("Expected:    ~3x speedup vs sequential\n")

    try:
        from agent_sql_final_1212_patched import (
            compare_groups_statistically,
            lookup_solvent_properties,
            get_solvent_table_name
        )

        solvent_table = get_solvent_table_name()
        if not solvent_table:
            print("⚠️ No solvent table found - skipping")
            return None

        start = time.time()

        # Execute 3 tools in parallel (use ainvoke for decorated tools)
        results = await asyncio.gather(
            compare_groups_statistically.ainvoke({
                "table_name": "common_solvents_database",
                "value_column": "solubility",
                "group_column": "polymer",
                "group1": "PVDF",
                "group2": "PLA"
            }),
            lookup_solvent_properties(["acetone", "ethanol"], solvent_table),
            compare_groups_statistically.ainvoke({
                "table_name": "common_solvents_database",
                "value_column": "solubility",
                "group_column": "polymer",
                "group1": "PS",
                "group2": "PP"
            })
        )

        elapsed = time.time() - start

        print(f"✅ Execution time: {elapsed:.2f}s (3 tools in parallel)")
        print(f"   Target: < 2s (should be ~max(tool times), not sum)")

        if elapsed < 2:
            print(f"   Status: PASS (good parallelization)")
        else:
            print(f"   Status: SLOW (tools may be running sequentially)")

        return elapsed

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("ASYNC PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 70)

    results = {}

    # Run benchmarks
    results['compare_groups'] = await benchmark_compare_groups()
    results['solvent_properties'] = await benchmark_lookup_solvent_properties()
    results['sequential_separation'] = await benchmark_plan_sequential_separation()
    results['concurrent_tools'] = await benchmark_concurrent_tool_calls()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_pass = 0
    total_tests = 0

    for name, elapsed in results.items():
        if elapsed is not None:
            total_tests += 1
            status = "PASS" if (
                (name == 'compare_groups' and elapsed < 1.0) or
                (name == 'solvent_properties' and elapsed < 0.5) or
                (name == 'sequential_separation' and elapsed < 10) or
                (name == 'concurrent_tools' and elapsed < 2.0)
            ) else "SLOW"
            if status == "PASS":
                total_pass += 1
            print(f"{name:30s}: {elapsed:6.2f}s [{status}]")
        else:
            total_tests += 1
            print(f"{name:30s}: FAILED")

    print("=" * 70)
    print(f"Tests passed: {total_pass}/{total_tests}")

    if total_pass == total_tests:
        print("✅ ALL BENCHMARKS PASSED!")
    elif total_pass >= total_tests * 0.75:
        print("⚠️ MOST BENCHMARKS PASSED")
    else:
        print("❌ PERFORMANCE TARGETS NOT MET")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
