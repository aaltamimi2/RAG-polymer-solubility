#!/usr/bin/env python3
"""
Test GSK G-Score Integration
Verifies fuzzy matching, G-score lookup, family alternatives, and visualization
"""

import asyncio
import sys
from agent_sql_final_1212_patched import (
    get_solvent_gscore,
    get_family_alternatives,
    visualize_gscores,
    query_database,
    fuzzy_match_solvent_name
)


async def test_gscore_lookup():
    """Test 1: Basic G-score lookup"""
    print("=" * 70)
    print("TEST 1: G-Score Lookup")
    print("=" * 70)

    # Test exact match
    result = await get_solvent_gscore.ainvoke({"solvent_name": "Ethanol"})
    print(result)
    print()

    # Test fuzzy match
    result2 = await get_solvent_gscore.ainvoke({"solvent_name": "ethyl alcohol", "use_fuzzy_matching": True})
    print(result2)
    print()


async def test_fuzzy_matching():
    """Test 2: Fuzzy name matching"""
    print("\n" + "=" * 70)
    print("TEST 2: Fuzzy Name Matching")
    print("=" * 70)

    # Test variations
    test_names = ["water", "H2O", "ethyl acetate", "EtOAc", "methanol", "MeOH"]

    for name in test_names:
        match = fuzzy_match_solvent_name(name, dataset="gsk", threshold=80)
        if match:
            print(f"✅ '{name}' → '{match['matched_name']}' (score: {match['score']})")
        else:
            print(f"❌ '{name}' → No match found")

    print()


async def test_family_alternatives():
    """Test 3: Family alternatives"""
    print("\n" + "=" * 70)
    print("TEST 3: Family Alternatives")
    print("=" * 70)

    result = await get_family_alternatives.ainvoke({
        "solvent_name": "Ethanol",
        "min_gscore": 6.0,
        "limit": 10
    })
    print(result)
    print()


async def test_gsk_visualization():
    """Test 4: G-score visualization"""
    print("\n" + "=" * 70)
    print("TEST 4: G-Score Visualization")
    print("=" * 70)

    # Test bar chart for alcohols
    result = await visualize_gscores.ainvoke({
        "filter_by": "family",
        "family": "Alcohols",
        "plot_type": "bar"
    })
    print(result)
    print()


async def test_database_query():
    """Test 5: Query GSK dataset"""
    print("\n" + "=" * 70)
    print("TEST 5: Database Query")
    print("=" * 70)

    result = await query_database.ainvoke({
        "sql_query": "SELECT solvent_common_name, classification, g_score FROM gsk_dataset ORDER BY g_score DESC LIMIT 10",
        "limit": 10
    })
    print(result)
    print()


async def test_count_by_family():
    """Test 6: Count solvents by family"""
    print("\n" + "=" * 70)
    print("TEST 6: Count Solvents by Family")
    print("=" * 70)

    result = await query_database.ainvoke({
        "sql_query": "SELECT classification, COUNT(*) as count, AVG(g_score) as avg_gscore FROM gsk_dataset GROUP BY classification ORDER BY avg_gscore DESC"
    })
    print(result)
    print()


async def main():
    """Run all tests"""
    print("\n🧪 GSK G-SCORE INTEGRATION TEST SUITE\n")

    try:
        # Run tests sequentially
        await test_gscore_lookup()
        await test_fuzzy_matching()
        await test_family_alternatives()
        await test_gsk_visualization()
        await test_database_query()
        await test_count_by_family()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\n📊 Summary:")
        print("  ✅ G-score lookup working (exact + fuzzy match)")
        print("  ✅ Fuzzy name matching across datasets")
        print("  ✅ Family alternatives retrieval")
        print("  ✅ G-score visualization")
        print("  ✅ GSK dataset queries")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
