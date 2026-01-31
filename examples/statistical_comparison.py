#!/usr/bin/env python3
"""
Example: Statistical Comparison of Polymer Groups
Compares solubility characteristics between different polymer types

Usage:
    python examples/statistical_comparison.py

Requirements:
    - Server must be running (python app_server.py)
    - Data files must be loaded in ./data directory
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_sql_final_1212_patched import compare_groups_statistically


async def main():
    """Run statistical comparison of polymer groups"""
    print("=" * 70)
    print("Statistical Comparison Example")
    print("=" * 70)
    print("\nComparing solubility profiles: PVDF vs LDPE")
    print("Statistical test: Independent t-test")
    print()

    result = await compare_groups_statistically(
        table_name="common_solvents_database",
        value_column="solubility",
        group_column="polymer",
        group1="PVDF",
        group2="LDPE"
    )

    print(result)

    print("\n" + "=" * 70)
    print("Comparing solubility profiles: PET vs PP")
    print("=" * 70)
    print()

    result2 = await compare_groups_statistically(
        table_name="common_solvents_database",
        value_column="solubility",
        group_column="polymer",
        group1="PET",
        group2="PP"
    )

    print(result2)

    print("\n" + "=" * 70)
    print("Statistical analysis complete!")
    print("=" * 70)
    print("\nInterpretation:")
    print("- p < 0.05: Statistically significant difference")
    print("- p ≥ 0.05: No significant difference")
    print("- Cohen's d > 0.8: Large effect size")
    print("- Cohen's d 0.5-0.8: Medium effect size")
    print("- Cohen's d < 0.5: Small effect size")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
