#!/usr/bin/env python3
"""
Example: Temperature Optimization Analysis
Analyzes solubility trends across temperature ranges for polymer-solvent pairs

Usage:
    python examples/temperature_optimization.py

Requirements:
    - Server must be running (python app_server.py)
    - Data files must be loaded in ./data directory
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_sql_final_1212_patched import query_database


async def main():
    """Run temperature optimization analysis"""
    print("=" * 70)
    print("Temperature Optimization Analysis Example")
    print("=" * 70)
    print("\nAnalyzing PVDF solubility in NMP across temperature range...")
    print()

    # Query solubility data across temperature range
    query = """
    SELECT
        temperature,
        solubility,
        ROUND(solubility, 2) as solubility_rounded
    FROM common_solvents_database
    WHERE polymer = 'PVDF'
      AND solvent = 'NMP'
    ORDER BY temperature ASC
    """

    result = await query_database(
        sql_query=query,
        export_csv=True  # Export results to CSV
    )

    print(result)
    print("\n" + "=" * 70)
    print("Temperature sweep complete! Check exports/ directory for CSV.")
    print("=" * 70)

    # Additional analysis: Compare multiple solvents
    print("\n" + "=" * 70)
    print("Comparing PVDF solubility in common solvents at 80°C...")
    print("=" * 70)
    print()

    compare_query = """
    SELECT
        solvent,
        solubility,
        temperature
    FROM common_solvents_database
    WHERE polymer = 'PVDF'
      AND temperature = 80
    ORDER BY solubility DESC
    LIMIT 10
    """

    result2 = await query_database(
        sql_query=compare_query,
        export_csv=True
    )

    print(result2)
    print("\n" + "=" * 70)
    print("Solvent comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
