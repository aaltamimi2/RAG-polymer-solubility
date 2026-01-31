#!/usr/bin/env python3
"""
Example: PVDF/PET Separation Analysis
Reproduces optimal solvent selection for separating PVDF from PET contamination

Usage:
    python examples/separation_pvdf_pet.py

Requirements:
    - Server must be running (python app_server.py)
    - Data files must be loaded in ./data directory
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_sql_final_1212_patched import find_optimal_separation_conditions


async def main():
    """Run PVDF/PET separation analysis"""
    print("=" * 70)
    print("PVDF/PET Separation Analysis Example")
    print("=" * 70)
    print("\nAnalyzing optimal solvents for separating PVDF from PET...")
    print("Temperature range: 25-160°C")
    print("Target selectivity: ≥30\n")

    result = await find_optimal_separation_conditions(
        table_name="common_solvents_database",
        polymer_column="polymer",
        solvent_column="solvent",
        temperature_column="temperature",
        solubility_column="solubility",
        target_polymer="PVDF",
        comparison_polymers="PET",
        start_temperature=25.0,
        initial_selectivity=30.0,
        export_csv=True  # Export results to CSV
    )

    print(result)
    print("\n" + "=" * 70)
    print("Analysis complete! Check exports/ directory for CSV output.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
