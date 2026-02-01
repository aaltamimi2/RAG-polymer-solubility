"""
Greedy Polymer Separation Algorithm

At each step:
1. For each remaining polymer, find the best solvent to separate it from ALL others
2. Pick the polymer with highest selectivity (easiest to separate)
3. Remove it and repeat

Complexity: O(n²) vs O(n!) for exhaustive search
"""

import asyncio
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()


@dataclass
class SeparationStep:
    """Result of one separation step."""
    target_polymer: str
    remaining_polymers: List[str]
    best_solvent: str
    selectivity: float
    temperature: float
    step_number: int


async def get_selectivity_for_polymer(
    target: str,
    others: List[str],
    temperature: float,
    db_connection
) -> Tuple[str, float, str]:
    """
    Find the best solvent to separate `target` from `others` at given temperature.

    Returns: (target_polymer, best_selectivity, best_solvent)
    """
    # Query to find solvents that dissolve target but NOT others
    # Selectivity = solubility(target) - max(solubility(others))

    query = f"""
    WITH target_solubility AS (
        SELECT solvent, solubility as target_sol
        FROM common_solvents_database
        WHERE UPPER(polymer) = UPPER('{target}')
    ),
    other_max AS (
        SELECT solvent, MAX(solubility) as max_other_sol
        FROM common_solvents_database
        WHERE UPPER(polymer) IN ({','.join([f"UPPER('{p}')" for p in others])})
        GROUP BY solvent
    )
    SELECT
        t.solvent,
        t.target_sol,
        COALESCE(o.max_other_sol, 0) as max_other_sol,
        (t.target_sol - COALESCE(o.max_other_sol, 0)) as selectivity
    FROM target_solubility t
    LEFT JOIN other_max o ON t.solvent = o.solvent
    WHERE t.target_sol > 0
    ORDER BY selectivity DESC
    LIMIT 1
    """

    try:
        result = db_connection.execute(query).fetchone()
        if result:
            return (target, result[3], result[0])  # selectivity, solvent
        return (target, -999, "none")
    except Exception as e:
        # Fallback: return low selectivity
        return (target, -999, "error")


async def greedy_separation(
    polymers: List[str],
    temperature: float = 80.0
) -> List[SeparationStep]:
    """
    Greedy algorithm for polymer separation.

    At each step, separate the polymer that has the highest selectivity
    (i.e., easiest to separate from all remaining polymers).
    """
    import duckdb

    # Connect to the database
    conn = duckdb.connect(':memory:')

    # Load the CSV data
    import os
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_path = os.path.join(data_dir, 'COMMON-SOLVENTS-DATABASE.csv')

    if os.path.exists(csv_path):
        conn.execute(f"CREATE TABLE common_solvents_database AS SELECT * FROM read_csv_auto('{csv_path}')")
    else:
        print(f"Error: Could not find {csv_path}")
        return []

    remaining = list(polymers)
    sequence = []
    step = 0

    print(f"\n{'='*70}")
    print(f"GREEDY SEPARATION ALGORITHM")
    print(f"{'='*70}")
    print(f"Polymers: {', '.join(polymers)}")
    print(f"Temperature: {temperature}°C")
    print(f"Evaluations needed: ~{sum(range(1, len(polymers)+1))} (vs {len(polymers)}! exhaustive)")
    print(f"{'='*70}\n")

    while len(remaining) > 1:
        step += 1
        print(f"\n--- Step {step}: {len(remaining)} polymers remaining ---")
        print(f"    Mixture: {{{', '.join(remaining)}}}")

        # Evaluate each polymer as potential target
        candidates = []

        for target in remaining:
            others = [p for p in remaining if p != target]

            # Query database for selectivity
            # Column names: "Solvent", "Polymer", "Solubility (%)", "Temperature (°C)"
            query = f"""
            WITH target_sol AS (
                SELECT "Solvent" as solvent, "Solubility (%)" as t_sol
                FROM common_solvents_database
                WHERE UPPER("Polymer") = UPPER('{target}')
                AND "Solubility (%)" > 0
            ),
            others_max AS (
                SELECT "Solvent" as solvent, MAX("Solubility (%)") as o_max
                FROM common_solvents_database
                WHERE UPPER("Polymer") IN ({','.join([f"UPPER('{p}')" for p in others])})
                GROUP BY "Solvent"
            )
            SELECT
                t.solvent,
                t.t_sol as target_solubility,
                COALESCE(o.o_max, 0) as other_max,
                (t.t_sol - COALESCE(o.o_max, 0)) as selectivity
            FROM target_sol t
            LEFT JOIN others_max o ON LOWER(t.solvent) = LOWER(o.solvent)
            ORDER BY selectivity DESC
            LIMIT 1
            """

            try:
                result = conn.execute(query).fetchone()
                if result:
                    solvent = result[0]
                    selectivity = result[3] if result[3] else 0
                    candidates.append({
                        'polymer': target,
                        'solvent': solvent,
                        'selectivity': selectivity,
                        'others': others
                    })
                    print(f"    → {target}: selectivity={selectivity:.1f}% with {solvent}")
                else:
                    candidates.append({
                        'polymer': target,
                        'solvent': 'none',
                        'selectivity': -999,
                        'others': others
                    })
                    print(f"    → {target}: no data found")
            except Exception as e:
                print(f"    → {target}: error - {e}")
                candidates.append({
                    'polymer': target,
                    'solvent': 'unknown',
                    'selectivity': -999,
                    'others': others
                })

        # Pick the polymer with highest selectivity
        if candidates:
            best = max(candidates, key=lambda x: x['selectivity'])

            print(f"\n    ✓ SELECTED: {best['polymer']} (selectivity={best['selectivity']:.1f}%)")
            print(f"      Solvent: {best['solvent']}")

            sequence.append(SeparationStep(
                target_polymer=best['polymer'],
                remaining_polymers=best['others'],
                best_solvent=best['solvent'],
                selectivity=best['selectivity'],
                temperature=temperature,
                step_number=step
            ))

            remaining.remove(best['polymer'])
        else:
            print("    ✗ No valid candidates found")
            break

    # Last polymer is isolated
    if remaining:
        print(f"\n--- Step {step+1}: {remaining[0]} is isolated ✓ ---")
        sequence.append(SeparationStep(
            target_polymer=remaining[0],
            remaining_polymers=[],
            best_solvent="N/A",
            selectivity=100.0,
            temperature=temperature,
            step_number=step+1
        ))

    return sequence


def print_sequence_summary(sequence: List[SeparationStep]):
    """Print a summary of the separation sequence."""
    print(f"\n{'='*70}")
    print("GREEDY SEPARATION SEQUENCE SUMMARY")
    print(f"{'='*70}")

    print(f"\nSequence: {' → '.join([s.target_polymer for s in sequence])}")

    total_selectivity = 0
    min_selectivity = 100
    solvents_used = set()

    print(f"\nStep-by-step breakdown:")
    for step in sequence:
        if step.remaining_polymers:
            print(f"  {step.step_number}. Separate {step.target_polymer}")
            print(f"     Solvent: {step.best_solvent}")
            print(f"     Selectivity: {step.selectivity:.1f}%")
            total_selectivity += step.selectivity
            min_selectivity = min(min_selectivity, step.selectivity)
            solvents_used.add(step.best_solvent)
        else:
            print(f"  {step.step_number}. {step.target_polymer} isolated ✓")

    n_steps = len([s for s in sequence if s.remaining_polymers])

    print(f"\n{'='*70}")
    print("METRICS")
    print(f"{'='*70}")
    print(f"  Total steps: {len(sequence)}")
    print(f"  Unique solvents: {len(solvents_used)} ({', '.join(solvents_used)})")
    print(f"  Minimum selectivity: {min_selectivity:.1f}%")
    print(f"  Average selectivity: {total_selectivity/n_steps:.1f}%" if n_steps > 0 else "  N/A")


async def main():
    """Test the greedy separation algorithm."""

    # Test with 10 polymers
    polymers = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "PA6", "PA66", "PET", "PMMA"]

    print("\n" + "="*70)
    print("TESTING GREEDY SEPARATION ON 10-LAYER FILM")
    print("="*70)

    sequence = await greedy_separation(polymers, temperature=80.0)

    print_sequence_summary(sequence)

    return sequence


if __name__ == "__main__":
    asyncio.run(main())
