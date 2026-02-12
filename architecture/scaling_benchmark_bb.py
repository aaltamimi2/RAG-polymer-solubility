"""Benchmark Branch & Bound algorithm scaling for 2-9 polymers.

Runs B&B directly (no LLM). Measures wall time and nodes explored.
"""
import json
import sys
import time
import heapq
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

from strap.solubility import get_all_solvents_selectivity

POLYMERS = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "Nylon6", "Nylon66", "PET"]
HERE = Path(__file__).parent


def bb_sequence(polymers, temperature=120.0, time_limit_s=60.0):
    """Run Branch & Bound, return sequence + metrics."""
    n = len(polymers)
    n_lookups = 0
    nodes_explored = 0
    nodes_pruned = 0

    # Priority queue: (-min_sel, seq_indices, remaining_set)
    pq = [(0.0, [], set(range(n)))]
    best_min = -float("inf")
    best_seq = None
    n_complete = 0
    start = time.time()

    while pq:
        if time.time() - start > time_limit_s:
            break

        neg_min, seq, remaining = heapq.heappop(pq)
        cur_min = -neg_min if seq else float("inf")
        nodes_explored += 1

        # Pruning
        if cur_min < best_min:
            nodes_pruned += 1
            continue

        # Complete solution
        if not remaining:
            n_complete += 1
            if cur_min > best_min:
                best_min = cur_min
                best_seq = list(seq)
            continue

        # Expand
        for pidx in sorted(remaining):
            others = remaining - {pidx}
            target = polymers[pidx]
            other_names = [polymers[i] for i in sorted(others)]

            if other_names:
                results = get_all_solvents_selectivity(target, other_names, temperature)
                n_lookups += 1
                sel = results[0]["selectivity"] if results else 0.0
            else:
                sel = 100.0  # last polymer isolated

            new_min = min(cur_min, sel) if seq else sel

            if new_min >= best_min:
                heapq.heappush(pq, (-new_min, seq + [pidx], others))
            else:
                nodes_pruned += 1

    seq_str = " → ".join(polymers[i] for i in best_seq) if best_seq else "N/A"

    return {
        "sequence": seq_str,
        "min_selectivity": best_min if best_min > -float("inf") else 0.0,
        "n_lookups": n_lookups,
        "nodes_explored": nodes_explored,
        "nodes_pruned": nodes_pruned,
        "n_complete": n_complete,
        "timed_out": time.time() - start > time_limit_s,
    }


if __name__ == "__main__":
    results = []
    print(f"{'N':>2}  {'Time':>8}  {'Lookups':>8}  {'Explored':>9}  {'Pruned':>7}  "
          f"{'Complete':>9}  {'Min sel':>8}  Sequence")
    print("=" * 120)

    for n in range(2, len(POLYMERS) + 1):
        poly = POLYMERS[:n]
        t0 = time.time()
        data = bb_sequence(poly)
        elapsed = time.time() - t0

        r = {
            "n_polymers": n,
            "polymers": poly,
            "wall_time_s": elapsed,
            **data,
        }
        results.append(r)

        timeout_flag = " TIMEOUT" if data["timed_out"] else ""
        print(f"{n:>2}  {elapsed:>7.3f}s  {data['n_lookups']:>8}  {data['nodes_explored']:>9}  "
              f"{data['nodes_pruned']:>7}  {data['n_complete']:>9}  "
              f"{data['min_selectivity']:>7.1f}%  {data['sequence']}{timeout_flag}")

    out = HERE / "scaling_benchmark_bb_run1.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
