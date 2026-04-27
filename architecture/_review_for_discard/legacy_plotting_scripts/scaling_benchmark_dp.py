"""Benchmark DP algorithm scaling for 2-9 polymers.

Runs the bitmask DP directly (no LLM). Measures wall time and
selectivity evaluations. Two runs for variance.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

from strap.solubility import get_all_solvents_selectivity

POLYMERS = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "Nylon6", "Nylon66", "PET"]
HERE = Path(__file__).parent


def dp_sequence(polymers, temperature=120.0):
    """Run bitmask DP, return sequence + metrics."""
    n = len(polymers)
    full_mask = (1 << n) - 1
    n_evals = 0

    # Precompute selectivities
    sel_cache = {}
    for tidx in range(n):
        for mask in range(1, 1 << n):
            if not (mask & (1 << tidx)):
                continue
            others_mask = mask ^ (1 << tidx)
            if others_mask == 0:
                continue
            target = polymers[tidx]
            others = [polymers[i] for i in range(n) if others_mask & (1 << i)]
            results = get_all_solvents_selectivity(target, others, temperature)
            n_evals += 1
            if results:
                sel_cache[(tidx, mask)] = (results[0]["solvent"], results[0]["selectivity"])
            else:
                sel_cache[(tidx, mask)] = ("N/A", 0.0)

    # DP
    dp = {}
    for i in range(n):
        rem = full_mask ^ (1 << i)
        _, sel = sel_cache.get((i, full_mask), ("N/A", 0.0))
        if rem not in dp or sel > dp[rem][0]:
            dp[rem] = (sel, i, full_mask)

    for mask in range(full_mask - 1, -1, -1):
        if mask not in dp:
            continue
        cur_min = dp[mask][0]
        if mask == 0:
            continue
        pc = bin(mask).count("1")
        if pc == 1:
            idx = next(i for i in range(n) if mask & (1 << i))
            if 0 not in dp or cur_min > dp[0][0]:
                dp[0] = (cur_min, idx, mask)
            continue
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            new_mask = mask ^ (1 << i)
            _, sel = sel_cache.get((i, mask), ("N/A", 0.0))
            new_min = min(cur_min, sel)
            if new_mask not in dp or new_min > dp[new_mask][0]:
                dp[new_mask] = (new_min, i, mask)

    # Reconstruct
    path = []
    cur = 0
    visited = set()
    while cur in dp and cur not in visited:
        visited.add(cur)
        _, ridx, came = dp[cur]
        solv, sel = sel_cache.get((ridx, came), ("N/A", 0.0))
        path.append({"polymer": polymers[ridx], "solvent": solv, "selectivity": sel})
        if came == full_mask:
            break
        cur = came
    path.reverse()

    min_sel = dp[0][0] if 0 in dp else 0.0
    seq_str = " → ".join(s["polymer"] for s in path)

    return {
        "sequence": seq_str,
        "min_selectivity": min_sel,
        "n_evals": n_evals,
        "n_states": 1 << n,
    }


def run_one_benchmark():
    """Run DP for 2-9 polymers, return results list."""
    results = []
    for n in range(2, len(POLYMERS) + 1):
        poly = POLYMERS[:n]
        t0 = time.time()
        data = dp_sequence(poly)
        elapsed = time.time() - t0

        r = {
            "n_polymers": n,
            "polymers": poly,
            "wall_time_s": elapsed,
            "n_evals": data["n_evals"],
            "n_states": data["n_states"],
            "sequence": data["sequence"],
            "min_selectivity": data["min_selectivity"],
        }
        results.append(r)
        print(f"  n={n}: {elapsed:.3f}s | {data['n_evals']} evals | "
              f"min_sel={data['min_selectivity']:.1f}% | {data['sequence']}")

    return results


if __name__ == "__main__":
    print("=== DP Scaling Benchmark Run 1 ===")
    run1 = run_one_benchmark()
    with open(HERE / "scaling_benchmark_dp_run1.json", "w") as f:
        json.dump(run1, f, indent=2)

    print("\n=== DP Scaling Benchmark Run 2 ===")
    run2 = run_one_benchmark()
    with open(HERE / "scaling_benchmark_dp_run2.json", "w") as f:
        json.dump(run2, f, indent=2)

    print("\nDone. Saved run1 and run2 JSON files.")
