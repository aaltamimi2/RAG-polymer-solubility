"""Quick comparison: Greedy vs DP sequences for 2-9 polymers."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

from strap.solubility import get_all_solvents_selectivity

POLYMERS = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "Nylon6", "Nylon66", "PET"]


# ── Greedy ────────────────────────────────────────────────────────

def greedy_sequence(polymers, temperature=120.0):
    remaining = list(polymers)
    steps = []
    while len(remaining) > 1:
        best_sel = -1
        best_polymer = None
        best_solvent = None
        for target in remaining:
            others = [p for p in remaining if p != target]
            results = get_all_solvents_selectivity(target, others, temperature)
            if results and results[0]["selectivity"] > best_sel:
                best_sel = results[0]["selectivity"]
                best_polymer = target
                best_solvent = results[0]["solvent"]
        steps.append((best_polymer, best_solvent, best_sel))
        remaining.remove(best_polymer)
    steps.append((remaining[0], "(isolated)", None))
    return steps


# ── DP (bitmask) ──────────────────────────────────────────────────

def dp_sequence(polymers, temperature=120.0):
    n = len(polymers)
    full_mask = (1 << n) - 1
    INF = float("inf")

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
        path.append((polymers[ridx], solv, sel))
        if came == full_mask:
            break
        cur = came
    path.reverse()

    # Min selectivity
    min_sel = dp[0][0] if 0 in dp else 0.0

    return path, min_sel, len(sel_cache)


# ── Compare ───────────────────────────────────────────────────────

print(f"{'N':>2}  {'Greedy seq':<55} {'DP seq':<55} {'Same?':>5}  "
      f"{'Greedy min':>10}  {'DP min':>10}  {'DP evals':>9}  {'DP time':>8}")
print("=" * 170)

for n in range(2, 10):
    poly = POLYMERS[:n]

    # Greedy
    t0 = time.time()
    g_steps = greedy_sequence(poly)
    g_time = time.time() - t0
    g_seq = [s[0] for s in g_steps]
    g_sels = [s[2] for s in g_steps if s[2] is not None]
    g_min = min(g_sels) if g_sels else 0.0

    # DP
    t0 = time.time()
    d_steps, d_min, d_evals = dp_sequence(poly)
    d_time = time.time() - t0
    d_seq = [s[0] for s in d_steps]

    same = "YES" if g_seq == d_seq else "NO"

    g_str = " → ".join(g_seq)
    d_str = " → ".join(d_seq)

    print(f"{n:>2}  {g_str:<55} {d_str:<55} {same:>5}  "
          f"{g_min:>9.1f}%  {d_min:>9.1f}%  {d_evals:>9}  {d_time:>7.1f}s")
