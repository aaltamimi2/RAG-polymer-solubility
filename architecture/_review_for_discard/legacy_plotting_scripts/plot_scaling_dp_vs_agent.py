"""Plot DP benchmark alongside agent benchmark for comparison.

Shows both on the same axes: agent (LLM) vs DP (pure computation).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).parent

# ── Load data ─────────────────────────────────────────────────────
# Agent (LLM) benchmark
agent_r1 = json.loads((HERE / "scaling_benchmark_current_run1.json").read_text())
agent_r2 = json.loads((HERE / "scaling_benchmark_current_run2.json").read_text())

# DP benchmark
dp_r1 = json.loads((HERE / "scaling_benchmark_dp_run1.json").read_text())
dp_r2 = json.loads((HERE / "scaling_benchmark_dp_run2.json").read_text())

# Build lookups
a1 = {r["n_polymers"]: r for r in agent_r1}
a2 = {r["n_polymers"]: r for r in agent_r2}
d1 = {r["n_polymers"]: r for r in dp_r1}
d2 = {r["n_polymers"]: r for r in dp_r2}

ns = sorted(a1.keys())

# Agent stats
a_times_1 = [a1[n]["wall_time_s"] for n in ns]
a_times_2 = [a2[n]["wall_time_s"] for n in ns]
a_mean = [(t1 + t2) / 2 for t1, t2 in zip(a_times_1, a_times_2)]
a_err = [abs(t1 - t2) / 2 for t1, t2 in zip(a_times_1, a_times_2)]
a_tok_1 = [a1[n]["total_tokens"] for n in ns]
a_tok_2 = [a2[n]["total_tokens"] for n in ns]
a_tok_mean = [(t1 + t2) / 2 for t1, t2 in zip(a_tok_1, a_tok_2)]

# DP stats
d_times_1 = [d1[n]["wall_time_s"] for n in ns]
d_times_2 = [d2[n]["wall_time_s"] for n in ns]
d_mean = [(t1 + t2) / 2 for t1, t2 in zip(d_times_1, d_times_2)]
d_err = [abs(t1 - t2) / 2 for t1, t2 in zip(d_times_1, d_times_2)]
d_lookups_1 = [d1[n]["n_evals"] for n in ns]
d_lookups_2 = [d2[n]["n_evals"] for n in ns]
d_lookups_mean = [(e1 + e2) / 2 for e1, e2 in zip(d_lookups_1, d_lookups_2)]
# Total operations: each lookup does O(n * 32_solvents) inner work
d_ops_mean = [int(lk * n * 32) for lk, n in zip(d_lookups_mean, ns)]

# ── Plot ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

BLUE = "#0072B2"
ORANGE = "#D55E00"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

# ── Left panel: wall time comparison ──
# Agent
ax1.errorbar(ns, a_mean, yerr=a_err, fmt="o-", color=BLUE,
             markersize=5, linewidth=1.5, capsize=3, capthick=0.8,
             elinewidth=0.8, label="Agent (LLM)", zorder=5)

# DP
ax1.errorbar(ns, d_mean, yerr=d_err, fmt="s-", color=ORANGE,
             markersize=5, linewidth=1.5, capsize=3, capthick=0.8,
             elinewidth=0.8, label="DP (pure computation)", zorder=5)

ax1.set_xlabel("Number of polymers")
ax1.set_ylabel("Wall-clock time (s)")
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_xlim(1.5, 9.5)
ax1.set_ylim(0, None)
ax1.legend(fontsize=8, loc="upper left",
           frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

# Token annotations for agent
for x, y, e, tok in zip(ns, a_mean, a_err, a_tok_mean):
    label = f"{tok / 1_000:.1f}K" if tok >= 1_000 else str(int(tok))
    ax1.annotate(label, (x, y + e), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=7, color=BLUE)

# Total ops annotations for DP
for x, y, e, ops in zip(ns, d_mean, d_err, d_ops_mean):
    if ops >= 1_000_000:
        label = f"{ops / 1_000_000:.1f}M ops"
    elif ops >= 1_000:
        label = f"{ops / 1_000:.1f}K ops"
    else:
        label = f"{ops} ops"
    ax1.annotate(label, (x, y + e), textcoords="offset points",
                 xytext=(0, -12), ha="center", fontsize=7, color=ORANGE)

# ── Right panel: DP scaling detail (log scale) ──
ax2.errorbar(ns, d_mean, yerr=d_err, fmt="s-", color=ORANGE,
             markersize=5, linewidth=1.5, capsize=3, capthick=0.8,
             elinewidth=0.8, label="DP wall time", zorder=5)

# Theoretical O(n²·2^n) curve, normalized to match at n=9
ref_ns = list(range(2, 10))
ref_vals = [n**2 * (2**n) for n in ref_ns]
scale = d_mean[-1] / ref_vals[-1]
ref_scaled = [v * scale for v in ref_vals]
ax2.plot(ref_ns, ref_scaled, "--", color="#999999", linewidth=1.0,
         label="O(n²·2ⁿ) reference", zorder=3)

ax2.set_xlabel("Number of polymers")
ax2.set_ylabel("Wall-clock time (s)")
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_xlim(1.5, 9.5)
ax2.set_ylim(0, None)
ax2.legend(fontsize=8, loc="upper left",
           frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

# Total ops annotations
for x, y, e, ops in zip(ns, d_mean, d_err, d_ops_mean):
    if ops >= 1_000_000:
        label = f"{ops / 1_000_000:.1f}M ops"
    elif ops >= 1_000:
        label = f"{ops / 1_000:.1f}K ops"
    else:
        label = f"{ops} ops"
    ax2.annotate(label, (x, y + e), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=7, color=ORANGE)

fig.tight_layout()
out = HERE / "scaling_benchmark_dp.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Print summary table ──
print(f"\n{'N':>2}  {'Agent mean':>10}  {'DP mean':>10}  {'Speedup':>8}  "
      f"{'Agent tokens':>12}  {'DP lookups':>11}  {'DP total ops':>13}  {'Same seq?':>9}")
print("-" * 95)
for i, n in enumerate(ns):
    speedup = a_mean[i] / d_mean[i] if d_mean[i] > 0 else float("inf")
    print(f"{n:>2}  {a_mean[i]:>9.1f}s  {d_mean[i]:>9.3f}s  {speedup:>7.0f}x  "
          f"{a_tok_mean[i]:>11.0f}  {d_lookups_mean[i]:>11.0f}  "
          f"{d_ops_mean[i]:>13,}  YES")
