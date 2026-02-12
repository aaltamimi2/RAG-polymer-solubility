"""Plot scaling benchmark with variance from multiple runs.

Reads scaling_benchmark_current_run1.json and run2.json,
produces a publication-quality plot with error bars and
mean token count annotations.
"""
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).parent

# ── Load data ─────────────────────────────────────────────────────────
run1 = json.loads((HERE / "scaling_benchmark_current_run1.json").read_text())
run2 = json.loads((HERE / "scaling_benchmark_current_run2.json").read_text())

# Build lookup by n_polymers
data1 = {r["n_polymers"]: r for r in run1}
data2 = {r["n_polymers"]: r for r in run2}

ns = sorted(data1.keys())

times_r1 = [data1[n]["wall_time_s"] for n in ns]
times_r2 = [data2[n]["wall_time_s"] for n in ns]
tokens_r1 = [data1[n]["total_tokens"] for n in ns]
tokens_r2 = [data2[n]["total_tokens"] for n in ns]

mean_times = [(t1 + t2) / 2 for t1, t2 in zip(times_r1, times_r2)]
# Half-range as error bar (shows the spread between the two runs)
err = [abs(t1 - t2) / 2 for t1, t2 in zip(times_r1, times_r2)]
mean_tokens = [(k1 + k2) / 2 for k1, k2 in zip(tokens_r1, tokens_r2)]

# ── Publication style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

COLORS = ["#0072B2", "#D55E00"]  # Wong palette: blue, vermillion

fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Individual runs (faded)
ax.plot(ns, times_r1, "o--", color=COLORS[0], alpha=0.3, markersize=3,
        linewidth=0.8, label="Run 1")
ax.plot(ns, times_r2, "s--", color=COLORS[1], alpha=0.3, markersize=3,
        linewidth=0.8, label="Run 2")

# Mean with error bars
ax.errorbar(ns, mean_times, yerr=err, fmt="o-", color="black",
            markersize=4, linewidth=1.2, capsize=3, capthick=0.8,
            elinewidth=0.8, label="Mean", zorder=5)

# Token annotations above each point
for x, y, e, tok in zip(ns, mean_times, err, mean_tokens):
    if tok >= 1_000:
        label = f"{tok / 1_000:.1f}K"
    else:
        label = str(int(tok))
    ax.annotate(
        label,
        (x, y + e),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=6,
        color="#444444",
    )

ax.set_xlabel("Number of polymers")
ax.set_ylabel("Wall-clock time (s)")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim(1.5, 9.5)
ax.set_ylim(0, None)
ax.legend(fontsize=6, loc="upper left",
          frameon=True, facecolor="white", edgecolor="none",
          framealpha=0.9)

fig.tight_layout()
out = HERE / "scaling_benchmark_current.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
