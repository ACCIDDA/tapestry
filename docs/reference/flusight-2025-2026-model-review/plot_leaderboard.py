"""Reproduce the supplied evaluation table and its descriptive comparison plot."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT = Path(__file__).resolve().parent
with (OUT / "evaluation-table.csv").open(newline="") as stream:
    rows = list(csv.reader(stream))

top = rows[1:9]
labels = [row[0].split(" (")[0] for row in top]
values = np.array([[float(x) for x in row[1:]] for row in top])
y = np.arange(len(top))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True,
                         gridspec_kw={"width_ratios": [1.15, 1, 1]})
colors = ["#246c8c", "#ad5f20", "#246c8c", "#60787f", "#ad5f20", "#60787f", "#60787f", "#60787f"]
axes[0].scatter(values[:, 0], y, s=65, c=colors)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlim(.54, .66)
axes[0].set_title("Relative WIS · lower is better")
axes[0].set_xlabel("Log-count evaluation; baseline = 1")
for yi, (score, _, _, completeness) in zip(y, values):
    axes[0].annotate(f"{score:.2f}  ({completeness:.0f}% submitted)", (score, yi),
                     xytext=(5, -14), textcoords="offset points", fontsize=8.5)
for ax, col, nominal in [(axes[1], 1, 50), (axes[2], 2, 95)]:
    ax.axvline(nominal, color="#777777", linestyle="--", linewidth=1.2)
    ax.scatter(values[:, col], y, s=65, c=colors)
    ax.set_title(f"{nominal}% interval coverage")
    ax.set_xlabel("Observed coverage (%)")
    ax.set_xlim((30, 57) if nominal == 50 else (78, 99))
    for yi, val in zip(y, values[:, col]):
        ax.annotate(f"{val:.2f}", (val, yi), xytext=(5, -14),
                    textcoords="offset points", fontsize=9)
for ax in axes:
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
fig.suptitle("FluSight 2025–2026 · Leading models and interval coverage", fontsize=16, x=.55)
fig.text(.02, .025, "Source: supplied FluSight_2025-26_Evaluation.docx, Table 1. Dashed lines: nominal coverage.\n"
         "Transcribed scores, not rescored forecasts. Unequal submission sets; no significance estimates available.", fontsize=9)
fig.tight_layout(rect=(0, .10, 1, .94))
fig.savefig(OUT / "leaderboard.png", dpi=180)
fig.savefig(OUT / "leaderboard.pdf")
plt.close(fig)
