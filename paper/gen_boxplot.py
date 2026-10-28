"""
Box-plot figure for the four-dataset main comparison, requested as a separate
deliverable during manuscript revision (professor feedback, Sec. V-A).

Reuses the existing, validated data-loading path from gen_figures.py
(seeds_f1, same raw per-seed Kaggle result JSONs already used for Table III /
Fig. 2) without modifying that script. Produces a NEW figure only
(fig_boxplot.pdf); no existing figure, script, or result file is touched.

n = 3 seeds (42, 43, 44) per dataset -- a floor, not a distributional sample.
The box (quartiles/median) is drawn only as a compact visual summary of the
three observed values, which are also plotted individually as points, so the
figure does not visually imply a distribution estimate that three
observations cannot support.

Run from FedMAGNN/paper/:  python gen_boxplot.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gen_figures import (
    HERE, SILOS, SILO_LABELS, seeds_f1,
    C_BLUE, C_ORANGE, C_INK,
)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": C_INK, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200,
    "legend.frameon": False, "pdf.fonttype": 42,
})

CONFIG = "fedavg"          # FedAvg row of Table III: the main federated result
CONFIG_LABEL = "FedGNN (FedAvg)"


def fig_boxplot():
    data = seeds_f1(CONFIG)          # {silo: array of 3 seed F1 values}
    n_seeds = len(data[SILOS[0]])
    assert n_seeds == 3, f"expected 3 seeds, found {n_seeds}"

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    box_data = [data[s] for s in SILOS]
    bp = ax.boxplot(box_data, positions=np.arange(len(SILOS)), widths=0.42,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color=C_INK, lw=1.2),
                     whiskerprops=dict(color=C_INK, lw=0.8),
                     capprops=dict(color=C_INK, lw=0.8),
                     boxprops=dict(facecolor="#EAF1F8", edgecolor=C_BLUE, lw=0.9))

    rng = np.random.default_rng(0)
    for i, s in enumerate(SILOS):
        jitter = rng.uniform(-0.07, 0.07, size=len(data[s]))
        ax.scatter(np.full(len(data[s]), i) + jitter, data[s],
                    s=14, color=C_ORANGE, edgecolor=C_INK, linewidth=0.4,
                    zorder=5, label="individual seed" if i == 0 else None)

    ax.set_xticks(np.arange(len(SILOS)), SILO_LABELS, fontsize=7, rotation=12)
    ax.set_ylabel("Test F1 (%)")
    ax.set_title(f"{CONFIG_LABEL}, $n{{=}}3$ seeds/dataset", fontsize=7.5)
    ax.legend(fontsize=6.4, loc="lower left")

    fig.tight_layout()
    fig.savefig(HERE / "fig_boxplot.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_boxplot()
    print("figure written: fig_boxplot.pdf")
