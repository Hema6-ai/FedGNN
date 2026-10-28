"""
Generate all paper figures from the raw Kaggle result JSONs.
Run from FedMAGNN/paper/:  python gen_figures.py
Outputs flat PDF files (referenced by exact name in manuscript.tex).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).parent
RES = HERE.parent / "results"

# Okabe-Ito (colorblind-safe), fixed categorical order
C_BLUE, C_ORANGE, C_GREEN, C_PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
C_GRAY, C_INK, C_GRID = "#7A7A7A", "#222222", "#DDDDDD"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.edgecolor": C_GRAY, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200,
    "legend.frameon": False, "pdf.fonttype": 42,
})

SILOS = ["ctu13", "cicids", "dohbrw", "iscxvpn"]
SILO_LABELS = ["CTU-13", "CICIDS2017", "DoHBrw-2020", "ISCXVPN2016"]
OG = {"ctu13": 97.58, "cicids": 97.70, "dohbrw": 98.35, "iscxvpn": 97.12}


def load(path):
    with open(path) as f:
        return json.load(f)


def load_stats():
    p = HERE / "stats.json"
    if not p.exists():
        raise FileNotFoundError("run_stats.py must be run first (writes stats.json)")
    return load(p)


def seeds_f1(config):
    """Per-silo F1 arrays over seeds 42 (v9) / 43 / 44 (v10)."""
    runs = [load(RES / "kaggle_v9/results" / f"fed_results_{config}.json")]
    for s in (43, 44):
        runs.append(load(RES / "kaggle_v10/results" / f"fed_results_{config}_s{s}.json"))
    return {silo: np.array([r["test_metrics"][silo]["f1"] for r in runs]) for silo in SILOS}


# ── Figure: main results — absolute F1 by silo ──────────────────────────────
# Statistical significance against the published MAGNN value is reported in
# Table~tab:sig in the manuscript, not annotated on this figure.
def fig_main_results():
    configs = [("fedprox_align", "FedProx + align", C_BLUE),
               ("fedavg", "FedAvg", C_ORANGE),
               ("local_only", "Local only", C_GREEN)]
    data = {tag: seeds_f1(tag) for tag, _, _ in configs}

    fig, ax = plt.subplots(figsize=(3.6, 3.3))

    x = np.arange(len(SILOS))
    w = 0.24
    for i, (tag, label, color) in enumerate(configs):
        means = [data[tag][s].mean() for s in SILOS]
        stds = [data[tag][s].std(ddof=1) for s in SILOS]  # sample sd (n=3),
        # matches run_stats.py / Table III convention -- NOT population std
        pos = x + (i - 1) * w
        ax.bar(pos, means, w * 0.92, color=color, label=label,
               yerr=stds, error_kw=dict(lw=0.8, capsize=2, ecolor=C_INK))
    for xi, silo in zip(x, SILOS):
        ax.hlines(OG[silo], xi - 1.6 * w, xi + 1.6 * w,
                   color=C_INK, lw=1.1, linestyle=(0, (4, 2)))
    ax.plot([], [], color=C_INK, lw=1.1, linestyle=(0, (4, 2)), label="MAGNN (centralized)")
    ax.set_xticks(x, SILO_LABELS, fontsize=7, rotation=12)
    ax.set_ylabel("Test F1 (%)")
    ax.set_ylim(40, 118)         # truncated at 40 (requested); disclosed in
                                  # the caption so the cut is not silent
    ax.legend(ncol=2, loc="upper center", fontsize=7.5, handlelength=1.4,
              columnspacing=1.0, handletextpad=0.5, frameon=False,
              bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout()
    fig.savefig(HERE / "fig_main_results.pdf")
    plt.close(fig)


# ── Figure 2: privacy–utility ────────────────────────────────────────────────
def fig_privacy():
    v9 = RES / "kaggle_v9/results"
    base = load(v9 / "fed_results_fedprox_align.json")["test_metrics"]
    dp05 = load(v9 / "fed_results_fed_dp_sigma05.json")
    dp10 = load(v9 / "fed_results_fed_dp_sigma10.json")

    sigmas = [0.0, 0.5, 1.0]
    # epsilon values account for BOTH Gaussian-mechanism releases per round
    # (update + prototype, same sigma) -- see manuscript Sec. IV-C-3.
    eps_labels = ["no DP", "$\\varepsilon\\approx$298", "$\\varepsilon\\approx$98"]
    series = {s: [base[s]["f1"], dp05["test_metrics"][s]["f1"], dp10["test_metrics"][s]["f1"]]
              for s in SILOS}

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PINK]
    for s, lbl, c in zip(SILOS, SILO_LABELS, colors):
        ax.plot(sigmas, series[s], marker="o", ms=4, lw=1.4, color=c, label=lbl)
        ax.annotate(lbl, (sigmas[-1], series[s][-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=6.6, color=C_INK, va="center")
    ax.set_xticks(sigmas, [f"$\\sigma$={s}\n({e})" for s, e in zip(sigmas, eps_labels)],
                  fontsize=7)
    ax.set_ylabel("Test F1 (%)")
    ax.set_ylim(55, 103)
    ax.set_xlim(-0.08, 1.45)
    fig.tight_layout()
    fig.savefig(HERE / "fig_privacy.pdf")
    plt.close(fig)


# ── Figure 3: ISCXVPN audit ladder ──────────────────────────────────────────
def fig_iscxvpn_audit():
    rows = [
        ("Original paper (contaminated data)", 97.12, C_GRAY, "//"),
        ("Random forest (clean)", 95.5, C_BLUE, None),
        ("2-layer MLP (clean)", 85.8, C_BLUE, None),
        ("FedGNN (clean)", 78.2, C_BLUE, None),
    ]
    fig, ax = plt.subplots(figsize=(3.7, 1.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, v, c, hatch) in zip(y, rows):
        ax.barh(yi, v, 0.6, color=c, hatch=hatch,
                edgecolor="white" if hatch else "none", linewidth=0.5)
        ax.text(v + 1.5, yi, f"{v:.1f}", va="center", fontsize=7, color=C_INK)
    # Labels as ordinary y-tick labels, not overlaid on the bars: the
    # previous white-text-on-hatch styling was unreadable where the
    # diagonal hatch pattern crossed the letters (the contaminated-data
    # row).
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, 108)
    ax.set_xlabel("ISCXVPN2016 test F1 (%)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_iscxvpn_audit.pdf")
    plt.close(fig)


# ── Figure 4: convergence ────────────────────────────────────────────────────
def fig_convergence():
    hist = load(RES / "kaggle_v9/results/fed_results_fedprox_align.json")["history"]
    rounds = [h["round"] for h in hist]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PINK]
    for s, lbl, c in zip(SILOS, SILO_LABELS, colors):
        ax.plot(rounds, [h["val"][s]["f1"] for h in hist], lw=1.2, color=c, label=lbl)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation F1 (%)")
    ax.set_ylim(30, 103)
    ax.legend(fontsize=6.8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "fig_convergence.pdf")
    plt.close(fig)


# ── Figure 5: zero-shot — naive vs calibrated (v12, no-align config, 3 seeds) ─
# The with-/without-alignment split was dropped from this figure: the two
# arms are statistically indistinguishable (Sec. V-D reports the two-sample
# test, t=-2.61, p=0.065) and, at this scale, visually redundant bars added
# nothing the text does not already state precisely; the comparison is
# reported in prose instead. This plots the no-alignment arm, which is also
# the configuration behind the headline number in the abstract/introduction.
def fig_zeroshot():
    v12 = RES / "kaggle_v12/results"
    seeds = (42, 43, 44)
    vals = {m: [] for m in ("naive", "calibrated")}
    for s in seeds:
        r = load(v12 / f"fed_results_zsc_noalign_s{s}.json")
        for m in ("naive", "calibrated"):
            vals[m].append(r["zeroshot"][m]["f1"])

    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    x = np.arange(2)                      # naive, calibrated
    means = [np.mean(vals[m]) for m in ("naive", "calibrated")]
    stds = [np.std(vals[m], ddof=1) for m in ("naive", "calibrated")]  # sample
    # sd (n=3), matches run_stats.py convention -- NOT population std
    ax.bar(x, means, 0.55, color=[C_ORANGE, C_BLUE],
           yerr=stds, error_kw=dict(lw=0.8, capsize=3, ecolor=C_INK))
    for xi, m in zip(x, means):
        ax.text(xi, m + 2.0, f"{m:.1f}", ha="center", fontsize=7.2, color=C_INK)

    ax.set_xticks(x, ["Naive\nprototype-NN", "Calibrated\n(benign 95th pct.)"],
                  fontsize=7.5)
    ax.set_ylabel("Zero-shot test F1 (%)")
    ax.set_ylim(0, 108)
    fig.tight_layout()
    fig.savefig(HERE / "fig_zeroshot.pdf")
    plt.close(fig)


# ── Figure 6: architecture diagram ──────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.6, 4.55))
    ax.set_xlim(0, 11.35)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    ax.grid(visible=False)

    silo_names = ["CTU-13\n(botnet)", "CICIDS2017\n(intrusion)",
                  "DoHBrw-2020\n(DoH tunnel)", "ISCXVPN2016\n(VPN)"]
    box_w, box_top, box_bot = 1.85, 4.10, 2.40
    xs = [0.35, 2.55, 4.75, 6.95]
    server_top = 1.60

    server_x0, server_w = 4.95, 2.6
    for i, (x0, nm) in enumerate(zip(xs, silo_names)):
        cx = x0 + box_w / 2
        ax.add_patch(FancyBboxPatch((x0, box_bot), box_w, box_top - box_bot,
                     boxstyle="round,pad=0.06", fc="#EAF1F8", ec=C_BLUE, lw=1))
        # silo name, upper band
        ax.text(cx, 3.85, nm, ha="center", va="center", fontsize=6.8)
        # shared-encoder label, middle band (clear of name above and heads below)
        ax.text(cx, 3.18, "shared\nencoder", ha="center", va="center",
                fontsize=6.3, color=C_BLUE, style="italic")
        # private-heads box, lower band
        heads_top, heads_bot = 2.80, 2.46
        ax.add_patch(FancyBboxPatch((x0 + 0.12, heads_bot), box_w - 0.24,
                     heads_top - heads_bot,
                     boxstyle="round,pad=0.03", fc="#FBF0DC", ec=C_ORANGE, lw=0.8))
        ax.text(cx, (heads_top + heads_bot) / 2, "private heads",
                ha="center", va="center", fontsize=6.0, color="#7A5A10")
        # arrow to server: fan out across the server's top edge so the four
        # arrowheads land at distinct, non-overlapping points
        end_x = server_x0 + 0.35 + i * (server_w - 0.7) / 3
        ax.add_patch(FancyArrowPatch((cx, box_bot), (end_x, server_top),
                     arrowstyle="<->", mutation_scale=8, color=C_GRAY, lw=0.9))

    server_x0, server_w, server_h = 4.95, 2.6, 0.85
    ax.add_patch(FancyBboxPatch((server_x0, 0.75), server_w, server_h,
                 boxstyle="round,pad=0.06", fc="#E7F3ED", ec=C_GREEN, lw=1.1))
    ax.text(server_x0 + server_w / 2, 1.175,
            "Server: FedAvg / FedProx\n+ DP (clip & noise, RDP $\\varepsilon$)",
            ha="center", va="center", fontsize=6.8)

    proto_x0, proto_w, proto_top, proto_bot = 9.45, 1.65, 4.10, 2.40
    ax.add_patch(FancyBboxPatch((proto_x0, proto_bot), proto_w, proto_top - proto_bot,
                 boxstyle="round,pad=0.06", fc="#F8ECF3", ec=C_PINK, lw=1))
    proto_cx = proto_x0 + proto_w / 2
    ax.text(proto_cx, 3.85, "Prototype\nexchange", ha="center", va="center",
            fontsize=6.9, fontweight="bold")
    ax.text(proto_cx, 2.98,
            "2 DP-noised 128-d\nclass prototypes /\nsilo / round\n"
            "$\\rightarrow$ alignment\n$\\rightarrow$ zero-shot",
            ha="center", va="center", fontsize=6.0)
    # arrow lives entirely in the whitespace between the two boxes, clear of
    # any text on either side
    last_x0 = xs[-1]
    ax.add_patch(FancyArrowPatch((last_x0 + box_w, 3.18), (proto_x0, 3.18),
                 arrowstyle="<->", mutation_scale=8, color=C_PINK, lw=1.0))

    ax.text(0.35, 0.42,
            "Shared per round: encoder deltas (clipped + noised)\n"
            "Never shared: raw traffic, topology, node/edge embeddings, label counts",
            fontsize=6.5, va="center", ha="left", color=C_INK)

    # Zero-shot inference: NOT a fifth silo (the mechanism uses one of the
    # four existing silos, e.g. an attack-blind CTU-13, per Sec. V-C) --
    # drawn as a dashed, inference-time-only path off the prototype channel
    # so it is not confused with the per-round training loop above.
    zs_x0, zs_w, zs_top, zs_bot = 9.30, 2.00, 1.55, 0.20
    ax.add_patch(FancyBboxPatch((zs_x0, zs_bot), zs_w, zs_top - zs_bot,
                 boxstyle="round,pad=0.05", fc="#FFF6E9", ec=C_ORANGE,
                 lw=0.9, linestyle=(0, (3, 2))))
    ax.text(zs_x0 + zs_w / 2, (zs_top + zs_bot) / 2,
            "Zero-shot\ninference:\nattack-blind silo\n(e.g. CTU-13)\nscores own traffic\nagainst foreign\nprototypes;\nthreshold = own\nbenign 95th pct.",
            ha="center", va="center", fontsize=5.4, color="#7A5A10", linespacing=1.35)
    ax.add_patch(FancyArrowPatch((proto_cx, proto_bot), (zs_x0 + zs_w / 2, zs_top),
                 arrowstyle="-|>", mutation_scale=7, color=C_ORANGE, lw=0.9,
                 linestyle=(0, (3, 2))))

    # ── Zoom-in: what "shared encoder" / "private heads" actually contain,
    # per silo (Sec. IV-A/IV-F). Same pipeline, independently weighted,
    # runs identically inside every one of the four silo boxes below.
    pl_y0, pl_h = 5.15, 1.05
    pl_top = pl_y0 + pl_h
    stages = [
        ("Local graph\n(private,\nnever shared)", "#EDEDED", C_GRAY, C_INK),
        ("Input proj. +\npositional\nencoding", "#EAF1F8", C_BLUE, C_BLUE),
        ("Graph\ntransformer\n$\\times 3$, $H{=}4$", "#EAF1F8", C_BLUE, C_BLUE),
        ("Temporal\ntransformer\n(host silos)", "#FBF0DC", C_ORANGE, "#7A5A10"),
        ("Task heads\n(focal + reg.)", "#FBF0DC", C_ORANGE, "#7A5A10"),
        ("Output:\nnode score", "#EDEDED", C_GRAY, C_INK),
    ]
    n = len(stages)
    gap, pl_x0, pl_x1 = 0.18, 0.35, 11.10
    bw = ((pl_x1 - pl_x0) - gap * (n - 1)) / n
    centers = []
    for i, (label, fc, ec, tc) in enumerate(stages):
        x0 = pl_x0 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x0, pl_y0), bw, pl_h,
                     boxstyle="round,pad=0.045", fc=fc, ec=ec, lw=1))
        ax.text(x0 + bw / 2, pl_y0 + pl_h / 2, label, ha="center", va="center",
                fontsize=5.6, color=tc)
        centers.append(x0 + bw / 2)
        if i > 0:
            ax.add_patch(FancyArrowPatch((centers[i - 1] + bw / 2, pl_y0 + pl_h / 2),
                         (x0, pl_y0 + pl_h / 2), arrowstyle="-|>",
                         mutation_scale=6, color=C_GRAY, lw=0.7))
    # boundary marker between graph transformer (shared) and temporal
    # transformer (private): this is the literal aggregation cut point
    boundary_x = (centers[2] + bw / 2 + centers[3] - bw / 2) / 2
    ax.plot([boundary_x, boundary_x], [pl_y0 - 0.06, pl_top + 0.06],
            color=C_INK, lw=1.1, linestyle=(0, (1, 1)))
    ax.text(centers[1] + (centers[2] - centers[1]) / 2, pl_top + 0.16,
            "shared (aggregated by server)", ha="center", va="bottom",
            fontsize=5.6, color=C_BLUE, style="italic")
    ax.text(centers[3] + (centers[4] - centers[3]) / 2, pl_top + 0.16,
            "private (never leaves silo)", ha="center", va="bottom",
            fontsize=5.6, color="#7A5A10", style="italic")
    ax.text(pl_x0, pl_top + 0.34,
            "Inside each silo's MAGNN$^{+}$ (identical architecture, "
            "independently weighted per silo):",
            ha="left", va="bottom", fontsize=6.4, color=C_INK)
    # bracket down to the CTU-13 box, showing this is a zoom of every silo
    # box above, not a separate fifth component
    ax.add_patch(FancyArrowPatch((xs[0] + box_w / 2, pl_y0 - 0.05),
                 (xs[0] + box_w / 2, box_top + 0.05),
                 arrowstyle="-|>", mutation_scale=7, color=C_GRAY, lw=0.8,
                 linestyle=(0, (2, 1.5)), connectionstyle="arc3,rad=0.0"))

    fig.savefig(HERE / "fig_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_main_results()
    fig_privacy()
    fig_zeroshot()
    fig_architecture()
    print("figures written:", sorted(p.name for p in HERE.glob("fig_*.pdf")))
