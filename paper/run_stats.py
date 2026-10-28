"""
Statistical analysis for the manuscript's Results section.
Run from FedMAGNN/paper/: python run_stats.py
Recomputes every t-test, p-value, and parameter/communication count cited
in the paper directly from results/kaggle_v9-v12 JSONs and the live model.
Writes stats.json (consumed by gen_figures.py for significance markers).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
ROOT = HERE.parent
RES = ROOT / "results"

SILOS = ["ctu13", "cicids", "dohbrw", "iscxvpn"]
OG = {"ctu13": 97.58, "cicids": 97.70, "dohbrw": 98.35, "iscxvpn": 97.12}


def load(p):
    with open(p) as f:
        return json.load(f)


def seeds_f1(tag):
    runs = [load(RES / "kaggle_v9/results" / f"fed_results_{tag}.json")]
    for s in (43, 44):
        runs.append(load(RES / "kaggle_v10/results" / f"fed_results_{tag}_s{s}.json"))
    return {silo: np.array([r["test_metrics"][silo]["f1"] for r in runs]) for silo in SILOS}


def one_sample_vs_og(tag):
    """One-sample t-test: is the 3-seed mean significantly > the published
    MAGNN value? One-tailed (H1: mean > OG), df=2."""
    data = seeds_f1(tag)
    out = {}
    for s in SILOS:
        x = data[s]
        t, p_two = stats.ttest_1samp(x, OG[s])
        p_one = p_two / 2 if t > 0 else 1 - p_two / 2
        out[s] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                   "t": float(t), "df": 2, "p_one_tailed": float(p_one),
                   "significant": bool(p_one < 0.05 and t > 0)}
    return out


def two_sample(tag_a, tag_b, label):
    a, b = seeds_f1(tag_a), seeds_f1(tag_b)
    out = {}
    for s in SILOS:
        t, p = stats.ttest_ind(a[s], b[s], equal_var=False)
        out[s] = {"mean_a": float(a[s].mean()), "mean_b": float(b[s].mean()),
                   "t": float(t), "p": float(p), "significant": bool(p < 0.05)}
    return out


def zeroshot_stats():
    seeds = (42, 43, 44)
    align = np.array([load(RES / "kaggle_v12/results" / f"fed_results_zsc_align_s{s}.json")
                       ["zeroshot"]["calibrated"]["f1"] for s in seeds])
    noalign = np.array([load(RES / "kaggle_v12/results" / f"fed_results_zsc_noalign_s{s}.json")
                         ["zeroshot"]["calibrated"]["f1"] for s in seeds])
    naive = np.array([load(RES / "kaggle_v12/results" / f"fed_results_zsc_align_s{s}.json")
                       ["zeroshot"]["naive"]["f1"] for s in seeds])
    t_align, p_align = stats.ttest_ind(align, noalign, equal_var=False)
    t_cal, p_cal = stats.ttest_rel(align, naive)
    return {
        "align_mean": float(align.mean()), "align_sd": float(align.std(ddof=1)),
        "noalign_mean": float(noalign.mean()), "noalign_sd": float(noalign.std(ddof=1)),
        "align_vs_noalign_t": float(t_align), "align_vs_noalign_p": float(p_align),
        "naive_mean": float(naive.mean()),
        "calibrated_vs_naive_t": float(t_cal), "calibrated_vs_naive_p": float(p_cal),
    }


def param_counts():
    sys.path.insert(0, str(ROOT))
    import torch, yaml
    from train import build_model
    with open(ROOT / "configs/fed_default.yaml") as f:
        cfg = yaml.safe_load(f)
    m = build_model("magnn_plus", cfg, in_dim=15)
    enc_prefixes = ("in_proj.", "pe.", "gnn.", "head_projs.")
    enc = sum(p.numel() for n, p in m.named_parameters() if n.startswith(enc_prefixes))
    total = sum(p.numel() for p in m.parameters())
    return {
        "total_params": int(total), "encoder_params": int(enc),
        "private_params": int(total - enc),
        "encoder_pct": float(enc / total * 100),
        "encoder_mb_fp32": float(enc * 4 / 1e6),
        "fullmodel_mb_fp32": float(total * 4 / 1e6),
        "prototype_bytes": 2 * 128 * 4,
        "communication_reduction_x": float(total / enc),
    }


if __name__ == "__main__":
    out = {
        "one_sample_vs_og": {tag: one_sample_vs_og(tag)
                              for tag in ("fedprox_align", "fedavg", "local_only")},
        "align_vs_fedavg": two_sample("fedprox_align", "fedavg", "alignment effect"),
        "fedavg_vs_local": two_sample("fedavg", "local_only", "federated vs. isolated (FedAvg)"),
        "align_vs_local": two_sample("fedprox_align", "local_only", "federated vs. isolated (FedProx+align)"),
        "zeroshot": zeroshot_stats(),
        "params": param_counts(),
    }
    with open(HERE / "stats.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=== One-sample t-test vs published MAGNN (df=2, one-tailed) ===")
    for tag, res in out["one_sample_vs_og"].items():
        print(f"-- {tag} --")
        for s, v in res.items():
            flag = "*" if v["significant"] else " "
            print(f"  {s:10s} mean={v['mean']:.2f} t={v['t']:+.2f} "
                  f"p={v['p_one_tailed']:.4f} {flag}")
    print("\n=== Params ===")
    for k, v in out["params"].items():
        print(f"  {k}: {v}")
    print("\nWrote", HERE / "stats.json")
