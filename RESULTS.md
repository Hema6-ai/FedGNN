# Results summary

The numbers below are reproduced exactly as reported in the paper. All
headline numbers are mean ± std over three random seeds (42, 43, 44),
computed by [paper/run_stats.py](paper/run_stats.py) from the per-seed
JSON files in `results/`.

## Main comparison (paper Table III)

Test F1 (%), mean ± std over 3 seeds. ★ = significantly exceeds the
originally reported centralized MAGNN (p < 0.05, one-sample t-test;
paper Table IV).

| Configuration | CTU-13 | CICIDS2017 | DoHBrw-2020 | ISCXVPN2016 |
|---|---|---|---|---|
| MAGNN (centralized, originally reported) | 97.58 | 97.70 | 98.35 | 97.12 |
| Local only | 99.6 ± 0.1 ★ | 99.8 ± 0.0 ★ | 99.2 ± 0.1 ★ | 78.2 ± 0.3 |
| FedProx + alignment | 98.5 ± 1.2 | 99.8 ± 0.0 ★ | 98.4 ± 0.5 | 77.3 ± 1.2 |
| FedAvg | 99.3 ± 0.2 ★ | 99.6 ± 0.0 ★ | 98.2 ± 0.6 | 77.0 ± 1.1 |

Source: `results/kaggle_v9/results/` (seed 42), `results/kaggle_v10/results/`
(seeds 43, 44). ISCXVPN2016's gap is a property of the benchmark, not of
federation (see Benchmark audit below); local-only matches it exactly.

Two-sample tests against local-only training show federation costs no
statistically significant accuracy on 7 of 8 (configuration, silo)
pairs; the one exception (FedAvg on CICIDS2017, p = 0.012) is a
0.11-F1 difference on a silo whose standard deviation is already
≤ 0.03.

## Feature comparison (paper Table V)

| Method | Personalized | DP | Zero-shot | Scale-validated | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Full-model FedAvg (diagnostic) | no | no | no | yes | 90.3 | 83.8 | 86.1 |
| MAGNN (centralized) | — | no | no | no | — | — | 97.88 |
| **FedGNN** | yes | yes | yes | yes | **99.12** | **99.03** | **99.05** |

FedGNN's P/R/F1 are the FedAvg row of Table III, averaged over the
three structurally sound silos (CTU-13, CICIDS2017, DoHBrw-2020); the
MAGNN F1 is the same average of its originally reported per-silo
values. The full-model FedAvg row is a smaller diagnostic run (single
seed, 6k-flow subsample, 20 rounds; see Personalization ablation
below), not the main protocol.

## Personalization ablation

Full-model FedAvg (encoder and task heads both aggregated, no
personalization) was trained as a diagnostic run: one seed, a 6k-flow
subsample per silo, 20 rounds. Mean test F1 was 86.1, roughly eight
points below the personalized configurations above despite a far
smaller scale and budget; CICIDS2017's per-round F1 oscillated between
80.2 and 97.5 without stabilizing. Source: `results/fed_results_fullshare_diagnostic.json`.

## Privacy-utility trade-off

Update and prototype releases are jointly accounted in one Renyi-DP
accountant over the full 50-round protocol (δ = 1e-5):

| σ | ε | Cost vs. non-private |
|---|---|---|
| 0.5 | ≈298 | smaller |
| 1.0 | ≈98 | at most 6 F1 points on structurally sound silos |

Source: `results/kaggle_v9/results/fed_results_fed_dp_sigma05.json`,
`fed_results_fed_dp_sigma10.json`.

## Zero-shot threat transfer

CTU-13 retrained with every malicious flow removed from its training
split; detection uses only foreign DP-noised class prototypes and the
silo's own benign traffic (no labels, no fine-tuning, no attack
examples).

| Detector | F1 |
|---|---|
| Nearest-prototype (naive, θ = 0), without alignment | 61.5 ± 2.6 |
| Nearest-prototype (naive, θ = 0), with alignment | 61.6 ± 1.2 |
| Benign-quantile calibrated, without alignment | 94.5 ± 0.05 |
| Benign-quantile calibrated, with alignment | 94.4 ± 0.06 |

The two calibrated arms are statistically indistinguishable
(two-sample t-test, t = -2.61, p = 0.065): the zero-shot capability
does not measurably depend on the InfoNCE alignment objective. The
calibrated-vs-naive gain is significant under a paired t-test
(t = +50.21, p = 0.0004). Source: `results/kaggle_v12/results/`
(`fed_results_zsc_align_s*.json`, `fed_results_zsc_noalign_s*.json`).

## Benchmark audit: ISCXVPN2016

The corpus's published 97.12 F1 was produced under two contaminations:
graph endpoints synthesized from the class label, and exact-duplicate
capture windows across scenario files with contradictory labels. On
the cleaned benchmark (Scenario A only, deduplicated, no synthetic
endpoints), the model ladder is:

| Model | Test F1 |
|---|---|
| Random forest | 95.5 |
| Two-layer MLP | 85.8 |
| GNN (all variants tested, 10-intervention ablation) | 78-80 |
| Original paper (with contamination) | 97.12 — unreachable on clean data |

Mixing scenario files without deduplication caps any model near 68%
accuracy from label noise alone; restricting to Scenario A and
deduplicating lifts a random forest from 68% to 96%. Source: `data/iscxvpn_loader.py`,
`data/preprocessing.py`.

## Reproducing these numbers

```bash
cd paper
python run_stats.py      # recomputes every t-test/p-value above from results/
python gen_figures.py    # regenerates fig_main_results.pdf, fig_privacy.pdf, fig_zeroshot.pdf
```

See [README.md](README.md) for the commands used to produce each
underlying run.
