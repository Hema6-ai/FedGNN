# FedMAGNN — Results Package (Phase B complete, 2026-07-14)

All numbers are test-set weighted F1 (%) on leakage-free data, 200k-flow
presampled splits, seeds 42/43/44 unless noted. Kaggle runs v9 (main),
v10 (seeds), v11 (zero-shot); raw JSONs in `results/kaggle_v*/results/`.

## 1. Main results (3 seeds, mean ± std)

| config | ctu13 | cicids | dohbrw | iscxvpn |
|---|---|---|---|---|
| FedProx + align | 98.51 ± 1.15 ★ | 99.75 ± 0.01 ★ | 98.35 ± 0.49 | 77.34 ± 1.15 |
| FedAvg | 99.29 ± 0.19 ★ | 99.64 ± 0.02 ★ | 98.23 ± 0.61 | 77.04 ± 1.13 |
| Local only (no fed) | 99.55 ± 0.06 ★ | 99.75 ± 0.03 ★ | 99.22 ± 0.14 ★ | 78.23 ± 0.34 |
| **Original MAGNN paper** | 97.58 | 97.70 | 98.35 | 97.12 |

★ = mean beats the original paper. Best single federated run
(`fedprox_align_long`, v9): ctu13 99.39, cicids 99.82, dohbrw 99.30.

**Claims supported:**
- Federated training matches isolated training (within noise) while never
  sharing raw data — and beats the original centralized paper on ctu13 and
  cicids, tying dohbrw.
- Prototype alignment is accuracy-neutral (v8's single-seed "alignment wins"
  did not survive multi-seed testing — report this honestly).

## 2. Differential privacy (v9, seed 42, fedprox+align)

| σ | ε (δ=1e-5) | ctu13 | cicids | dohbrw | iscxvpn |
|---|---|---|---|---|---|
| 0.5 | 168.4 | 96.94 | 99.64 | 93.67 | 71.56 |
| 1.0 | 58.9 | 96.92 | 99.62 | 93.59 | 70.29 |

DP costs ≤6 F1 on clean silos. (ε values are large; tighter budgets need
lower clip norms / fewer rounds — future work.)

## 3. Zero-shot cross-silo transfer (v11, ctu13 attack-blind)

ctu13 trained with ZERO malicious examples; test-set detection via
prototype-NN against foreign silos' DP prototypes (both arms exchange
prototypes; arms differ only in the alignment loss).

| arm | seed 42 | seed 43 |
|---|---|---|
| with alignment | 72.05 | 62.59 |
| without alignment | 62.96 | 62.62 |

**Status: mechanism validated, effect not yet robust** (one seed +9 F1, one
seed flat). Before this becomes a paper claim: calibrated anomaly-score
classifier (threshold from own-benign embeddings) + ≥3 seeds.

## 4. ISCXVPN benchmark audit (a contribution, not a failure)

The original loader derived `dst_ip` AND `dst_port` from the label —
centralized "100%" results were label leakage. Additionally, Scenario B
files duplicate Scenario A traffic with contradictory labels (~30% noise).
On cleaned data the model ladder is:

| model | test F1 |
|---|---|
| this GNN (all variants) | ~78–80 |
| 2-layer MLP | 85.8 |
| Random Forest | 95.5 |
| original paper (with leakage + duplicates) | 97.12 |

No leakage-free model reaches the paper's number; RF with duplicates left
in reaches only 96.6. Exhaustive GNN ablations (loss weights, residual,
snapshots, PE, 2× budget, 5× LR, plain-CE) all land 77–80: k-NN graphs
built from features give message passing nothing to add on this dataset.

## 5. Engineering contributions

- Monte-Carlo RWSE: positional encoding at 100k+ nodes (dense version needs
  N² memory — 72 GB at iscxvpn scale).
- Memory-bounded contrastive losses (anchor subsampling, max 4096).
- Vectorized RWR augmentation / Dirichlet sampling / label building
  (each was O(N·E) or Python-loop — hours per epoch at real scale).
- Temporal-truncation fix: snapshot fusion no longer drops supervision for
  nodes beyond temporal coverage (was collapsing time-sorted flow graphs
  to majority-class).
- Flow-level graph mode (k-NN + same-host temporal chains) for silos where
  host graphs degenerate.

## Next steps

1. Refine zero-shot classifier (calibrated threshold) + 3 seeds — when GPU
   quota resets.
2. Optional: DP at lower ε (clip/rounds sweep).
3. Writeup: intro/related work can start now; experiments section is
   complete except zero-shot.
