# FedMAGNN — Privacy-Preserving Federated Malicious Traffic Detection

Federated extension of MAGNN+ (`../MAGNN/magnn_improved`). Four real dataset
silos train a shared graph encoder **without any raw traffic leaving a silo**:

| client | dataset     | silo role   | traffic type        |
|--------|-------------|-------------|---------------------|
| 0      | CTU-13      | ISP         | botnet               |
| 1      | CICIDS2017  | enterprise  | intrusions           |
| 2      | DoHBrw2020  | resolver    | DoH tunneling        |
| 3      | ISCXVPN2016 | gateway     | VPN/non-VPN          |

This is genuine cross-silo **non-IID** federation — the silos differ in attack
families, label balance (CTU-13 is 1.4% malicious, ISCXVPN 86%), and traffic
statistics. Not a synthetic Dirichlet split of one dataset.

## Method

1. **FedPer-style personalization** — only the graph encoder
   (`in_proj`, positional encoding, graph transformer, head projections) is
   aggregated; classification/regression heads and the temporal encoder stay
   private per silo (`--share full` shares everything, as an ablation).
2. **FedAvg / FedProx** aggregation over shared-parameter *deltas*, weighted
   by silo size (uniform under DP).
3. **Client-level DP** (DP-FedAvg): per-client delta clipping + Gaussian
   noise on the aggregate; ε tracked with an RDP accountant
   ([fedmagnn/dp.py](fedmagnn/dp.py)).
4. **DP prototype alignment** (the novel piece,
   [fedmagnn/prototypes.py](fedmagnn/prototypes.py)): each round every silo
   releases two DP-noised 128-dim class prototypes (mean benign / mean
   malicious embedding). Clients add an InfoNCE term pulling their embeddings
   toward other silos' same-class prototypes — aligning latent spaces across
   silos without sharing topology, features, or per-node embeddings. The same
   channel is the future "embedding gossip" primitive for the agent layer.

## Experiment matrix (run automatically by the Kaggle notebook)

| tag             | mode    | align | DP σ | question answered                     |
|-----------------|---------|-------|------|---------------------------------------|
| `local_only`    | local   | —     | —    | lower bound: no communication          |
| `fedavg`        | fedavg  | no    | 0    | does federation help at all?           |
| `fedavg_align`  | fedavg  | yes   | 0    | does prototype alignment add on top?   |
| `fedprox_align` | fedprox | yes   | 0    | does the proximal term stabilize?      |
| `fed_dp_sigma05`| fedprox | yes   | 0.5  | accuracy cost of moderate privacy      |
| `fed_dp_sigma10`| fedprox | yes   | 1.0  | accuracy cost of strong privacy        |

Reference upper bound (centralized MAGNN+, from `../MAGNN/magnn_improved/results`):
CTU-13 97.26/97.74, CICIDS 99.55/99.64, DoHBrw 100/100, ISCXVPN 100/100 (acc/F1).

Budget parity: 50 rounds × 2 local epochs = 100 epoch-equivalents, matching
the centralized runs.

## Run locally

```bash
python run_federated.py --mode fedavg --align                 # full run
python run_federated.py --mode fedavg --rounds 2 --sample 1500 \
    --datasets ctu13 iscxvpn --device cpu --tag smoke          # quick dev
```

Data resolves automatically from `../MAGNN/presampled/` (or set
`FEDMAGNN_DATA` / `--data_root`).

## Run on Kaggle

```bash
python kaggle/stage_kaggle.py                       # stage code + data
kaggle datasets create -p kaggle/dataset_staging    # first time only
kaggle datasets version -p kaggle/dataset_staging -m "update"   # after edits
kaggle kernels push -p kaggle                       # launch training
kaggle kernels status hemapolumati/fedmagnn-federated-training
kaggle kernels output hemapolumati/fedmagnn-federated-training -p results/
```

The notebook ([kaggle/fedmagnn_train.ipynb](kaggle/fedmagnn_train.ipynb)) runs
the full matrix (~2–3 h on a P100/T4) and writes `summary.csv` + per-run JSONs
+ `fedmagnn_results.zip`.

## Design notes

- **Serial simulation, not Flower**: 4 cross-silo clients on one GPU gain
  nothing from Ray-based parallelism, and serial rounds are exactly
  reproducible. The `FedServer` round loop is the aggregation logic a real
  deployment would run.
- Graphs live on CPU and move to GPU per client turn, so 4 clients share one
  device without memory pressure.
- Under DP, delta averaging is uniform (not size-weighted): data-dependent
  weights would leak silo sizes and complicate sensitivity analysis.
