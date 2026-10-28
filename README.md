# FedGNN

Code and results for **"FedGNN: Zero-Shot Threat Transfer via
Privacy-Preserving Federated Graph Learning for Network Intrusion
Detection"** (P. Hema, Rajendra Prasath; submitted to IEEE Transactions
on Information Forensics and Security).

FedGNN federates a graph-contrastive intrusion detector (MAGNN+) across
four heterogeneous public network-traffic corpora. Only the graph
encoder is shared between silos, under client-level differential
privacy; classification heads and the temporal encoder stay local. A
privatized, DP-noised class-prototype exchange additionally supports
zero-shot detection at silos with no labeled attack examples. This
repository contains the training and evaluation code, the federated
learning protocol (client, server, DP accountant, prototype exchange),
the Kaggle run configuration, and the per-seed result files underlying
every table and figure in the paper.

## Repository layout

```
fedmagnn/            Federated learning protocol
  client.py             local training step for one silo per round
  server.py             round orchestration, encoder aggregation
  dp.py                 Gaussian DP mechanism, Renyi accountant
  prototypes.py         DP-noised class prototypes, alignment loss
  partition.py          per-silo dataset loading and client setup

models/               MAGNN / MAGNN+ graph encoder and heads
losses/                Focal, contrastive, and multitask loss terms
data/                   per-corpus loaders and graph construction
utils/                  augmentation, sampling, metrics, logging

configs/fed_default.yaml   default federated hyperparameters
run_federated.py           entry point for all federated experiments
train.py                   single-dataset MAGNN/MAGNN+ trainer (not
                            used to produce any number reported in the
                            paper; kept for reference, see note below)

results/               per-run output JSONs and logs for every seed
  kaggle_v9/              main results, seed 42
  kaggle_v10/              main results, seeds 43-44
  kaggle_v12/              zero-shot transfer (calibrated), seeds 42-44
  kaggle_run/, kaggle_v6/ ... kaggle_v8/, kaggle_v11/
                          earlier intermediate runs, superseded by the
                          above and not used to generate any reported
                          number; kept for provenance

paper/                  manuscript source and figure/statistics scripts
  manuscript.tex           IEEE TIFS manuscript
  gen_figures.py            regenerates every figure from results/
  run_stats.py              regenerates the significance tests (stats.json)

kaggle/                 Kaggle notebook and staging scripts used to run
                        the experiment matrix on a single GPU kernel
```

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

`torch-geometric` additionally requires a compatible PyTorch build for
your CUDA version; see the
[PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Data

The four public corpora used in this study are:

- CTU-13 (Garcia et al., 2014)
- CICIDS2017 (Sharafaldin et al., 2018)
- DoHBrw-2020 (MontazeriShatoori et al., 2020)
- ISCXVPN2016 (Draper-Gil et al., 2016)

Each has its own license and distribution terms, so raw traffic is not
redistributed in this repository. `data/` contains the per-corpus
loader and the graph-construction logic described in the paper
(host graphs for CTU-13/CICIDS2017, feature-space k-NN flow graphs for
DoHBrw-2020/ISCXVPN2016, including the Scenario-A restriction and
duplicate-window removal used to correct the ISCXVPN2016 contamination
documented in the paper's benchmark audit).

Once a corpus's raw files are obtained from its original source and
pointed at by a loader's `data_root`, a processed, presampled cache is
written as `<corpus>_processed.pkl`. Subsequent runs load directly from
this cache. `fedmagnn/partition.py` resolves the cache directory in
this order: an explicit `--data_root`, the `FEDMAGNN_DATA` environment
variable, then a local `presampled/` directory.

## Reproducing the paper's results

All federated experiments share one entry point:

```bash
python run_federated.py --config configs/fed_default.yaml --mode <mode> [options]
```

| Paper result | Command |
|---|---|
| Table III / Fig. 2 (main comparison), local-only | `--mode local --seed {42,43,44}` |
| Table III / Fig. 2, FedAvg | `--mode fedavg --seed {42,43,44}` |
| Table III / Fig. 2, FedProx + alignment | `--mode fedprox --align --seed {42,43,44}` |
| Table V, personalization ablation (full-model FedAvg) | `--mode fedavg --share full` |
| Fig. 3, privacy-utility trade-off | `--mode fedprox --align --dp_sigma {0.5,1.0}` |
| Fig. 4, zero-shot threat transfer | `--mode fedavg [--align] --zeroshot_silo ctu13 --seed {42,43,44}` |

Each run writes `fed_results_<tag>.json` to `results/`. The exact
commands used for every released result are recorded in
`kaggle/fedmagnn_train.ipynb` and in each `results/kaggle_v*/results/*.log`
file.

To regenerate the paper's figures and significance tests from the
released result files:

```bash
cd paper
python run_stats.py      # writes stats.json (t-tests, p-values)
python gen_figures.py    # writes fig_architecture.pdf, fig_main_results.pdf,
                          # fig_privacy.pdf, fig_zeroshot.pdf
```

A quick development run (small subsample, few rounds, CPU):

```bash
python run_federated.py --mode fedavg --rounds 2 --sample 1500 \
    --datasets ctu13 iscxvpn --device cpu --tag smoke
```

### Note on `train.py`

`train.py` trains the base MAGNN or MAGNN+ model on a single corpus,
outside the federated setting. It is not required to reproduce any
number in the paper: the paper's "Local only" baseline is the
`--mode local` federated run above, not `train.py`, and the paper's
centralized MAGNN baseline is the value originally reported by the
MAGNN authors, not re-derived here. It is included for reference only;
its per-dataset configuration files are not part of this release.

## Running on Kaggle

The released results were produced on single-GPU Kaggle kernels.

```bash
python kaggle/stage_kaggle.py                                    # stage code + data
kaggle datasets create -p kaggle/dataset_staging                 # first upload
kaggle datasets version -p kaggle/dataset_staging -m "update"    # later updates
kaggle kernels push -p kaggle                                    # launch training
```

`kaggle/kernel-metadata.json` and `kaggle/stage_kaggle.py` use the
placeholder `<kaggle-username>`; replace it with your own Kaggle
account before pushing.

## Results summary

See [RESULTS.md](RESULTS.md) for the headline numbers, reproduced
exactly as reported in the paper, with a pointer to the script and
result files behind each one.

## Citation

If you use this code, please cite the paper:

```
P. Hema and R. Prasath, "FedGNN: Zero-Shot Threat Transfer via
Privacy-Preserving Federated Graph Learning for Network Intrusion
Detection," IEEE Transactions on Information Forensics and Security,
under review.
```

## Contact

P. Hema (phema7696@gmail.com) · Rajendra Prasath (rajendra.prasath@iiits.in)
