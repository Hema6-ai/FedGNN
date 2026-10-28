"""
Client data partitioning for FedMAGNN.

Each federated client corresponds to one real dataset silo:
    client 0: ctu13    (botnet traffic          — "ISP" silo)
    client 1: cicids   (enterprise intrusions   — "enterprise" silo)
    client 2: dohbrw   (DoH tunneling           — "resolver" silo)
    client 3: iscxvpn  (VPN/non-VPN traffic     — "gateway" silo)

This is genuine cross-silo non-IID federation: the silos differ in attack
families, label balance, and traffic statistics — not a synthetic Dirichlet
split of a single dataset.

Data is loaded through the existing MAGNN loaders, which fall back to the
presampled *.pkl caches when `data_root` points at a directory containing
`{name}_processed.pkl`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from torch_geometric.data import Data

from data import get_dataset

CLIENT_DATASETS = ["ctu13", "cicids", "dohbrw", "iscxvpn"]


@dataclass
class ClientData:
    name: str
    train_graph: Data
    val_graph: Data
    test_graph: Data
    train_snapshots: List[Data] = field(default_factory=list)
    graph_mode: str = "host"

    @property
    def num_train_nodes(self) -> int:
        return self.train_graph.num_nodes

    @property
    def in_dim(self) -> int:
        return self.train_graph.x.shape[1]


def resolve_data_root(explicit: Optional[str] = None) -> str:
    """
    Locate the directory containing {name}_processed.pkl files.
    Priority: explicit arg > $FEDMAGNN_DATA > known local/Kaggle paths.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("FEDMAGNN_DATA"):
        candidates.append(os.environ["FEDMAGNN_DATA"])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates += [
        os.path.join(here, "presampled"),
        os.path.join(os.path.dirname(here), "MAGNN", "presampled"),
        "/kaggle/input/fedmagnn-code/presampled",
    ]
    for c in candidates:
        if os.path.isdir(c) and any(
            os.path.exists(os.path.join(c, f"{n}_processed.pkl")) for n in CLIENT_DATASETS
        ):
            return c
    raise FileNotFoundError(
        f"Could not find presampled data. Tried: {candidates}. "
        "Set FEDMAGNN_DATA or pass --data_root."
    )


def load_clients_data(
    data_root: Optional[str] = None,
    num_snapshots: int = 5,
    sample_size: Optional[int] = None,
    seed: int = 42,
    datasets: Optional[List[str]] = None,
    zeroshot_silo: Optional[str] = None,
) -> List[ClientData]:
    """Load one ClientData per dataset silo.

    zeroshot_silo: if set, that silo's TRAINING split is stripped of all
    malicious flows — it has never seen an attack. Used by the zero-shot
    cross-silo transfer experiment (detection must come from foreign
    prototypes, not local supervision).
    """
    root = resolve_data_root(data_root)
    names = datasets or CLIENT_DATASETS
    clients = []
    for name in names:
        ds = get_dataset(
            name=name,
            data_root=root,
            num_snapshots=num_snapshots,
            sample_size=sample_size,
            seed=seed,
            cache=True,
        )
        # When loading from a presampled cache, sample_size is ignored by the
        # loader (the cache stores pre-split DataFrames) — subsample here so
        # quick dev runs stay quick.
        if sample_size is not None:
            for attr in ("_train_df", "_val_df", "_test_df"):
                df = getattr(ds, attr)
                if df is not None and len(df) > sample_size:
                    setattr(ds, attr, df.sample(n=sample_size, random_state=seed).reset_index(drop=True))

        # Zero-shot silo: remove every malicious flow from training. All
        # remaining train nodes are benign (host labels are recomputed from
        # the remaining flows during graph construction).
        if zeroshot_silo == name:
            df = ds._train_df
            ds._train_df = df[df["label"] == 0].reset_index(drop=True)

        # Flow-mode graphs put one node per flow; cap training-split size so a
        # single silo (dohbrw has 140k train flows -> ~1M edges) cannot blow
        # past CPU/GPU memory during the 3-view contrastive forward passes.
        FLOW_TRAIN_CAP = 60000
        if getattr(ds, "graph_mode", "host") == "flow" and sample_size is None:
            df = ds._train_df
            if df is not None and len(df) > FLOW_TRAIN_CAP:
                ds._train_df = df.sample(n=FLOW_TRAIN_CAP, random_state=seed).reset_index(drop=True)
        train_g, val_g, test_g = ds.get_graphs()
        mode = getattr(ds, "graph_mode", "host")
        # Temporal snapshots assume node k is the same entity in every time
        # window. Flow-graph nodes are individual flows that never recur
        # across windows, so snapshot fusion mixes unrelated nodes — skip it.
        snaps = ds.get_snapshots("train") if mode != "flow" else []
        clients.append(ClientData(
            name=name,
            train_graph=train_g,
            val_graph=val_g,
            test_graph=test_g,
            train_snapshots=snaps,
            graph_mode=mode,
        ))
    return clients
