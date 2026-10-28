"""
Graph augmentation strategies for contrastive learning views:
  1. Edge Modification (EM): randomly drop + add edges
  2. Random Walk with Restart (RWR): generate a subgraph by personalized random walk
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import scipy.sparse as sp
from typing import Optional


def edge_modification(
    data: Data,
    drop_ratio: float = 0.1,
    add_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Data:
    """
    Edge Modification (EM) augmentation.
    - Randomly drops `drop_ratio` fraction of existing edges.
    - Randomly adds `add_ratio * E` new random edges (to simulate noise/new connections).

    Returns a new Data object sharing the same node features but with a modified edge_index.
    """
    rng = np.random.default_rng(seed)
    edge_index = data.edge_index.numpy()
    E = edge_index.shape[1]
    N = data.num_nodes

    # Drop edges
    num_drop = int(E * drop_ratio)
    keep_mask = np.ones(E, dtype=bool)
    if num_drop > 0:
        drop_idx = rng.choice(E, size=num_drop, replace=False)
        keep_mask[drop_idx] = False

    new_src = list(edge_index[0, keep_mask])
    new_dst = list(edge_index[1, keep_mask])
    edge_attr_keep = data.edge_attr[keep_mask] if data.edge_attr is not None else None

    # Add edges
    num_add = int(E * add_ratio)
    if num_add > 0:
        add_src = rng.integers(0, N, size=num_add).tolist()
        add_dst = rng.integers(0, N, size=num_add).tolist()
        new_src.extend(add_src)
        new_dst.extend(add_dst)

        if edge_attr_keep is not None:
            mean_attr = data.edge_attr.mean(dim=0, keepdim=True).expand(num_add, -1)
            edge_attr_keep = torch.cat([edge_attr_keep, mean_attr], dim=0)

    new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)

    aug = Data(
        x=data.x,
        edge_index=new_edge_index,
        edge_attr=edge_attr_keep,
        y=data.y,
        num_nodes=N,
    )
    return aug


def random_walk_restart(
    data: Data,
    restart_prob: float = 0.15,
    num_steps: int = 100,
    seed: Optional[int] = None,
) -> Data:
    """
    Random Walk with Restart (RWR) augmentation.
    Simulates a personalized PageRank walk to generate an alternative subgraph view.

    Starting from all nodes simultaneously, walks `num_steps` steps with restart
    probability `restart_prob`. Edges traversed during the walk are kept; the rest
    are dropped. This produces a topology-aware augmented view.
    """
    rng = np.random.default_rng(seed)
    N = data.num_nodes
    edge_index_np = data.edge_index.numpy()
    E = edge_index_np.shape[1]

    if E == 0:
        pairs = edge_index_np[:, :0]
    else:
        # Vectorized: advance all N walks simultaneously via CSR arrays.
        # The per-node Python loop is O(N * num_steps) interpreter iterations
        # (~13M at 134k nodes — over an hour per epoch).
        order = np.argsort(edge_index_np[0], kind="stable")
        col = edge_index_np[1][order]
        rowcount = np.bincount(edge_index_np[0], minlength=N)
        rowptr = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(rowcount, out=rowptr[1:])

        start = np.arange(N, dtype=np.int64)
        current = start.copy()
        visited_src, visited_dst = [], []
        for _ in range(num_steps):
            restart = rng.random(N) < restart_prob
            cnt = rowcount[current]
            dead = cnt == 0
            reset = restart | dead
            current = np.where(reset, start, current)
            # Step the walks that didn't reset this iteration
            step = ~reset
            if step.any():
                cur_s = current[step]
                cnt_s = rowcount[cur_s]
                offset = rng.integers(0, cnt_s)          # cnt_s > 0 guaranteed
                nxt = col[rowptr[cur_s] + offset]
                visited_src.append(cur_s)
                visited_dst.append(nxt)
                current[step] = nxt

        if visited_src:
            all_pairs = np.stack([np.concatenate(visited_src), np.concatenate(visited_dst)])
            pairs = np.unique(all_pairs, axis=1)          # dedupe (src, dst)
        else:
            pairs = edge_index_np[:, :0]

    # Edge features for kept edges (use mean of original as proxy for new/kept edges)
    if data.edge_attr is not None:
        mean_attr = data.edge_attr.mean(dim=0, keepdim=True).expand(pairs.shape[1], -1)
    else:
        mean_attr = None

    aug = Data(
        x=data.x,
        edge_index=torch.tensor(pairs, dtype=torch.long),
        edge_attr=mean_attr,
        y=data.y,
        num_nodes=N,
    )
    return aug
