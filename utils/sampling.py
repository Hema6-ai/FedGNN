"""
Dirichlet neighbor sampling as described in the MAGNN paper (sampling size=30).

Each node's neighbors are sampled with weights drawn from a Dirichlet distribution,
giving a probabilistic, diverse neighborhood that prevents over-smoothing from always
picking the same top-k neighbors.
"""

import numpy as np
from typing import List


def dirichlet_sample_neighbors(
    neighbors: np.ndarray,
    size: int = 30,
    alpha: float = 1.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Sample up to `size` neighbors with Dirichlet-distributed weights.

    Args:
        neighbors: array of neighbor node indices
        size: target sample size (paper default: 30)
        alpha: Dirichlet concentration parameter (1.0 = uniform prior)
        rng: numpy random generator for reproducibility
    Returns:
        sampled neighbor indices (may be smaller than size if fewer neighbors exist)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(neighbors)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n <= size:
        return neighbors.copy()

    weights = rng.dirichlet(np.full(n, alpha))
    sampled_idx = rng.choice(n, size=size, replace=False, p=weights)
    return neighbors[sampled_idx]


def batch_dirichlet_sample(
    edge_index: np.ndarray,
    num_nodes: int,
    size: int = 30,
    alpha: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply Dirichlet sampling to all nodes and return a new edge_index.

    Args:
        edge_index: (2, E) array of [src, dst] edges
        num_nodes: total number of nodes
        size: max neighbors per node
        alpha: Dirichlet concentration
        seed: RNG seed
    Returns:
        filtered edge_index (2, E') with at most `size` neighbors per node
    """
    rng = np.random.default_rng(seed)
    src, dst = edge_index[0], edge_index[1]
    E = edge_index.shape[1]
    if E == 0:
        return edge_index[:, :0]

    # Group edges by source via CSR instead of scanning all E edges per node
    # (the per-node scan is O(N*E) — hours at 134k nodes). Nodes with <= size
    # neighbors keep everything; only over-degree nodes are subsampled, in
    # ascending node order so the RNG stream matches the original per-node loop.
    order = np.argsort(src, kind="stable")
    dst_sorted = dst[order]
    rowcount = np.bincount(src, minlength=num_nodes)
    rowptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(rowcount, out=rowptr[1:])

    keep_mask = np.ones(E, dtype=bool)
    for node in np.where(rowcount > size)[0]:
        s, e = rowptr[node], rowptr[node + 1]
        neighbors = dst_sorted[s:e]
        n = e - s
        weights = rng.dirichlet(np.full(n, alpha))
        sampled_idx = rng.choice(n, size=size, replace=False, p=weights)
        keep_mask[s:e] = np.isin(neighbors, neighbors[sampled_idx])

    keep = np.sort(order[keep_mask])
    if keep.size == 0:
        return edge_index[:, :0]
    return edge_index[:, keep]
