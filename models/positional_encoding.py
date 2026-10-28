"""
Learnable positional encodings for MAGNN-Plus.

Two complementary encodings are concatenated:
  1. Degree encoding  — captures local connectivity (in-degree + out-degree)
  2. Random-walk structural encoding (RWSE) — approximates eigenvector centrality
     via k-step landing probabilities of a random walk from each node to itself.

These encodings are projected to `hidden_dim` and added to the initial node features,
giving the model positional awareness without relying on absolute node IDs.

Design note: The paper mentions "degree + centrality features" but does not specify
the exact encoding. We follow the GPS (General, Powerful, Scalable) graph Transformer
convention: degree PE + RWSE as a well-validated lightweight alternative to Laplacian PE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, to_dense_adj
from typing import Optional


class DegreeEncoding(nn.Module):
    def __init__(self, max_degree: int = 512, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(max_degree + 1, hidden_dim, padding_idx=0)
        self.max_degree = max_degree

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Returns (N, hidden_dim) degree embedding."""
        d_out = degree(edge_index[0], num_nodes=num_nodes).long().clamp(max=self.max_degree)
        d_in = degree(edge_index[1], num_nodes=num_nodes).long().clamp(max=self.max_degree)
        return self.embedding(d_out) + self.embedding(d_in)   # (N, H)


class RandomWalkStructuralEncoding(nn.Module):
    """
    k-step random walk landing probabilities as structural positional encodings.
    pe_i = [RW^1[i,i], RW^2[i,i], ..., RW^k[i,i]] projected to hidden_dim.

    Graphs up to `dense_threshold` nodes use the exact dense computation.
    Larger graphs use a Monte-Carlo estimate (`num_walks` sampled walks per
    node): the dense N x N transition matrix is O(N^2) memory and O(k*N^3)
    compute — at the 30k-135k nodes of the full traffic graphs that is
    tens of GB and hours per forward pass, so sampling is the only viable
    path. Landing probabilities are estimated to within ~1/sqrt(num_walks).
    """
    def __init__(self, walk_length: int = 16, hidden_dim: int = 64,
                 dense_threshold: int = 2000, num_walks: int = 32):
        super().__init__()
        self.walk_length = walk_length
        self.dense_threshold = dense_threshold
        self.num_walks = num_walks
        self.proj = nn.Linear(walk_length, hidden_dim)

    @torch.no_grad()
    def _exact_dense(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)   # (N, N)
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        RW = A / deg         # row-stochastic
        del A

        rw_pe = []
        P = RW.clone()
        for _ in range(self.walk_length):
            rw_pe.append(P.diagonal())   # (N,) landing prob at self
            P = P @ RW
        return torch.stack(rw_pe, dim=-1)    # (N, walk_length)

    @torch.no_grad()
    def _monte_carlo(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        device = edge_index.device
        landing = torch.zeros(num_nodes, self.walk_length, device=device)
        if edge_index.size(1) == 0:
            return landing

        # CSR neighbor structure
        src, dst = edge_index[0], edge_index[1]
        perm = torch.argsort(src)
        col = dst[perm]
        rowcount = torch.bincount(src, minlength=num_nodes)
        rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        rowptr[1:] = rowcount.cumsum(0)

        # Deterministic per-graph seed so repeated forwards on the same graph
        # (e.g. every validation pass) produce identical encodings.
        gen = torch.Generator(device=device)
        gen.manual_seed(int(num_nodes) * 1000003 + int(edge_index.size(1)))

        W = self.num_walks
        start = torch.arange(num_nodes, device=device).repeat_interleave(W)
        cur = start.clone()
        alive = rowcount[start] > 0
        for t in range(self.walk_length):
            cnt = rowcount[cur]
            alive = alive & (cnt > 0)
            r = torch.rand(cur.size(0), device=device, generator=gen)
            offset = torch.minimum((r * cnt.clamp(min=1)).long(), (cnt - 1).clamp(min=0))
            nxt = col[(rowptr[cur] + offset).clamp(max=col.size(0) - 1)]
            cur = torch.where(alive, nxt, cur)
            returned = alive & (cur == start)
            if returned.any():
                landing[:, t].index_add_(
                    0, start[returned],
                    torch.ones(int(returned.sum()), device=device),
                )
        landing /= float(W)
        return landing

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if num_nodes <= self.dense_threshold:
            rw_pe = self._exact_dense(edge_index, num_nodes)
        else:
            rw_pe = self._monte_carlo(edge_index, num_nodes)
        return self.proj(rw_pe)              # (N, hidden_dim)


class LearnablePositionalEncoding(nn.Module):
    """
    Combined learnable PE: degree + RWSE, projected to `out_dim`.
    Added to input node features before the first GNN layer.
    """
    def __init__(
        self,
        out_dim: int = 128,
        max_degree: int = 512,
        walk_length: int = 16,
        pe_dim: int = 64,
    ):
        super().__init__()
        self.degree_enc = DegreeEncoding(max_degree=max_degree, hidden_dim=pe_dim)
        self.rwse = RandomWalkStructuralEncoding(walk_length=walk_length, hidden_dim=pe_dim)
        self.proj = nn.Linear(2 * pe_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Returns (N, out_dim) positional encoding to be added to node features."""
        import os
        if os.environ.get("FEDMAGNN_NO_PE"):   # probe switch: PE ablation
            p = next(self.proj.parameters())
            return torch.zeros(num_nodes, self.proj.out_features, device=p.device, dtype=p.dtype)
        d_enc = self.degree_enc(edge_index, num_nodes)       # (N, pe_dim)
        rw_enc = self.rwse(edge_index, num_nodes)             # (N, pe_dim)
        combined = torch.cat([d_enc, rw_enc], dim=-1)         # (N, 2*pe_dim)
        return self.norm(self.proj(combined))                 # (N, out_dim)
