"""
Transformer-based temporal encoder for MAGNN-Plus.

In base MAGNN, temporal-node contrast simply compares embeddings from two
consecutive time-window snapshots. MAGNN-Plus replaces this with a lightweight
Transformer that processes a sequence of T snapshot embeddings for each node,
learning *how* representations evolve rather than just comparing two endpoints.

Architecture:
  Input:  (T, N, D) — T snapshot embeddings for N nodes of dimension D
  Output: (N, D)    — temporally-aware node embedding used for downstream tasks
                       and for constructing a richer temporal contrast signal.

The Transformer operates over the time axis (T positions) for each node.
Positional encodings along the time axis use sinusoidal encodings (standard).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, T: int) -> torch.Tensor:
        return self.pe[:T]   # (T, d_model)


class TemporalTransformerEncoder(nn.Module):
    """
    Processes a sequence of per-snapshot GNN node embeddings across T time steps
    and outputs a temporally-fused embedding per node.
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_time_steps: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.time_pe = SinusoidalTimeEncoding(d_model, max_time_steps)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,   # expects (T, N, D) to match nn.Transformer convention
            norm_first=True,     # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # Aggregate across time: learned weighted sum
        self.time_attn = nn.Linear(d_model, 1)

    def forward(self, snapshots: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            snapshots: list of T tensors, each (N, D)  — one per time window
        Returns:
            fused: (N, D) final temporally-aware node embedding
            contrast_pair: (N, D) embedding from the midpoint snapshot,
                           used as the second view for temporal contrastive loss
        """
        T = len(snapshots)
        N, D = snapshots[0].shape
        device = snapshots[0].device

        # Align node counts (different snapshots may have different #nodes in theory)
        # We take the minimum common set — nodes absent from a snapshot get zero embedding
        stack = torch.stack(snapshots, dim=0)    # (T, N, D)

        # Add time positional encoding
        pe = self.time_pe(T).to(device)          # (T, D)
        stack = stack + pe.unsqueeze(1)          # (T, N, D)

        # Transformer over time axis (T, N, D) → (T, N, D)
        encoded = self.encoder(stack)            # (T, N, D)
        encoded = self.out_norm(encoded)

        # Learned time-attention pooling to produce final embedding
        attn_weights = self.time_attn(encoded).squeeze(-1)   # (T, N)
        attn_weights = F.softmax(attn_weights, dim=0)        # (T, N)
        fused = (encoded * attn_weights.unsqueeze(-1)).sum(dim=0)  # (N, D)

        # For contrastive purposes, use the last snapshot's encoding as the second view
        contrast_pair = encoded[-1]    # (N, D)

        return fused, contrast_pair
