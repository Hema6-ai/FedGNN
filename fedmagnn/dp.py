"""
Client-level differential privacy for federated updates.

Mechanism (DP-FedAvg, McMahan et al. 2018 style):
  1. Each client computes its model delta  Δ_k = w_k_local - w_global.
  2. The delta is flat-clipped to L2 norm C:   Δ_k ← Δ_k · min(1, C / ||Δ_k||).
  3. The server adds Gaussian noise N(0, (σC)²/K) to the *average* of the K
     clipped deltas (equivalently, noise N(0, (σC)²) to the sum).

Privacy accounting: Rényi DP of the Gaussian mechanism.
  Per round, at order α:   ε_RDP(α) = α / (2σ²)
  Over R rounds (adaptive composition): R·α / (2σ²)
  Convert to (ε, δ)-DP:    ε = min_α [ R·α/(2σ²) + log(1/δ)/(α-1) ]

All 4 clients participate every round (cross-silo, no subsampling), so no
subsampling amplification is applied — the accounting is conservative/exact
for full participation.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

_ALPHA_GRID = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def rdp_epsilon(noise_multiplier: float, rounds: int, delta: float = 1e-5) -> float:
    """(ε, δ)-DP guarantee after `rounds` Gaussian mechanism invocations."""
    if noise_multiplier <= 0:
        return float("inf")
    best = float("inf")
    for alpha in _ALPHA_GRID:
        if alpha <= 1:
            continue
        rdp = rounds * alpha / (2.0 * noise_multiplier ** 2)
        eps = rdp + math.log(1.0 / delta) / (alpha - 1.0)
        best = min(best, eps)
    return best


class GaussianDPMechanism:
    """Clip per-client deltas; noise the aggregate."""

    def __init__(self, clip_norm: float = 1.0, noise_multiplier: float = 0.0,
                 delta: float = 1e-5, seed: int = 42):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.rounds_used = 0
        self._gen = torch.Generator().manual_seed(seed)

    @property
    def enabled(self) -> bool:
        return self.noise_multiplier > 0

    def clip_delta(self, delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flat-clip a state-dict delta to L2 norm <= clip_norm."""
        total_sq = sum(float(v.detach().float().pow(2).sum()) for v in delta.values())
        norm = math.sqrt(total_sq)
        scale = min(1.0, self.clip_norm / (norm + 1e-12))
        return {k: v * scale for k, v in delta.items()}

    def noise_aggregate(
        self, avg_delta: Dict[str, torch.Tensor], num_clients: int
    ) -> Dict[str, torch.Tensor]:
        """Add N(0, (σC/K)²) per-coordinate noise to the averaged delta."""
        self.rounds_used += 1
        if not self.enabled:
            return avg_delta
        std = self.noise_multiplier * self.clip_norm / num_clients
        noised = {}
        for k, v in avg_delta.items():
            noise = torch.normal(
                mean=0.0, std=std, size=v.shape, generator=self._gen
            ).to(v.dtype)
            noised[k] = v + noise
        return noised

    def epsilon(self, extra_rounds: int = 0) -> float:
        """Cumulative (ε, δ)-DP spent so far (optionally projected forward)."""
        if not self.enabled:
            return float("inf")
        return rdp_epsilon(self.noise_multiplier, self.rounds_used + extra_rounds, self.delta)


def dp_noised_mean_embedding(
    embeddings: torch.Tensor,
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    DP release of a mean embedding (used for class prototypes).

    Each row is clipped to L2 norm <= clip_norm, so the mean over n rows has
    per-row sensitivity clip_norm / n; Gaussian noise with std
    σ · clip_norm / n gives one Gaussian-mechanism release.
    """
    n = embeddings.shape[0]
    if n == 0:
        return torch.zeros(embeddings.shape[1], device=embeddings.device)
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)
    scale = (clip_norm / norms).clamp(max=1.0)
    clipped = embeddings * scale
    mean = clipped.mean(dim=0)
    if noise_multiplier > 0:
        std = noise_multiplier * clip_norm / n
        noise = torch.normal(mean=0.0, std=std, size=mean.shape, generator=generator)
        mean = mean + noise.to(mean.device)
    return mean
