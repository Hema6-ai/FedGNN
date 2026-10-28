"""
Abstract base class for all four dataset loaders.

Concrete loaders must implement:
  - load_raw() → pd.DataFrame  (raw CSV read with dataset-specific column names)
  - canonicalize() → pd.DataFrame  (rename to standard schema defined in graph_construction.py)

The base class handles: normalization, splitting, graph construction, and
temporal snapshot creation. All loaders thus share the same post-processing pipeline.
"""

from __future__ import annotations

import abc
import os
import pickle
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import RobustScaler

from .preprocessing import (
    stratified_split, normalize_features, fill_and_clip, safe_cast_float
)
from utils.graph_construction import build_host_graph, build_flow_graph, temporal_graph_snapshots

# Canonical column names (must be produced by canonicalize())
CANONICAL_COLS = [
    "src_ip", "dst_ip", "label",
    "timestamp", "duration", "pkt_count", "byte_count",
    "protocol", "src_port", "dst_port", "tcp_flags",
]
FEATURE_COLS = ["duration", "pkt_count", "byte_count", "src_port", "dst_port", "tcp_flags"]


class BaseTrafficDataset(abc.ABC):
    """
    Base class providing a standard preprocessing + graph-construction pipeline.

    Usage:
        ds = CTU13Dataset(data_root="path/to/data")
        train_graph, val_graph, test_graph = ds.get_graphs()
        train_snaps = ds.get_snapshots("train", num_snapshots=5)
    """

    name: str = "base"
    graph_mode: str = "host"          # "host" (IP nodes) | "flow" (flow nodes)
    flow_feature_cols: list = []      # 15 rich per-flow features (flow mode only)
    chain_by_host: bool = False       # temporal same-host edges (needs real IPs)

    def __init__(
        self,
        data_root: str,
        num_snapshots: int = 5,
        sample_size: Optional[int] = None,  # downsample for quick dev runs
        seed: int = 42,
        cache: bool = True,
    ):
        self.data_root = Path(data_root)
        self.num_snapshots = num_snapshots
        self.sample_size = sample_size
        self.seed = seed
        self.cache = cache
        self._scaler: Optional[RobustScaler] = None

        self._train_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None

        cache_path = self.data_root / f"{self.name}_processed.pkl"
        # Also check sibling presampled/ directory (used when raw data is not available)
        presampled_path = Path(__file__).parent.parent.parent / "presampled" / f"{self.name}_processed.pkl"
        if cache and cache_path.exists():
            self._load_cache(cache_path)
        elif cache and presampled_path.exists():
            print(f"[{self.name}] Loading from presampled cache: {presampled_path}")
            self._load_cache(presampled_path)
        else:
            self._build()
            if cache:
                self._save_cache(cache_path)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Read raw CSVs from self.data_root and return a single DataFrame."""

    @abc.abstractmethod
    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename / derive columns to match CANONICAL_COLS.
        Must produce at minimum: src_ip, dst_ip, label (binary 0/1).
        """

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _build(self) -> None:
        df = self.load_raw()
        df = self.canonicalize(df)

        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.seed).reset_index(drop=True)

        norm_cols = FEATURE_COLS + [c for c in self.flow_feature_cols if c not in FEATURE_COLS]
        df = safe_cast_float(df, norm_cols)
        df = fill_and_clip(df, norm_cols)

        train, val, test = stratified_split(df, label_col="label", seed=self.seed)

        # Fit scaler on training data only
        train, self._scaler = normalize_features(train, norm_cols, scaler=None)
        val, _ = normalize_features(val, norm_cols, scaler=self._scaler)
        test, _ = normalize_features(test, norm_cols, scaler=self._scaler)

        self._train_df = train
        self._val_df = val
        self._test_df = test

    def _save_cache(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "train": self._train_df,
                "val": self._val_df,
                "test": self._test_df,
                "scaler": self._scaler,
            }, f)

    def _load_cache(self, path: Path) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._train_df = d["train"]
        self._val_df = d["val"]
        self._test_df = d["test"]
        self._scaler = d["scaler"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_graph(self, df: pd.DataFrame) -> Data:
        if self.graph_mode == "flow":
            return build_flow_graph(df, self.flow_feature_cols, chain_by_host=self.chain_by_host)
        return build_host_graph(df)

    def get_graphs(self) -> Tuple[Data, Data, Data]:
        """Returns (train_graph, val_graph, test_graph) as PyG Data objects."""
        return (
            self._build_graph(self._train_df),
            self._build_graph(self._val_df),
            self._build_graph(self._test_df),
        )

    def get_snapshots(self, split: str = "train") -> List[Data]:
        """Returns a list of T graph snapshots for temporal contrastive learning."""
        df_map = {"train": self._train_df, "val": self._val_df, "test": self._test_df}
        df = df_map[split]
        return temporal_graph_snapshots(
            df, num_snapshots=self.num_snapshots,
            graph_mode=self.graph_mode,
            flow_feature_cols=self.flow_feature_cols,
            chain_by_host=self.chain_by_host,
        )

    @property
    def feature_dim(self) -> int:
        """Node feature dimension after graph construction."""
        if self.graph_mode == "flow":
            return len(self.flow_feature_cols)
        from utils.graph_construction import _NODE_FEATURE_COLS
        return len(_NODE_FEATURE_COLS)

    @property
    def train_df(self) -> pd.DataFrame:
        return self._train_df

    @property
    def val_df(self) -> pd.DataFrame:
        return self._val_df

    @property
    def test_df(self) -> pd.DataFrame:
        return self._test_df

    def __repr__(self) -> str:
        n_train = len(self._train_df) if self._train_df is not None else 0
        n_val = len(self._val_df) if self._val_df is not None else 0
        n_test = len(self._test_df) if self._test_df is not None else 0
        return (
            f"{self.__class__.__name__}("
            f"train={n_train}, val={n_val}, test={n_test})"
        )
