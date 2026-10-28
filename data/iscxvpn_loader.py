"""
ISCXVPN2016 Dataset Loader.

Actual data location:
  ISCXVPN2016/Scenario A1-ARFF/Scenario A1-ARFF/TimeBasedFeatures-Dataset-*s-VPN.csv
  ISCXVPN2016/Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-*s-VPN.csv
  ISCXVPN2016/Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-*s-NO-VPN.csv
  ISCXVPN2016/Scenario B-ARFF/Scenario B-ARFF/TimeBasedFeatures-Dataset-*.csv

Format: Time-aggregated flow statistics (NOT per-flow; NO IP addresses)
Columns: duration,total_fiat,total_biat,min_fiat,min_biat,max_fiat,max_biat,
         mean_fiat,mean_biat,flowPktsPerSecond,flowBytesPerSecond,
         min_flowiat,max_flowiat,mean_flowiat,std_flowiat,
         min_active,mean_active,max_active,std_active,
         min_idle,mean_idle,max_idle,std_idle,label

Label: determined by FILENAME (not the label column which may be sub-categories):
  Files with "-VPN" but NOT "-NO-VPN" in name → label=1 (VPN/malicious)
  Files with "-NO-VPN" in name OR Scenario B AllInOne → label from label column

Graph construction strategy (no IPs → flow-level graph):
  Each row = one flow node carrying 15 rich time-based features; edges are
  k-NN in feature space (labels never touched — the earlier host-graph hack
  derived dst_ip AND dst_port from the label, leaking it into the topology).

Relative path from magnn_improved/: ../ISCXVPN2016/
"""

from __future__ import annotations

import glob
import re
import numpy as np
import pandas as pd
from pathlib import Path

from .base_dataset import BaseTrafficDataset

ISCXVPN_FEATURE_COLS = [
    "duration", "total_fiat", "total_biat",
    "min_fiat", "min_biat", "max_fiat", "max_biat", "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond",
    "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle",
]


def _label_from_filename(filename: str) -> int | None:
    """
    Determine label from filename.
    Returns 1 for VPN, 0 for NO-VPN, None if ambiguous (use label column).
    """
    name = filename.lower()
    if "-no-vpn" in name or "_no_vpn" in name or "novpn" in name:
        return 0
    if "-vpn" in name or "_vpn" in name:
        return 1
    return None   # ambiguous — use label column (Scenario B AllInOne)


def _label_from_column(label_series: pd.Series) -> pd.Series:
    """
    Parse the 'label' column in Scenario B (multi-class: VPN sub-categories vs non-VPN).
    VPN sub-categories (VOIP-VPN, P2P-VPN, etc.) → 1
    Non-VPN categories → 0
    """
    vpn_keywords = ("vpn", "tunnel")
    return label_series.astype(str).apply(
        lambda x: 1 if any(kw in x.lower() for kw in vpn_keywords) else 0
    ).astype(np.int32)


class ISCXVPNDataset(BaseTrafficDataset):
    name = "iscxvpn"
    graph_mode = "flow"       # no real IPs exist — k-NN feature-similarity graph
    chain_by_host = False
    # Exactly 15 features (must match host-mode node dim for the shared encoder)
    flow_feature_cols = [
        "duration", "total_fiat", "total_biat", "mean_fiat", "mean_biat",
        "flowPktsPerSecond", "flowBytesPerSecond",
        "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
        "mean_active", "std_active", "mean_idle", "std_idle",
    ]

    def load_raw(self) -> pd.DataFrame:
        # Find all TimeBasedFeatures CSV files recursively
        pattern = str(self.data_root / "**" / "TimeBasedFeatures-*.csv")
        csv_files = sorted(glob.glob(pattern, recursive=True))

        if not csv_files:
            raise FileNotFoundError(
                f"No TimeBasedFeatures-*.csv files found under {self.data_root}.\n"
                f"Expected: ISCXVPN2016/Scenario A*-ARFF/Scenario A*-ARFF/TimeBasedFeatures-*.csv"
            )

        print(f"[ISCXVPN] Found {len(csv_files)} CSV files")
        dfs = []
        for f in sorted(csv_files):
            fname = Path(f).name
            file_label = _label_from_filename(fname)
            # Scenario B / AllInOne files duplicate Scenario A traffic with a
            # different labeling scheme — mixing them poisons ~30% of labels
            # (a plain RF drops from 96% to 68% accuracy). Use only files with
            # unambiguous filename labels.
            if file_label is None:
                print(f"  {fname}: skipped (mixed/duplicated Scenario B)")
                continue
            try:
                df = pd.read_csv(f, low_memory=False)
                df["_file_label"] = file_label
                df["_source_file"] = fname
                dfs.append(df)
                label_str = "VPN(1)" if file_label == 1 else "NO-VPN(0)"
                print(f"  {fname}: {len(df):,} rows → {label_str}")
            except Exception as e:
                print(f"  Warning: {f}: {e}")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"[ISCXVPN] Total rows: {len(combined):,}")
        return combined

    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        df = df.copy()

        # Determine label
        file_label_col = "_file_label" if "_file_label" in df.columns else None
        if file_label_col and df[file_label_col].notna().all():
            df["label"] = df[file_label_col].astype(np.int32)
        elif "label" in df.columns:
            df["label"] = _label_from_column(df["label"])
        else:
            df["label"] = 0

        # Normalize feature columns (they may have slightly different case)
        feat_rename = {}
        for col in df.columns:
            lower = col.lower().replace(" ", "_")
            for feat in ISCXVPN_FEATURE_COLS:
                if feat.lower() == lower:
                    feat_rename[col] = feat
        df = df.rename(columns=feat_rename)

        # Keep only feature columns that exist
        existing_feats = [c for c in ISCXVPN_FEATURE_COLS if c in df.columns]
        self._flow_feature_cols = existing_feats

        # The same capture appears at multiple aggregation windows across
        # scenario folders — drop exact feature-level duplicates.
        before = len(df)
        df = df.drop_duplicates(subset=existing_feats).reset_index(drop=True)
        if len(df) < before:
            print(f"[ISCXVPN] Deduped {before:,} -> {len(df):,} rows")

        for col in existing_feats:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[existing_feats] = df[existing_feats].replace([np.inf, -np.inf], np.nan)
        for col in existing_feats:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)

        # No real endpoints exist in this dataset. Placeholders only — the
        # flow-level graph never reads them, and nothing is derived from the
        # label (the old scheme leaked it via dst_ip and dst_port).
        n = len(df)
        idx = np.arange(n, dtype=np.int64)
        df["src_ip"] = "0.0.0.0"
        df["dst_ip"] = "0.0.0.1"

        # Timestamp: row order
        df["timestamp"] = idx.astype(np.float64)

        # Canonical flow columns (label-independent)
        df["duration"] = df.get("duration", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["pkt_count"] = df.get("flowPktsPerSecond", pd.Series(1.0, index=df.index)).fillna(1.0)
        df["byte_count"] = df.get("flowBytesPerSecond", pd.Series(64.0, index=df.index)).fillna(64.0)
        df["src_port"] = 0.0
        df["dst_port"] = 0.0
        df["protocol"] = 6.0    # TCP
        df["tcp_flags"] = 0.0

        # Ensure all 15 flow features exist (missing ones → 0)
        for col in self.flow_feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        print(f"[ISCXVPN] After canonicalize: {len(df):,} rows | "
              f"VPN={df['label'].sum():,} ({df['label'].mean()*100:.1f}%) | "
              f"features={len(existing_feats)}")

        keep = ["src_ip", "dst_ip", "label", "timestamp",
                "duration", "pkt_count", "byte_count",
                "protocol", "src_port", "dst_port", "tcp_flags"]
        keep += [c for c in self.flow_feature_cols if c not in keep]
        return df[keep]
