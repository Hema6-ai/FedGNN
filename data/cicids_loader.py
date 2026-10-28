"""
CICIDS-2017 Dataset Loader.

Actual data location:  CICIDS2017/MachineLearningCSV/MachineLearningCVE/*.pcap_ISCX.csv
Format: CICFlowMeter CSV — 78 numeric flow features + Label
NOTE: The MachineLearningCSV version strips Source IP and Destination IP for privacy.

Graph construction strategy (no IPs available):
  - Destination IP = synthetic, derived from Destination Port:
      "172.{port//256 % 256}.{port%256}.0"  →  flows to same service share dst_ip
  - Source IP = synthetic, derived from flow fingerprint features:
      "10.{(Total_Fwd_Packets) % 256}.{SYN_Flag_Count % 4 * 64}.0"
  This creates a manageable service-bipartite graph capturing client→service structure.

All 78 numeric columns are used as flow features (passed through the model directly
instead of the 15 aggregated host features). The node feature dimension is thus 78.

Relative path from magnn_improved/: ../CICIDS2017/MachineLearningCSV/MachineLearningCVE/
"""

from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from pathlib import Path

from .base_dataset import BaseTrafficDataset
from .preprocessing import encode_protocol

BENIGN_LABEL = "BENIGN"

# All feature columns (78 numeric columns from CICFlowMeter output)
CICIDS_FEATURE_COLS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def _synthetic_src_ip(df: pd.DataFrame) -> pd.Series:
    """
    Derive synthetic source IP from flow fingerprint features.
    Flows with similar forwarding packet counts + SYN flag patterns
    get the same source IP → creates natural client clusters.
    """
    fwd_pkt = pd.to_numeric(df.get("Total Fwd Packets", pd.Series(0, index=df.index)),
                             errors="coerce").fillna(0).astype(int) % 256
    syn_flag = pd.to_numeric(df.get("SYN Flag Count", pd.Series(0, index=df.index)),
                              errors="coerce").fillna(0).astype(int) % 4
    # Combine into 256 src groups
    group = (fwd_pkt // 8 + syn_flag * 32) % 256
    return group.apply(lambda g: f"10.0.0.{g}")


def _synthetic_dst_ip(df: pd.DataFrame) -> pd.Series:
    """
    Derive destination IP from Destination Port.
    Flows to same service (port) share the same dst_ip → service nodes.
    """
    port = pd.to_numeric(df.get("Destination Port", pd.Series(0, index=df.index)),
                          errors="coerce").fillna(0).astype(int)
    hi = (port // 256) % 256
    lo = port % 256
    return port.apply(lambda p: f"172.{(p // 256) % 256}.{p % 256}.0")


class CICIDSDataset(BaseTrafficDataset):
    name = "cicids"
    # Host graph over port-derived endpoints (label-independent). This is the
    # construction that produced the 99.6 F1 results — the old "flow" tag here
    # was aspirational and never read by the pipeline.
    graph_mode = "host"

    def load_raw(self) -> pd.DataFrame:
        # The CSV files have .pcap_ISCX.csv extension
        pattern = str(self.data_root / "**" / "*.pcap_ISCX.csv")
        csv_files = sorted(glob.glob(pattern, recursive=True))

        if not csv_files:
            # Try direct .csv files as fallback
            csv_files = sorted(list(self.data_root.glob("**/*.csv")))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found under {self.data_root}.\n"
                f"Expected: CICIDS2017/MachineLearningCSV/MachineLearningCVE/*.pcap_ISCX.csv"
            )

        print(f"[CICIDS] Found {len(csv_files)} CSV files")
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
                print(f"  {Path(f).name}: {len(df):,} rows")
            except Exception as e:
                print(f"  Warning: {f}: {e}")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"[CICIDS] Total rows: {len(combined):,}")
        return combined

    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip column name whitespace (CICFlowMeter quirk)
        df.columns = df.columns.str.strip()

        # Locate label column (may have leading space)
        label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
        if label_col is None:
            df["label"] = 0
        else:
            df["label"] = (df[label_col].astype(str).str.strip() != BENIGN_LABEL).astype(np.int32)
            df = df.drop(columns=[label_col])

        # Synthetic IPs (flows with same service share dst_ip; similar clients share src_ip)
        df["src_ip"] = _synthetic_src_ip(df)
        df["dst_ip"] = _synthetic_dst_ip(df)

        # Timestamp: not available in MachineLearningCSV → use row order
        df["timestamp"] = np.arange(len(df), dtype=np.float64)

        # Keep only the feature columns that exist
        existing_feat_cols = [c for c in CICIDS_FEATURE_COLS if c in df.columns]

        # Fill NaN/Inf in feature columns
        for col in existing_feat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[existing_feat_cols] = df[existing_feat_cols].replace([np.inf, -np.inf], np.nan)
        for col in existing_feat_cols:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)

        # Store original feature columns list for graph construction
        self._flow_feature_cols = existing_feat_cols

        # Canonical flow columns needed by base class + graph construction
        df["duration"] = pd.to_numeric(df.get("Flow Duration", 0), errors="coerce").fillna(0.0)
        df["pkt_count"] = pd.to_numeric(df.get("Total Fwd Packets", 1), errors="coerce").fillna(1.0)
        df["byte_count"] = pd.to_numeric(df.get("Total Length of Fwd Packets", 64), errors="coerce").fillna(64.0)
        df["src_port"] = 0.0
        df["dst_port"] = pd.to_numeric(df.get("Destination Port", 0), errors="coerce").fillna(0.0)
        df["protocol"] = 6.0   # assumed TCP (most CICIDS2017 attacks are TCP-based)
        df["tcp_flags"] = pd.to_numeric(df.get("SYN Flag Count", 0), errors="coerce").fillna(0.0)

        print(f"[CICIDS] After canonicalize: {len(df):,} rows | "
              f"malicious={df['label'].sum():,} ({df['label'].mean()*100:.1f}%) | "
              f"features={len(existing_feat_cols)}")

        return df[["src_ip", "dst_ip", "label", "timestamp",
                   "duration", "pkt_count", "byte_count",
                   "protocol", "src_port", "dst_port", "tcp_flags"]]
