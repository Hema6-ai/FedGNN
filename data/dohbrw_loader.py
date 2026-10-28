"""
CIRA-CIC-DoHBrw-2020 Dataset Loader.

Actual data location (after extracting CSVs.zip → Total_CSVs.zip):
  CIC-DoHBrw2020/CSVs/Total/l2-benign.csv    (Benign DoH traffic)
  CIC-DoHBrw2020/CSVs/Total/l2-malicious.csv (Malicious DoH traffic)

We use level-2 files only (benign vs malicious DoH) which match the paper's
binary classification task. Level-1 files (DoH vs non-DoH) are ignored.

Columns (real IPs available → host-centric graph):
  SourceIP, DestinationIP, SourcePort, DestinationPort, TimeStamp,
  Duration, FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate,
  PacketLengthVariance, PacketLengthStandardDeviation, PacketLengthMean,
  PacketLengthMedian, PacketLengthMode, PacketLengthSkewFromMedian,
  PacketLengthSkewFromMode, PacketLengthCoefficientofVariation,
  PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean,
  PacketTimeMedian, PacketTimeMode, PacketTimeSkewFromMedian,
  PacketTimeSkewFromMode, PacketTimeCoefficientofVariation,
  ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation, ResponseTimeTimeMean,
  ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian,
  ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation, Label

Relative path from magnn_improved/: ../CIC-DoHBrw2020/CSVs/Total/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .base_dataset import BaseTrafficDataset

MALICIOUS_FILE = "l2-malicious.csv"
BENIGN_FILE    = "l2-benign.csv"

DOH_FEATURE_COLS = [
    "duration", "flowbytessent", "flowsentrate", "flowbytesreceived", "flowreceivedrate",
    "packetlengthvariance", "packetlengthstandarddeviation", "packetlengthmean",
    "packetlengthmedian", "packetlengthmode",
    "packettimevariance", "packettimestandarddeviation", "packettimemean",
    "responsetimetimevariance", "responsetimetimestandarddeviation", "responsetimetimemean",
]


class DoHBrwDataset(BaseTrafficDataset):
    name = "dohbrw"
    # Only ~15 unique hosts exist — a host graph collapses to nothing.
    # Flow-level graph: k-NN feature edges + temporal chains per real host.
    graph_mode = "flow"
    chain_by_host = True
    # Exactly 15 features (must match host-mode node dim for the shared encoder)
    flow_feature_cols = [
        "duration", "flowbytessent", "flowsentrate", "flowbytesreceived",
        "flowreceivedrate", "packetlengthvariance", "packetlengthstandarddeviation",
        "packetlengthmean", "packetlengthmedian", "packetlengthmode",
        "packettimevariance", "packettimestandarddeviation", "packettimemean",
        "responsetimetimestandarddeviation", "responsetimetimemean",
    ]

    def load_raw(self) -> pd.DataFrame:
        malicious_path = self.data_root / MALICIOUS_FILE
        benign_path    = self.data_root / BENIGN_FILE

        dfs = []
        for path, file_label in [(malicious_path, 1), (benign_path, 0)]:
            if not path.exists():
                print(f"[DoHBrw] WARNING: {path} not found, skipping")
                continue
            try:
                df = pd.read_csv(path, low_memory=False)
                df["_file_label"] = file_label
                dfs.append(df)
                print(f"[DoHBrw] {path.name}: {len(df):,} rows → label={file_label}")
            except Exception as e:
                print(f"[DoHBrw] Warning: {path}: {e}")

        if not dfs:
            raise FileNotFoundError(
                f"Neither {MALICIOUS_FILE} nor {BENIGN_FILE} found in {self.data_root}.\n"
                f"Expected: CIC-DoHBrw2020/CSVs/Total/{MALICIOUS_FILE} and {BENIGN_FILE}\n"
                f"Run: Expand-Archive CIC-DoHBrw2020\\CSVs.zip → then extract Total_CSVs.zip"
            )

        combined = pd.concat(dfs, ignore_index=True)
        print(f"[DoHBrw] Total rows: {len(combined):,}")
        return combined

    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize column names
        df.columns = df.columns.str.strip()

        # Label from _file_label (overrides any label column in the CSV)
        df["label"] = df["_file_label"].astype(np.int32)

        # IP addresses (real, available in this dataset)
        df["src_ip"] = df.get("SourceIP", pd.Series("0.0.0.0", index=df.index)).fillna("0.0.0.0").astype(str)
        df["dst_ip"] = df.get("DestinationIP", pd.Series("0.0.0.1", index=df.index)).fillna("0.0.0.1").astype(str)

        # Timestamp
        if "TimeStamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
            df["timestamp"] = df["timestamp"].astype(np.int64) / 1e9
            df["timestamp"] = df["timestamp"].fillna(0.0)
        else:
            df["timestamp"] = np.arange(len(df), dtype=np.float64)

        # Core flow columns
        df["duration"] = pd.to_numeric(df.get("Duration", 0), errors="coerce").fillna(0.0)
        df["pkt_count"] = pd.to_numeric(df.get("FlowSentRate", 1), errors="coerce").fillna(1.0)
        df["byte_count"] = pd.to_numeric(df.get("FlowBytesSent", 64), errors="coerce").fillna(64.0)
        df["src_port"]  = pd.to_numeric(df.get("SourcePort", 0), errors="coerce").fillna(0.0)
        df["dst_port"]  = pd.to_numeric(df.get("DestinationPort", 443), errors="coerce").fillna(443.0)
        df["protocol"]  = 6.0   # DoH is always TCP/HTTPS (port 443)
        df["tcp_flags"] = 0.0

        # Rich per-flow features for the flow-level graph (lowercased names)
        rename = {}
        for col in df.columns:
            low = col.lower()
            if low in self.flow_feature_cols and col != low and low not in df.columns:
                rename[col] = low
        df = df.rename(columns=rename)
        for col in self.flow_feature_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        print(f"[DoHBrw] After canonicalize: {len(df):,} rows | "
              f"malicious={df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")

        keep = ["src_ip", "dst_ip", "label", "timestamp",
                "duration", "pkt_count", "byte_count",
                "protocol", "src_port", "dst_port", "tcp_flags"]
        keep += [c for c in self.flow_feature_cols if c not in keep]
        return df[keep]
