from .ctu13_loader import CTU13Dataset
from .iscxvpn_loader import ISCXVPNDataset
from .cicids_loader import CICIDSDataset
from .dohbrw_loader import DoHBrwDataset

DATASET_REGISTRY = {
    "ctu13": CTU13Dataset,
    "iscxvpn": ISCXVPNDataset,
    "cicids": CICIDSDataset,
    "dohbrw": DoHBrwDataset,
}

def get_dataset(name: str, data_root: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](data_root=data_root, **kwargs)
