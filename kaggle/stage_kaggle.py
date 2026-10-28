"""
Stage the FedMAGNN code + presampled data as a Kaggle dataset upload.

Usage (from FedMAGNN/):
    python kaggle/stage_kaggle.py
    kaggle datasets create -p kaggle/dataset_staging          # first upload
    kaggle datasets version -p kaggle/dataset_staging -m "update"  # later updates
    kaggle kernels push -p kaggle                             # push the notebook
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # FedMAGNN/
STAGE = ROOT / "kaggle" / "dataset_staging"
PRESAMPLED = ROOT.parent / "MAGNN" / "presampled"

CODE_ITEMS = [
    "fedmagnn", "models", "losses", "utils", "data",
    "configs", "train.py", "run_federated.py", "requirements.txt",
]


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for item in CODE_ITEMS:
        src = ROOT / item
        dst = STAGE / item
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)

    # Presampled per-silo data caches
    dst_pre = STAGE / "presampled"
    dst_pre.mkdir()
    for pkl in sorted(PRESAMPLED.glob("*_processed.pkl")):
        shutil.copy2(pkl, dst_pre / pkl.name)
        print(f"  + presampled/{pkl.name} ({pkl.stat().st_size / 1e6:.1f} MB)")

    # Kaggle dataset metadata
    (STAGE / "dataset-metadata.json").write_text(
        '{\n'
        '  "title": "FedMAGNN Code",\n'
        '  "id": "<kaggle-username>/fedmagnn-code",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n'
        '}\n'
    )
    print(f"\nStaged to {STAGE}")


if __name__ == "__main__":
    main()
