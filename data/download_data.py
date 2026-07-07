#!/usr/bin/env python3
"""
Download SIQA benchmark data from HuggingFace.

Downloads:
  - bench_SIQA-U.json   (SIQA-U VQA benchmark)
  - bench_SIQA-S.json   (SIQA-S quality scoring benchmark)
  - images/             (all benchmark images)

Usage
-----
    python data/download_data.py --output_dir data/

Requires: pip install huggingface_hub
"""

import argparse
import json
import os

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    raise ImportError("Install huggingface_hub: pip install huggingface_hub")


DATASET_REPO  = "SIQA/TrainSet"       # HuggingFace dataset repo
BENCH_U_FILE  = "bench_SIQA-U.json"
BENCH_S_FILE  = "bench_SIQA-S.json"


def download_data(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading SIQA benchmark from huggingface.co/datasets/{DATASET_REPO} ...")
    local_dir = snapshot_download(
        repo_id   = DATASET_REPO,
        repo_type = "dataset",
        local_dir = output_dir,
    )
    print(f"✅ Dataset downloaded to: {local_dir}")

    # Quick sanity check
    for fname in [BENCH_U_FILE, BENCH_S_FILE]:
        path = os.path.join(output_dir, fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"  {fname}: {len(data)} items")
        else:
            print(f"  [WARN] {fname} not found — check the repo structure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SIQA benchmark data")
    parser.add_argument("--output_dir", default="data/",
                        help="Local directory to save downloaded data")
    args = parser.parse_args()
    download_data(args.output_dir)
