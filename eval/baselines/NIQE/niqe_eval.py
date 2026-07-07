#!/usr/bin/env python3
"""
NIQE baseline evaluation for SIQA-S.

NIQE is a no-reference image quality metric. Lower NIQE → better quality.
Scores are linearly remapped to [1, 5] (inverted) for SIQA-S correlation.

Usage
-----
    python eval/baselines/NIQE/niqe_eval.py \\
        --input_json  data/bench_SIQA-S.json \\
        --output_json outputs/SIQA-S_niqe.json \\
        --image_root  data/images/
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    import torch
    import pyiqa
except ImportError:
    raise ImportError("Install dependencies: pip install pyiqa torch torchvision")

MIN_DIM_SIZE = 96  # NIQE requires at least this edge length


# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_path):
    try:
        img = Image.open(img_path).convert("L")
        w, h = img.size
        if w < MIN_DIM_SIZE or h < MIN_DIM_SIZE:
            scale = max(MIN_DIM_SIZE / w, MIN_DIM_SIZE / h)
            img = img.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)
        return img
    except Exception:
        return None


def get_niqe_score(metric_model, img_path):
    try:
        if not os.path.exists(img_path):
            return float("nan")
        img_pil = preprocess_image(img_path)
        if img_pil is None:
            return float("nan")
        score = metric_model(img_pil)
        if isinstance(score, (torch.Tensor, np.ndarray)):
            score = float(score.item()) if hasattr(score, "item") else float(score)
        return float(score) if math.isfinite(float(score)) else float("nan")
    except Exception:
        return float("nan")


def map_to_1_5_inverted(raw_scores):
    """Invert and remap raw NIQE scores (lower=better) to [1,5] (higher=better)."""
    valid = [s for s in raw_scores if not math.isnan(s)]
    if len(valid) < 2:
        return [None] * len(raw_scores)
    lo, hi = min(valid), max(valid)
    rng = hi - lo
    mapped = []
    for s in raw_scores:
        if math.isnan(s):
            mapped.append(None)
        elif rng < 1e-6:
            mapped.append(3.0)
        else:
            m = 1.0 + 4.0 * (hi - s) / rng
            mapped.append(round(max(1.0, min(5.0, m)), 4))
    return mapped


def compute_correlations(data, mapped_scores):
    preds, gt_p, gt_k = [], [], []
    for item, score in zip(data, mapped_scores):
        p = item.get("perception_rating")
        k = item.get("knowledge_rating")
        if score is not None and p is not None and k is not None:
            preds.append(score)
            gt_p.append(float(p))
            gt_k.append(float(k))
    if len(preds) < 2:
        return {}
    y = np.array(preds)
    srcc_p, _ = spearmanr(gt_p, y)
    plcc_p, _ = pearsonr(gt_p, y)
    srcc_k, _ = spearmanr(gt_k, y)
    plcc_k, _ = pearsonr(gt_k, y)
    return {
        "Perception_SRCC": abs(srcc_p), "Perception_PLCC": abs(plcc_p),
        "Knowledge_SRCC":  abs(srcc_k), "Knowledge_PLCC":  abs(plcc_k),
        "Overall_SRCC":    (abs(srcc_p) + abs(srcc_k)) / 2,
        "Overall_PLCC":    (abs(plcc_p) + abs(plcc_k)) / 2,
        "n_samples": len(preds),
    }


def main(args):
    print("=" * 55)
    print("  NIQE Baseline Evaluation (SIQA-S)")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    niqe_metric = pyiqa.create_metric("niqe", device=device)
    print(f"✅ NIQE model loaded on {device}")

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} items from {args.input_json}")

    raw_scores = []
    for item in tqdm(data, desc="NIQE scoring"):
        img_rel  = item.get("image_path", "")
        img_path = os.path.join(args.image_root, img_rel.strip())
        raw_scores.append(get_niqe_score(niqe_metric, img_path))

    mapped_scores = map_to_1_5_inverted(raw_scores)

    # Embed scores into data items
    for item, raw, mapped in zip(data, raw_scores, mapped_scores):
        item.setdefault("precision", {})["NIQE"] = {
            "raw_score":        None if math.isnan(raw) else round(raw, 4),
            "score_mapped_1_5": mapped,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to {args.output_json}")

    stats = compute_correlations(data, mapped_scores)
    if stats:
        print(f"\n📊 NIQE Results (n={stats['n_samples']}):")
        print(f"  Perception  SRCC/PLCC: {stats['Perception_SRCC']:.4f} / {stats['Perception_PLCC']:.4f}")
        print(f"  Knowledge   SRCC/PLCC: {stats['Knowledge_SRCC']:.4f}  / {stats['Knowledge_PLCC']:.4f}")
        print(f"  Overall     SRCC/PLCC: {stats['Overall_SRCC']:.4f}   / {stats['Overall_PLCC']:.4f}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIQE evaluation for SIQA-S")
    parser.add_argument("--input_json",  default="data/bench_SIQA-S.json")
    parser.add_argument("--output_json", default="outputs/SIQA-S_niqe.json")
    parser.add_argument("--image_root",  default="data/images/")
    main(parser.parse_args())
