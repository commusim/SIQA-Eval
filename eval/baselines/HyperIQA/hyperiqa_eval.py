#!/usr/bin/env python3
"""
HyperIQA baseline evaluation for SIQA-S (inference-only, trained weights).

HyperIQA was fine-tuned on SIQA training data; this script runs inference
on the benchmark split and reports SRCC/PLCC for Perception and Knowledge.

Usage
-----
    # Download weights from HuggingFace first:
    #   model/hyperiqa/hyperiqa_siqa.pth

    python eval/baselines/HyperIQA/hyperiqa_eval.py \\
        --checkpoint  model/hyperiqa/hyperiqa_siqa.pth \\
        --input_json  data/bench_SIQA-S.json \\
        --output_json outputs/SIQA-S_hyperiqa.json \\
        --image_root  data/images/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Allow importing models.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))
import models  # noqa: E402 (local import)


# ─────────────────────────────────────────────────────────────────────────────

TEST_PATCH_NUM  = 25
PATCH_SIZE      = 224

_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 512)),
    torchvision.transforms.CenterCrop(PATCH_SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])


def load_model(checkpoint_path, device):
    model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Support checkpoints saved as {"model": ...} or bare state_dict
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_single(model_hyper, img_path, device):
    """Return (perception_score, knowledge_score) averaged over TEST_PATCH_NUM crops."""
    img = Image.open(img_path).convert("RGB")
    preds_p, preds_k = [], []
    for _ in range(TEST_PATCH_NUM):
        tensor = _transforms(img).unsqueeze(0).to(device)
        paras  = model_hyper(tensor)
        model_target = models.TargetNet(paras).to(device)
        for p in model_target.parameters():
            p.requires_grad_(False)
        pred = model_target(paras["target_in_vec"])   # [1, 2]
        preds_p.append(pred[0, 0].item())
        preds_k.append(pred[0, 1].item())
    return float(np.mean(preds_p)), float(np.mean(preds_k))


def compute_correlations(data, perception_scores, knowledge_scores):
    pred_p, pred_k, gt_p, gt_k = [], [], [], []
    for item, sp, sk in zip(data, perception_scores, knowledge_scores):
        p = item.get("perception_rating")
        k = item.get("knowledge_rating")
        if sp is not None and sk is not None and p is not None and k is not None:
            pred_p.append(sp); gt_p.append(float(p))
            pred_k.append(sk); gt_k.append(float(k))
    if len(pred_p) < 2:
        return {}
    srcc_p, _ = spearmanr(gt_p, pred_p)
    plcc_p, _ = pearsonr(gt_p, pred_p)
    srcc_k, _ = spearmanr(gt_k, pred_k)
    plcc_k, _ = pearsonr(gt_k, pred_k)
    return {
        "Perception_SRCC": float(srcc_p), "Perception_PLCC": float(plcc_p),
        "Knowledge_SRCC":  float(srcc_k), "Knowledge_PLCC":  float(plcc_k),
        "Overall_SRCC":    (abs(srcc_p) + abs(srcc_k)) / 2,
        "Overall_PLCC":    (abs(plcc_p) + abs(plcc_k)) / 2,
        "n_samples": len(pred_p),
    }


def main(args):
    print("=" * 55)
    print("  HyperIQA Baseline Evaluation (SIQA-S)")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    print(f"✅ HyperIQA loaded from {args.checkpoint} on {device}")

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} items")

    perception_scores = []
    knowledge_scores  = []

    for item in tqdm(data, desc="HyperIQA scoring"):
        rel      = item.get("image_path", "")
        img_path = os.path.join(args.image_root, rel.strip())
        if not os.path.exists(img_path):
            perception_scores.append(None)
            knowledge_scores.append(None)
            continue
        try:
            sp, sk = predict_single(model, img_path, device)
        except Exception as e:
            print(f"  [WARN] failed on {img_path}: {e}")
            sp = sk = None
        perception_scores.append(round(sp, 4) if sp is not None else None)
        knowledge_scores.append(round(sk, 4) if sk is not None else None)

    for item, sp, sk in zip(data, perception_scores, knowledge_scores):
        item.setdefault("precision", {})["HyperIQA"] = {
            "perception": sp,
            "knowledge":  sk,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to {args.output_json}")

    stats = compute_correlations(data, perception_scores, knowledge_scores)
    if stats:
        print(f"\n📊 HyperIQA Results (n={stats['n_samples']}):")
        print(f"  Perception  SRCC/PLCC: {stats['Perception_SRCC']:.4f} / {stats['Perception_PLCC']:.4f}")
        print(f"  Knowledge   SRCC/PLCC: {stats['Knowledge_SRCC']:.4f}  / {stats['Knowledge_PLCC']:.4f}")
        print(f"  Overall     SRCC/PLCC: {stats['Overall_SRCC']:.4f}   / {stats['Overall_PLCC']:.4f}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperIQA evaluation for SIQA-S")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to trained HyperIQA checkpoint (.pth)")
    parser.add_argument("--input_json",  default="data/bench_SIQA-S.json")
    parser.add_argument("--output_json", default="outputs/SIQA-S_hyperiqa.json")
    parser.add_argument("--image_root",  default="data/images/")
    main(parser.parse_args())
