#!/usr/bin/env python3
"""
CLIP-IQA / CLIP-IQA+ baseline evaluation for SIQA-S.

- Zero-shot mode  (--checkpoint not set): uses the original CLIP-IQA weights.
- Fine-tuned mode (--checkpoint path):    uses the SIQA-trained CLIP-IQA+ checkpoint.

Depends on the mmedit library (MMEditing ≤ 1.x). Install via:
    pip install openmim && mim install mmcv-full==1.5.0
    pip install -e eval/baselines/CLIP-IQA/

Usage
-----
    # Zero-shot
    python eval/baselines/CLIP-IQA/clip_iqa_eval.py \\
        --config  eval/baselines/CLIP-IQA/configs/clipiqa_siqa_test.py \\
        --input_json  data/bench_SIQA-S.json \\
        --output_json outputs/SIQA-S_clip_iqa.json \\
        --image_root  data/images/

    # Fine-tuned (CLIP-IQA+)
    python eval/baselines/CLIP-IQA/clip_iqa_eval.py \\
        --config     eval/baselines/CLIP-IQA/configs/clipiqa_siqa_test.py \\
        --checkpoint model/clip_iqa_plus.pth \\
        --input_json  data/bench_SIQA-S.json \\
        --output_json outputs/SIQA-S_clip_iqa_plus.json \\
        --image_root  data/images/
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    from mmedit.apis import init_model, restoration_inference
except ImportError:
    raise ImportError(
        "mmedit not found. Install via:\n"
        "  pip install openmim && mim install mmcv-full==1.5.0\n"
        "  pip install -e eval/baselines/CLIP-IQA/"
    )


# ─────────────────────────────────────────────────────────────────────────────

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
    model_tag = "CLIP-IQA+" if args.checkpoint else "CLIP-IQA"
    print("=" * 55)
    print(f"  {model_tag} Baseline Evaluation (SIQA-S)")
    print("=" * 55)

    device = torch.device("cuda", args.device) if torch.cuda.is_available() else torch.device("cpu")
    model  = init_model(args.config, args.checkpoint, device=device)
    print(f"✅ {model_tag} model loaded")

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} items")

    perception_scores = []
    knowledge_scores  = []

    for item in tqdm(data, desc=f"{model_tag} scoring"):
        rel      = item.get("image_path", "")
        img_path = os.path.join(args.image_root, rel.strip())
        if not os.path.exists(img_path):
            perception_scores.append(None)
            knowledge_scores.append(None)
            continue
        try:
            _, attrs = restoration_inference(model, img_path, return_attributes=True)
            attrs = attrs.float().detach().cpu().numpy()[0]   # shape [2]
            # attrs[0] → subjective/perception, attrs[1] → objective/knowledge
            # CLIP-IQA outputs are in [0,1]; scale to [1,5] for SIQA
            sp = round(float(attrs[0]) * 5.0, 4)
            sk = round(float(attrs[1]) * 5.0, 4)
        except Exception as e:
            print(f"  [WARN] inference failed for {img_path}: {e}")
            sp = sk = None
        perception_scores.append(sp)
        knowledge_scores.append(sk)

    for item, sp, sk in zip(data, perception_scores, knowledge_scores):
        item.setdefault("precision", {})[model_tag] = {
            "perception": sp,
            "knowledge":  sk,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to {args.output_json}")

    stats = compute_correlations(data, perception_scores, knowledge_scores)
    if stats:
        print(f"\n📊 {model_tag} Results (n={stats['n_samples']}):")
        print(f"  Perception  SRCC/PLCC: {stats['Perception_SRCC']:.4f} / {stats['Perception_PLCC']:.4f}")
        print(f"  Knowledge   SRCC/PLCC: {stats['Knowledge_SRCC']:.4f}  / {stats['Knowledge_PLCC']:.4f}")
        print(f"  Overall     SRCC/PLCC: {stats['Overall_SRCC']:.4f}   / {stats['Overall_PLCC']:.4f}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-IQA / CLIP-IQA+ evaluation for SIQA-S")
    parser.add_argument("--config",      required=True,
                        help="mmedit test config (e.g. configs/clipiqa_siqa_test.py)")
    parser.add_argument("--checkpoint",  default=None,
                        help="Checkpoint path for fine-tuned CLIP-IQA+ (omit for zero-shot)")
    parser.add_argument("--input_json",  default="data/bench_SIQA-S.json")
    parser.add_argument("--output_json", default="outputs/SIQA-S_clip_iqa.json")
    parser.add_argument("--image_root",  default="data/images/")
    parser.add_argument("--device",      type=int, default=0,
                        help="CUDA device id (integer)")
    main(parser.parse_args())
