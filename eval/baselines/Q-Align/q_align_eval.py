#!/usr/bin/env python3
"""
Q-Align (One-Align) baseline evaluation for SIQA-S.

Q-Align scores images on two axes:
  - "quality"     → mapped to Perception (subjective)
  - "aesthetics"  → mapped to Knowledge  (objective)

Scores are already in [1,5] and used directly for correlation.

Usage
-----
    # Download the model first:
    #   git clone https://huggingface.co/q-future/one-align model/one-align/
    #   cp eval/baselines/Q-Align/modeling_mplug_owl2.py  model/one-align/

    python eval/baselines/Q-Align/q_align_eval.py \\
        --model_path  model/one-align/ \\
        --input_json  data/bench_SIQA-S.json \\
        --output_json outputs/SIQA-S_q_align.json \\
        --image_root  data/images/ \\
        --batch_size  16
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM


# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_batch(model, pil_images, task):
    scores = model.score(images=pil_images, task_=task, input_="image")
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().float().tolist()
    else:
        scores = [float(scores)]
    if len(pil_images) > 1 and len(scores) == 1:
        scores = scores * len(pil_images)
    return scores


def compute_correlations(data, perception_scores, knowledge_scores):
    preds_p, preds_k, gt_p, gt_k = [], [], [], []
    for item, sp, sk in zip(data, perception_scores, knowledge_scores):
        p = item.get("perception_rating")
        k = item.get("knowledge_rating")
        if sp is not None and sk is not None and p is not None and k is not None:
            preds_p.append(sp); gt_p.append(float(p))
            preds_k.append(sk); gt_k.append(float(k))
    if len(preds_p) < 2:
        return {}
    srcc_p, _ = spearmanr(gt_p, preds_p)
    plcc_p, _ = pearsonr(gt_p, preds_p)
    srcc_k, _ = spearmanr(gt_k, preds_k)
    plcc_k, _ = pearsonr(gt_k, preds_k)
    return {
        "Perception_SRCC": float(srcc_p), "Perception_PLCC": float(plcc_p),
        "Knowledge_SRCC":  float(srcc_k), "Knowledge_PLCC":  float(plcc_k),
        "Overall_SRCC":    (abs(srcc_p) + abs(srcc_k)) / 2,
        "Overall_PLCC":    (abs(plcc_p) + abs(plcc_k)) / 2,
        "n_samples": len(preds_p),
    }


def main(args):
    print("=" * 55)
    print("  Q-Align Baseline Evaluation (SIQA-S)")
    print("=" * 55)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    print(f"✅ Q-Align model loaded from {args.model_path}")

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} items")

    # Flatten to batches: each item → 2 entries (quality, aesthetics)
    img_paths  = []
    task_names = []
    item_idxs  = []
    for idx, item in enumerate(data):
        rel = item.get("image_path", "")
        fp  = os.path.join(args.image_root, rel.strip())
        if not os.path.exists(fp):
            print(f"  [WARN] Not found: {fp}")
            continue
        img_paths.extend([fp, fp])
        task_names.extend(["quality", "aesthetics"])
        item_idxs.extend([idx, idx])

    all_scores = []
    pbar = tqdm(total=len(img_paths), desc="Q-Align scoring")
    for i in range(0, len(img_paths), args.batch_size):
        batch_paths = img_paths[i: i + args.batch_size]
        batch_tasks = task_names[i: i + args.batch_size]

        # Group by task (Q-Align takes one task per call)
        task2imgs  = {}
        task2local = {}
        pil_images = []
        valid_mask = []
        for j, (p, task) in enumerate(zip(batch_paths, batch_tasks)):
            try:
                img = Image.open(p).convert("RGB")
                pil_images.append(img)
                valid_mask.append(True)
            except (UnidentifiedImageError, OSError):
                pil_images.append(None)
                valid_mask.append(False)

        batch_scores = [float("nan")] * len(batch_paths)
        for j, (img, task, valid) in enumerate(zip(pil_images, batch_tasks, valid_mask)):
            if not valid:
                continue
            task2imgs.setdefault(task, []).append(img)
            task2local.setdefault(task, []).append(j)

        for task, imgs in task2imgs.items():
            try:
                scores = score_batch(model, imgs, task)
                for local_j, score in zip(task2local[task], scores):
                    batch_scores[local_j] = score
            except Exception as e:
                print(f"  [WARN] Scoring failed for task '{task}': {e}")

        all_scores.extend(batch_scores)
        pbar.update(len(batch_paths))
    pbar.close()

    # Assign back (pairs of quality, aesthetics)
    per_scores = [None] * len(data)
    kno_scores = [None] * len(data)
    for j in range(0, len(all_scores), 2):
        if j + 1 >= len(all_scores):
            break
        idx   = item_idxs[j]
        s_per = all_scores[j]       # quality → perception
        s_kno = all_scores[j + 1]   # aesthetics → knowledge
        per_scores[idx] = None if math.isnan(s_per) else round(float(s_per), 4)
        kno_scores[idx] = None if math.isnan(s_kno) else round(float(s_kno), 4)

    for item, sp, sk in zip(data, per_scores, kno_scores):
        item.setdefault("precision", {})["Q-Align"] = {
            "perception": sp,
            "knowledge":  sk,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to {args.output_json}")

    stats = compute_correlations(data, per_scores, kno_scores)
    if stats:
        print(f"\n📊 Q-Align Results (n={stats['n_samples']}):")
        print(f"  Perception  SRCC/PLCC: {stats['Perception_SRCC']:.4f} / {stats['Perception_PLCC']:.4f}")
        print(f"  Knowledge   SRCC/PLCC: {stats['Knowledge_SRCC']:.4f}  / {stats['Knowledge_PLCC']:.4f}")
        print(f"  Overall     SRCC/PLCC: {stats['Overall_SRCC']:.4f}   / {stats['Overall_PLCC']:.4f}")

    return stats


if __name__ == "__main__":
    # Patch for older torch versions
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False

    parser = argparse.ArgumentParser(description="Q-Align evaluation for SIQA-S")
    parser.add_argument("--model_path",  default="model/one-align/")
    parser.add_argument("--input_json",  default="data/bench_SIQA-S.json")
    parser.add_argument("--output_json", default="outputs/SIQA-S_q_align.json")
    parser.add_argument("--image_root",  default="data/images/")
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--device",      default="auto")
    main(parser.parse_args())
