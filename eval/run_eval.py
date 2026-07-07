"""
One-click evaluation script for SIQA benchmark.

Runs inference on SIQA-U (VQA understanding) and SIQA-S (image quality scoring)
using any HuggingFace-compatible vision-language model, then computes official
SIQA metrics.

Usage
-----
    python eval/run_eval.py \\
        --model  /path/to/model-or-hf-id \\
        --model_name  My_Model \\
        --data_dir  data/ \\
        --image_root  data/images/ \\
        --output_dir  outputs/ \\
        --result_dir  results/ \\
        --SIQA_U --SIQA_S
"""

import argparse
import json
import os
import sys

# Allow importing BaseModel from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from BaseModel import ScoreModel, UnderstandModel
from eval_pipeline import evaluate_SIQA_U, evaluate_SIQA_S


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_siqa_u(model_path, data_dir, image_root, model_name, output_dir):
    """Run SIQA-U inference and save intermediate predictions."""
    bench_path = os.path.join(data_dir, "bench_SIQA-U.json")
    if not os.path.exists(bench_path):
        raise FileNotFoundError(f"SIQA-U benchmark not found: {bench_path}")

    with open(bench_path, encoding="utf-8") as f:
        dataset = json.load(f)

    understander = UnderstandModel.from_pretrained(model_path)
    predictions  = []

    print(f"\n🔍 Running SIQA-U inference on {len(dataset)} items ...")
    for item in dataset:
        image_path = os.path.join(image_root, item["image_path"])
        answer     = understander.predict_answer(image_path, item["question"], item["option"])
        new_item   = dict(item)
        new_item[model_name] = answer.strip().upper()
        predictions.append(new_item)

    out_path = os.path.join(output_dir, "SIQA-U.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"✅ SIQA-U predictions saved to {out_path}")
    return predictions


def run_siqa_s(model_path, data_dir, image_root, model_name, output_dir):
    """Run SIQA-S inference and save intermediate predictions."""
    bench_path = os.path.join(data_dir, "bench_SIQA-S.json")
    if not os.path.exists(bench_path):
        raise FileNotFoundError(f"SIQA-S benchmark not found: {bench_path}")

    with open(bench_path, encoding="utf-8") as f:
        dataset = json.load(f)

    scorer      = ScoreModel.from_pretrained(model_path)
    predictions = []

    print(f"\n🔍 Running SIQA-S inference on {len(dataset)} items ...")
    for item in dataset:
        image_path = os.path.join(image_root, item["image_path"])
        perception = scorer.predict_score(image_path, "perception")
        knowledge  = scorer.predict_score(image_path, "knowledge")
        new_item   = dict(item)
        new_item[model_name] = {
            "perception": perception,
            "knowledge":  knowledge,
        }
        predictions.append(new_item)

    out_path = os.path.join(output_dir, "SIQA-S.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"✅ SIQA-S predictions saved to {out_path}")
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    results     = {}
    predict_u   = None
    predict_s   = None

    # ── Inference ────────────────────────────────────────────────────────────
    if args.SIQA_U:
        predict_u = run_siqa_u(
            args.model, args.data_dir, args.image_root,
            args.model_name, args.output_dir
        )

    if args.SIQA_S:
        predict_s = run_siqa_s(
            args.model, args.data_dir, args.image_root,
            args.model_name, args.output_dir
        )

    # ── Evaluation ───────────────────────────────────────────────────────────
    if args.SIQA_U and predict_u:
        u_metrics = evaluate_SIQA_U(predict_u, args.model_name)
        results["SIQA_U"] = u_metrics
        print("\n📊 SIQA-U Results:")
        for k, v in u_metrics.items():
            print(f"  {k}: {v:.4f}")

    if args.SIQA_S and predict_s:
        s_metrics = evaluate_SIQA_S(predict_s, args.model_name)
        results["SIQA_S"] = s_metrics
        print("\n📊 SIQA-S Results:")
        print(f"  Perceptual SRCC/PLCC: {s_metrics['Perceptual_SRCC']:.4f} / {s_metrics['Perceptual_PLCC']:.4f}")
        print(f"  Knowledge  SRCC/PLCC: {s_metrics['Knowledge_SRCC']:.4f}  / {s_metrics['Knowledge_PLCC']:.4f}")
        print(f"  Perceptual Score:     {s_metrics['Perceptual_Score']:.2f}")
        print(f"  Factual Score:        {s_metrics['Factual_Score']:.2f}")
        print(f"  SIQA-S Final:         {s_metrics['SIQA_S_Score']:.2f}")

    # ── Save final results ───────────────────────────────────────────────────
    out_file = os.path.join(args.result_dir, "results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Final results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-click SIQA evaluation (SIQA-U & SIQA-S).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default="model/InternVL3.5-4B-hf/full/train_set",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--model_name", default="My_Model",
                        help="Short name tag written into prediction JSON and used for metrics")
    parser.add_argument("--data_dir",   default="data/",
                        help="Directory containing bench_SIQA-U.json and bench_SIQA-S.json")
    parser.add_argument("--image_root", default="data/images/",
                        help="Root directory of benchmark images")
    parser.add_argument("--output_dir", default="outputs/",
                        help="Directory for intermediate prediction JSON files")
    parser.add_argument("--result_dir", default="results/",
                        help="Directory for final results.json")
    parser.add_argument("--SIQA_U",     action="store_true",
                        help="Evaluate SIQA-U (VQA understanding)")
    parser.add_argument("--SIQA_S",     action="store_true",
                        help="Evaluate SIQA-S (quality scoring)")
    args = parser.parse_args()

    if not args.SIQA_U and not args.SIQA_S:
        parser.error("Specify at least one of --SIQA_U / --SIQA_S")

    main(args)
