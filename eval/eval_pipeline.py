"""
SIQA Evaluation Pipeline
Computes official SIQA-U and SIQA-S metrics from prediction files.
"""

from scipy.stats import spearmanr, pearsonr
import argparse
import json
import os


def evaluate_SIQA_U(predicted_data, model_name):
    correct = {"yes/no": 0, "what": 0, "how": 0}
    total   = {"yes/no": 0, "what": 0, "how": 0}

    for item in predicted_data:
        q_type = item.get("type", "")
        if q_type not in total:
            continue
        gt   = item["answer"].strip().upper()
        pred = item.get(model_name, "").strip().upper()

        total[q_type] += 1
        if gt == pred:
            correct[q_type] += 1

    acc = {t: (correct[t] / total[t] if total[t] > 0 else 0.0) for t in total}

    score_u = (
        0.2 * acc.get("yes/no", 0)
        + 0.3 * acc.get("what", 0)
        + 0.5 * acc.get("how", 0)
    )
    return {
        "ACC_yes/no":    acc.get("yes/no", 0),
        "ACC_what":      acc.get("what", 0),
        "ACC_how":       acc.get("how", 0),
        "SIQA_U_Score":  score_u,
    }


def evaluate_SIQA_S(predicted_data, model_name):
    gt_perception   = []
    pred_perception = []
    gt_knowledge    = []
    pred_knowledge  = []

    for item in predicted_data:
        gt_p = item.get("perception_rating")
        gt_k = item.get("knowledge_rating")

        pred_dict = item.get(model_name, {})
        pred_p    = pred_dict.get("perception")
        pred_k    = pred_dict.get("knowledge")

        if all(v is not None for v in [gt_p, gt_k, pred_p, pred_k]):
            gt_perception.append(gt_p)
            pred_perception.append(pred_p)
            gt_knowledge.append(gt_k)
            pred_knowledge.append(pred_k)

    if len(gt_perception) == 0:
        return {"SIQA_S_Score": 0.0, "Perceptual_Score": 0.0, "Factual_Score": 0.0}

    srcc_p, _ = spearmanr(gt_perception, pred_perception)
    plcc_p, _ = pearsonr(gt_perception, pred_perception)
    score_p   = max((srcc_p + plcc_p) / 2, 0) * 100

    srcc_k, _ = spearmanr(gt_knowledge, pred_knowledge)
    plcc_k, _ = pearsonr(gt_knowledge, pred_knowledge)
    score_k   = max((srcc_k + plcc_k) / 2, 0) * 100

    return {
        "Perceptual_SRCC":  float(srcc_p),
        "Perceptual_PLCC":  float(plcc_p),
        "Perceptual_Score": score_p,
        "Knowledge_SRCC":   float(srcc_k),
        "Knowledge_PLCC":   float(plcc_k),
        "Factual_Score":    score_k,
        "SIQA_S_Score":     (score_p + score_k) / 2,
    }


def main(args):
    model_name = args.model_name
    os.makedirs(args.output, exist_ok=True)

    results = {}

    # ── SIQA-U ──────────────────────────────────────────────────────────────
    if args.SIQA_U:
        u_pred_path = os.path.join(args.input, "SIQA-U.json")
        if not os.path.exists(u_pred_path):
            print(f"[WARN] SIQA-U prediction file not found: {u_pred_path}")
        else:
            with open(u_pred_path, encoding="utf-8") as f:
                predict_u = json.load(f)
            u_metrics = evaluate_SIQA_U(predict_u, model_name)
            results["SIQA_U"] = u_metrics
            print("\n📊 SIQA-U Results:")
            for k, v in u_metrics.items():
                print(f"  {k}: {v:.4f}")

    # ── SIQA-S ──────────────────────────────────────────────────────────────
    if args.SIQA_S:
        s_pred_path = os.path.join(args.input, "SIQA-S.json")
        if not os.path.exists(s_pred_path):
            print(f"[WARN] SIQA-S prediction file not found: {s_pred_path}")
        else:
            with open(s_pred_path, encoding="utf-8") as f:
                predict_s = json.load(f)
            s_metrics = evaluate_SIQA_S(predict_s, model_name)
            results["SIQA_S"] = s_metrics
            print("\n📊 SIQA-S Results:")
            print(f"  Perceptual SRCC/PLCC: {s_metrics['Perceptual_SRCC']:.4f} / {s_metrics['Perceptual_PLCC']:.4f}")
            print(f"  Knowledge  SRCC/PLCC: {s_metrics['Knowledge_SRCC']:.4f}  / {s_metrics['Knowledge_PLCC']:.4f}")
            print(f"  Perceptual Score:     {s_metrics['Perceptual_Score']:.2f}")
            print(f"  Factual Score:        {s_metrics['Factual_Score']:.2f}")
            print(f"  SIQA-S Final:         {s_metrics['SIQA_S_Score']:.2f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_file = os.path.join(args.output, "results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Final evaluation results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SIQA metrics from prediction files.")
    parser.add_argument("--input",      type=str, default="outputs/",        help="Directory containing SIQA-U.json and SIQA-S.json")
    parser.add_argument("--output",     type=str, default="results/",        help="Directory to write results.json")
    parser.add_argument("--model_name", type=str, default="The_Best_IQA",   help="Model name key used during prediction")
    parser.add_argument("--SIQA_U",     action="store_true",                 help="Evaluate SIQA-U")
    parser.add_argument("--SIQA_S",     action="store_true",                 help="Evaluate SIQA-S")
    args = parser.parse_args()
    main(args)
