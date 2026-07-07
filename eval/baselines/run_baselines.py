#!/usr/bin/env python3
"""
One-click baseline evaluation for SIQA-S.

Runs one or more traditional IQA baselines and prints a summary table.

Usage
-----
    # Run all baselines (each requires its model to be downloaded first):
    python eval/baselines/run_baselines.py \\
        --baselines all \\
        --input_json  data/bench_SIQA-S.json \\
        --image_root  data/images/ \\
        --output_dir  outputs/

    # Run specific baselines:
    python eval/baselines/run_baselines.py \\
        --baselines niqe q_align \\
        --input_json  data/bench_SIQA-S.json \\
        --image_root  data/images/ \\
        --output_dir  outputs/

    # CLIP-IQA+ requires checkpoint path:
    python eval/baselines/run_baselines.py \\
        --baselines clip_iqa_plus \\
        --input_json   data/bench_SIQA-S.json \\
        --image_root   data/images/ \\
        --output_dir   outputs/ \\
        --clip_iqa_config      eval/baselines/CLIP-IQA/configs/clipiqa_siqa_test.py \\
        --clip_iqa_checkpoint  model/clip_iqa_plus.pth \\
        --hyperiqa_checkpoint  model/hyperiqa/hyperiqa_siqa.pth \\
        --q_align_model        model/one-align/
"""

import argparse
import importlib.util
import json
import os
import sys


BASELINES = ["niqe", "q_align", "clip_iqa", "clip_iqa_plus", "hyperiqa"]
BASELINE_DIR = os.path.dirname(__file__)


def _load_module(path, name):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_namespace(**kwargs):
    import argparse
    ns = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def run_niqe(args):
    mod  = _load_module(os.path.join(BASELINE_DIR, "NIQE", "niqe_eval.py"), "niqe_eval")
    ns   = make_namespace(
        input_json  = args.input_json,
        output_json = os.path.join(args.output_dir, "SIQA-S_niqe.json"),
        image_root  = args.image_root,
    )
    return mod.main(ns)


def run_q_align(args):
    mod = _load_module(os.path.join(BASELINE_DIR, "Q-Align", "q_align_eval.py"), "q_align_eval")
    ns  = make_namespace(
        model_path  = args.q_align_model,
        input_json  = args.input_json,
        output_json = os.path.join(args.output_dir, "SIQA-S_q_align.json"),
        image_root  = args.image_root,
        batch_size  = args.batch_size,
        device      = "auto",
    )
    return mod.main(ns)


def run_clip_iqa(args, plus=False):
    mod = _load_module(os.path.join(BASELINE_DIR, "CLIP-IQA", "clip_iqa_eval.py"), "clip_iqa_eval")
    tag = "clip_iqa_plus" if plus else "clip_iqa"
    ns  = make_namespace(
        config      = args.clip_iqa_config,
        checkpoint  = args.clip_iqa_checkpoint if plus else None,
        input_json  = args.input_json,
        output_json = os.path.join(args.output_dir, f"SIQA-S_{tag}.json"),
        image_root  = args.image_root,
        device      = 0,
    )
    return mod.main(ns)


def run_hyperiqa(args):
    mod = _load_module(os.path.join(BASELINE_DIR, "HyperIQA", "hyperiqa_eval.py"), "hyperiqa_eval")
    ns  = make_namespace(
        checkpoint  = args.hyperiqa_checkpoint,
        input_json  = args.input_json,
        output_json = os.path.join(args.output_dir, "SIQA-S_hyperiqa.json"),
        image_root  = args.image_root,
    )
    return mod.main(ns)


def print_table(all_stats):
    header  = f"{'Model':<18} {'Per.SRCC':>9} {'Per.PLCC':>9} {'Kno.SRCC':>9} {'Kno.PLCC':>9} {'Ovr.SRCC':>9} {'Ovr.PLCC':>9}"
    divider = "-" * len(header)
    print(f"\n{'='*70}")
    print("  SIQA-S Baseline Summary")
    print(f"{'='*70}")
    print(header)
    print(divider)
    for name, s in all_stats.items():
        if not s:
            print(f"{name:<18}  (no results)")
            continue
        print(
            f"{name:<18}"
            f" {s.get('Perception_SRCC', 0):>9.4f}"
            f" {s.get('Perception_PLCC', 0):>9.4f}"
            f" {s.get('Knowledge_SRCC',  0):>9.4f}"
            f" {s.get('Knowledge_PLCC',  0):>9.4f}"
            f" {s.get('Overall_SRCC',    0):>9.4f}"
            f" {s.get('Overall_PLCC',    0):>9.4f}"
        )
    print(divider)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    targets   = BASELINES if "all" in args.baselines else args.baselines
    all_stats = {}

    for b in targets:
        print(f"\n{'='*55}")
        print(f"  Running: {b}")
        print(f"{'='*55}")
        try:
            if b == "niqe":
                all_stats["NIQE"] = run_niqe(args)
            elif b == "q_align":
                all_stats["Q-Align"] = run_q_align(args)
            elif b == "clip_iqa":
                all_stats["CLIP-IQA"] = run_clip_iqa(args, plus=False)
            elif b == "clip_iqa_plus":
                all_stats["CLIP-IQA+"] = run_clip_iqa(args, plus=True)
            elif b == "hyperiqa":
                all_stats["HyperIQA"] = run_hyperiqa(args)
        except Exception as e:
            print(f"  [ERROR] {b} failed: {e}")
            all_stats[b] = {}

    print_table(all_stats)

    summary_path = os.path.join(args.output_dir, "baselines_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-click SIQA-S baseline evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baselines",  nargs="+", default=["all"],
                        choices=BASELINES + ["all"],
                        help="Which baselines to run")
    parser.add_argument("--input_json",  default="data/bench_SIQA-S.json")
    parser.add_argument("--image_root",  default="data/images/")
    parser.add_argument("--output_dir",  default="outputs/")
    # Optional model paths
    parser.add_argument("--q_align_model",        default="model/one-align/")
    parser.add_argument("--clip_iqa_config",       default="eval/baselines/CLIP-IQA/configs/clipiqa_siqa_test.py")
    parser.add_argument("--clip_iqa_checkpoint",   default=None,
                        help="Required for clip_iqa_plus")
    parser.add_argument("--hyperiqa_checkpoint",   default="model/hyperiqa/hyperiqa_siqa.pth",
                        help="Required for hyperiqa")
    parser.add_argument("--batch_size", type=int,  default=8,
                        help="Batch size for Q-Align")
    args = parser.parse_args()
    main(args)
