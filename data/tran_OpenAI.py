#!/usr/bin/env python3
"""
Convert SIQA benchmark data to OpenAI / LLaMA-Factory SFT format.

Produces two JSONL-style files:
  - <out_path>_SIQA-U.json   (VQA understanding, single-turn)
  - <out_path>_SIQA-S.json   (quality scoring, two turns: perception + knowledge)

Usage
-----
    python data/tran_OpenAI.py \\
        --SIQA_U_json  data/bench_SIQA-U.json \\
        --SIQA_S_json  data/bench_SIQA-S.json \\
        --root_path    data/images/ \\
        --out_path     data/sft/openai_bench
"""

import argparse
import json
import os


RATING_TO_WORD = {1: "Bad", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}
TERMS_STR      = "Bad, Poor, Fair, Good, Excellent"


def float_to_rating_word(score: float) -> str:
    score     = max(1.0, min(5.0, score))
    bin_width = 0.8
    rating    = int((score - 1.0) // bin_width) + 1
    return RATING_TO_WORD[min(5, max(1, rating))]


def tran_SIQA_U_OpenAI(item, root_path):
    for k in ("image_path", "question", "option"):
        if k not in item or item[k] is None:
            raise ValueError(f"Missing key '{k}' in item: {item}")

    # Support both "answer" and "annotation" field names
    answer = (item.get("annotation") or item.get("answer") or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid answer '{answer}' in item: {item}")

    user_text = (
        f"<image>Answer the following question based on the image.\n"
        f"Question: {item['question'].strip()}\n"
        f"Choices: {item['option'].strip()}\n"
        f"Respond with ONLY one uppercase letter: A, B, C, or D."
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in scientific image analysis. Your task is to answer "
                    "visual question answering (VQA) questions based on the given image. "
                    "Respond with ONLY a single uppercase letter: A, B, C, or D. "
                    "Do not include any explanations, punctuation, spaces, or additional characters."
                ),
            },
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": answer},
        ],
        "images": [os.path.join(root_path, item["image_path"].strip())],
    }


def tran_SIQA_S_OpenAI(item, root_path):
    for k in ("image_path", "perception_rating", "knowledge_rating"):
        if k not in item or item[k] is None:
            raise ValueError(f"Missing key '{k}' in item: {item}")

    img_full  = os.path.join(root_path, item["image_path"].strip())
    per_word  = float_to_rating_word(item["perception_rating"])
    kno_word  = float_to_rating_word(item["knowledge_rating"])

    sub_entry = {
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are an expert in scientific image analysis. Evaluate the given image "
                    f"on **Subjective Quality** only:\n\n"
                    f"- Consider technical quality (sharpness, lighting, legibility) and aesthetic "
                    f"quality (visual appeal, layout balance, information density).\n"
                    f"- Ignore scientific correctness.\n\n"
                    f"Use exactly one of these five terms: [{TERMS_STR}]. Respond ONLY as:\n\n"
                    f"Subjective: [Quality Word]\n"
                ),
            },
            {"role": "user",      "content": "How would you rate the subjective quality of this image?\n<image>"},
            {"role": "assistant", "content": f"Subjective: {per_word}"},
        ],
        "images": [img_full],
    }

    obj_entry = {
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are an expert in scientific image analysis. Evaluate the given image "
                    f"on **Objective Quality** only:\n\n"
                    f"- Assess scientific rigor: completeness (e.g., scale bars, axis labels, units), "
                    f"correctness of data, and avoidance of redundancy.\n"
                    f"- Ignore aesthetics or technical rendering.\n\n"
                    f"Use exactly one of these five terms: [{TERMS_STR}]. Respond ONLY as:\n\n"
                    f"Objective: [Quality Word]\n"
                ),
            },
            {"role": "user",      "content": "How would you rate the objective quality of this image?\n<image>"},
            {"role": "assistant", "content": f"Objective: {kno_word}"},
        ],
        "images": [img_full],
    }

    return sub_entry, obj_entry


def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)

    # ── SIQA-U ──────────────────────────────────────────────────────────────
    with open(args.SIQA_U_json, encoding="utf-8") as f:
        u_data = json.load(f)

    u_out, errors = [], 0
    for item in u_data:
        try:
            u_out.append(tran_SIQA_U_OpenAI(item, args.root_path))
        except ValueError as e:
            errors += 1
            print(f"  [WARN] Skipped SIQA-U item: {e}")

    u_path = args.out_path + "_SIQA-U.json"
    with open(u_path, "w", encoding="utf-8") as f:
        json.dump(u_out, f, ensure_ascii=False, indent=2)
    print(f"✅ SIQA-U SFT data: {len(u_out)} items (skipped {errors}) → {u_path}")

    # ── SIQA-S ──────────────────────────────────────────────────────────────
    with open(args.SIQA_S_json, encoding="utf-8") as f:
        s_data = json.load(f)

    s_out, errors = [], 0
    for item in s_data:
        try:
            sub, obj = tran_SIQA_S_OpenAI(item, args.root_path)
            s_out.append(sub)
            s_out.append(obj)
        except ValueError as e:
            errors += 1
            print(f"  [WARN] Skipped SIQA-S item: {e}")

    s_path = args.out_path + "_SIQA-S.json"
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(s_out, f, ensure_ascii=False, indent=2)
    print(f"✅ SIQA-S SFT data: {len(s_out)} items (skipped {errors}) → {s_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SIQA data to OpenAI SFT format")
    parser.add_argument("--SIQA_U_json", default="data/bench_SIQA-U.json")
    parser.add_argument("--SIQA_S_json", default="data/bench_SIQA-S.json")
    parser.add_argument("--root_path",   default="data/images/")
    parser.add_argument("--out_path",    default="data/sft/openai_bench")
    main(parser.parse_args())
