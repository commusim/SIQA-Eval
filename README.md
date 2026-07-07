<div align="center">

# SIQA: Scientific Image Quality Assessment

<p>
  <a href="https://arxiv.org/abs/2603.06700"><img src="https://img.shields.io/badge/arXiv-2603.06700-b31b1b.svg?logo=arxiv" alt="arXiv"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/SIQA/TrainSet"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-blue" alt="HF Dataset"></a>
  &nbsp;
  <a href="https://huggingface.co/commusim-hf/SIQA-Finetune"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-blue" alt="HF Model"></a>
  &nbsp;
  <a href="https://github.com/commusim/SIQA-Eval"><img src="https://img.shields.io/badge/GitHub-SIQA--Eval-black?logo=github" alt="GitHub"></a>
  &nbsp;
  <a href="https://siqa-competition.github.io/"><img src="https://img.shields.io/badge/Project-Page-green?logo=googlechrome" alt="Project Page"></a>
</p>

**Official evaluation toolkit for the Scientific Image Quality Assessment (SIQA) benchmark.**  
Supports MLLM inference and metrics for **SIQA-U** (visual understanding) and **SIQA-S** (quality scoring), along with five traditional IQA baselines.

</div>

---

## Overview

SIQA evaluates models on two complementary tasks:

| Task | Description | Metric |
|------|-------------|--------|
| **SIQA-U** | VQA understanding — yes-or-no, what, how questions about scientific images | Mean accuracy across three question types |
| **SIQA-S** | Quality scoring — predict perceptual & knowledge ratings on a 1–5 scale | SRCC / PLCC (Perception + Knowledge) |

### SIQA-S Scoring Baselines

| Model | Perception SRCC | Perception PLCC | Knowledge SRCC | Knowledge PLCC |
|-------|:-:|:-:|:-:|:-:|
| NIQE (zero-shot)       | 0.345 | 0.235 | 0.447 | 0.410 |
| Q-Align (zero-shot)    | 0.749 | 0.762 | 0.285 | 0.400 |
| CLIP-IQA (zero-shot)   | 0.496 | 0.520 | 0.362 | 0.435 |
| CLIP-IQA+ (trained)    | 0.724 | 0.676 | 0.862 | 0.801 |
| HyperIQA (trained)     | 0.773 | 0.783 | 0.897 | 0.895 |
| **InternVL3.5 (fine-tuned)** | **0.857** | **0.881** | **0.915** | **0.937** |

---

## Repository Structure

```
SIQA-Eval/
├── data/
│   ├── download_data.py        # Download benchmark from HuggingFace
│   └── tran_OpenAI.py          # Convert to OpenAI / LLaMA-Factory SFT format
├── model/
│   └── download_models.sh      # One-click model download (fine-tuned + baselines)
├── eval/
│   ├── BaseModel.py            # MLLM wrappers: ScoreModel, UnderstandModel
│   ├── eval_pipeline.py        # Metric computation (SIQA-U / SIQA-S)
│   ├── run_eval.py             # One-click MLLM inference + evaluation
│   └── baselines/
│       ├── run_baselines.py    # One-click baseline evaluation (all or single)
│       ├── NIQE/niqe_eval.py
│       ├── Q-Align/q_align_eval.py
│       ├── CLIP-IQA/clip_iqa_eval.py
│       └── HyperIQA/hyperiqa_eval.py
├── outputs/                    # Intermediate prediction JSON files
├── results/                    # Final evaluation results
├── requirements.txt
└── install.sh
```

---

## Quick Start

### 1. Install Dependencies

```bash
bash install.sh
# For CLIP-IQA/CLIP-IQA+ baselines (requires mmcv):
bash install.sh --with-clip-iqa
```

### 2. Download Benchmark Data

```bash
python data/download_data.py --output_dir data/
```

Expected layout after download:
```
data/
├── bench_SIQA-U.json
├── bench_SIQA-S.json
└── images/
    ├── fig1.png
    └── ...
```

### 3. Download Models

```bash
# All models (fine-tuned MLLM + all baselines)
bash model/download_models.sh

# Only the fine-tuned MLLM (skip baselines)
bash model/download_models.sh --skip-baselines

# Only baselines (skip fine-tuned MLLM)
bash model/download_models.sh --skip-finetune
```

---

## MLLM Evaluation (SIQA-U + SIQA-S)

### One-click with the provided fine-tuned model

```bash
python eval/run_eval.py \
    --model        model/InternVL3.5-4B-hf/full/train_set \
    --model_name   InternVL3_5_4B \
    --data_dir     data/ \
    --image_root   data/images/ \
    --output_dir   outputs/ \
    --result_dir   results/ \
    --SIQA_U --SIQA_S
```

### Use your own model

Any HuggingFace vision-language model that supports `AutoModelForImageTextToText` and `AutoProcessor` works:

```bash
python eval/run_eval.py \
    --model        Qwen/Qwen2.5-VL-7B-Instruct \
    --model_name   Qwen25_VL_7B \
    --data_dir     data/ \
    --image_root   data/images/ \
    --SIQA_U --SIQA_S
```

**Verified models:** `Qwen2.5-VL`, `Qwen3-VL`, `InternVL3.5`

### Argument reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID or local path |
| `--model_name` | `My_Model` | Tag written into prediction files |
| `--data_dir` | `data/` | Folder with benchmark JSONs |
| `--image_root` | `data/images/` | Root folder of benchmark images |
| `--output_dir` | `outputs/` | Intermediate predictions |
| `--result_dir` | `results/` | Final `results.json` |
| `--SIQA_U` | off | Evaluate SIQA-U |
| `--SIQA_S` | off | Evaluate SIQA-S |

---

## SIQA-S Baseline Evaluation

### One-click (all baselines)

```bash
python eval/baselines/run_baselines.py \
    --baselines all \
    --input_json  data/bench_SIQA-S.json \
    --image_root  data/images/ \
    --output_dir  outputs/
```

### Individual baselines

#### NIQE (no model download needed)

```bash
python eval/baselines/NIQE/niqe_eval.py \
    --input_json  data/bench_SIQA-S.json \
    --output_json outputs/SIQA-S_niqe.json \
    --image_root  data/images/
```

#### Q-Align (zero-shot)

```bash
# Model is downloaded by download_models.sh into model/one-align/
python eval/baselines/Q-Align/q_align_eval.py \
    --model_path  model/one-align/ \
    --input_json  data/bench_SIQA-S.json \
    --output_json outputs/SIQA-S_q_align.json \
    --image_root  data/images/
```

#### CLIP-IQA (zero-shot) / CLIP-IQA+ (trained)

```bash
# Zero-shot
python eval/baselines/CLIP-IQA/clip_iqa_eval.py \
    --config     eval/baselines/CLIP-IQA/configs/clipiqa_siqa_test.py \
    --input_json data/bench_SIQA-S.json \
    --image_root data/images/ \
    --output_json outputs/SIQA-S_clip_iqa.json

# Fine-tuned (CLIP-IQA+)
python eval/baselines/CLIP-IQA/clip_iqa_eval.py \
    --config      eval/baselines/CLIP-IQA/configs/clipiqa_siqa_test.py \
    --checkpoint  model/clip_iqa/clip_iqa_plus.pth \
    --input_json  data/bench_SIQA-S.json \
    --image_root  data/images/ \
    --output_json outputs/SIQA-S_clip_iqa_plus.json
```

#### HyperIQA (trained)

```bash
python eval/baselines/HyperIQA/hyperiqa_eval.py \
    --checkpoint  model/hyperiqa/hyperiqa_siqa.pth \
    --input_json  data/bench_SIQA-S.json \
    --image_root  data/images/ \
    --output_json outputs/SIQA-S_hyperiqa.json
```

---

## Fine-tuning Your Own Model

Generate SFT training data in OpenAI / LLaMA-Factory format:

```bash
python data/tran_OpenAI.py \
    --SIQA_U_json data/bench_SIQA-U.json \
    --SIQA_S_json data/bench_SIQA-S.json \
    --root_path   data/images/ \
    --out_path    data/sft/openai_bench
# Outputs: data/sft/openai_bench_SIQA-U.json
#          data/sft/openai_bench_SIQA-S.json
```

The resulting files are compatible with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and similar frameworks.

---

## SIQA-S Scoring Methodology

Given a scientific image, the model outputs a distribution over five quality levels:

$$\{\text{Bad}, \text{Poor}, \text{Fair}, \text{Good}, \text{Excellent}\}$$

The predicted score is computed as a soft expectation:

$$S_{\text{pred}} = \sum_{i=1}^{5} i \cdot \frac{e^{x_i}}{\sum_{j=1}^{5} e^{x_j}}$$

where $x_i$ is the model logit for quality class $i$. See `eval/BaseModel.py: ScoreSoftHead` for the implementation.

---

## Data Format

### SIQA-U benchmark (`bench_SIQA-U.json`)

```json
{
  "image_path": "fig1.png",
  "question":   "Is the scale bar present?",
  "option":     "A. Yes  B. No",
  "answer":     "A",
  "type":       "yes/no"
}
```

### SIQA-S benchmark (`bench_SIQA-S.json`)

```json
{
  "image_path":        "fig2.png",
  "perception_rating": 4.2,
  "knowledge_rating":  3.8
}
```

---

## Citation

If you use SIQA-Eval in your research, please cite:

```bibtex
@misc{siqa2024,
  title  = {SIQA: Scientific Image Quality Assessment Benchmark},
  author = {SIQA Team},
  year   = {2024},
  url    = {https://siqa-competition.github.io/}
}
```

---

## Contact

- Challenge website: [siqa-competition.github.io](https://siqa-competition.github.io/)
- HuggingFace dataset: [huggingface.co/datasets/SIQA/TrainSet](https://huggingface.co/datasets/SIQA/TrainSet)
- Fine-tuned model: [huggingface.co/commusim-hf/SIQA-Finetune](https://huggingface.co/commusim-hf/SIQA-Finetune)
- Issues / questions: open an issue in this repository
