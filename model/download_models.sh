#!/usr/bin/env bash
# One-click model download script for SIQA-Eval
# Usage: bash model/download_models.sh [--skip-finetune] [--skip-baselines]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$ROOT_DIR/model"

SKIP_FINETUNE=false
SKIP_BASELINES=false

for arg in "$@"; do
    case $arg in
        --skip-finetune)  SKIP_FINETUNE=true  ;;
        --skip-baselines) SKIP_BASELINES=true ;;
    esac
done

echo "========================================================"
echo "  SIQA-Eval: Model Download"
echo "========================================================"

# ── 1. Fine-tuned InternVL3.5-4B (for SIQA-U & SIQA-S MLLM evaluation) ─────
if [ "$SKIP_FINETUNE" = false ]; then
    echo ""
    echo "[1/4] Downloading SIQA fine-tuned model (InternVL3.5-4B-hf/full/train_set) ..."
    mkdir -p "$MODEL_DIR"
    python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="commusim-hf/SIQA-Finetune",
    local_dir="model/",
    repo_type="model",
    allow_patterns="InternVL3.5-4B-hf/full/train_set/*",
)
print("✅  InternVL3.5-4B-hf/full/train_set downloaded to model/InternVL3.5-4B-hf/full/train_set/")
EOF
fi

if [ "$SKIP_BASELINES" = false ]; then
    # ── 2. Q-Align (One-Align) ───────────────────────────────────────────────
    echo ""
    echo "[2/4] Downloading Q-Align (q-future/one-align) ..."
    mkdir -p "$MODEL_DIR/one-align"
    python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="q-future/one-align",
    local_dir="model/one-align",
    repo_type="model",
)
print("✅  Q-Align downloaded to model/one-align/")
EOF
    # Replace modeling file with SIQA-adapted version
    if [ -f "$ROOT_DIR/eval/baselines/Q-Align/modeling_mplug_owl2.py" ]; then
        cp "$ROOT_DIR/eval/baselines/Q-Align/modeling_mplug_owl2.py" \
           "$MODEL_DIR/one-align/modeling_mplug_owl2.py"
        echo "✅  Patched modeling_mplug_owl2.py in model/one-align/"
    fi

    # ── 3. HyperIQA checkpoint ───────────────────────────────────────────────
    echo ""
    echo "[3/4] Downloading HyperIQA SIQA checkpoint ..."
    mkdir -p "$MODEL_DIR/hyperiqa"
    python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="commusim-hf/SIQA-Finetune",
    filename="hyperiqa/hyperiqa_siqa.pth",
    local_dir="model/hyperiqa",
    repo_type="model",
)
print("✅  HyperIQA checkpoint downloaded to model/hyperiqa/")
EOF

    # ── 4. CLIP-IQA+ checkpoint ──────────────────────────────────────────────
    echo ""
    echo "[4/4] Downloading CLIP-IQA+ SIQA checkpoint ..."
    mkdir -p "$MODEL_DIR/clip_iqa"
    python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="commusim-hf/SIQA-Finetune",
    filename="clip_iqa/clip_iqa_plus.pth",
    local_dir="model/clip_iqa",
    repo_type="model",
)
print("✅  CLIP-IQA+ checkpoint downloaded to model/clip_iqa/")
EOF
fi

echo ""
echo "========================================================"
echo "  All models downloaded successfully."
echo "========================================================"
