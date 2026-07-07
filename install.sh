#!/usr/bin/env bash
# One-click environment setup for SIQA-Eval
# Usage: bash install.sh [--with-clip-iqa]
set -euo pipefail

WITH_CLIP_IQA=false
for arg in "$@"; do
    [ "$arg" = "--with-clip-iqa" ] && WITH_CLIP_IQA=true
done

echo "========================================================"
echo "  SIQA-Eval: Environment Setup"
echo "========================================================"

# ── Core dependencies ────────────────────────────────────────────────────────
echo ""
echo "[1/3] Installing core dependencies ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 Pillow numpy scipy tqdm huggingface_hub accelerate

# ── NIQE baseline ────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Installing NIQE dependency (pyiqa) ..."
pip install pyiqa

# ── CLIP-IQA baseline (optional, requires mmcv) ──────────────────────────────
if [ "$WITH_CLIP_IQA" = true ]; then
    echo ""
    echo "[3/3] Installing CLIP-IQA dependencies (mmedit / mmcv-full) ..."
    pip install openmim
    mim install "mmcv-full==1.5.0"
    pip install -e eval/baselines/CLIP-IQA/
else
    echo ""
    echo "[3/3] Skipping CLIP-IQA deps (pass --with-clip-iqa to enable)."
fi

echo ""
echo "========================================================"
echo "  Setup complete."
echo "  Next: download data and models:"
echo "    python data/download_data.py"
echo "    bash   model/download_models.sh"
echo "========================================================"
