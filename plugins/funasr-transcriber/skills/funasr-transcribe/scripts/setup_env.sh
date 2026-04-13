#!/usr/bin/env bash
# Set up Python venv with FunASR and dependencies.
# Detects GPU/CUDA automatically; falls back to CPU-only PyTorch.
#
# Usage:
#   bash setup_env.sh            # auto-detect
#   bash setup_env.sh cpu        # force CPU-only
#   bash setup_env.sh cu121      # force CUDA 12.1

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
FORCE_VARIANT="${1:-auto}"

echo "=== FunASR Environment Setup ==="

# Ensure ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
    elif command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "Error: ffmpeg not found. Install it manually."
        exit 1
    fi
fi

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Detect CUDA
if [ "$FORCE_VARIANT" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        if [ "$MAJOR" -ge 12 ] 2>/dev/null; then
            FORCE_VARIANT="cu121"
        elif [ "$MAJOR" -ge 11 ] 2>/dev/null; then
            FORCE_VARIANT="cu118"
        else
            FORCE_VARIANT="cpu"
        fi
        echo "Detected CUDA $CUDA_VER -> $FORCE_VARIANT"
    else
        FORCE_VARIANT="cpu"
        echo "No NVIDIA GPU detected -> CPU mode"
    fi
fi

# Install PyTorch
echo "Installing PyTorch ($FORCE_VARIANT)..."
if [ "$FORCE_VARIANT" = "cpu" ]; then
    pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    pip install -q torch torchaudio --index-url "https://download.pytorch.org/whl/$FORCE_VARIANT"
fi

# Install FunASR + deps
echo "Installing FunASR and dependencies..."
pip install -q -U funasr modelscope boto3

# Patch clustering for long audio
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/patch_clustering.py" ]; then
    echo "Applying clustering optimization patch..."
    python3 "$SCRIPT_DIR/patch_clustering.py"
fi

echo ""
echo "=== Setup complete ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import funasr
print(f'FunASR {funasr.__version__}')
"
echo ""
echo "Activate with: source $VENV_DIR/bin/activate"
