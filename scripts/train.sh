#!/bin/bash
# Fine-tune SmolLM for Datalevin NLQ on Mac Mini M1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
DATA_DIR="$PROJECT_DIR/training-data"

# Model configuration.
# Qwen2.5-Coder is heavily pretrained on code, which matters more than parameter
# count here: the target language is s-expressions, and a code-pretrained base
# produces balanced brackets and well-formed EDN far more reliably than a
# general-purpose model of the same size. Costs ~300MB at 4-bit vs ~70MB.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
MLX_MODEL_DIR="$MODELS_DIR/smollm-mlx"
ADAPTER_DIR="$MODELS_DIR/datalevin-adapter"
FUSED_DIR="$MODELS_DIR/datalevin-fused"

echo "=== Datalevin NLQ Training Pipeline ==="
echo "Project: $PROJECT_DIR"
echo "Base model: $BASE_MODEL"

# Check for virtual environment.
# MLX has no wheels for Python 3.14 (the current system default on this Mac), so
# pin to a version that does rather than letting `python3` pick.
PYBIN=""
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then PYBIN="$candidate"; break; fi
done
if [ -z "$PYBIN" ]; then
    echo "Error: need Python 3.10-3.12 for MLX (found $(python3 --version))."
    echo "Install one, e.g.: brew install python@3.12"
    exit 1
fi

if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo ""
    echo "Creating virtual environment with $PYBIN..."
    "$PYBIN" -m venv "$PROJECT_DIR/venv"
fi

# Activate venv
source "$PROJECT_DIR/venv/bin/activate"

# Install dependencies if needed
if ! python3 -c "import mlx_lm" 2>/dev/null; then
    echo ""
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install mlx mlx-lm transformers datasets huggingface_hub
fi

# Download and convert base model if needed
if [ ! -d "$MLX_MODEL_DIR" ]; then
    echo ""
    echo "Converting base model to MLX format..."
    mkdir -p "$MODELS_DIR"
    mlx_lm.convert --hf-path "$BASE_MODEL" --mlx-path "$MLX_MODEL_DIR"
fi

# Verify training data exists
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo ""
    echo "Generating training data..."
    python3 "$SCRIPT_DIR/generate-data.py"
fi

echo ""
echo "Training data:"
echo "  Train: $(wc -l < "$DATA_DIR/train.jsonl") examples"
echo "  Valid: $(wc -l < "$DATA_DIR/valid.jsonl") examples"

# Fine-tune with LoRA
echo ""
echo "=== Starting LoRA Fine-tuning ==="
# ~2600 training examples at batch 4 is ~650 steps/epoch, so 2000 iters is
# roughly 3 epochs. Validation is schema-disjoint: val loss measures
# generalization to an unseen schema, so watch it for divergence rather than
# expecting it to track train loss.
mlx_lm.lora \
    --model "$MLX_MODEL_DIR" \
    --data "$DATA_DIR" \
    --train \
    --batch-size 4 \
    --lora-layers 8 \
    --iters 2000 \
    --adapter-path "$ADAPTER_DIR" \
    --learning-rate 1e-4 \
    --steps-per-eval 200 \
    --seed 42

echo ""
echo "=== Fusing LoRA Adapter ==="
mlx_lm.fuse \
    --model "$MLX_MODEL_DIR" \
    --adapter-path "$ADAPTER_DIR" \
    --save-path "$FUSED_DIR"

echo ""
echo "=== Training Complete ==="
echo "Fused model saved to: $FUSED_DIR"
echo ""
echo "Next steps:"
echo "  1. Test the model: python3 scripts/test-model.py"
echo "  2. Quantize: ./scripts/quantize.sh"
echo "  3. Export to ONNX: ./scripts/export-onnx.sh"
