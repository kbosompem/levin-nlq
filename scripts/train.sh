#!/bin/bash
# Fine-tune SmolLM for Datalevin NLQ on Mac Mini M1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
DATA_DIR="$PROJECT_DIR/training-data"

# Model configuration
BASE_MODEL="HuggingFaceTB/SmolLM-135M-Instruct"
MLX_MODEL_DIR="$MODELS_DIR/smollm-mlx"
ADAPTER_DIR="$MODELS_DIR/datalevin-adapter"
FUSED_DIR="$MODELS_DIR/datalevin-fused"

echo "=== Datalevin NLQ Training Pipeline ==="
echo "Project: $PROJECT_DIR"
echo "Base model: $BASE_MODEL"

# Check for virtual environment
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/venv"
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
mlx_lm.lora \
    --model "$MLX_MODEL_DIR" \
    --data "$DATA_DIR" \
    --train \
    --batch-size 4 \
    --lora-layers 4 \
    --iters 500 \
    --adapter-path "$ADAPTER_DIR" \
    --learning-rate 1e-4 \
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
echo "  1. Test the model: mlx_lm.generate --model $FUSED_DIR --prompt '<|user|>Schema: ...'"
echo "  2. Quantize: ./scripts/quantize.sh"
echo "  3. Export to ONNX: ./scripts/export-onnx.sh"
