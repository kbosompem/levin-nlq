#!/bin/bash
# Quantize the fused model to 4-bit for smaller size

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

FUSED_DIR="$MODELS_DIR/datalevin-fused"
QUANTIZED_DIR="$MODELS_DIR/datalevin-4bit"

echo "=== Quantizing Datalevin NLQ Model ==="

# Activate venv
source "$PROJECT_DIR/venv/bin/activate"

if [ ! -d "$FUSED_DIR" ]; then
    echo "Error: Fused model not found at $FUSED_DIR"
    echo "Run train.sh first"
    exit 1
fi

echo "Input: $FUSED_DIR"
echo "Output: $QUANTIZED_DIR"

# Quantize to 4-bit
mlx_lm.convert \
    --hf-path "$FUSED_DIR" \
    --mlx-path "$QUANTIZED_DIR" \
    --quantize \
    --q-bits 4

echo ""
echo "=== Quantization Complete ==="
echo "Quantized model: $QUANTIZED_DIR"

# Show size comparison
echo ""
echo "Size comparison:"
du -sh "$FUSED_DIR" 2>/dev/null || true
du -sh "$QUANTIZED_DIR" 2>/dev/null || true
