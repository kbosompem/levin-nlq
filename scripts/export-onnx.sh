#!/bin/bash
# Export the fine-tuned model to ONNX format for use in VS Code extension

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# Use the fused (non-quantized) model for ONNX export
# ONNX runtime can do its own quantization
FUSED_DIR="$MODELS_DIR/datalevin-fused"
ONNX_DIR="$MODELS_DIR/datalevin-onnx"
RELEASE_DIR="$MODELS_DIR/releases"

echo "=== Exporting to ONNX Format ==="

# Activate venv
source "$PROJECT_DIR/venv/bin/activate"

# Install ONNX export dependencies if needed
if ! python3 -c "import optimum" 2>/dev/null; then
    echo "Installing ONNX export dependencies..."
    pip install optimum onnx onnxruntime
fi

if [ ! -d "$FUSED_DIR" ]; then
    echo "Error: Fused model not found at $FUSED_DIR"
    echo "Run train.sh first"
    exit 1
fi

echo "Input: $FUSED_DIR"
echo "Output: $ONNX_DIR"

# Export to ONNX
optimum-cli export onnx \
    --model "$FUSED_DIR" \
    --task causal-lm \
    "$ONNX_DIR"

echo ""
echo "=== ONNX Export Complete ==="
echo "ONNX model: $ONNX_DIR"

# Show size
echo ""
echo "Model size:"
du -sh "$ONNX_DIR" 2>/dev/null || true

# Create release bundle
echo ""
echo "Creating release bundle..."
mkdir -p "$RELEASE_DIR"

# Copy essential files for the extension
RELEASE_NAME="datalevin-nlq-$(date +%Y%m%d)"
RELEASE_BUNDLE="$RELEASE_DIR/$RELEASE_NAME"
mkdir -p "$RELEASE_BUNDLE"

cp "$ONNX_DIR"/*.onnx "$RELEASE_BUNDLE/" 2>/dev/null || true
cp "$ONNX_DIR"/config.json "$RELEASE_BUNDLE/" 2>/dev/null || true
cp "$ONNX_DIR"/tokenizer*.json "$RELEASE_BUNDLE/" 2>/dev/null || true
cp "$ONNX_DIR"/special_tokens_map.json "$RELEASE_BUNDLE/" 2>/dev/null || true

echo "Release bundle: $RELEASE_BUNDLE"
ls -la "$RELEASE_BUNDLE"

echo ""
echo "=== Ready for VS Code Extension ==="
echo ""
echo "To use in Levin extension:"
echo "  1. Copy $RELEASE_BUNDLE to levin/models/"
echo "  2. Or publish as GitHub release and download on first use"
