#!/bin/bash

# Docker entrypoint script for ONNX Model Tester
# Handles command-line arguments in a user-friendly format

set -e

# Default values
PROMPT=""
REFINEMENT_CYCLES=1
GENERATE_EDGE_CASES="true"
LANG="english"
MODEL_PATH="/app/models"
OUTPUT_DIR="/app/outputs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p=*|--prompt=*)
            PROMPT="${1#*=}"
            shift
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        --refinement-cycles=*)
            REFINEMENT_CYCLES="${1#*=}"
            shift
            ;;
        --refinement-cycles)
            REFINEMENT_CYCLES="$2"
            shift 2
            ;;
        --generate-edge-cases=*)
            GENERATE_EDGE_CASES="${1#*=}"
            shift
            ;;
        --generate-edge-cases)
            GENERATE_EDGE_CASES="$2"
            shift 2
            ;;
        --lang=*|--language=*)
            LANG="${1#*=}"
            shift
            ;;
        --lang|--language)
            LANG="$2"
            shift 2
            ;;
        --model-path=*)
            MODEL_PATH="${1#*=}"
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "ONNX Model Tester"
            echo "=================="
            echo ""
            echo "Usage: docker run [OPTIONS] IMAGE [ARGUMENTS]"
            echo ""
            echo "Arguments:"
            echo "  -p, --prompt=TEXT              Classification task description (required)"
            echo "  --refinement-cycles=N          Number of test cycles (default: 1)"
            echo "  --generate-edge-cases=BOOL     Generate edge cases (default: true)"
            echo "  --lang=LANGUAGE               Language for samples (default: english)"
            echo "  --model-path=PATH             Path to model folder (default: /app/models)"
            echo "  --output-dir=PATH             Output directory (default: /app/outputs)"
            echo ""
            echo "Environment Variables:"
            echo "  OPENROUTER_API_KEY            Required for LLM text generation"
            echo "  OPEN_ROUTER_API_KEY           Alternative name for API key"
            echo ""
            echo "Example:"
            echo "  docker run --rm \\"
            echo "    -v \$(pwd):/app/models \\"
            echo "    -e OPENROUTER_API_KEY=\"your-key\" \\"
            echo "    your-image \\"
            echo "    -p=\"Classify customer feedback as positive or negative\" \\"
            echo "    --refinement-cycles=2 \\"
            echo "    --generate-edge-cases=true \\"
            echo "    --lang=english"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$PROMPT" ]; then
    echo "Error: --prompt is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check for API key
if [ -z "$OPENROUTER_API_KEY" ] && [ -z "$OPEN_ROUTER_API_KEY" ]; then
    echo "Warning: No OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY found"
    echo "LLM text generation will be skipped"
fi

# Find ONNX model in the model path
ONNX_MODEL=""
if [ -d "$MODEL_PATH" ]; then
    # Look for .onnx files
    ONNX_FILES=$(find "$MODEL_PATH" -name "*.onnx" -type f | head -1)
    if [ -n "$ONNX_FILES" ]; then
        ONNX_MODEL="$ONNX_FILES"
        echo "✓ Found ONNX model: $ONNX_MODEL"
    else
        echo "Error: No .onnx files found in $MODEL_PATH"
        exit 1
    fi
else
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Prepare arguments for the Python script
PYTHON_ARGS=(
    "--model-path" "$ONNX_MODEL"
    "--prompt" "$PROMPT"
    "--generate-samples" "10"
    "--lang" "$LANG"
    "--generate-edge-cases" "$GENERATE_EDGE_CASES"
    "--refinement-cycles" "$REFINEMENT_CYCLES"
    "--output-dir" "$OUTPUT_DIR"
    "--use-llm-text"
    "--verbose"
)

# Set config directory to the same as model directory
MODEL_DIR=$(dirname "$ONNX_MODEL")
PYTHON_ARGS+=("--config-dir" "$MODEL_DIR")

echo "🚀 Starting ONNX Model Testing"
echo "================================"
echo "Model: $ONNX_MODEL"
echo "Task: $PROMPT"
echo "Language: $LANG"
echo "Refinement cycles: $REFINEMENT_CYCLES"
echo "Generate edge cases: $GENERATE_EDGE_CASES"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the Python script
exec python onnx_model_tester.py "${PYTHON_ARGS[@]}" 