#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Export Qwen2.5 model from HuggingFace safetensors to OpenVINO IR format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CP_BUILD="${SCRIPT_DIR}/../openvino.pipeline.mx/build/cp310-cp310-linux_x86_64"
OV_LIB="${CP_BUILD}/build_ov/install/runtime/lib/intel64"
CP_LIB="${CP_BUILD}/src/cpp"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_dir> <output_ir_path>"
    echo ""
    echo "Examples:"
    echo "  $0 ~/.cache/huggingface/models/Qwen--Qwen2.5-0.5B qwen25-0.5b.xml"
    echo "  $0 /mnt/yxu28/models/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/... qwen25-0.5b.xml"
    exit 1
fi

MODEL_DIR="$1"
OUTPUT_IR="$2"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Error: config.json not found in $MODEL_DIR"
    exit 1
fi

echo "Exporting Qwen2.5 model to OpenVINO IR format"
echo "  Model dir: $MODEL_DIR"
echo "  Output IR: $OUTPUT_IR"
echo ""

export LD_LIBRARY_PATH="${OV_LIB}:${CP_LIB}:${LD_LIBRARY_PATH}"

"${SCRIPT_DIR}/build/run_qwen25" "$MODEL_DIR" --export-ir "$OUTPUT_IR"

if [ -f "$OUTPUT_IR" ]; then
    BIN_FILE="${OUTPUT_IR%.xml}.bin"
    echo ""
    echo "Export successful!"
    echo "  XML file: $OUTPUT_IR ($(du -h "$OUTPUT_IR" | cut -f1))"
    echo "  BIN file: $BIN_FILE ($(du -h "$BIN_FILE" | cut -f1))"
    echo ""
    echo "To use the exported model:"
    echo "  ./build/run_qwen25 \"$MODEL_DIR\" --load-ir \"$OUTPUT_IR\" --device CPU TOKEN_IDS..."
else
    echo "Error: Export failed"
    exit 1
fi
