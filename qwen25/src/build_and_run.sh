#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Build and run Qwen2.5 with OpenVINO C++ modeling
#
# Usage:
#   ./build_and_run.sh <model_dir> [prompt_token_ids...]
#
# Example:
#   ./build_and_run.sh ~/.cache/huggingface/qwen25/Qwen--Qwen2.5-0.5B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CP_ROOT="${SCRIPT_DIR}/../openvino.pipeline.mx"

# Activate virtual environment
source ~/projects/venv-ovmx/bin/activate

# Find OpenVINO
OV_DIR=""
if [ -d "/home/gta/ov_build/openvino/runtime/cmake" ]; then
    OV_DIR="/home/gta/ov_build/openvino/runtime/cmake"
elif [ -d "/opt/intel/openvino/runtime/cmake" ]; then
    OV_DIR="/opt/intel/openvino/runtime/cmake"
fi

echo "[Build] Script dir: ${SCRIPT_DIR}"
echo "[Build] Build dir: ${BUILD_DIR}"
echo "[Build] CP root: ${CP_ROOT}"
echo "[Build] OpenVINO dir: ${OV_DIR:-auto-detect}"

# Configure
mkdir -p "${BUILD_DIR}"
cmake_args="-DCMAKE_BUILD_TYPE=Release"
if [ -n "${OV_DIR}" ]; then
    cmake_args="${cmake_args} -DOpenVINO_DIR=${OV_DIR}"
fi

echo "[Build] Configuring..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" ${cmake_args}

echo "[Build] Building..."
cmake --build "${BUILD_DIR}" -j$(nproc)

echo "[Build] Done!"
echo ""

# Run if model dir provided
if [ -n "$1" ]; then
    echo "[Run] Executing: ${BUILD_DIR}/run_qwen25 $@"
    "${BUILD_DIR}/run_qwen25" "$@"
fi
