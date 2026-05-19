# qwen25 — Qwen2.5 OpenVINO C++ Modeling

Run Qwen2.5 models directly from HuggingFace safetensors using C++ OpenVINO modeling API.

## Commands

| Command | Description |
|---------|-------------|
| `/qwen25:text` | Text generation with tokenizer (Python pipeline) |
| `/qwen25:setup` | Build/check C++ binary and dependencies |
| `/qwen25:run-cpp` | Run C++ binary directly with raw token IDs |

## Architecture

This skill loads raw safetensors weights directly (no IR/XML export), builds the OpenVINO model graph via the C++ composable_pipeline modeling API, and runs inference on CPU or Intel Arc GPU.

**Key features:**
- Direct safetensors loading (no preprocessing needed)
- C++ model definition: Qwen2.5Attention, Qwen2.5MLP, Qwen2.5DecoderLayer
- KV-cache via `append_kv_cache`
- Grouped Query Attention (GQA) with `repeat_kv`
- RoPE positional embeddings
- SwiGLU MLP (gate + up → silu → down)
- RMSNorm pre/post layers
- Tied word embeddings (lm_head shares embed_tokens weights)

**Performance (Qwen2.5-0.5B):**
- CPU: ~83 tokens/sec
- GPU (Intel Arc A770): ~40-53 tokens/sec

## Requirements

### Software

- **CMake:** 3.23+
- **C++ Compiler:** C++17 support (GCC 9+, Clang 10+)
- **OpenVINO Runtime:** 2025.0+ (or build from source)
- **composable_pipeline:** From [openvino.pipeline.mx](https://github.com/openvinotoolkit/openvino.genai)
- **nlohmann_json:** Header-only (bundled with OpenVINO or install separately)
- **Python:** 3.10+ with transformers, huggingface_hub, openvino

### Hardware

- **CPU:** Any x86_64 with AVX2+ (recommended)
- **GPU (optional):** Intel Arc (A-series), Intel Data Center GPU Max, Intel Iris Xe

## Setup

### 1. Install OpenVINO and composable_pipeline

```bash
# Clone and build openvino.pipeline.mx (includes OpenVINO)
git clone https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai
git submodule update --init --recursive

# Build composable_pipeline
cmake --preset developer
cmake --build build -j$(nproc)

# Or specify existing OpenVINO installation
cmake --preset developer -DOpenVINO_DIR=/path/to/openvino/runtime/cmake
cmake --build build -j$(nproc)
```

### 2. Build the Qwen2.5 C++ runner

```bash
# Clone this skill and create project
mkdir -p ~/projects/run_qwen25
cd ~/projects/run_qwen25

# Copy source files (modeling_qwen25.hpp, modeling_qwen25.cpp, run_qwen25.cpp)
# Or clone from your fork/repo

# Create CMakeLists.txt pointing to composable_pipeline
# See example in this repo: CMakeLists.txt.example

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 3. Set up Python environment

```bash
python3 -m venv ~/projects/venv-ovmx
source ~/projects/venv-ovmx/bin/activate
pip install transformers huggingface_hub openvino
```

### 4. Install the skill

```bash
# User-level
ln -s ~/claude.skills/qwen25 ~/.claude/commands/qwen25

# Or project-level
mkdir -p .claude/commands
ln -s ~/claude.skills/qwen25 .claude/commands/qwen25
```

## Usage

### Text generation

```bash
# Basic usage
/qwen25:text "Hello, how are you?"

# With GPU acceleration
/qwen25:text "Explain quantum computing" --device GPU

# Larger model with more tokens
/qwen25:text "Write Python code for a web scraper" --model-id Qwen/Qwen2.5-1.5B --max-tokens 200

# Interactive mode
/qwen25:text --interactive --device GPU

# List available devices
/qwen25:text --list-devices
```

### Build and setup

```bash
# Build C++ binary
/qwen25:setup build

# Check all components
/qwen25:setup check

# Show architecture info
/qwen25:setup info
```

### Direct C++ execution (raw token IDs)

```bash
# Run with token IDs (151643 = <|im_start|>, 108386 = "Hello", 198 = newline)
/qwen25:run-cpp /path/to/model --device GPU 151643 108386 198
```

## Project Structure

Expected layout when installed:

```
~/projects/run_qwen25/
├── modeling_qwen25.hpp          # Qwen2.5 model definition
├── modeling_qwen25.cpp          # Implementation
├── run_qwen25.cpp               # C++ runner
├── hf_to_openvino.py            # Python pipeline script
├── CMakeLists.txt               # Build configuration
├── build/
│   └── run_qwen25               # Compiled binary
└── build_and_run.sh             # Build helper script
```

## Supported Models

Any Qwen2 or Qwen2.5 model from HuggingFace with safetensors format:

- `Qwen/Qwen2.5-0.5B` (fast, 896 hidden size)
- `Qwen/Qwen2.5-1.5B`
- `Qwen/Qwen2.5-3B`
- `Qwen/Qwen2.5-7B`
- Custom Qwen2-based models

## Troubleshooting

### Build errors

**nlohmann_json not found:**
```bash
# Use the one bundled with OpenVINO
export JSON_INCLUDE=/path/to/openvino/thirdparty/json/nlohmann_json/include
```

**composable_pipeline not found:**
```bash
# Ensure you built it and set CP_ROOT in CMakeLists.txt
export CP_ROOT=~/projects/openvino.pipeline.mx
```

### Runtime errors

**Library not found:**
```bash
export LD_LIBRARY_PATH=/path/to/openvino/runtime/lib/intel64:/path/to/composable_pipeline/build/src/cpp:$LD_LIBRARY_PATH
```

**Model not cached:**
The skill auto-downloads models to `~/.cache/huggingface/models/` or uses cached models in `/mnt/yxu28/models/.cache/huggingface/hub/`.

## License

- C++ source files: Apache-2.0 (Intel Corporation, 2025)
- Skill definitions (.md files): MIT
- Dependencies: See respective project licenses
