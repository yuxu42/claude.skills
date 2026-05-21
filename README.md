# Claude Code Skills

Custom skills for [Claude Code](https://claude.ai/code) — reusable slash commands that extend Claude's capabilities.

## Available Skills

### qwen25 — Qwen2.5 OpenVINO C++ Modeling

Run Qwen2.5 models directly from HuggingFace safetensors using C++ OpenVINO modeling API (no IR export needed).

| Skill | Description |
|-------|-------------|
| `/qwen25:text` | Text generation with tokenizer (Python pipeline) |
| `/qwen25:setup` | Build/check C++ binary and dependencies |
| `/qwen25:run-cpp` | Run C++ binary directly with raw token IDs |

**Features:**
- Loads safetensors weights directly (no openvino_model.xml needed)
- Builds OV model graph via composable_pipeline C++ modeling API
- Supports KV-cache, GQA, fused RoPE, SwiGLU, RMSNorm, tied embeddings
- Runs on CPU or Intel Arc GPU (A770)
- Performance: ~129 tok/s (CPU), ~122-127 tok/s (GPU) for Qwen2.5-0.5B
- Optional IR export/load for faster startup (`--export-ir` / `--load-ir`)
- Optional FP16 weight compression for FP32 models (`--compress-weights`)

**Key optimizations applied:**
1. Shared causal mask across all layers (-19% graph ops, +15-50% GPU speed)
2. Fused internal RoPE op via `OpPolicy::use_internal_rope` (-14% ops, 2× GPU speed)
3. Greedy KV cache without beam_idx Gather (-22% ops, +22% GPU decode)

See [qwen25/GPU_OPTIMIZATION.md](qwen25/GPU_OPTIMIZATION.md) for the full step-by-step optimization guide (40 → 125 tok/s on Intel Arc A770).

**Supported models:** Qwen/Qwen2.5-{0.5B,1.5B,3B,7B} (any qwen2 safetensors model)

---

## Research Notes

### Gemma4 KV Cache Compression

[GEMMA4_KV_CACHE_RESULTS.md](GEMMA4_KV_CACHE_RESULTS.md) documents an accuracy investigation for `google/gemma-4-E2B-it` on OpenVINO CPU.

**Key finding:** The default per-channel KV cache quantization (`KEY_CACHE_GROUP_SIZE=0`) causes severely degraded output (premature EOS, truncated text). Setting `group_size=64` or `128` fully restores quality.

```python
pipe = ov_genai.VLMPipeline(model_dir, "CPU", **{
    "KV_CACHE_PRECISION": "u8",
    "KEY_CACHE_GROUP_SIZE": "64",
    "VALUE_CACHE_GROUP_SIZE": "64",
})
```

Accuracy tiers (WWB similarity, 24 samples): FP16 (0.99) > INT8 (0.95) >> INT4 (~0.84)

### Gemma4 Weight Compression

[GEMMA4_WEIGHT_COMPRESSION.md](GEMMA4_WEIGHT_COMPRESSION.md) documents INT4/INT8 weight compression experiments for Gemma4.

**Key findings:**
- Symmetric INT4 is ~2% better than asymmetric for this model
- Group size 64 is optimal (matches optimum-intel default for Gemma4)
- Sensitive layer precision (`per_layer_input_gate`, `per_layer_projection`) has minimal impact at full evaluation scale
- INT4 accuracy ceiling is ~0.84-0.85 regardless of mixed-precision tweaks
- 5-sample WWB is unreliable for absolute accuracy; always validate with 24+ samples

---

## Installation

### User-level (all projects)

```bash
# Clone this repo
git clone https://github.com/yuxu42/claude.skills.git ~/claude.skills

# Symlink or copy skills to ~/.claude/commands/
ln -s ~/claude.skills/qwen25 ~/.claude/commands/qwen25
```

### Project-level (specific project)

```bash
# In your project directory
mkdir -p .claude/commands
ln -s ~/claude.skills/qwen25 .claude/commands/qwen25
```

---

## Usage Examples

### qwen25:text

```bash
/qwen25:text "Hello, how are you?"
/qwen25:text "Explain quantum computing" --device GPU
/qwen25:text "Write Python code" --model-id Qwen/Qwen2.5-1.5B --max-tokens 200
/qwen25:text --interactive --device GPU
```

### qwen25:setup

```bash
/qwen25:setup build    # Build C++ binary
/qwen25:setup check    # Verify all components
/qwen25:setup info     # Show architecture overview
```

### qwen25:run-cpp

```bash
# Basic run
/qwen25:run-cpp /path/to/model --device GPU 151643 108386 198

# Export optimized graph to IR (faster subsequent loads)
/qwen25:run-cpp /path/to/model --export-ir qwen25-optimized.xml

# Load from IR
/qwen25:run-cpp /path/to/model --load-ir qwen25-optimized.xml --device GPU

# Compress FP32 weights to FP16 (do NOT use for BF16 models)
/qwen25:run-cpp /path/to/model --compress-weights --device GPU
```

---

## Requirements

### qwen25

**C++ toolchain:**
- CMake 3.23+
- C++17 compiler
- OpenVINO 2025.0+ (or from source)
- [openvino.pipeline.mx](https://github.com/openvinotoolkit/openvino.genai/tree/master/src/cpp) (composable_pipeline)

**Python:**
- Python 3.10+
- transformers, huggingface_hub (for tokenizer)
- openvino runtime

**Setup:**
See [qwen25/README.md](qwen25/README.md) for detailed installation instructions.

**Source files** (in [qwen25/src/](qwen25/src/)):

| File | Description |
|------|-------------|
| `modeling_qwen25.hpp/cpp` | Qwen2.5 architecture (GQA, fused RoPE, shared causal mask, KV-cache, SwiGLU, RMSNorm) |
| `run_qwen25.cpp` | CLI runner with `--export-ir`, `--load-ir`, `--compress-weights` |
| `hf_to_openvino.py` | Python pipeline (download, tokenize, run) |
| `CMakeLists.txt` | Build config linking composable_pipeline + OpenVINO |
| `build_and_run.sh` / `export_to_ir.sh` | Helper scripts |

---

## Contributing

To add your own skills:

1. Fork this repo
2. Create a new directory for your skill: `my-skill/`
3. Add markdown files: `my-skill/command1.md`, `my-skill/command2.md`
4. Follow the format (YAML frontmatter + markdown body)
5. Submit a PR

### Skill File Format

```markdown
---
description: Short description of what this command does
allowed-tools: Bash, Read, Write, Edit
---

Brief explanation of what this skill does.

## Context

- Key paths and configuration
- Dependencies and requirements

## Instructions

Concrete steps Claude should follow:

1. Do this
2. Then do that

\`\`\`bash
# Example command
cd /path && ./run.sh
\`\`\`

$ARGUMENTS
```

---

## License

MIT License - see individual skill directories for component licenses.
