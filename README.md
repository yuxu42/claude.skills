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
- Supports KV-cache, GQA, RoPE, SwiGLU, RMSNorm, tied embeddings
- Runs on CPU or Intel Arc GPU (A770)
- Performance: ~83 tok/s (CPU), ~40-53 tok/s (GPU) for Qwen2.5-0.5B

**Supported models:** Qwen/Qwen2.5-{0.5B,1.5B,3B,7B} (any qwen2 safetensors model)

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
/qwen25:run-cpp /path/to/model --device GPU 151643 108386 198
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
See [qwen25/SETUP.md](qwen25/SETUP.md) for detailed installation instructions.

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
