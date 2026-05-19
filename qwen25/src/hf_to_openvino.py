#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# hf_to_openvino.py — Full pipeline skill: HuggingFace → OpenVINO inference
#
# This script provides a complete workflow for running Qwen2.5 models:
#   1. Download model from HuggingFace (safetensors format)
#   2. Build OpenVINO model graph from safetensors weights (C++ modeling API)
#   3. Compile and run inference on CPU or Intel Arc GPU
#
# Supported models:
#   - Qwen/Qwen2.5-0.5B
#   - Qwen/Qwen2.5-1.5B
#   - Qwen/Qwen2.5-3B
#   - Qwen/Qwen2.5-7B
#   - Any qwen2 model_type from HuggingFace
#
# Usage:
#   source ~/projects/venv-ovmx/bin/activate
#   python hf_to_openvino.py --model-id Qwen/Qwen2.5-0.5B --device GPU --prompt "Hello"
#   python hf_to_openvino.py --model-dir /path/to/model --device CPU --max-tokens 100
#   python hf_to_openvino.py --model-id Qwen/Qwen2.5-0.5B --interactive
#
# Environment:
#   PYTHONPATH must include the composable_pipeline build directory
#   LD_LIBRARY_PATH must include the OpenVINO runtime libraries

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
CP_BUILD = Path.home() / "projects/openvino.pipeline.mx/build/cp310-cp310-linux_x86_64"
OV_LIB = CP_BUILD / "build_ov/install/runtime/lib/intel64"
OV_PYTHON = CP_BUILD / "build_ov/install/python"
LOCAL_SITE = Path.home() / ".local/lib/python3.10/site-packages"
DEFAULT_CACHE = Path.home() / ".cache/huggingface/models"
RUNNER_BINARY = SCRIPT_DIR / "build/run_qwen25"


def setup_paths():
    """Add required paths for imports."""
    for p in [str(LOCAL_SITE), str(CP_BUILD), str(OV_PYTHON)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if str(OV_LIB) not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{OV_LIB}:{ld_path}"


# ============================================================================
# Step 1: Model Download
# ============================================================================

def download_model(model_id: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier (e.g., 'Qwen/Qwen2.5-0.5B')
        cache_dir: Local cache directory (default: ~/.cache/huggingface/models)

    Returns:
        Path to the downloaded model directory
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE

    model_dir = cache_dir / model_id.replace("/", "--")

    if (model_dir / "config.json").exists():
        print(f"[Download] Model already cached at: {model_dir}")
        return model_dir

    print(f"[Download] Downloading {model_id}...")
    print(f"[Download] Target: {model_dir}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            ignore_patterns=["*.gguf", "*.bin", "*.msgpack", "*.pt", "*.h5"],
        )
    except ImportError:
        print("[Error] huggingface_hub not installed. Install with: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        sys.exit(1)

    print(f"[Download] Complete. Files: {list(f.name for f in model_dir.iterdir())}")
    return model_dir


def find_local_model(model_id: str) -> Optional[Path]:
    """Search common cache locations for an already-downloaded model."""
    candidates = [
        DEFAULT_CACHE / model_id.replace("/", "--"),
        Path(f"/mnt/yxu28/models/.cache/huggingface/hub/models--{model_id.replace('/', '--')}"),
    ]

    # Check HuggingFace hub cache structure (with snapshots)
    hub_dir = Path(f"/mnt/yxu28/models/.cache/huggingface/hub/models--{model_id.replace('/', '--')}")
    if hub_dir.exists():
        snapshots = hub_dir / "snapshots"
        if snapshots.exists():
            for snap in snapshots.iterdir():
                if (snap / "config.json").exists():
                    return snap

    for c in candidates:
        if (c / "config.json").exists():
            return c

    return None


# ============================================================================
# Step 2: Model Validation & Config
# ============================================================================

def validate_model(model_dir: Path) -> dict:
    """Validate the model directory and return parsed config."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"[Error] config.json not found in {model_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "unknown")
    if model_type != "qwen2":
        print(f"[Warning] model_type is '{model_type}', expected 'qwen2' for Qwen2.5")

    # Check for safetensors
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        print(f"[Error] No .safetensors files found in {model_dir}")
        sys.exit(1)

    # Check for tokenizer
    has_tokenizer = (model_dir / "tokenizer.json").exists()

    print(f"[Config] Model type: {model_type}")
    print(f"[Config] Architecture: {config.get('architectures', ['unknown'])[0]}")
    print(f"[Config] Hidden size: {config.get('hidden_size')}")
    print(f"[Config] Layers: {config.get('num_hidden_layers')}")
    print(f"[Config] Attention heads: {config.get('num_attention_heads')}")
    print(f"[Config] KV heads: {config.get('num_key_value_heads')}")
    print(f"[Config] Vocab size: {config.get('vocab_size')}")
    print(f"[Config] Safetensors files: {len(safetensors_files)}")
    print(f"[Config] Tokenizer: {'yes' if has_tokenizer else 'no'}")

    return config


# ============================================================================
# Step 3: Tokenization
# ============================================================================

class Tokenizer:
    """Simple tokenizer wrapper using HuggingFace tokenizers library."""

    def __init__(self, model_dir: Path):
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

        from tokenizers import Tokenizer as HFTokenizer
        self._tokenizer = HFTokenizer.from_file(str(tokenizer_path))

        # Load tokenizer config for special tokens
        config_path = model_dir / "tokenizer_config.json"
        self._config = {}
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)

        self.eos_token_id = self._get_token_id("eos_token", 151643)
        self.bos_token_id = self._get_token_id("bos_token", None)
        self.pad_token_id = self._get_token_id("pad_token", self.eos_token_id)

    def _get_token_id(self, key: str, default):
        """Get token ID from config."""
        token = self._config.get(key)
        if token is None:
            return default
        if isinstance(token, dict):
            token = token.get("content", token)
        if isinstance(token, str):
            enc = self._tokenizer.encode(token)
            return enc.ids[0] if enc.ids else default
        return default

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(token_ids)

    def apply_chat_template(self, messages: List[dict]) -> str:
        """Apply chat template for instruction-tuned models."""
        # Qwen2.5 chat format: <|im_start|>role\ncontent<|im_end|>
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)


# ============================================================================
# Step 4: OpenVINO Inference (C++ backend via subprocess)
# ============================================================================

def run_cpp_inference(model_dir: Path, token_ids: List[int], device: str,
                     max_tokens: int = 50) -> Tuple[List[int], dict]:
    """Run inference using the C++ runner binary.

    Returns:
        Tuple of (generated_token_ids, stats_dict)
    """
    if not RUNNER_BINARY.exists():
        print(f"[Error] C++ runner not found at {RUNNER_BINARY}")
        print(f"[Error] Build it first: cd {SCRIPT_DIR} && ./build_and_run.sh")
        sys.exit(1)

    cmd = [
        str(RUNNER_BINARY),
        str(model_dir),
        "--device", device,
    ] + [str(t) for t in token_ids]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{OV_LIB}:{env.get('LD_LIBRARY_PATH', '')}"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)

    if result.returncode != 0:
        print(f"[Error] C++ runner failed:\n{result.stderr}")
        sys.exit(1)

    # Parse output token IDs from stdout
    generated_ids = []
    stats = {}
    for line in result.stdout.split("\n"):
        if "Output token IDs:" in line:
            ids_str = line.split("Output token IDs:")[1].strip()
            generated_ids = [int(t) for t in ids_str.split() if t.strip()]
        elif "Decode throughput:" in line:
            parts = line.split("Decode throughput:")
            if len(parts) > 1:
                stats["throughput"] = parts[1].strip()
        elif "Prefill:" in line and "Decode:" in line:
            # Extract: "Prefill: 193ms, Decode: 931ms"
            idx = line.find("Prefill:")
            if idx >= 0:
                stats["timing"] = line[idx:].strip()
        elif "Compilation done in" in line:
            # Extract: "[Main] Compilation done in 6795ms"
            idx = line.find("Compilation done in")
            if idx >= 0:
                stats["compile_time"] = line[idx + len("Compilation done in"):].strip()

    return generated_ids, stats


# ============================================================================
# Step 5: OpenVINO Inference (Python composable_pipeline)
# ============================================================================

def run_pipeline_inference(model_dir: Path, prompt: str, device: str,
                          max_tokens: int = 50) -> str:
    """Run inference using composable_pipeline LLMPipeline (Python API).

    Note: LLMPipeline requires the model directory to contain either:
      - openvino_model.xml (pre-converted IR), OR
      - pipeline.yaml (composable pipeline config)

    For raw safetensors models, use --backend cpp instead.
    """
    os.environ.setdefault("OV_GENAI_USE_MODELING_API", "1")

    # Check if model has OV IR files or pipeline.yaml
    has_ov = (model_dir / "openvino_model.xml").exists()
    has_yaml = (model_dir / "pipeline.yaml").exists()

    if not has_ov and not has_yaml:
        print("[Pipeline] No openvino_model.xml or pipeline.yaml found.")
        print("[Pipeline] Falling back to C++ backend (direct safetensors loading).")
        print()
        tokenizer = Tokenizer(model_dir)
        token_ids = tokenizer.encode(prompt)
        generated_ids, stats = run_cpp_inference(model_dir, token_ids, device, max_tokens)
        output_text = tokenizer.decode(generated_ids)
        if stats.get("throughput"):
            print(f"  [{stats['throughput']}]")
        return output_text

    try:
        import composable_pipeline as cp
    except ImportError:
        sys.path.insert(0, str(CP_BUILD))
        import composable_pipeline as cp

    print(f"[Pipeline] Initializing LLMPipeline on {device}...")
    start = time.time()
    pipe = cp.LLMPipeline(str(model_dir), device)
    print(f"[Pipeline] Init time: {time.time() - start:.2f}s")

    print(f"[Pipeline] Generating (max {max_tokens} tokens)...")
    gen_start = time.time()
    result = pipe.generate(prompt, max_new_tokens=max_tokens)
    gen_time = time.time() - gen_start
    print(f"[Pipeline] Generation time: {gen_time:.2f}s")

    return str(result)


# ============================================================================
# Step 6: Interactive Mode
# ============================================================================

def interactive_mode(model_dir: Path, device: str, max_tokens: int,
                    use_pipeline: bool = True):
    """Run interactive chat loop."""
    print("\n" + "=" * 60)
    print("  Qwen2.5 Interactive Mode")
    print(f"  Model: {model_dir.name}")
    print(f"  Device: {device}")
    print(f"  Max tokens: {max_tokens}")
    print("=" * 60)
    print("Type your message (or 'quit' to exit)\n")

    if use_pipeline:
        os.environ.setdefault("OV_GENAI_USE_MODELING_API", "1")
        try:
            import composable_pipeline as cp
        except ImportError:
            sys.path.insert(0, str(CP_BUILD))
            import composable_pipeline as cp

        print(f"[Init] Loading model on {device}...")
        start = time.time()
        pipe = cp.LLMPipeline(str(model_dir), device)
        print(f"[Init] Ready in {time.time() - start:.2f}s\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            if not user_input:
                continue

            gen_start = time.time()
            result = pipe.generate(user_input, max_new_tokens=max_tokens)
            gen_time = time.time() - gen_start
            print(f"Assistant: {result}")
            print(f"  [{gen_time:.2f}s]\n")
    else:
        tokenizer = Tokenizer(model_dir)
        print(f"[Init] Using C++ runner with tokenizer\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            if not user_input:
                continue

            token_ids = tokenizer.encode(user_input)
            gen_start = time.time()
            generated_ids, stats = run_cpp_inference(
                model_dir, token_ids, device, max_tokens
            )
            gen_time = time.time() - gen_start

            output_text = tokenizer.decode(generated_ids)
            print(f"Assistant: {output_text}")
            if stats.get("throughput"):
                print(f"  [{gen_time:.2f}s, {stats['throughput']}]\n")
            else:
                print(f"  [{gen_time:.2f}s]\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace → OpenVINO: Download, build, and run Qwen2.5 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default model (Qwen2.5-0.5B) on CPU
  python hf_to_openvino.py --prompt "What is Python?"

  # Run on Intel Arc A770 GPU
  python hf_to_openvino.py --device GPU --prompt "Hello"

  # Use a specific local model directory
  python hf_to_openvino.py --model-dir /path/to/model --device GPU.0

  # Interactive chat mode
  python hf_to_openvino.py --interactive --device GPU

  # Download a larger model
  python hf_to_openvino.py --model-id Qwen/Qwen2.5-7B --device GPU --prompt "Hi"

  # Use C++ backend directly (faster startup, no pipeline overhead)
  python hf_to_openvino.py --backend cpp --prompt "Hello world"
""")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Local model directory (skips download)")
    parser.add_argument("--device", default="CPU",
                        help="OpenVINO device: CPU, GPU, GPU.0, GPU.1 (default: CPU)")
    parser.add_argument("--prompt", default=None,
                        help="Text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum new tokens to generate (default: 50)")
    parser.add_argument("--backend", choices=["pipeline", "cpp"], default="pipeline",
                        help="Inference backend (default: pipeline)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive chat mode")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Model cache directory")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available OpenVINO devices and exit")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate model directory, don't run inference")

    args = parser.parse_args()

    setup_paths()

    # List devices
    if args.list_devices:
        try:
            import openvino as ov
            core = ov.Core()
            print("Available OpenVINO devices:")
            for d in core.available_devices:
                try:
                    name = core.get_property(d, "FULL_DEVICE_NAME")
                    print(f"  {d}: {name}")
                except Exception:
                    print(f"  {d}")
        except Exception as e:
            print(f"[Error] Cannot query devices: {e}")
        return

    # Resolve model directory
    if args.model_dir:
        model_dir = args.model_dir
        if not (model_dir / "config.json").exists():
            print(f"[Error] config.json not found in {model_dir}")
            sys.exit(1)
    else:
        # Try to find locally first
        local = find_local_model(args.model_id)
        if local:
            model_dir = local
            print(f"[Info] Found local model: {model_dir}")
        else:
            model_dir = download_model(args.model_id, args.cache_dir)

    # Validate
    print(f"\n{'='*60}")
    print(f"  Model: {model_dir}")
    print(f"{'='*60}")
    config = validate_model(model_dir)

    if args.validate_only:
        print("\n[Done] Validation passed.")
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(model_dir, args.device, args.max_tokens,
                        use_pipeline=(args.backend == "pipeline"))
        return

    # Single inference
    if args.prompt is None:
        args.prompt = "Hello, how are you?"

    print(f"\n[Inference] Prompt: \"{args.prompt}\"")
    print(f"[Inference] Device: {args.device}")
    print(f"[Inference] Backend: {args.backend}")
    print(f"[Inference] Max tokens: {args.max_tokens}")
    print()

    if args.backend == "pipeline":
        result = run_pipeline_inference(model_dir, args.prompt, args.device, args.max_tokens)
        print(f"\n{'='*60}")
        print(f"Output: {result}")
        print(f"{'='*60}")
    else:
        tokenizer = Tokenizer(model_dir)
        token_ids = tokenizer.encode(args.prompt)
        print(f"[Tokenize] Input: {len(token_ids)} tokens")

        generated_ids, stats = run_cpp_inference(
            model_dir, token_ids, args.device, args.max_tokens
        )

        output_text = tokenizer.decode(generated_ids)
        print(f"\n{'='*60}")
        print(f"Output: {output_text}")
        print(f"{'='*60}")
        if stats:
            for k, v in stats.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
