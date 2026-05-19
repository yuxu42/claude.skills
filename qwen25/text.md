---
description: Run Qwen2.5 text generation from HuggingFace safetensors via OpenVINO C++ modeling
allowed-tools: Bash, Read
---

Run Qwen2.5 text generation. Parse $ARGUMENTS for prompt text, --device, --model-id, --max-tokens, --interactive flags.

Defaults: model=Qwen/Qwen2.5-0.5B, device=CPU, max-tokens=50.

```bash
cd /home/yxu28/projects/run_qwen25
source ~/projects/venv-ovmx/bin/activate
python3 hf_to_openvino.py --model-id Qwen/Qwen2.5-0.5B --device CPU --max-tokens 50 --prompt "PROMPT"
```

Supported flags:
- `--device GPU|GPU.0|GPU.1` — Intel Arc A770
- `--model-id Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}`
- `--max-tokens N`
- `--interactive` — multi-turn mode
- `--list-devices` — show available OpenVINO devices
- `--compress-weights` — compress BF16/FP32 weights to FP16 (mainly for FP32 models)

Performance (Qwen2.5-0.5B): CPU ~129 tok/s, GPU ~100-119 tok/s

Cached model: `/mnt/yxu28/models/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/`

$ARGUMENTS
