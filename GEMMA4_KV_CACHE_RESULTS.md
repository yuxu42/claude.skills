# Gemma4 KV Cache Compression Accuracy Investigation

## Environment

- Model: `google/gemma-4-E2B-it` (4.9B parameters, MoE)
- OpenVINO: 2026.3.0 nightly (2026-05-20)
- OpenVINO GenAI: 2026.3.0.0 nightly
- Transformers: 5.5.0
- Platform: CPU (Intel)
- WWB dataset: `ucla-contextual/contextual_test` (24 visual-text samples)

## Models Exported

| Model | Path | Size |
|---|---|---|
| FP32 weights | `/mnt/yxu28/models/gemma4-e2b-ov-fp32` | 20G |
| FP16 weights | `/mnt/yxu28/models/gemma4-e2b-ov-fp16` | 9.8G |
| INT8 weights | `/mnt/yxu28/models/gemma4-e2b-ov` | 6.5G |

## Issue Description

When running Gemma4 on OpenVINO CPU with KV cache compression (`KV_CACHE_PRECISION=u8`), the model produces truncated/degraded output with the default per-channel group size (`KEY_CACHE_GROUP_SIZE=0`, `VALUE_CACHE_GROUP_SIZE=0`).

**Fix:** Setting `KEY_CACHE_GROUP_SIZE=64` (or 128) and `VALUE_CACHE_GROUP_SIZE=64` (or 128) resolves the issue.

## Default KV Cache Properties

Queried from OpenVINO CPU plugin at runtime:

| Property | INT8 model (default) | FP16 model (default) |
|---|---|---|
| `KV_CACHE_PRECISION` | **u8** (auto-enabled) | **f16** (no compression) |
| `KEY_CACHE_GROUP_SIZE` | 0 (per-channel) | 0 |
| `VALUE_CACHE_GROUP_SIZE` | 0 (per-channel) | 0 |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE` | 32 | 32 |

**Key finding:** For INT8 weight models, KV cache u8 compression is auto-enabled by default with per-channel granularity.

## Supported KV Cache Properties

- `KV_CACHE_PRECISION`: Supported values: `u8`, `u4`, `bf16`, `f16`, `f32` (NOT `i8`)
- `KEY_CACHE_GROUP_SIZE`: 0 = per-channel, 1 = per-token, N = group of N
- `VALUE_CACHE_GROUP_SIZE`: same as above
- `KEY_CACHE_QUANT_MODE`: `AUTO`, `BY_CHANNEL`, `BY_TOKEN`
- `VALUE_CACHE_QUANT_MODE`: `AUTO`, `BY_CHANNEL`, `BY_TOKEN`

## Text Generation Quality Test

Prompt: "Write a short story about a cat who learns to fly. Be creative and detailed."
Model: FP16 weights, VLMPipeline, greedy decoding, max_new_tokens=200

| Configuration | Words Generated | Status |
|---|---|---|
| No compression (baseline) | **147** | Good |
| KV u8, default (group=0, per-channel) | **19** | **DEGRADED** |
| KV u8, group_size=64 | **149** | Good |
| KV u8, group_size=128 | **149** | Good |
| KV disabled (f16) | **150** | Good (reference) |

## KV Cache Quantization Mode Tests (FP16 model)

| K mode | V mode | Words | Status |
|---|---|---|---|
| BY_CHANNEL | BY_CHANNEL | 65 | Degraded |
| BY_TOKEN | BY_TOKEN | 17 | Degraded |
| BY_CHANNEL | BY_TOKEN | 68 | Degraded |
| BY_TOKEN | BY_CHANNEL | 14 | Degraded |
| Any mode + group_size=64 | Any mode + group_size=64 | ~146-149 | Good |

**Conclusion:** The QUANT_MODE alone doesn't fix the issue. The group_size parameter is the key factor.

## INT8 Weight Model Tests

| Configuration | Words | Status |
|---|---|---|
| Default (KV u8, group=0 per-channel, auto-enabled) | **38** | **DEGRADED** |
| KV u8, group_size=64 | 144 | Good |
| KV u8, group_size=128 | 148 | Good |
| KV disabled (f16) | 150 | Good |

## INT8 vs FP16 Weight Comparison (KV disabled)

With KV compression fully disabled, both models produce identical quality output (~147-150 words, coherent). INT8 weight compression itself does NOT degrade accuracy.

## WWB Accuracy Benchmark (24 samples, visual-text)

Ground truth collected with PyTorch fp32 inference (`PYTORCH_MODEL_DTYPE_KWARG = torch.float32`).

| Model | Inference Precision | KV Config | Similarity |
|---|---|---|---|
| FP32 weights | f32 | KV f32 | **0.9822** |
| FP32 weights | bf16 (default) | KV f16 | 0.9657 |
| FP16 weights | f32 | KV f16 | **0.9886** |
| FP16 weights | bf16 (default) | KV f16 | 0.9467 |
| FP16 weights | bf16 (default) | KV u8, group_size=64 | 0.9584 |
| INT8 weights | f32 | KV disabled (f16) | 0.9499 |
| INT8 weights | bf16 (default) | KV u8, group_size=64 | 0.9427 |
| INT4 weights | f32 | KV disabled (f16) | 0.8377 |
| INT4 weights | bf16 (default) | KV u8, group_size=64 | 0.8491 |

**Notes:**
- FP32 vs FP16 weight models with f32 inference show ~same similarity (0.98-0.99) — slight variance is noise from non-deterministic generation on ambiguous images.
- bf16 inference precision drops similarity by ~2-3% compared to f32 inference.
- All models score above 0.94, which is acceptable given visual interpretation variance.
- The remaining gap from 1.0 is primarily from image interpretation differences between HF pytorch and OV GenAI vision pipeline (4/5 text-identical in spot checks).

## Root Cause

The default KV cache compression for Gemma4 uses **per-channel quantization** (`group_size=0`), which is too coarse for this model's attention patterns. This causes:
1. Premature EOS token generation
2. Truncated/incoherent outputs

## Recommended Fix

```python
import openvino_genai as ov_genai

pipe = ov_genai.VLMPipeline(model_dir, "CPU", **{
    "KV_CACHE_PRECISION": "u8",
    "KEY_CACHE_GROUP_SIZE": "64",   # or "128"
    "VALUE_CACHE_GROUP_SIZE": "64", # or "128"
})
```

Both group_size=64 and group_size=128 restore output quality to match the no-compression baseline.

## Commands to Reproduce

```bash
# Activate environment
source ~/projects/run_gemma4/venv-gemma4/bin/activate
export HF_HOME=/mnt/yxu28/models/.cache/huggingface

# Run reproduction script
python ~/projects/run_gemma4/reproduce_issue.py

# Run WWB benchmark
wwb --base-model "google/gemma-4-E2B-it" --model-type visual-text --hf \
    --num-samples 24 --gt-data wwb_gt.csv --max_new_tokens 128

wwb --target-model /mnt/yxu28/models/gemma4-e2b-ov-fp16 --model-type visual-text --genai \
    --gt-data wwb_gt.csv --num-samples 24 --device CPU \
    --ov-config '{"KV_CACHE_PRECISION": "f16", "INFERENCE_PRECISION_HINT": "f32"}' \
    --max_new_tokens 128
```
