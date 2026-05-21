# Gemma4 Weight Compression Accuracy Investigation

## Environment

- Model: `google/gemma-4-E2B-it` (4.9B parameters, MoE)
- OpenVINO: 2026.3.0 nightly (2026-05-20)
- OpenVINO GenAI: 2026.3.0.0 nightly
- optimum-intel (NNCF weight compression)
- Transformers: 5.5.0
- Platform: CPU (Intel)
- WWB dataset: `ucla-contextual/contextual_test` (visual-text)
- Ground truth: PyTorch fp32 inference
- WWB results use f32 inference precision, KV cache disabled
- Sample count noted per table (5 or 24 samples)

## Gemma4 Model Architecture (Weight Compression Perspective)

Gemma4 E2B is a VLM with multiple sub-models:

| Sub-model | OV XML | Role |
|---|---|---|
| `lm_model` | `openvino_language_model.xml` | Main language model (MoE, 35 blocks) |
| `vision_embeddings_model` | `openvino_vision_embeddings_model.xml` | SigLIP2 vision encoder |
| `text_embeddings_model` | `openvino_text_embeddings_model.xml` | Text embedding |
| `text_embeddings_per_layer_model` | `openvino_text_embeddings_per_layer_model.xml` | Per-layer text embedding |

### Language Model Layer Types

Within the LM (35 transformer blocks), weights fall into categories with different sensitivity:

| Layer type | Shape (typical) | Count | Sensitivity |
|---|---|---|---|
| `self_attn.q_proj` | [2048, 1536] | 35 | Normal |
| `self_attn.k_proj` | [256, 1536] | 35 | Normal |
| `self_attn.v_proj` | [256, 1536] | 35 | Normal |
| `self_attn.o_proj` | [1536, 2048] | 35 | Normal |
| `mlp.gate_proj` | [6144, 1536] | 35 | Normal |
| `mlp.up_proj` | [6144, 1536] | 35 | Normal |
| `mlp.down_proj` | [1536, 6144] | 35 | Normal |
| `per_layer_input_gate` | [256, 1536] | 35 | Low |
| `per_layer_projection` | [1536, 256] | 35 | **High** |
| `per_layer_model_projection` | [8960, 1536] | 1 | **High** |
| `embed_tokens` (tied lm_head) | [262144, 1536] | 1 | Medium |

## GGUF Q4_0 Reference (Unsloth)

Reference: `unsloth/gemma-4-E2B-it-GGUF` Q4_0 (2.83 GB)

| Layer type | GGUF precision |
|---|---|
| Attention Q/K/V/O, FFN gate/up/down | Q4_0 (4-bit sym, group_size=32) |
| FFN down (blocks 0-3) | Q4_1 (4-bit asym, group_size=32) |
| `token_embd.weight` | Q4_K (~4.5-bit k-quant) |
| `per_layer_token_embd.weight` | Q5_K (5-bit) |
| `per_layer_model_proj.weight` | BF16 |
| All norms, input gates, projections, scales | F32 |

Key insight: GGUF keeps `per_layer_projection`, `per_layer_input_gate`, and `per_layer_model_projection` at **full precision** — these are identified as accuracy-sensitive.

## INT4 Weight Compression Results

### Quick Screening (5 samples)

Used for rapid iteration to identify trends before running full 24-sample validation.

#### Factor 1: Symmetric vs Asymmetric

| Quantization | Group Size | Similarity (5s) |
|---|---|---|
| INT4-asym | 128 | 0.8339 |
| INT4-sym | 128 | 0.8549 |

**Finding:** Symmetric quantization is ~2% better than asymmetric for this model.

#### Factor 2: Group Size

| Quantization | Group Size | Similarity (5s) |
|---|---|---|
| INT4-sym | 128 | 0.8549 |
| INT4-sym | 64 | **0.8854** |
| INT4-sym | 32 | 0.8743 |

**Finding:** Group size 64 is the sweet spot for 5-sample tests. This matches the optimum-intel default config for Gemma4 models (group_size=64).

#### Factor 3: Sensitive Layer Precision (5 samples)

All use INT4-sym-g32 for main attention/FFN layers:

| per_layer_input_gate | per_layer_projection | Similarity (5s) |
|---|---|---|
| INT4 | INT4 | 0.8743 |
| INT8-sym per-ch | INT8-sym per-ch | 0.8886 |
| FP32 | INT8-sym per-ch | 0.8909 |
| **FP32** | **FP32** | **0.9101** |

**5-sample finding:** Sensitive layers at FP32 show ~3.6% improvement over all-INT4 in quick tests.

### Full 24-Sample Validation

Full 24-sample runs reveal that 5-sample estimates were **overly optimistic** for mixed-precision configurations. The variance in small samples masked the true behavior.

#### INT4 Weight Compression (24 samples, f32 inference, KV disabled)

| Model | gates/proj precision | lm_head | Main layers | Similarity (24s) |
|---|---|---|---|---|
| INT4-sym-g64 (all layers) | INT4 | INT8 | INT4-sym-g64 | 0.8476 |
| gates/proj=FP32, lm_head=INT8 | **FP32** | INT8 | INT4-sym-g64 | 0.8459 |
| gates/proj=INT8, lm_head=INT8 | INT8 | INT8 | INT4-sym-g64 | 0.8389 |
| INT4-asym-g128 (all layers) | INT4 | INT8 | INT4-asym-g128 | 0.8377 |

**24-sample finding:** Keeping `per_layer_input_gate`/`per_layer_projection` at FP32 or INT8 makes **no meaningful difference** compared to all-INT4 at the full 24-sample scale. The INT4 compression of main attention/FFN layers is the dominant accuracy factor.

#### All Weight Formats Comparison (24 samples)

| Model | Inference | KV Config | Similarity (24s) |
|---|---|---|---|
| FP16 weights | f32 | KV f16 | **0.9886** |
| FP16 weights | bf16 | KV f16 | 0.9467 |
| FP16 weights | bf16 | KV u8, group_size=64 | 0.9584 |
| INT8 weights | f32 | KV disabled (f16) | 0.9499 |
| INT8 weights | bf16 | KV u8, group_size=64 | 0.9427 |
| INT4-sym-g64 (all layers) | f32 | KV disabled (f16) | 0.8476 |
| INT4-sym-g64 + gates/proj=FP32 | f32 | KV disabled (f16) | 0.8459 |
| INT4-asym-g128 (all layers) | f32 | KV disabled (f16) | 0.8377 |

### 5-Sample vs 24-Sample Discrepancy

| Configuration | 5 samples | 24 samples | Delta |
|---|---|---|---|
| INT4-sym-g64 (all layers) | 0.8854 | 0.8476 | -0.038 |
| INT4-sym-g32 + gates/proj=FP32 | 0.9101 | — | — |
| INT4-sym-g64 + gates/proj=FP32 | — | 0.8459 | — |

**Lesson:** 5-sample WWB tests are useful for directional screening but **unreliable for absolute values**. Always validate findings with full sample set before drawing conclusions.

## Recommended INT4 Configuration for Gemma4

Based on the GGUF Q4_0 scheme and our experiments:

```python
from optimum.intel import OVModelForVisualCausalLM
from optimum.intel.openvino.configuration import OVWeightQuantizationConfig, OVPipelineQuantizationConfig

quant_config = OVPipelineQuantizationConfig(
    quantization_configs={
        'lm_model': OVWeightQuantizationConfig(
            bits=4,
            sym=True,
            group_size=32,  # or 64
            ratio=1.0,
            ignored_scope={
                "patterns": [
                    ".*per_layer_input_gate.*",
                    ".*per_layer_projection.*",
                    ".*per_layer_model_projection.*",
                ],
            },
        ),
    },
    processor='google/gemma-4-E2B-it',
)

model = OVModelForVisualCausalLM.from_pretrained(
    'google/gemma-4-E2B-it',
    export=True,
    quantization_config=quant_config,
)
model.save_pretrained('/path/to/output')
```

This keeps sensitive layers at FP32 (following GGUF Q4_0's approach) while compressing the bulk of the model to INT4.

## optimum-intel Default Config for Gemma4

From `optimum/intel/openvino/configuration.py`:

```python
# For gemma-4-E4B-it (similar architecture)
"google/gemma-4-E4B-it": {
    "bits": 4,
    "sym": False,           # asym (our tests show sym is better)
    "group_size": 64,       # good choice
    "dataset": "contextual",
    "quant_method": OVQuantizationMethod.AWQ,
    "scale_estimation": True,
}

# For gemma-4-26B-A4B (MoE variant)
"google/gemma-4-26B-A4B-it": {
    "bits": 4,
    "sym": False,
    "group_size": 64,
    "quant_method": OVQuantizationMethod.AWQ,
    "group_size_fallback": "adjust",
    "dq_group_size": 64,
}
```

Note: The defaults use AWQ with calibration data which likely improves accuracy beyond what naive weight-only compression achieves. Data-aware quantization (AWQ, scale_estimation) is the recommended path for production INT4 models.

## NNCF Limitations

- INT8 mode only supports **per-channel** quantization (no group_size parameter)
- `ratio < 1.0` without calibration dataset assigns INT8 to layers by heuristic (typically early layers), not by name pattern
- To target specific layers for INT8, use two-step approach: export with `ignored_scope`, then post-process with `nncf.compress_weights(mode=INT8_SYM)`

## Models Exported

| Model | Path | Size | Config |
|---|---|---|---|
| INT4-asym-g128 | `/mnt/yxu28/models/gemma4-e2b-ov-int4` | 4.1G | Default optimum-cli |
| INT4-sym-g128 | `/mnt/yxu28/models/gemma4-e2b-ov-int4-sym` | 4.1G | --sym |
| INT4-sym-g64 | `/mnt/yxu28/models/gemma4-e2b-ov-int4-sym-g64` | 4.1G | --sym --group-size 64 |
| INT4-sym-g32 | `/mnt/yxu28/models/gemma4-e2b-ov-int4-sym-g32` | 4.2G | --sym --group-size 32 |
| Q4_0-style | `/mnt/yxu28/models/gemma4-e2b-ov-int4sym-g32-q4_0style` | 13G* | gates/proj=FP32, main=INT4-sym-g32 |
| Mixed (all INT8) | `/mnt/yxu28/models/gemma4-e2b-ov-int4sym-g32-allint8` | ~4.5G | gates/proj=INT8, main=INT4-sym-g32 |

*Q4_0-style is 13G because vision encoder is FP32 (unquantized). LM portion alone is ~4.5G.

## Key Conclusions

1. **Symmetric > Asymmetric** for Gemma4 INT4 weight compression (+1-2% on 5 samples)
2. **Group size 64** is optimal for INT4 (matches optimum-intel default for Gemma4)
3. **Sensitive layer precision (gates/proj) has minimal impact** at full 24-sample scale — the improvement seen in 5-sample tests does not hold up
4. **GGUF Q4_0 is a useful reference** for identifying per-layer precision decisions, but the accuracy gain from keeping small layers at FP32 is negligible in OpenVINO with this evaluation set
5. **INT4 accuracy ceiling is ~0.84-0.85** regardless of mixed-precision tweaks on gates/projections
6. **Clear accuracy tiers**: FP16 (0.99) > INT8 (0.95) >> INT4 (~0.84) — the gap between INT8 and INT4 is fundamental to the 4-bit compression of main attention/FFN layers
7. **5-sample WWB is unreliable** for absolute accuracy — use for quick directional screening only, validate with full 24 samples
