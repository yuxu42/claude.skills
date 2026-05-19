# GPU Performance Optimization Guide for OpenVINO LLM Inference

Step-by-step record of optimizing Qwen2.5-0.5B on Intel Arc A770 GPU, from 40 tok/s to 119 tok/s.

## Starting Point

- Model: Qwen2.5-0.5B (24 layers, 896 hidden, 14 heads, 2 KV heads, BF16 weights)
- Framework: OpenVINO composable_pipeline C++ modeling API
- Device: Intel Arc A770 (16GB, 560 GB/s bandwidth)
- Baseline: **CPU ~83 tok/s, GPU ~40-53 tok/s** (GPU slower than CPU)

---

## Step 1: Analyze the Graph

Export the model to OpenVINO IR and count operations:

```bash
./build/run_qwen25 /path/to/model --export-ir model.xml
grep -c "<layer " model.xml  # → 3424 ops
grep '<layer ' model.xml | sed 's/.*type="\([^"]*\)".*/\1/' | sort | uniq -c | sort -rn
```

**Findings:**
```
671 Const
412 Convert       ← type conversions preventing fusion
241 Unsqueeze
217 Multiply
217 Gather
216 Concat
192 Add
169 MatMul
168 ShapeOf       ← dynamic shape ops forcing CPU↔GPU sync
145 Reshape
120 Slice
 96 Transpose
 24 Select        ← per-layer causal mask rebuild
 24 LogicalAnd
 24 LessEqual
 24 Abs
```

**Root causes identified:**
1. **168 ShapeOf** — dynamic shape queries that force CPU execution on GPU
2. **24 Select + LogicalAnd + LessEqual + Abs** — causal mask rebuilt every layer
3. **48× Slice + Multiply + Concat** — decomposed RoPE per Q/K per layer

---

## Step 2: Share the Causal Mask Across Layers

**Problem:** `build_kv_causal_mask_with_attention(q_heads, cached.first, attention_mask)` was called inside each attention layer. It uses `shape::dim()` → `ShapeOf` + `Gather` to get q_len and kv_len dynamically, plus `Range` + `LessEqual` + `Select` to build the mask. That's ~20 dynamic ops × 24 layers = 480 ops that break GPU fusion.

**Key insight:** All layers share the same q_len (input sequence length) and kv_len (attention_mask length). The mask can be computed once.

**Fix:** Compute the mask once in `Qwen25Model::forward()` using `build_kv_causal_mask_with_attention_from_q_len()`:

```cpp
// In Qwen25Model::forward() — compute ONCE before the layer loop
auto q_len = Tensor(shape::dim(input_ids, 1), input_ids.context());
auto kv_len = Tensor(shape::dim(attention_mask, 1), attention_mask.context());
auto causal_mask = ops::llm::build_kv_causal_mask_with_attention_from_q_len(
    q_len, kv_len, attention_mask);

// Pass precomputed mask to all layers
for (auto& layer : layers_) {
    hidden_states = layer.forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, &causal_mask);
}
```

In `Qwen25Attention::forward()`, remove mask building and pass the precomputed mask directly to SDPA:

```cpp
// Before: built mask per-layer
// Tensor mask = ops::llm::build_kv_causal_mask_with_attention(q_heads, cached.first, *attention_mask);

// After: use precomputed mask directly
auto attn = ops::llm::sdpa(q_heads, k_expanded, v_expanded, scaling_, 3, causal_mask, false, policy);
```

**Result:** 3424 → 2780 ops (-19%). GPU: 40-53 → **55-60 tok/s** (+15-50%)

---

## Step 3: Fuse RoPE into Single Internal Op

**Problem:** `ops::llm::apply_rope()` with default policy decomposes into ~12 ops per call:
- 2 Slice (split head into halves)
- 4 Multiply (x1×cos, x2×sin, x1×sin, x2×cos)
- 1 Subtract + 1 Add (rotate)
- 1 Concat (rejoin halves)
- Unsqueezes for broadcast

Called 2× per layer (Q and K) × 24 layers = 48 invocations × ~12 ops = ~576 small ops.

Each small op on GPU means a separate kernel launch + potential sync. The GPU can't fuse across these because they're independent graph nodes.

**Fix:** The framework already has an internal fused `RoPE` op (`ov::op::internal::RoPE`). Enable it via `OpPolicy`:

```cpp
std::shared_ptr<ov::Model> create_qwen25_model(...) {
    OpPolicy policy;
    policy.use_internal_rope = true;  // ← fuse RoPE into single kernel
    BuilderContext ctx(policy);
    // ...
}
```

That's it — `apply_rope()` checks `policy->use_internal_rope` and emits a single `RoPE` node instead of the decomposition.

**Result:** 2780 → 2395 ops (-14%). GPU: 55-60 → **92-119 tok/s** (2× improvement!)

This was the biggest single win because it eliminated the most frequent small-op chains that caused CPU↔GPU synchronization on every layer.

---

## Step 4: GPU Compilation Hints

Set FP16 inference precision and performance mode for the GPU plugin:

```cpp
ov::AnyMap compile_props;
if (device.find("GPU") != std::string::npos) {
    compile_props[ov::hint::inference_precision.name()] = ov::element::f16;
    compile_props[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
}
auto compiled_model = core.compile_model(ov_model, device, compile_props);
```

Also set KV cache precision hint in the model's runtime info:

```cpp
ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
```

**Impact:** Marginal for this model (GPU plugin already defaults to FP16 on Arc), but ensures consistent behavior.

---

## Step 5: Weight Compression (FP32 Models Only)

For models with FP32 weights, compressing to FP16 halves memory bandwidth:

```cpp
void compress_weights_to_fp16(std::shared_ptr<ov::Model>& model) {
    for (auto& op : model->get_ordered_ops()) {
        auto constant = std::dynamic_pointer_cast<ov::opset13::Constant>(op);
        if (!constant) continue;
        if (constant->get_element_type() != ov::element::f32) continue;
        if (ov::shape_size(constant->get_shape()) < 512) continue;

        // Convert FP32 → FP16
        const float* src = constant->get_data_ptr<float>();
        std::vector<ov::float16> fp16_data(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
            fp16_data[i] = ov::float16(src[i]);

        auto new_const = std::make_shared<ov::opset13::Constant>(ov::element::f16, shape, fp16_data.data());
        auto convert = std::make_shared<ov::opset13::Convert>(new_const->output(0), ov::element::f32);
        ov::replace_node(constant, convert);
    }
}
```

**Important note:** For BF16 models (like Qwen2.5-0.5B from HuggingFace), this is NOT needed — the GPU plugin already handles BF16→FP16 conversion internally during compilation. Manually converting BF16→FP16 and adding Convert nodes actually hurts performance (74 tok/s vs 119 tok/s).

---

## Step 6: IR Export for Faster Startup

Export the optimized graph to IR format for faster repeated runs:

```bash
# Export once (includes all graph optimizations)
./build/run_qwen25 /path/to/model --export-ir qwen25-optimized.xml

# Load from IR (6x faster than safetensors loading)
./build/run_qwen25 /path/to/model --load-ir qwen25-optimized.xml --device GPU
```

**Startup time comparison:**
| Step | Safetensors | IR |
|------|-------------|-----|
| Load weights | 650ms | 105ms |
| Build graph | 15ms | 0ms |
| Compile | 3200ms | 3200ms |
| **Total** | **~3865ms** | **~3305ms** |

---

## Final Results

| Configuration | Graph Ops | GPU tok/s | CPU tok/s |
|---|---|---|---|
| Original (per-layer mask, decomposed RoPE) | 3424 | 40-53 | 83-86 |
| + Shared causal mask | 2780 | 55-60 | 91-96 |
| **+ Fused RoPE (final)** | **2395** | **92-119** | **129** |

**Total GPU improvement: 2-3× faster** (40 → 100+ tok/s)

---

## Key Principles for GPU LLM Optimization

### 1. Minimize Dynamic Shape Ops

`ShapeOf` + `Gather` forces the GPU plugin to execute those ops on CPU and sync results back. For operations that produce the same result across layers (mask, RoPE tables), compute them once and share.

**How to detect:** Export IR, count `ShapeOf` ops. Each one is a potential sync point.

### 2. Fuse Small Op Chains into Single Kernels

A sequence of Slice → Multiply → Add → Concat creates 4 GPU kernel launches with memory round-trips. A single fused op (like `RoPE`) does it in one kernel.

**How to detect:** Look for repeated patterns in the op type breakdown. If you see N×(Slice+Mul+Add+Concat) where N = num_layers, that's a fusion opportunity.

### 3. Don't Fight the Plugin

The GPU plugin already:
- Converts BF16 weights to FP16 internally
- Defaults to FP16 inference precision on Arc
- Fuses Convert + MatMul patterns

Adding explicit Convert nodes at the graph level can actually prevent the plugin's built-in optimizations from triggering.

### 4. Graph Size Matters

Fewer ops = less compilation time + better plugin optimization opportunities. The GPU plugin's graph partitioning and kernel fusion work better on smaller, cleaner graphs.

### 5. Model Size vs GPU Utilization

For very small models (0.5B), the GPU may not be faster than CPU because:
- Weight matrices (896×896) are too small to saturate GPU parallelism
- Kernel launch overhead dominates compute time
- PCIe transfer overhead for dynamic shape ops

For larger models (3B+), GPU should significantly outperform CPU as compute density improves.

---

## Remaining Optimization Opportunities

Still 122 `ShapeOf` ops remaining from:
- **KV cache append** (`append_kv_cache`) — uses shape queries to track cache state
- **RoPE cos/sin computation** — some reshape ops still use dynamic shapes

These are in the framework (`openvino.pipeline.mx`) and would require upstream changes to fix.

Additional future work:
- INT4/INT8 weight quantization (via `SafetensorsWeightFinalizer` with `QuantizationConfig`)
- Paged attention for longer sequences
- Speculative decoding for higher throughput
