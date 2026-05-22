import numpy as np
import torch
from safetensors import safe_open
from pathlib import Path
import json
from collections import defaultdict
import sys

MODEL_DIR = Path("/mnt/yxu28/models/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0")

with open(MODEL_DIR / "model.safetensors.index.json") as f:
    index = json.load(f)

weight_map = index["weight_map"]
shard_files = sorted(set(weight_map.values()))

def categorize(name):
    if "visual" in name:
        return "visual"
    if "layernorm" in name or ("norm" in name and "proj" not in name and "fc" not in name):
        return "layernorm"
    if "embed_tokens" in name:
        return "embed_tokens"
    if "lm_head" in name:
        return "lm_head"
    if "self_attn.q_proj" in name: return "self_attn.q_proj"
    if "self_attn.k_proj" in name: return "self_attn.k_proj"
    if "self_attn.v_proj" in name: return "self_attn.v_proj"
    if "self_attn.o_proj" in name: return "self_attn.o_proj"
    if "linear_attn.in_proj_qkv" in name: return "linear_attn.in_proj_qkv"
    if "linear_attn.in_proj_a" in name: return "linear_attn.in_proj_a"
    if "linear_attn.in_proj_b" in name: return "linear_attn.in_proj_b"
    if "linear_attn.in_proj_z" in name: return "linear_attn.in_proj_z"
    if "linear_attn.out_proj" in name: return "linear_attn.out_proj"
    if "linear_attn.conv1d" in name: return "linear_attn.conv1d"
    if "mlp.experts.gate_up_proj" in name: return "mlp.experts.gate_up"
    if "mlp.experts.down_proj" in name: return "mlp.experts.down"
    if "mlp.shared_expert.gate_proj" in name: return "shared_expert.gate"
    if "mlp.shared_expert.up_proj" in name: return "shared_expert.up"
    if "mlp.shared_expert.down_proj" in name: return "shared_expert.down"
    if "mlp.gate" in name: return "mlp.router_gate"
    return None

results = defaultdict(lambda: {
    "count": 0, "means": [], "stds": [], "mins": [], "maxs": [],
    "skewness": [], "kurtosis": [], "abs_means": [],
    "row_range_cv": [], "col_range_cv": [],
    "max_over_std": [],
    "shapes": [], "dtypes": []
})

print("Analyzing Qwen3.6-35B-A3B weights (single pass)...")
print(f"Model dir: {MODEL_DIR}")
print(f"Shards: {len(shard_files)}")
sys.stdout.flush()

# Sample every Nth layer for row/col analysis to save time
SAMPLE_LAYER_INTERVAL = 4

for shard_idx, shard_file in enumerate(shard_files):
    filepath = MODEL_DIR / shard_file
    if not filepath.exists():
        print(f"  MISSING: {shard_file}")
        continue

    print(f"  Processing shard {shard_idx+1}/{len(shard_files)}: {shard_file}")
    sys.stdout.flush()

    with safe_open(str(filepath), framework="pt") as f:
        for name in f.keys():
            cat = categorize(name)
            if cat is None:
                continue

            tensor_pt = f.get_tensor(name).to(torch.float32)
            shape = tuple(tensor_pt.shape)
            numel = tensor_pt.numel()
            if numel < 10:
                continue

            # For very large tensors (MoE experts), sample
            if numel > 50_000_000:
                flat = tensor_pt.flatten()
                indices = torch.randperm(numel)[:5_000_000]
                sample = flat[indices].numpy()
            else:
                sample = tensor_pt.flatten().numpy()

            r = results[cat]
            r["count"] += 1
            r["shapes"].append(shape)
            r["dtypes"].append(str(f.get_tensor(name).dtype))

            mean_val = float(np.mean(sample))
            std_val = float(np.std(sample))
            r["means"].append(mean_val)
            r["stds"].append(std_val)
            r["mins"].append(float(np.min(sample)))
            r["maxs"].append(float(np.max(sample)))
            r["abs_means"].append(float(np.mean(np.abs(sample))))

            if std_val > 0:
                r["max_over_std"].append(float(np.max(np.abs(sample)) / std_val))
                centered = (sample - mean_val) / std_val
                r["skewness"].append(float(np.mean(centered ** 3)))
                r["kurtosis"].append(float(np.mean(centered ** 4) - 3))

            # Row vs Column range analysis (sample layers)
            # Extract layer number to decide if we sample
            import re
            layer_match = re.search(r'layers\.(\d+)\.', name)
            layer_num = int(layer_match.group(1)) if layer_match else 0
            do_rowcol = (layer_num % SAMPLE_LAYER_INTERVAL == 0)

            if do_rowcol and cat not in ("layernorm", "embed_tokens", "lm_head", "mlp.router_gate"):
                if tensor_pt.ndim == 2 and shape[0] > 1 and shape[1] > 1:
                    # Standard 2D weight: [out, in]
                    t2d = tensor_pt.numpy()
                    row_ranges = np.ptp(t2d, axis=1)
                    col_ranges = np.ptp(t2d, axis=0)
                    rmean = np.mean(row_ranges)
                    cmean = np.mean(col_ranges)
                    if rmean > 0:
                        r["row_range_cv"].append(float(np.std(row_ranges) / rmean))
                    if cmean > 0:
                        r["col_range_cv"].append(float(np.std(col_ranges) / cmean))
                elif tensor_pt.ndim == 3 and shape[0] > 1 and shape[1] > 1 and shape[2] > 1:
                    # MoE expert weights: [num_experts, out, in] or [num_experts, in, out]
                    # Analyze a few experts
                    num_experts = shape[0]
                    for eidx in range(0, num_experts, max(1, num_experts // 4)):
                        expert_w = tensor_pt[eidx].numpy()
                        row_ranges = np.ptp(expert_w, axis=1)
                        col_ranges = np.ptp(expert_w, axis=0)
                        rmean = np.mean(row_ranges)
                        cmean = np.mean(col_ranges)
                        if rmean > 0:
                            r["row_range_cv"].append(float(np.std(row_ranges) / rmean))
                        if cmean > 0:
                            r["col_range_cv"].append(float(np.std(col_ranges) / cmean))

            del tensor_pt

print(f"\nDone. Analyzed {sum(r['count'] for r in results.values())} tensors.\n")

# === PRINT RESULTS ===
print("=" * 120)
print("WEIGHT DISTRIBUTION SUMMARY")
print("=" * 120)
print(f"{'Category':<28} {'#':>3} {'Shape Example':<28} {'Mean':>10} {'Std':>10} {'AbsMean':>10} {'Min':>10} {'Max':>10} {'Skew':>7} {'Kurt':>7}")
print("-" * 120)

for cat in sorted(results.keys()):
    r = results[cat]
    if r["count"] == 0:
        continue
    print(f"{cat:<28} {r['count']:>3} {str(r['shapes'][0]):<28} "
          f"{np.mean(r['means']):>10.6f} {np.mean(r['stds']):>10.6f} {np.mean(r['abs_means']):>10.6f} "
          f"{np.mean(r['mins']):>10.4f} {np.mean(r['maxs']):>10.4f} "
          f"{np.mean(r['skewness']) if r['skewness'] else 0:>7.3f} "
          f"{np.mean(r['kurtosis']) if r['kurtosis'] else 0:>7.2f}")

# === QUANTIZATION ANALYSIS ===
print("\n")
print("=" * 120)
print("QUANTIZATION FRIENDLINESS ANALYSIS")
print("=" * 120)
print(f"\n{'Category':<28} {'RowRangeCV':>11} {'ColRangeCV':>11} {'Row<Col?':>9} {'|Mean|/Std':>11} {'MinMaxSym':>10} {'Quant Rec':<35}")
print("-" * 120)

for cat in sorted(results.keys()):
    r = results[cat]
    if r["count"] == 0 or not r["row_range_cv"]:
        continue

    avg_row_cv = np.mean(r["row_range_cv"])
    avg_col_cv = np.mean(r["col_range_cv"])
    avg_mean = np.mean(r["means"])
    avg_std = np.mean(r["stds"])
    avg_min = np.mean(r["mins"])
    avg_max = np.mean(r["maxs"])

    # Zero-centered check
    bias_ratio = abs(avg_mean) / avg_std if avg_std > 0 else float('inf')

    # Min/Max symmetry
    absmin = abs(avg_min)
    absmax = abs(avg_max)
    range_sym = abs(absmin - absmax) / max(absmin, absmax) if max(absmin, absmax) > 0 else 0

    # Row vs col preference
    row_better = "Yes" if avg_row_cv < avg_col_cv else "No"

    # Quantization recommendation
    recs = []
    if avg_row_cv < avg_col_cv * 0.8:
        recs.append("per-row")
    elif avg_col_cv < avg_row_cv * 0.8:
        recs.append("per-col")
    else:
        recs.append("row~col")

    if bias_ratio < 0.02 and range_sym < 0.15:
        recs.append("sym")
    elif bias_ratio < 0.05:
        recs.append("sym(~)")
    else:
        recs.append("asym")

    rec_str = ", ".join(recs)
    print(f"{cat:<28} {avg_row_cv:>11.4f} {avg_col_cv:>11.4f} {row_better:>9} {bias_ratio:>11.5f} {range_sym:>10.4f} {rec_str:<35}")

# === OUTLIER ANALYSIS ===
print("\n")
print("=" * 120)
print("OUTLIER ANALYSIS (Max/Std ratio — lower is better for quantization)")
print("=" * 120)
print(f"\n{'Category':<28} {'Avg Max/Std':>12} {'Worst Max/Std':>14} {'Median Max/Std':>15} {'Outlier Level':<15}")
print("-" * 90)

for cat in sorted(results.keys()):
    r = results[cat]
    if not r["max_over_std"]:
        continue
    vals = r["max_over_std"]
    avg_v = np.mean(vals)
    max_v = np.max(vals)
    med_v = np.median(vals)

    if avg_v < 5:
        level = "Low (good)"
    elif avg_v < 10:
        level = "Moderate"
    elif avg_v < 20:
        level = "High"
    else:
        level = "Very High (!)"

    print(f"{cat:<28} {avg_v:>12.2f} {max_v:>14.2f} {med_v:>15.2f} {level:<15}")

# === SUMMARY ===
print("\n")
print("=" * 120)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 120)

# Aggregate
all_means = []
all_bias_ratios = []
for cat in results:
    r = results[cat]
    if r["count"] == 0 or cat in ("layernorm", "visual"):
        continue
    all_means.extend(r["means"])
    if r["stds"]:
        for m, s in zip(r["means"], r["stds"]):
            if s > 0:
                all_bias_ratios.append(abs(m) / s)

overall_bias = np.mean(all_bias_ratios) if all_bias_ratios else 0

print(f"\n1. Zero-centered: Overall |mean|/std = {overall_bias:.5f}")
if overall_bias < 0.02:
    print("   => Weights are WELL zero-centered. Symmetric quantization is appropriate.")
elif overall_bias < 0.05:
    print("   => Weights are approximately zero-centered. Symmetric quant OK for most layers.")
else:
    print("   => Weights show notable bias. Asymmetric quantization recommended.")

# Row vs col
all_row = []
all_col = []
for cat in results:
    r = results[cat]
    all_row.extend(r["row_range_cv"])
    all_col.extend(r["col_range_cv"])

if all_row and all_col:
    avg_r = np.mean(all_row)
    avg_c = np.mean(all_col)
    print(f"\n2. Per-row vs Per-col: Avg RowRangeCV={avg_r:.4f}, ColRangeCV={avg_c:.4f}")
    if avg_r < avg_c * 0.8:
        print("   => Per-ROW (per-token) quantization is clearly preferred.")
        print("   => Rows have more uniform ranges = less quantization error per group.")
    elif avg_c < avg_r * 0.8:
        print("   => Per-COLUMN (per-channel) quantization is clearly preferred.")
    else:
        print("   => Row and column granularity are similar. Either works; per-channel is standard for weights.")

# Outlier severity
all_outlier = []
for cat in results:
    r = results[cat]
    if cat not in ("layernorm", "visual", "embed_tokens"):
        all_outlier.extend(r["max_over_std"])
if all_outlier:
    avg_out = np.mean(all_outlier)
    print(f"\n3. Outlier severity: Avg Max/Std = {avg_out:.2f}")
    if avg_out < 6:
        print("   => Low outlier levels. Standard INT8/INT4 quantization should work well.")
    elif avg_out < 12:
        print("   => Moderate outliers. Consider group quantization (g128) or mixed-precision.")
    else:
        print("   => Significant outliers. Recommend SmoothQuant, AWQ, or GPTQ with group size.")

print(f"\n4. Architecture notes:")
print(f"   - Hybrid model: self-attention + linear attention (Mamba-like) + MoE")
print(f"   - MoE experts are 3D tensors [num_experts, hidden, intermediate]")
print(f"   - Linear attention has conv1d, A_log, dt_bias (special handling needed)")
print(f"   - No bias terms in language model linear layers (only in visual encoder)")
print()
