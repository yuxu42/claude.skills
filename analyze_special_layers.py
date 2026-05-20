import numpy as np
import torch
from safetensors import safe_open
from pathlib import Path
import json
import sys

MODEL_DIR = Path("/mnt/yxu28/models/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0")

with open(MODEL_DIR / "model.safetensors.index.json") as f:
    index = json.load(f)

weight_map = index["weight_map"]

# Find the specific layers
targets = {}
for name, shard in weight_map.items():
    if "embed_tokens" in name:
        targets[name] = shard
    elif "lm_head" in name:
        targets[name] = shard
    elif "mlp.gate.weight" in name:
        targets[name] = shard

print(f"Found {len(targets)} target tensors")
print(f"  embed_tokens: {sum(1 for n in targets if 'embed_tokens' in n)}")
print(f"  lm_head: {sum(1 for n in targets if 'lm_head' in n)}")
print(f"  router_gate: {sum(1 for n in targets if 'mlp.gate' in n)}")
sys.stdout.flush()

# Group by shard for efficient loading
shard_to_names = {}
for name, shard in targets.items():
    shard_to_names.setdefault(shard, []).append(name)


def analyze_tensor_detailed(name, tensor_pt):
    """Deep analysis of a single tensor."""
    t = tensor_pt.to(torch.float32)
    arr = t.numpy()
    flat = arr.flatten()

    print(f"\n{'='*100}")
    print(f"  {name}")
    print(f"  Shape: {tuple(t.shape)}, Dtype: {tensor_pt.dtype}, Size: {t.numel():,} params")
    print(f"{'='*100}")

    # Basic stats
    mean = np.mean(flat)
    std = np.std(flat)
    print(f"\n  Basic Stats:")
    print(f"    Mean:     {mean:.8f}")
    print(f"    Std:      {std:.8f}")
    print(f"    |Mean|/Std: {abs(mean)/std:.6f}  {'(well centered)' if abs(mean)/std < 0.01 else '(biased!)'}")
    print(f"    Min:      {np.min(flat):.6f}")
    print(f"    Max:      {np.max(flat):.6f}")
    print(f"    AbsMean:  {np.mean(np.abs(flat)):.8f}")

    # Distribution shape
    skew = np.mean(((flat - mean) / std) ** 3)
    kurt = np.mean(((flat - mean) / std) ** 4) - 3
    print(f"\n  Distribution Shape:")
    print(f"    Skewness: {skew:.4f}  {'(symmetric)' if abs(skew) < 0.1 else '(skewed!)'}")
    print(f"    Kurtosis: {kurt:.4f}  {'(Gaussian-like)' if abs(kurt) < 1 else '(heavy tails!)' if kurt > 3 else '(light tails)'}")

    # Percentiles
    percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    pvals = np.percentile(flat, percentiles)
    print(f"\n  Percentiles:")
    print(f"    {'P0.1':>6} {'P1':>8} {'P5':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P95':>8} {'P99':>8} {'P99.9':>8}")
    print(f"    {pvals[0]:>6.4f} {pvals[1]:>8.4f} {pvals[2]:>8.4f} {pvals[3]:>8.4f} {pvals[4]:>8.4f} {pvals[5]:>8.4f} {pvals[6]:>8.4f} {pvals[7]:>8.4f} {pvals[8]:>8.4f}")

    # Outlier analysis
    max_abs = np.max(np.abs(flat))
    p999_abs = np.percentile(np.abs(flat), 99.9)
    p99_abs = np.percentile(np.abs(flat), 99)
    print(f"\n  Outlier Analysis:")
    print(f"    Max|val|:    {max_abs:.6f}")
    print(f"    Max/Std:     {max_abs/std:.2f}x")
    print(f"    P99.9/Std:   {p999_abs/std:.2f}x")
    print(f"    P99/Std:     {p99_abs/std:.2f}x")
    print(f"    Max/P99.9:   {max_abs/p999_abs:.2f}x  {'(smooth tail)' if max_abs/p999_abs < 1.5 else '(spike outliers!)'}")

    # Zero/near-zero analysis
    zero_frac = np.mean(np.abs(flat) < 1e-8)
    near_zero_frac = np.mean(np.abs(flat) < std * 0.01)
    print(f"\n  Sparsity:")
    print(f"    Exact zeros: {zero_frac*100:.4f}%")
    print(f"    Near-zero (<0.01*std): {near_zero_frac*100:.2f}%")

    # Row vs Column analysis (for 2D tensors)
    if arr.ndim == 2:
        nrows, ncols = arr.shape
        print(f"\n  Per-Row (output channel) Analysis [{nrows} rows]:")
        row_means = np.mean(arr, axis=1)
        row_stds = np.std(arr, axis=1)
        row_ranges = np.ptp(arr, axis=1)
        row_absmax = np.max(np.abs(arr), axis=1)
        print(f"    Row means:  min={np.min(row_means):.6f}, max={np.max(row_means):.6f}, std={np.std(row_means):.6f}")
        print(f"    Row stds:   min={np.min(row_stds):.6f}, max={np.max(row_stds):.6f}, std={np.std(row_stds):.6f}")
        print(f"    Row ranges: min={np.min(row_ranges):.4f}, max={np.max(row_ranges):.4f}, CV={np.std(row_ranges)/np.mean(row_ranges):.4f}")
        print(f"    Row absmax: min={np.min(row_absmax):.4f}, max={np.max(row_absmax):.4f}, CV={np.std(row_absmax)/np.mean(row_absmax):.4f}")

        print(f"\n  Per-Col (input channel) Analysis [{ncols} cols]:")
        col_means = np.mean(arr, axis=0)
        col_stds = np.std(arr, axis=0)
        col_ranges = np.ptp(arr, axis=0)
        col_absmax = np.max(np.abs(arr), axis=0)
        print(f"    Col means:  min={np.min(col_means):.6f}, max={np.max(col_means):.6f}, std={np.std(col_means):.6f}")
        print(f"    Col stds:   min={np.min(col_stds):.6f}, max={np.max(col_stds):.6f}, std={np.std(col_stds):.6f}")
        print(f"    Col ranges: min={np.min(col_ranges):.4f}, max={np.max(col_ranges):.4f}, CV={np.std(col_ranges)/np.mean(col_ranges):.4f}")
        print(f"    Col absmax: min={np.min(col_absmax):.4f}, max={np.max(col_absmax):.4f}, CV={np.std(col_absmax)/np.mean(col_absmax):.4f}")

        # Quantization simulation: per-row vs per-col INT8 error
        # Per-row: each row quantized with its own scale
        row_scales = row_absmax / 127.0
        row_quant_err = np.mean([np.mean(np.abs(arr[i] - np.clip(np.round(arr[i] / row_scales[i]) * row_scales[i], -row_absmax[i], row_absmax[i]))) for i in range(min(nrows, 1000))])

        # Per-col: each col quantized with its own scale
        col_scales = col_absmax / 127.0
        col_quant_err = np.mean([np.mean(np.abs(arr[:, j] - np.clip(np.round(arr[:, j] / col_scales[j]) * col_scales[j], -col_absmax[j], col_absmax[j]))) for j in range(min(ncols, 1000))])

        # Per-tensor: single scale
        tensor_scale = max_abs / 127.0
        tensor_quant_err = np.mean(np.abs(flat - np.clip(np.round(flat / tensor_scale) * tensor_scale, -max_abs, max_abs)))

        print(f"\n  Simulated INT8 Quantization Error (MAE):")
        print(f"    Per-tensor:  {tensor_quant_err:.8f}  (relative: {tensor_quant_err/np.mean(np.abs(flat))*100:.3f}%)")
        print(f"    Per-row:     {row_quant_err:.8f}  (relative: {row_quant_err/np.mean(np.abs(flat))*100:.3f}%)")
        print(f"    Per-col:     {col_quant_err:.8f}  (relative: {col_quant_err/np.mean(np.abs(flat))*100:.3f}%)")
        print(f"    Winner:      {'per-row' if row_quant_err < col_quant_err else 'per-col'} ({abs(row_quant_err-col_quant_err)/min(row_quant_err,col_quant_err)*100:.1f}% better)")

        # Symmetric vs Asymmetric simulation
        # Symmetric: scale = max(|min|, |max|) / 127
        sym_scale = max_abs / 127.0
        sym_err = np.mean(np.abs(flat - np.round(flat / sym_scale) * sym_scale))

        # Asymmetric: scale = (max - min) / 255, zero_point adjusted
        tmin, tmax = np.min(flat), np.max(flat)
        asym_scale = (tmax - tmin) / 255.0
        asym_zp = np.round(-tmin / asym_scale)
        asym_quant = np.clip(np.round(flat / asym_scale + asym_zp), 0, 255)
        asym_dequant = (asym_quant - asym_zp) * asym_scale
        asym_err = np.mean(np.abs(flat - asym_dequant))

        print(f"\n  Symmetric vs Asymmetric INT8 (per-tensor):")
        print(f"    Symmetric MAE:   {sym_err:.8f}  (relative: {sym_err/np.mean(np.abs(flat))*100:.3f}%)")
        print(f"    Asymmetric MAE:  {asym_err:.8f}  (relative: {asym_err/np.mean(np.abs(flat))*100:.3f}%)")
        print(f"    Winner:          {'symmetric' if sym_err <= asym_err else 'asymmetric'} ({abs(sym_err-asym_err)/min(sym_err,asym_err)*100:.1f}% better)")

        # Check if lm_head == embed_tokens (tied weights)
        return arr


# Load and analyze
print("\n" + "#"*100)
print("# DETAILED ANALYSIS: embed_tokens, lm_head, router_gate")
print("#"*100)

embed_arr = None
lm_head_arr = None
router_stats = []

for shard, names in sorted(shard_to_names.items()):
    filepath = MODEL_DIR / shard
    print(f"\nLoading {shard}...")
    sys.stdout.flush()

    with safe_open(str(filepath), framework="pt") as f:
        for name in names:
            arr = analyze_tensor_detailed(name, f.get_tensor(name))
            if "embed_tokens" in name:
                embed_arr = arr
            elif "lm_head" in name:
                lm_head_arr = arr
            elif "mlp.gate" in name:
                router_stats.append((name, arr))
            sys.stdout.flush()

# Check weight tying
if embed_arr is not None and lm_head_arr is not None:
    print(f"\n\n{'='*100}")
    print("WEIGHT TYING CHECK: embed_tokens vs lm_head")
    print(f"{'='*100}")
    if embed_arr.shape == lm_head_arr.shape:
        diff = np.max(np.abs(embed_arr - lm_head_arr))
        print(f"  Same shape: {embed_arr.shape}")
        print(f"  Max absolute difference: {diff:.10f}")
        if diff < 1e-6:
            print(f"  => TIED (identical weights) — quantize once, share")
        else:
            print(f"  => NOT tied (different weights)")
            # Check correlation
            flat_e = embed_arr.flatten()
            flat_l = lm_head_arr.flatten()
            corr = np.corrcoef(flat_e[:100000], flat_l[:100000])[0, 1]
            print(f"  Correlation (sample): {corr:.6f}")
    else:
        print(f"  Different shapes: embed={embed_arr.shape}, lm_head={lm_head_arr.shape}")
        print(f"  => NOT tied")

# Router gate summary across layers
if router_stats:
    print(f"\n\n{'='*100}")
    print(f"ROUTER GATE ANALYSIS ACROSS LAYERS ({len(router_stats)} layers)")
    print(f"{'='*100}")

    all_router_means = []
    all_router_stds = []
    all_router_maxabs = []
    all_router_skew = []
    all_router_kurt = []
    all_router_row_cv = []
    all_router_col_cv = []

    for name, arr in router_stats:
        flat = arr.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        all_router_means.append(mean)
        all_router_stds.append(std)
        all_router_maxabs.append(np.max(np.abs(flat)))
        centered = (flat - mean) / std
        all_router_skew.append(np.mean(centered**3))
        all_router_kurt.append(np.mean(centered**4) - 3)
        if arr.ndim == 2:
            row_ranges = np.ptp(arr, axis=1)
            col_ranges = np.ptp(arr, axis=0)
            all_router_row_cv.append(np.std(row_ranges) / np.mean(row_ranges))
            all_router_col_cv.append(np.std(col_ranges) / np.mean(col_ranges))

    print(f"\n  Per-layer mean range: [{np.min(all_router_means):.6f}, {np.max(all_router_means):.6f}]")
    print(f"  Per-layer std range:  [{np.min(all_router_stds):.6f}, {np.max(all_router_stds):.6f}]")
    print(f"  Per-layer max|w| range: [{np.min(all_router_maxabs):.4f}, {np.max(all_router_maxabs):.4f}]")
    print(f"  Mean skewness: {np.mean(all_router_skew):.4f} (range: [{np.min(all_router_skew):.4f}, {np.max(all_router_skew):.4f}])")
    print(f"  Mean kurtosis: {np.mean(all_router_kurt):.4f} (range: [{np.min(all_router_kurt):.4f}, {np.max(all_router_kurt):.4f}])")

    if all_router_row_cv:
        print(f"\n  Row range CV (across experts): mean={np.mean(all_router_row_cv):.4f}")
        print(f"  Col range CV (across hidden):  mean={np.mean(all_router_col_cv):.4f}")
        print(f"  => {'Per-row (per-expert)' if np.mean(all_router_row_cv) < np.mean(all_router_col_cv) else 'Per-col (per-hidden-dim)'} is more uniform")

    # Check if router weights vary significantly across layers
    print(f"\n  Cross-layer consistency:")
    print(f"    Std of means: {np.std(all_router_means):.6f}")
    print(f"    Std of stds:  {np.std(all_router_stds):.6f}")
    print(f"    => {'Consistent' if np.std(all_router_stds)/np.mean(all_router_stds) < 0.1 else 'Variable'} across layers")

    # Router sensitivity note
    print(f"\n  QUANTIZATION NOTE for routers:")
    print(f"    Routers compute softmax(W @ x) to select experts.")
    print(f"    Small weight errors can change expert selection (top-k routing).")
    print(f"    Mean kurtosis={np.mean(all_router_kurt):.1f} indicates {'heavy tails' if np.mean(all_router_kurt) > 3 else 'moderate tails' if np.mean(all_router_kurt) > 1 else 'light tails'}.")
    max_ratio = np.mean(all_router_maxabs) / np.mean(all_router_stds)
    print(f"    Avg Max/Std = {max_ratio:.1f}x => {'HIGH outlier risk' if max_ratio > 15 else 'moderate outlier risk' if max_ratio > 10 else 'low outlier risk'}")
    print(f"    RECOMMENDATION: Keep router at higher precision (FP16 or INT8 with per-row scale)")
