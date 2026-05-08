#!/usr/bin/env python3
"""
PRT Phase 15B-A: INT4 Sidecar Offline Prototype and Parity Gate.

Goal: Determine whether selected Qwen2.5-7B FFN_UP sidecar weights can be
quantized to INT4 with acceptable offline parity versus the reference GGUF
weights, before attempting any runtime canary.

INT4 format:
- 2 weights per byte (nibble packing: high nibble = w[i], low nibble = w[i+1])
- Per-row scale: one float32 per M row (same as INT8 per-row approach)
- File layout: [M*K/2 bytes nibble data][M*4 bytes float32 scales]

Scale: scale[row] = max(|W[row][:]) / 7.0
Quantize: q[row][col] = round(W[row][col] / scale[row]), clamped to [-7, 7]
Dequantize: W_dq[row][col] = q[row][col] * scale[row]

Parity gate:
- weight_cosine >= 0.990 → INT4 SAFE for layer
- weight_cosine >= 0.980 → INT4 CONDITIONAL for layer
- weight_cosine < 0.980 → INT4 REJECTED for layer

If all probed layers pass SAFE or CONDITIONAL, runtime canary is justified.
If any layer passes REJECTED, INT4 prototype fails and should not proceed to runtime.
"""

import os, sys, struct, json, numpy as np, time

INT8_DIR = "/tmp/prt_sidecars_7b_int8/"
PROBE_OUT_DIR = "/tmp/prt_sidecars_7b_int4_probe"

# 7B model shape
N_LAYERS = 28
HIDDEN = 3584   # K = hidden = input dim
FFN = 18944     # M = ffn = intermediate dim


def get_layer_name(li):
    return f"ffn_up_layer{li}_prt"


def load_int8_reference(int8_dir, layer_idx, M, K):
    """Load reference from existing INT8 sidecar (dequantized to float32)."""
    name = get_layer_name(layer_idx)
    int8_path = os.path.join(int8_dir, f"{name}.int8")
    
    if not os.path.exists(int8_path):
        return None
    
    with open(int8_path, 'rb') as f:
        raw = np.frombuffer(f.read(M * K), dtype=np.int8)
        scales = np.frombuffer(f.read(M * 4), dtype=np.float32)
    
    # Dequantize: [M][K] row-major, per-row scale
    W_int8 = raw.reshape(M, K)
    W_ref = W_int8.astype(np.float32) * scales[:, np.newaxis]
    return W_ref


def quantize_to_int4_vectorized(W_ref, M, K):
    """
    Quantize float32 reference to INT4 with per-row scales.
    Vectorized implementation for speed with large matrices.
    
    Returns: (nibble_data, scales, info_dict)
    """
    # Per-row scale: max abs in row / 7
    row_max = np.abs(W_ref).max(axis=1)  # [M]
    scales = np.where(row_max > 1e-10, row_max / 7.0, 1.0)  # [M]
    
    # Quantize: q = round(W / scale), clamp to [-7, 7]
    W_q = np.clip(np.round(W_ref / scales[:, np.newaxis]), -7, 7).astype(np.int8)
    
    # Pack into nibbles: 2 per byte using vectorized reshape + bitwise
    # Layout: [M][K] row-major -> reshape to [M][K/2][2] -> interleave
    K_half = K // 2
    
    # Reshape to [M, K/2, 2], then pack
    W_reshaped = W_q[:, :K_half * 2].reshape(M, K_half, 2)  # [M, K/2, 2]
    
    # High nibble = col 0, Low nibble = col 1
    high = (W_reshaped[:, :, 0].astype(np.uint8) & 0x0F)  # [M, K/2]
    low = (W_reshaped[:, :, 1].astype(np.uint8) & 0x0F)   # [M, K/2]
    
    # Pack: high nibble shifted left = 4 bits
    packed = (high << 4) | low  # [M, K/2] uint8
    
    # Flatten to [M*K/2] contiguous
    nibble_data = packed.flatten()
    
    # Reconstruction check (for parity assessment)
    # Unpack nibbles back to int8
    high_unpack = (nibble_data >> 4).astype(np.int8)
    low_unpack = (nibble_data & 0x0F).astype(np.int8)
    
    # Sign-extend: if high >= 8, it's negative (range -8..-1)
    high_se = np.where(high_unpack >= 8, high_unpack - 16, high_unpack)
    low_se = np.where(low_unpack >= 8, low_unpack - 16, low_unpack)
    
    # Reconstruct: [M*K] row-major
    W_dq_flat = np.empty(M * K, dtype=np.float32)
    W_dq_flat[0::2] = high_se * scales.repeat(K_half)   # even indices
    W_dq_flat[1::2] = low_se * scales.repeat(K_half)    # odd indices
    W_dq = W_dq_flat.reshape(M, K)
    
    # Parity metrics
    weight_cosine = float(np.dot(W_ref.flatten(), W_dq.flatten()) / 
                          (np.linalg.norm(W_ref) * np.linalg.norm(W_dq) + 1e-8))
    max_err = float(np.abs(W_ref - W_dq).max())
    mean_err = float(np.abs(W_ref - W_dq).mean())
    
    return nibble_data, scales, {
        'weight_cosine': round(weight_cosine, 6),
        'max_abs_error': round(max_err, 6),
        'mean_abs_error': round(mean_err, 6),
    }


def probe_layer(layer_idx, M, K):
    """Probe one layer: load INT8 reference, quantize to INT4, measure parity."""
    W_ref = load_int8_reference(INT8_DIR, layer_idx, M, K)
    if W_ref is None:
        return {'layer': layer_idx, 'error': 'no INT8 reference found'}
    
    nibble_data, scales, q_info = quantize_to_int4_vectorized(W_ref, M, K)
    
    # Save probe sidecar
    name = get_layer_name(layer_idx)
    out_path = os.path.join(PROBE_OUT_DIR, f"{name}.int4")
    with open(out_path, 'wb') as f:
        f.write(nibble_data.tobytes())
        f.write(scales.astype(np.float32).tobytes())
    
    file_size = os.path.getsize(out_path)
    
    # Size comparison
    int8_size = M * K + M * 4        # int8 + scales
    fp32_size = M * K * 4           # float32
    int4_size = M * K // 2 + M * 4  # nibble + scales
    
    return {
        'layer': layer_idx,
        'weight_cosine': q_info['weight_cosine'],
        'max_abs_error': q_info['max_abs_error'],
        'mean_abs_error': q_info['mean_abs_error'],
        'fp32_bytes': fp32_size,
        'int8_bytes': int8_size,
        'int4_bytes': int4_size,
        'int4_vs_fp32_comp': round(fp32_size / int4_size, 3),
        'int4_vs_int8_comp': round(int8_size / int4_size, 3),
        'file_saved': out_path,
        'file_size': file_size,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', default='0,5,10,15,20,27', 
                        help='comma-separated layer indices to probe')
    parser.add_argument('--output-dir', default='/tmp/prt_sidecars_7b_int4_probe')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    layers = [int(l) for l in args.layers.split(',')]
    
    M, K = FFN, HIDDEN
    print(f"=== INT4 Sidecar Offline Prototype ===")
    print(f"Model: Qwen2.5-7B Q4_K_M (M={M}, K={K}, {N_LAYERS} layers)")
    print(f"Output: {args.output_dir}")
    print(f"Probing layers: {layers}")
    print()
    
    results = []
    for li in layers:
        print(f"--- Layer {li} ---")
        t0 = time.time()
        r = probe_layer(li, M, K)
        elapsed = time.time() - t0
        
        if 'error' in r:
            print(f"  ERROR: {r['error']}")
            results.append(r)
            continue
        
        print(f"  Weight cosine: {r['weight_cosine']}")
        print(f"  Max abs error: {r['max_abs_error']}")
        print(f"  Mean abs error: {r['mean_abs_error']}")
        print(f"  FP32 size: {r['fp32_bytes']/1e6:.1f}MB")
        print(f"  INT8 size: {r['int8_bytes']/1e6:.1f}MB")
        print(f"  INT4 size: {r['int4_bytes']/1e6:.1f}MB")
        print(f"  INT4 vs FP32: {r['int4_vs_fp32_comp']:.2f}x smaller")
        print(f"  INT4 vs INT8: {r['int4_vs_int8_comp']:.3f}x smaller")
        print(f"  Probe file: {r['file_saved']}")
        print(f"  Elapsed: {elapsed:.1f}s")
        results.append(r)
    
    # Parity gate assessment
    print(f"\n=== Parity Gate Assessment ===")
    safe = []
    conditional = []
    rejected = []
    
    for r in results:
        if 'error' in r:
            continue
        wc = r['weight_cosine']
        if wc >= 0.990:
            safe.append(r['layer'])
        elif wc >= 0.980:
            conditional.append(r['layer'])
        else:
            rejected.append(r['layer'])
        tag = 'SAFE' if wc >= 0.990 else 'CONDITIONAL' if wc >= 0.980 else 'REJECTED'
        print(f"  Layer {r['layer']}: weight_cosine={wc} → {tag}")
    
    print(f"\n  SAFE: {sorted(safe)}")
    print(f"  CONDITIONAL: {sorted(conditional)}")
    print(f"  REJECTED: {sorted(rejected)}")
    
    # Recommendation
    if rejected:
        verdict = "INT4_PROTOTYPE_FAILED"
        reason = f"Layers {rejected} failed parity gate (cosine < 0.980)"
        recommendation = "Do NOT proceed to runtime canary"
    elif conditional:
        verdict = "INT4_PROTOTYPE_CONDITIONAL"
        reason = f"Layers {conditional} are conditional (0.980 <= cosine < 0.990)"
        recommendation = "Runtime canary justified but monitor quality closely"
    else:
        verdict = "INT4_PROTOTYPE_PASSED"
        reason = f"All {len(safe)} layers passed SAFE (cosine >= 0.990)"
        recommendation = "Runtime canary justified"
    
    print(f"\n  Verdict: {verdict}")
    print(f"  Reason: {reason}")
    print(f"  Recommendation: {recommendation}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'int4_probe_results.json')
    summary = {
        'verdict': verdict,
        'reason': reason,
        'recommendation': recommendation,
        'layers_probed': layers,
        'safe_layers': sorted(safe),
        'conditional_layers': sorted(conditional),
        'rejected_layers': sorted(rejected),
        'layer_results': results,
    }
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {results_file}")
    
    return 0 if verdict == "INT4_PROTOTYPE_PASSED" else 1


if __name__ == '__main__':
    sys.exit(main())