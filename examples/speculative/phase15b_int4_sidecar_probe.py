#!/usr/bin/env python3
"""
PRT Phase 15B: INT4 Sidecar Offline Probe
Direct GGUF extraction + INT4 quantization + parity metrics.

Reads ffn_up weights from GGUF Q4_K_M, quantizes to INT4 with per-row scales,
computes offline parity metrics. Does NOT rely on existing (buggy) INT8 sidecars.

INT4 format:
- 2 weights per byte (nibble packing)
- Per-row scale: float32 per ffn row (same as INT8 layout)
- File: [M*K/2 bytes nibble][M*4 bytes scales]  (M=ffn, K=hidden)
- Signed range: [-7, +7]
- Scale: scale[row] = row_max / 7.0
"""

import os, sys, struct, json, time
import numpy as np

# 7B model shape
N_LAYERS = 28
HIDDEN = 3584   # K = hidden = input dim
FFN = 18944      # M = ffn = intermediate dim
QK_K = 256       # Q4_K block size

MODEL_7B = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
PROBE_DIR = "/tmp/prt_phase15b_int4_probe"
os.makedirs(PROBE_DIR, exist_ok=True)


def read_gguf_tensor_metadata(path):
    """Read GGUF tensor metadata and return dict of tensor info."""
    tensors = {}
    with open(path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != 0x46554746:
            raise ValueError(f"Not a GGUF file: {hex(magic)}")
        version = struct.unpack('I', f.read(4))[0]
        num_tensors = struct.unpack('Q', f.read(8))[0]
        
        for _ in range(num_tensors):
            name_len = struct.unpack('I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8', errors='replace')
            n_dims = struct.unpack('I', f.read(4))[0]
            shape = [struct.unpack('Q', f.read(8))[0] for _ in range(n_dims)]
            while len(shape) < 4:
                shape.append(1)
            dtype = struct.unpack('I', f.read(4))[0]
            offset = struct.unpack('Q', f.read(8))[0]
            tensors[name] = {'shape': shape[:n_dims], 'dtype': dtype, 'offset': offset}
    
    return tensors


def dequantize_q4_k_block(block_bytes, block_size=QK_K):
    """Dequantize one Q4_K block (256 elements) to float32."""
    # Q4_K block layout (144 bytes for 256 elements):
    # [2] d = float16 delta
    # [2] dmin = float16 delta_min  
    # [16] scales (4 bits each, 8 values)
    # [128] nibbles (64 pairs)
    result = np.empty(block_size, dtype=np.float32)
    
    d_val = struct.unpack_from('<e', block_bytes, 0)[0]
    dmin_val = struct.unpack_from('<e', block_bytes, 2)[0]
    scales = block_bytes[4:20]
    
    qp = 0
    for j in range(block_size // 64):
        s0 = scales[j * 2]
        s1 = scales[j * 2 + 1]
        
        sc0 = s0 & 0xF
        m0 = (s0 >> 4) & 0xF
        sc1 = s1 & 0xF
        m1 = (s1 >> 4) & 0xF
        
        d1 = d_val * sc0
        m1v = dmin_val * m0
        d2 = d_val * sc1
        m2v = dmin_val * m1
        
        base_qp = 20 + j * 128
        
        # 32 low nibbles
        for l in range(32):
            q = (block_bytes[base_qp + l] & 0xF)
            result[qp] = d1 * q - m1v
            qp += 1
        # 32 high nibbles
        for l in range(32):
            q = (block_bytes[base_qp + l] >> 4)
            result[qp] = d2 * q - m2v
            qp += 1
    
    return result


def dequantize_ffn_up_layer(gguf_path, layer_idx, ffn, hidden):
    """
    Extract and dequantize ffn_up.weight for one layer from GGUF Q4_K_M.
    Returns: W_ref as [ffn][hidden] float32 row-major.
    """
    tensor_name = f"blk.{layer_idx}.ffn_up.weight"
    
    # Get tensor info
    tensors = read_gguf_tensor_metadata(gguf_path)
    if tensor_name not in tensors:
        raise ValueError(f"Tensor {tensor_name} not found. Available: {[k for k in tensors.keys() if 'ffn_up' in k][:5]}")
    
    info = tensors[tensor_name]
    shape = info['shape']  # [K, M] = [hidden, ffn] for weight
    offset = info['offset']
    
    # Shape should be [3584, 18944] for 7B
    if len(shape) != 2:
        raise ValueError(f"Unexpected shape {shape} for {tensor_name}")
    
    k_dim, m_dim = shape[0], shape[1]
    if k_dim != hidden or m_dim != ffn:
        raise ValueError(f"Shape mismatch: expected [{hidden}, {ffn}], got [{k_dim}, {m_dim}]")
    
    # Read Q4_K data
    n_elements = ffn * hidden
    n_blocks = n_elements // QK_K
    block_size_bytes = 144  # Q4_K block = 144 bytes
    
    with open(gguf_path, 'rb') as f:
        f.seek(offset)
        data = f.read(n_blocks * block_size_bytes)
    
    if len(data) != n_blocks * block_size_bytes:
        raise ValueError(f"Read {len(data)} bytes, expected {n_blocks * block_size_bytes}")
    
    # Dequantize each block
    W_rows = []
    for i in range(n_blocks):
        block = dequantize_q4_k_block(data[i * block_size_bytes : (i+1) * block_size_bytes])
        W_rows.append(block)
    
    # Stack into [ffn, hidden] — each block has 256 elements in hidden dim
    W = np.stack(W_rows, axis=0)  # [ffn, 256*blocks] = [ffn, hidden]
    return W  # [ffn][hidden] row-major


def quantize_to_int4(W, ffn, hidden):
    """
    Quantize float32 [ffn, hidden] to INT4 nibbles + per-row scales.
    Scale: row_max / 7.0 per row (row = ffn output dimension).
    Returns: (nibble_bytes, scales, metrics)
    """
    # Per-row scale
    row_max = np.abs(W).max(axis=1)  # [ffn]
    scales = np.where(row_max > 1e-10, row_max / 7.0, 1.0)  # [ffn]
    
    # Quantize
    W_q = np.clip(np.round(W / scales[:, np.newaxis]), -7, 7).astype(np.int8)
    
    # Pack nibbles
    k_half = hidden // 2
    W_clip = W_q[:, :k_half * 2].reshape(ffn, k_half, 2)
    high = (W_clip[:, :, 0].astype(np.uint8) & 0x0F)
    low = (W_clip[:, :, 1].astype(np.uint8) & 0x0F)
    nibble_bytes = ((high << 4) | low).flatten().tobytes()
    
    # Reconstruct for parity
    # Unpack nibbles
    high_u = np.frombuffer(nibble_bytes, dtype=np.uint8) >> 4
    low_u = np.frombuffer(nibble_bytes, dtype=np.uint8) & 0x0F
    high_s = np.where(high_u >= 8, high_u - 16, high_u)
    low_s = np.where(low_u >= 8, low_u - 16, low_u)
    
    W_dq = np.empty(ffn * hidden, dtype=np.float32)
    W_dq[0::2] = high_s * np.repeat(scales, k_half)
    W_dq[1::2] = low_s * np.repeat(scales, k_half)
    W_dq = W_dq.reshape(ffn, hidden)
    
    # Metrics
    weight_cos = float(np.dot(W.flatten(), W_dq.flatten()) / 
                       (np.linalg.norm(W) * np.linalg.norm(W_dq) + 1e-8))
    diff = W - W_dq
    max_err = float(np.abs(diff).max())
    mean_err = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff**2).mean()))
    
    zero_frac = float((W_q == 0).sum() / W_q.size)
    abs7_frac = float((np.abs(W_q) == 7).sum() / W_q.size)
    
    return nibble_bytes, scales, {
        'weight_cosine': round(weight_cos, 6),
        'max_abs_error': round(max_err, 6),
        'mean_abs_error': round(mean_err, 6),
        'rmse': round(rmse, 6),
        'zero_fraction': round(zero_frac, 4),
        'abs7_fraction': round(abs7_frac, 4),
    }


def matvec_parity(W_ref, scales, ffn, hidden, seeds=[42, 123, 456, 789, 1011, 2022, 3033, 4044]):
    """
    Compute matvec parity for multiple input vectors.
    Uses nibble unpack + scale multiply per row.
    """
    # Reconstruct W_dq from stored nibbles for speed
    k_half = hidden // 2
    scales_exp = np.repeat(scales, k_half)  # [ffn * k_half]
    
    results = []
    for seed in seeds:
        np.random.seed(seed)
        x = np.random.randn(hidden).astype(np.float32)
        
        y_ref = W_ref @ x
        y_dq = W_dq_from_nibbles_and_scales(scales, hidden, ffn, k_half, seeds=[seed])[0]
        
        cos = float(np.dot(y_ref, y_dq) / (np.linalg.norm(y_ref) * np.linalg.norm(y_dq) + 1e-8))
        rel_l2 = float(np.linalg.norm(y_ref - y_dq) / np.linalg.norm(y_ref))
        max_err = float(np.abs(y_ref - y_dq).max())
        
        results.append({
            'seed': seed,
            'cosine': round(cos, 6),
            'rel_l2': round(rel_l2, 6),
            'max_err': round(max_err, 6),
            'y_norm': round(float(np.linalg.norm(y_ref)), 4),
        })
    
    coses = [r['cosine'] for r in results]
    rels = [r['rel_l2'] for r in results]
    maxs = [r['max_err'] for r in results]
    
    return {
        'cosine_mean': round(float(np.mean(coses)), 6),
        'cosine_min': round(float(np.min(coses)), 6),
        'rel_l2_mean': round(float(np.mean(rels)), 6),
        'rel_l2_max': round(float(np.max(rels)), 6),
        'max_err_mean': round(float(np.mean(maxs)), 6),
        'max_err_max': round(float(np.max(maxs)), 6),
        'per_seed': results,
    }


# Global for matvec reuse
_cached_W_dq = None
_cached_scales = None
_cached_hidden = None
_cached_ffn = None
_cached_k_half = None


def _ensure_W_dq_cached(scales, hidden, ffn):
    global _cached_W_dq, _cached_scales, _cached_hidden, _cached_ffn, _cached_k_half
    if _cached_W_dq is not None and np.array_equal(_cached_scales, scales):
        return _cached_W_dq
    
    k_half = hidden // 2
    scales_exp = np.repeat(scales, k_half)
    
    # Load INT4 nibbles from probe file
    probe_path = os.path.join(PROBE_DIR, f"ffn_up_layer0_prt.int4")
    if os.path.exists(probe_path):
        with open(probe_path, 'rb') as f:
            nibble_bytes = f.read(ffn * k_half)
        high_u = np.frombuffer(nibble_bytes, dtype=np.uint8) >> 4
        low_u = np.frombuffer(nibble_bytes, dtype=np.uint8) & 0x0F
        high_s = np.where(high_u >= 8, high_u - 16, high_u)
        low_s = np.where(low_u >= 8, low_u - 16, low_u)
        W_flat = np.empty(ffn * hidden, dtype=np.float32)
        W_flat[0::2] = high_s * scales_exp
        W_flat[1::2] = low_s * scales_exp
        _cached_W_dq = W_flat.reshape(ffn, hidden)
    else:
        # Reconstruct from W_ref (callers must provide W_ref)
        return None
    
    _cached_scales = scales.copy()
    _cached_hidden = hidden
    _cached_ffn = ffn
    _cached_k_half = k_half
    return _cached_W_dq


def W_dq_from_nibbles_and_scales(scales, hidden, ffn, k_half, seeds=None):
    """Reconstruct dequantized W from INT4 nibbles and scales."""
    np.random.seed(seeds[0] if seeds else 42)
    x = np.random.randn(hidden).astype(np.float32)
    
    # Use W_ref directly for matvec
    return None  # Caller passes W_ref


def probe_layer(layer_idx, gguf_path, ffn, hidden):
    """Extract one layer from GGUF, quantize to INT4, measure parity."""
    print(f"  Layer {layer_idx}: extracting from GGUF...", end=" ", flush=True)
    t0 = time.time()
    
    try:
        W_ref = dequantize_ffn_up_layer(gguf_path, layer_idx, ffn, hidden)
    except Exception as e:
        return {'layer': layer_idx, 'error': str(e)}
    
    print(f"dequant={time.time()-t0:.1f}s ", end="", flush=True)
    
    # Quantize to INT4
    t1 = time.time()
    nibble_bytes, scales, q_info = quantize_to_int4(W_ref, ffn, hidden)
    print(f"quantize={time.time()-t1:.1f}s ", end="", flush=True)
    
    # Save INT4 probe file
    name = f"ffn_up_layer{layer_idx}_prt"
    out_path = os.path.join(PROBE_DIR, f"{name}.int4")
    with open(out_path, 'wb') as f:
        f.write(nibble_bytes)
        f.write(scales.astype(np.float32).tobytes())
    
    # Matvec parity (8 seeds)
    t2 = time.time()
    np.random.seed(42)
    seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044]
    mv_results = []
    for seed in seeds:
        np.random.seed(seed)
        x = np.random.randn(hidden).astype(np.float32)
        y_ref = W_ref @ x
        
        # Dequantize INT4 to float32 for matvec
        k_half = hidden // 2
        high_u = np.frombuffer(nibble_bytes, dtype=np.uint8) >> 4
        low_u = np.frombuffer(nibble_bytes, dtype=np.uint8) & 0x0F
        high_s = np.where(high_u >= 8, high_u.astype(np.int16) - 16, high_u.astype(np.int16)).astype(np.float32)
        low_s = np.where(low_u >= 8, low_u.astype(np.int16) - 16, low_u.astype(np.int16)).astype(np.float32)
        
        scales_exp = np.repeat(scales, k_half)
        W_flat = np.empty(ffn * hidden, dtype=np.float32)
        W_flat[0::2] = high_s * scales_exp
        W_flat[1::2] = low_s * scales_exp
        W_dq = W_flat.reshape(ffn, hidden)
        
        y_dq = W_dq @ x
        cos = float(np.dot(y_ref, y_dq) / (np.linalg.norm(y_ref) * np.linalg.norm(y_dq) + 1e-8))
        rel_l2 = float(np.linalg.norm(y_ref - y_dq) / np.linalg.norm(y_ref))
        max_err = float(np.abs(y_ref - y_dq).max())
        mv_results.append({'seed': seed, 'cosine': round(cos, 6), 'rel_l2': round(rel_l2, 6), 'max_err': round(max_err, 6)})
    
    print(f"matvec={time.time()-t2:.1f}s")
    
    coses = [r['cosine'] for r in mv_results]
    
    return {
        'layer': layer_idx,
        'finite': bool(np.isfinite(W_ref).all()),
        'weight_cosine': q_info['weight_cosine'],
        'max_abs_error': q_info['max_abs_error'],
        'mean_abs_error': q_info['mean_abs_error'],
        'rmse': q_info['rmse'],
        'zero_fraction': q_info['zero_fraction'],
        'abs7_fraction': q_info['abs7_fraction'],
        'scale_min': round(float(scales.min()), 8),
        'scale_max': round(float(scales.max()), 8),
        'scale_mean': round(float(scales.mean()), 8),
        'matvec_cosine_mean': round(float(np.mean(coses)), 6),
        'matvec_cosine_min': round(float(np.min(coses)), 6),
        'matvec_per_seed': mv_results,
        'int4_file': out_path,
        'int4_bytes': len(nibble_bytes) + len(scales) * 4,
    }


def main():
    layers = [0, 1, 10, 11, 15, 20, 27]
    
    print(f"=== INT4 Offline Parity Probe ===")
    print(f"Model: Qwen2.5-7B Q4_K_M")
    print(f"Shape: ffn={FFN}, hidden={HIDDEN}")
    print(f"Layers: {layers}")
    print(f"GGUF: {MODEL_7B}")
    print(f"Output: {PROBE_DIR}")
    print()
    
    # Verify GGUF
    tensors = read_gguf_tensor_metadata(MODEL_7B)
    sample = tensors.get(f"blk.0.ffn_up.weight")
    print(f"Sample tensor blk.0.ffn_up.weight: {sample}")
    
    results = []
    for li in layers:
        print(f"--- Layer {li} ---")
        r = probe_layer(li, MODEL_7B, FFN, HIDDEN)
        if 'error' in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Weight cosine:    {r['weight_cosine']}")
            print(f"  Matvec cos mean: {r['matvec_cosine_mean']}, min: {r['matvec_cosine_min']}")
            print(f"  MAE: {r['mean_abs_error']}, Max: {r['max_abs_error']}")
            print(f"  Zero frac: {r['zero_fraction']}, |7| frac: {r['abs7_fraction']}")
            print(f"  Scales: [{r['scale_min']:.2e}, {r['scale_max']:.2e}], mean={r['scale_mean']:.2e}")
        results.append(r)
    
    # Assessment
    print(f"\n=== Assessment ===")
    min_wc = min(r['weight_cosine'] for r in results if 'error' not in r)
    min_mc = min(r['matvec_cosine_min'] for r in results if 'error' not in r)
    
    print(f"Min weight cosine: {min_wc}")
    print(f"Min matvec cosine: {min_mc}")
    
    # Per-layer verdict
    for r in results:
        if 'error' in r:
            print(f"  L{r['layer']}: ERROR")
            continue
        wc = r['weight_cosine']
        mc = r['matvec_cosine_min']
        if mc >= 0.995 and wc >= 0.995:
            tag = "PASS"
        elif mc >= 0.990 and wc >= 0.980:
            tag = "MAYBE"
        else:
            tag = "NO-GO"
        print(f"  L{r['layer']}: WC={wc} MC_min={mc} → {tag}")
    
    # Overall verdict
    all_pass = all(
        r.get('matvec_cosine_min', 0) >= 0.995 and r.get('weight_cosine', 0) >= 0.995
        for r in results if 'error' not in r
    )
    all_maybe_or_better = all(
        r.get('matvec_cosine_min', 0) >= 0.990 and r.get('weight_cosine', 0) >= 0.980
        for r in results if 'error' not in r
    )
    
    if all_pass:
        verdict = "PASS_INT4_OFFLINE_PARITY"
    elif all_maybe_or_better:
        verdict = "MAYBE_INT4_OFFLINE_PARITY"
    else:
        verdict = "NO_GO_INT4_OFFLINE_PARITY"
    
    print(f"\n  Verdict: {verdict}")
    
    # Save results
    out_json = os.path.join(PROBE_DIR, 'int4_offline_parity_results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'verdict': verdict,
            'layers': layers,
            'model': 'Qwen2.5-7B-Instruct-Q4_K_M',
            'shape': {'ffn': FFN, 'hidden': HIDDEN},
            'int4_format': {
                'signed_range': '[-7, +7]',
                'scale_formula': 'row_max / 7.0',
                'packing': '2 nibbles per byte, row-major',
                'scale_storage': 'float32 per ffn row',
                'file_layout': '[M*K/2 bytes nibble][M*4 bytes scales]',
                'runtime_compatibility': 'NOT attempted in Phase 15B-A'
            },
            'results': results,
            'min_weight_cosine': min_wc,
            'min_matvec_cosine': min_mc,
        }, f, indent=2)
    
    print(f"Results: {out_json}")
    return 0 if verdict.startswith("PASS") else 1


if __name__ == '__main__':
    sys.exit(main())