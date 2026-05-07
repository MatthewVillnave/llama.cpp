#!/usr/bin/env python3
"""
PRT Phase 14A: Quantize float32 sidecars to INT8 per-row scale.

Sidecar layout: [ffn][hidden] row-major float32 binary.
For matvec y = W @ x where W is [hidden][ffn]:
  - Sidecar: stored as [ffn][hidden] = [M][K]
  - To get W [K][M] for matmul: transpose on read
  
Scale granularity: per hidden dimension (per row of [M][K] sidecar).
Each row i (hidden dim i) has one scale[i] applied to all ffn elements.

File format: 
  [M*K bytes int8 data] + [K*4 bytes float32 scales]
"""

import os, sys, json, numpy as np

def get_sidecar_name(layer_idx, suffix):
    if layer_idx < 10:
        return f'ffn_up_layer{layer_idx}_prt{suffix}'
    else:
        return f'ffn_up_layer{layer_idx:02d}_prt{suffix}'

def quantize_sidecar(input_path, output_path, layer_idx, hidden, ffn):
    """
    Quantize float32 sidecar to INT8 with per-row (per hidden dim) scales.
    Sidecar format: [M][K] row-major = [ffn][hidden]
    Scale format: one scale per row (K scales total, one per hidden dim)
    """
    M, K = ffn, hidden
    
    # Read sidecar as [M][K]
    data_fp32 = np.fromfile(input_path, dtype=np.float32).reshape(M, K)
    
    # Per-row scales: for each hidden dim i, scale[i] = max_j(|W[j][i]|) / 127
    row_max = np.abs(data_fp32).max(axis=1)  # shape [M] - max over ffn for each hidden dim
    # Wait - row_max over axis=1 gives [M] but we want [K]
    # Let's reconsider: data_fp32 is [M][K]. axis=1 is over K (hidden), gives max per row = [M]
    # But we want per hidden dim, not per ffn output
    
    # Actually: data_fp32[i][j] = weight for ffn output i, hidden dim j
    # For per hidden dim scale: need max over i for each j
    col_max = np.abs(data_fp32).max(axis=0)  # shape [K] - max over ffn for each hidden dim
    
    scales = np.where(col_max > 1e-8, col_max / 127.0, 1.0)  # shape [K]
    
    # Quantize per column: W[j][i] / scale[i]
    W_quant = np.round(data_fp32 / scales).astype(np.int8)
    W_quant = np.clip(W_quant, -127, 127)
    
    # Write: int8 data + scales (float32)
    with open(output_path, 'wb') as f:
        f.write(W_quant.tobytes())
        f.write(scales.astype(np.float32).tobytes())
    
    # Reconstruction check
    W_dq = W_quant.astype(np.float32) * scales  # [M][K] * [K] broadcast = [M][K]
    # Note: this broadcast works because scales is [K] and W_quant is [M][K]
    # W_dq[j][i] = W_quant[j][i] * scales[i]
    
    diff = data_fp32 - W_dq
    max_err = float(np.abs(diff).max())
    mean_err = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff**2).mean()))
    
    weight_cosine = float(np.dot(data_fp32.flatten(), W_dq.flatten()) / 
                          (np.linalg.norm(data_fp32) * np.linalg.norm(W_dq) + 1e-8))
    
    total_bytes = W_quant.nbytes + scales.nbytes
    compression = float(data_fp32.nbytes / total_bytes)
    
    return {
        'layer': layer_idx, 'hidden': hidden, 'ffn': ffn,
        'fp32_bytes': int(data_fp32.nbytes), 'int8_bytes': int(W_quant.nbytes),
        'scale_bytes': int(scales.nbytes), 'total_bytes': int(total_bytes),
        'compression': round(compression, 3),
        'weight_cosine': round(weight_cosine, 6),
        'max_abs_error': round(max_err, 6), 'mean_abs_error': round(mean_err, 6),
        'rmse': round(rmse, 6),
    }

def matvec_test(fp32_path, int8_path, H, M, n_iters=30):
    """Test matvec parity and timing."""
    np.random.seed(42)
    x = np.random.randn(H).astype(np.float32)
    
    # Float32
    W_fp = np.fromfile(fp32_path, dtype=np.float32).reshape(M, H)
    t_fp = []
    for _ in range(n_iters):
        t0 = time.time()
        y = W_fp @ x
        t_fp.append((time.time() - t0) * 1000)
    t_fp = np.median(t_fp)
    
    # INT8
    with open(int8_path, 'rb') as f:
        int8_data = np.frombuffer(f.read(M * H), dtype=np.int8)
        scales = np.frombuffer(f.read(H * 4), dtype=np.float32)
    W_int8 = int8_data.reshape(M, H)
    W_dq = W_int8.astype(np.float32) * scales  # broadcast [M][H] * [H] = [M][H]
    
    t_i8 = []
    for _ in range(n_iters):
        t0 = time.time()
        y = W_dq @ x
        t_i8.append((time.time() - t0) * 1000)
    t_i8 = np.median(t_i8)
    
    y_fp = W_fp @ x
    y_dq = W_dq @ x
    cosine = float(np.dot(y_fp, y_dq) / (np.linalg.norm(y_fp) * np.linalg.norm(y_dq) + 1e-8))
    rel_l2 = float(np.linalg.norm(y_fp - y_dq) / np.linalg.norm(y_fp))
    
    return {'fp32_ms': round(t_fp, 3), 'int8_ms': round(t_i8, 3),
            'ratio': round(t_i8 / t_fp, 3), 'cosine': round(cosine, 6), 'rel_l2': round(rel_l2, 6)}

if __name__ == '__main__':
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='/tmp/prt_sidecars_3b')
    parser.add_argument('--output-dir', default='/tmp/prt_sidecars_3b_int8')
    parser.add_argument('--model', choices=['3b', '05b'], default='3b')
    parser.add_argument('--layers', default='0,23,35')
    parser.add_argument('--mode', choices=['quantize', 'matvec', 'all'], default='all')
    args = parser.parse_args()
    
    H, M = (2048, 11008) if args.model == '3b' else (896, 4864)
    layers = [int(l) for l in args.layers.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    for li in layers:
        fp = os.path.join(args.input_dir, get_sidecar_name(li, '.bin'))
        ip = os.path.join(args.output_dir, get_sidecar_name(li, '.int8'))
        if not os.path.exists(fp):
            print(f"Layer {li}: not found"); continue
        
        print(f"\n=== Layer {li} ===")
        if args.mode in ['quantize', 'all']:
            r = quantize_sidecar(fp, ip, li, H, M)
            print(f"  Compression: {r['compression']}x, Weight cosine: {r['weight_cosine']}")
            print(f"  Max error: {r['max_abs_error']}, Mean error: {r['mean_abs_error']}")
            results.append(r)
        
        if args.mode in ['matvec', 'all']:
            m = matvec_test(fp, ip, H, M)
            print(f"  FP32: {m['fp32_ms']}ms, INT8: {m['int8_ms']}ms, Ratio: {m['ratio']}x")
            print(f"  Output cosine: {m['cosine']}, Rel L2: {m['rel_l2']}")
            if results and results[-1]['layer'] == li:
                results[-1]['matvec'] = m
            else:
                results.append({'layer': li, 'matvec': m})
    
    print(f"\n=== Summary ({args.model} {H}x{M}) ===")
    for r in results:
        print(f"Layer {r['layer']}: comp={r.get('compression','?')}x cos={r.get('weight_cosine', r['matvec']['cosine'])} "
              f"fp32={r['matvec']['fp32_ms']}ms int8={r['matvec']['int8_ms']}ms ratio={r['matvec']['ratio']}x")