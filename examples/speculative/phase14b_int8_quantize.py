#!/usr/bin/env python3
"""
PRT Phase 14B: Quantize float32 sidecars to INT8 per-row scale (fixed layout).

Sidecar file format (matches float32 PRT sidecar layout):
- File: [M][K] row-major float32 = [ffn][hidden]
- PRT kernel accesses: sidecar[j * K + k] → row j, element k
- Per-row scale: one scale per row j (one per ffn output dimension)
- Scale formula: scale[j] = max_k(|W[j][k]|) / 127.0
- Quantize: q[j][k] = round(W[j][k] / scale[j]), clamped to [-127, 127]
- Dequantize: W_dq[j][k] = q[j][k] * scale[j]
- Storage: [M*K bytes int8 data][M*4 bytes float32 scales] (same M*K+K*4 as original)

File naming: same as float32 (ffn_up_layer{N}_prt.bin → .int8)
"""

import os, json, numpy as np

def get_sidecar_name(layer_idx, suffix):
    if layer_idx < 10:
        return f'ffn_up_layer{layer_idx}_prt{suffix}'
    else:
        return f'ffn_up_layer{layer_idx:02d}_prt{suffix}'

def quantize_per_row(input_path, output_path, layer_idx, M, K):
    """
    Quantize float32 sidecar to INT8 with per-row scales.
    Input: [M][K] float32 row-major
    Output: [M][K] int8 + [M] float32 scales
    """
    # Read as [M][K]
    W = np.fromfile(input_path, dtype=np.float32).reshape(M, K)
    
    # Per-row scales: one scale per output dimension (row j = 0..M-1)
    row_max = np.abs(W).max(axis=1)  # [M]
    scales = np.where(row_max > 1e-8, row_max / 127.0, 1.0)  # [M]
    
    # Quantize: q[j][k] = round(W[j][k] / scale[j])
    W_q = np.clip(np.round(W / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    
    # Reconstruct for error check
    W_dq = W_q.astype(np.float32) * scales[:, np.newaxis]  # [M][K] * [M][1] → [M][K]
    diff = W - W_dq
    weight_cosine = float(np.dot(W.flatten(), W_dq.flatten()) / 
                          (np.linalg.norm(W) * np.linalg.norm(W_dq) + 1e-8))
    
    # Write: int8 data + scales
    with open(output_path, 'wb') as f:
        f.write(W_q.tobytes())
        f.write(scales.astype(np.float32).tobytes())
    
    int8_bytes = W_q.nbytes
    scale_bytes = scales.nbytes * 4  # float32
    total = int8_bytes + scale_bytes
    
    return {
        'layer': layer_idx, 'M': M, 'K': K,
        'fp32_bytes': W.nbytes,
        'int8_bytes': int8_bytes, 'scale_bytes': scale_bytes,
        'total_bytes': total,
        'compression': round(W.nbytes / total, 3),
        'weight_cosine': round(weight_cosine, 6),
        'max_abs_err': round(float(np.abs(diff).max()), 6),
        'mean_abs_err': round(float(np.abs(diff).mean()), 6),
    }

def matvec_parity(fp32_path, int8_path, M, K, n_iters=30):
    """Test matvec parity between float32 and INT8."""
    W_fp = np.fromfile(fp32_path, dtype=np.float32).reshape(M, K)
    
    with open(int8_path, 'rb') as f:
        int8_data = np.frombuffer(f.read(M * K), dtype=np.int8)
        scales = np.frombuffer(f.read(M * 4), dtype=np.float32)
    
    W_int8 = int8_data.reshape(M, K)
    W_dq = W_int8.astype(np.float32) * scales[:, np.newaxis]  # [M][K] * [M][1] → [M][K]
    
    np.random.seed(42)
    x = np.random.randn(K).astype(np.float32)
    
    # Time float32
    t_fp = []
    for _ in range(n_iters):
        t0 = __import__('time').time()
        y_fp = W_fp @ x
        t_fp.append((__import__('time').time() - t0) * 1000)
    t_fp = np.median(t_fp)
    
    # Time INT8
    t_i8 = []
    for _ in range(n_iters):
        t0 = __import__('time').time()
        y_i8 = W_dq @ x
        t_i8.append((__import__('time').time() - t0) * 1000)
    t_i8 = np.median(t_i8)
    
    y_fp = W_fp @ x
    y_i8 = W_dq @ x
    cosine = float(np.dot(y_fp, y_i8) / (np.linalg.norm(y_fp) * np.linalg.norm(y_i8) + 1e-8))
    rel_l2 = float(np.linalg.norm(y_fp - y_i8) / np.linalg.norm(y_fp))
    
    return {
        'fp32_ms': round(t_fp, 3), 'int8_ms': round(t_i8, 3),
        'ratio': round(t_i8 / t_fp, 3),
        'output_cosine': round(cosine, 6),
        'output_rel_l2': round(rel_l2, 6),
    }

if __name__ == '__main__':
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='/tmp/prt_sidecars_3b')
    parser.add_argument('--output-dir', default='/tmp/prt_sidecars_3b_int8')
    parser.add_argument('--model', choices=['3b', '05b'], default='3b')
    parser.add_argument('--layers', default='0,23,35')
    parser.add_argument('--mode', choices=['quantize', 'matvec', 'all'], default='all')
    parser.add_argument('--verify', action='store_true', help='verify existing sidecars')
    args = parser.parse_args()
    
    if args.model == '3b':
        M, K = 11008, 2048
    else:
        M, K = 4864, 896
    
    layers = [int(l) for l in args.layers.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== INT8 per-row quantized sidecars ===")
    print(f"Model: {args.model}, M={M}, K={K}")
    print(f"Output: {args.output_dir}")
    print(f"Layers: {layers}")
    print()
    
    for li in layers:
        fp = os.path.join(args.input_dir, get_sidecar_name(li, '.bin'))
        ip = os.path.join(args.output_dir, get_sidecar_name(li, '.int8'))
        
        if not os.path.exists(fp):
            print(f"Layer {li}: {fp} not found"); continue
        
        if args.mode in ['quantize', 'all']:
            r = quantize_per_row(fp, ip, li, M, K)
            print(f"Layer {li}: {r['compression']}x comp, "
                  f"fp32={r['fp32_bytes']/1e6:.1f}MB int8={r['total_bytes']/1e6:.1f}MB "
                  f"cosine={r['weight_cosine']}")
            res = r
        
        if args.mode in ['matvec', 'all', 'verify'] or args.verify:
            if os.path.exists(ip):
                m = matvec_parity(fp, ip, M, K)
                print(f"  matvec: FP32={m['fp32_ms']}ms INT8={m['int8_ms']}ms "
                      f"ratio={m['ratio']}x cosine={m['output_cosine']} relL2={m['output_rel_l2']}")
            else:
                print(f"  matvec: {ip} not found, skipping")
    
    print("\nDone.")