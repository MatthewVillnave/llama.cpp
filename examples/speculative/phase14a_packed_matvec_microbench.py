#!/usr/bin/env python3
"""
PRT Phase 14A: Packed Matvec Microbench
Compare float32 matvec vs int8 dequant matvec on PRT sidecar layers.
"""
import os, sys, time, json, numpy as np

def get_sidecar_name(layer_idx, suffix):
    if layer_idx < 10:
        return f'ffn_up_layer{layer_idx}_prt{suffix}'
    else:
        return f'ffn_up_layer{layer_idx:02d}_prt{suffix}'

def run_benchmark(fp32_dir, int8_dir, model, layers, n_iters=20):
    H, M = (2048, 11008) if model == '3b' else (896, 4864)
    
    results = []
    for layer_idx in layers:
        fp32_name = get_sidecar_name(layer_idx, '.bin')
        int8_name = get_sidecar_name(layer_idx, '.int8')
        fp32_path = os.path.join(fp32_dir, fp32_name)
        int8_path = os.path.join(int8_dir, int8_name)
        
        if not os.path.exists(fp32_path):
            print(f"Layer {layer_idx}: FP32 not found")
            continue
        if not os.path.exists(int8_path):
            print(f"Layer {layer_idx}: INT8 not found")
            continue
        
        np.random.seed(42 + layer_idx)
        x = np.random.randn(H).astype(np.float32)
        
        # Float32 sidecar: [M][K] stored. We need [K][M] for matvec W @ x.
        fp32_data = np.fromfile(fp32_path, dtype=np.float32).reshape(M, H).T  # [K][M]
        times = []
        for _ in range(n_iters):
            t0 = time.time()
            y_fp32 = fp32_data @ x
            times.append((time.time() - t0) * 1000)
        t_fp32 = np.median(times)
        
        # INT8 sidecar: [M][K] int8 + [M] float32 scales
        with open(int8_path, 'rb') as f:
            int8_arr = np.frombuffer(f.read(H * M), dtype=np.int8)
            scales = np.frombuffer(f.read(M * 4), dtype=np.float32)
        W_int8 = int8_arr.reshape(M, H)  # [M][K]
        W_dq = (W_int8.T.astype(np.float32)) * scales  # [K][M]
        
        times = []
        for _ in range(n_iters):
            t0 = time.time()
            y_int8 = W_dq @ x
            times.append((time.time() - t0) * 1000)
        t_int8 = np.median(times)
        
        # Compare outputs
        cosine = np.dot(y_fp32, y_int8) / (np.linalg.norm(y_fp32) * np.linalg.norm(y_int8) + 1e-8)
        rel_l2 = np.linalg.norm(y_fp32 - y_int8) / (np.linalg.norm(y_fp32) + 1e-8)
        
        fp32_bytes = H * M * 4
        int8_bytes = H * M * 1 + M * 4
        compression = fp32_bytes / int8_bytes
        
        print(f"Layer {layer_idx}: FP32={t_fp32:.3f}ms INT8={t_int8:.3f}ms ratio={t_int8/t_fp32:.3f}x "
              f"cosine={cosine:.6f} comp={compression:.3f}x")
        
        results.append({
            'layer': layer_idx,
            'fp32_ms': round(t_fp32, 3),
            'int8_ms': round(t_int8, 3),
            'ratio': round(t_int8 / t_fp32, 3),
            'compression': round(compression, 3),
            'fp32_bytes': fp32_bytes,
            'int8_bytes': int8_bytes,
            'output_cosine': round(cosine, 6),
            'output_rel_l2': round(rel_l2, 6),
        })
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['3b', '05b'], default='3b')
    parser.add_argument('--fp32-dir', default='/tmp/prt_sidecars_3b')
    parser.add_argument('--int8-dir', default='/tmp/prt_sidecars_3b_int8')
    parser.add_argument('--layers', default='0,23,35')
    parser.add_argument('--n-iters', type=int, default=20)
    parser.add_argument('--output-json')
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(',')]
    
    print("=== Phase 14A: Packed Matvec Microbench ===")
    model = args.model
    H, M = (2048, 11008) if model == '3b' else (896, 4864)
    print(f"Model: {model} ({H}x{M}), layers: {layers}, n_iters: {args.n_iters}")
    
    results = run_benchmark(args.fp32_dir, args.int8_dir, model, layers, args.n_iters)
    
    avg_fp32 = np.mean([r['fp32_ms'] for r in results])
    avg_int8 = np.mean([r['int8_ms'] for r in results])
    avg_cosine = np.mean([r['output_cosine'] for r in results])
    avg_comp = np.mean([r['compression'] for r in results])
    
    print(f"\nAggregate: FP32 avg={avg_fp32:.3f}ms INT8 avg={avg_int8:.3f}ms "
          f"ratio={avg_int8/avg_fp32:.3f}x cosine={avg_cosine:.6f} comp={avg_comp:.3f}x")
    
    if args.output_json:
        output = {'model': model, 'shape': f'{H}x{M}', 'layers': layers,
                  'n_iters': args.n_iters, 'results': results,
                  'aggregate': {'avg_fp32_ms': round(avg_fp32,3), 'avg_int8_ms': round(avg_int8,3),
                                'avg_ratio': round(avg_int8/avg_fp32,3), 'avg_cosine': round(avg_cosine,6),
                                'avg_compression': round(avg_comp,3)}}
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Written: {args.output_json}")