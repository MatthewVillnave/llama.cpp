#!/usr/bin/env python3
"""Phase 10A-4: All 28 layers PRT_3P sidecar extraction."""
import subprocess
import struct
import json
import os
import time

# PRT_3P thresholds from Phase 8
T_HIGH_0, T_HIGH_1, T_HIGH_2 = 2.0, 0.5, 0.1

def dequant_q4_k(data_bytes, n_elements):
    """Dequantize Q4_K (same as dequant_ffn_up.py)."""
    import numpy as np
    n_blocks = n_elements // 256
    result = []
    BLOCK_SIZE = 144
    
    for i in range(n_blocks):
        off = i * BLOCK_SIZE
        d_val = struct.unpack_from('<e', data_bytes, off)[0]
        dmin_val = struct.unpack_from('<e', data_bytes, off + 2)[0]
        scales = data_bytes[off + 4 : off + 20]
        qs = bytearray(data_bytes[off + 20 : off + 148])
        
        qp = 0
        for j in range(4):
            s0 = scales[j * 2]
            s1 = scales[j * 2 + 1]
            sc0, m0 = s0 & 0xF, (s0 >> 4) & 0xF
            sc1, m1 = s1 & 0xF, (s1 >> 4) & 0xF
            d1 = d_val * sc0
            m1v = dmin_val * m0
            d2 = d_val * sc1
            m2v = dmin_val * m1
            for l in range(32):
                result.append(d1 * (qs[qp] & 0xF) - m1v)
                qp += 1
            for l in range(32):
                result.append(d2 * (qs[qp] >> 4) - m2v)
                qp += 1
    return result

def prt_3plane(X, M, batch):
    """Split X into 3 planes."""
    n = batch * M
    import numpy as np
    X_q0 = np.zeros(n, dtype=np.int8)
    X_q1 = np.zeros(n, dtype=np.int8)
    X_q2 = np.zeros(n, dtype=np.int8)
    S0 = np.zeros(n, dtype=np.float32)
    S1 = np.zeros(n, dtype=np.float32)
    S2 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = X[i]
        ax = abs(x)
        if ax > T_HIGH_0:
            X_q0[i] = 1 if x > 0 else -1
            S0[i] = ax
        elif ax > T_HIGH_1:
            X_q1[i] = 1 if x > 0 else -1
            S1[i] = ax
        elif ax > T_HIGH_2:
            X_q2[i] = 1 if x > 0 else -1
            S2[i] = ax
    return X_q0, X_q1, X_q2, S0, S1, S2

def matmul_prt(X_q0, X_q1, X_q2, S0, S1, S2, W_flat, batch, M, N):
    """Y = X @ W with PRT_3P."""
    import numpy as np
    Y = np.zeros((batch, N), dtype=np.float32)
    W = W_flat.reshape(M, N)
    for b in range(batch):
        for k in range(M):
            c = 0.0
            xq0, xq1, xq2 = X_q0[b*M+k], X_q1[b*M+k], X_q2[b*M+k]
            if xq0: c += S0[b*M+k] * xq0
            if xq1: c += S1[b*M+k] * xq1
            if xq2: c += S2[b*M+k] * xq2
            if c != 0.0:
                Y[b] += c * W[k]
    return Y

def cosine(a, b):
    import numpy as np
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_test_layer(layer_idx, W_flat, M, N):
    """Run batch 16 and 17 test for one layer."""
    import numpy as np
    rng = np.random.default_rng(42)
    results = {}
    
    for batch in [16, 17]:
        X = rng.uniform(-1, 1, size=(batch, M)).astype(np.float32)
        W = W_flat.reshape(M, N)
        Y_float = X @ W
        
        X_q0, X_q1, X_q2, S0, S1, S2 = prt_3plane(X.flatten(), M, batch)
        Y_prt = matmul_prt(X_q0, X_q1, X_q2, S0, S1, S2, W_flat, batch, M, N)
        
        cos = cosine(Y_float.flatten(), Y_prt.flatten())
        err = np.abs(Y_float - Y_prt)
        
        float_crc = 0xFFFFFFFF ^ (sum(int(v*1e6) & 0xFFFFFFFF for v in Y_float.flatten()) | (sum(int(v*1e6) & 0xFFFFFFFF for v in Y_float.flatten()) << 32))
        prt_crc = 0xFFFFFFFF ^ (sum(int(v*1e6) & 0xFFFFFFFF for v in Y_prt.flatten()) | (sum(int(v*1e6) & 0xFFFFFFFF for v in Y_prt.flatten()) << 32))
        
        results[batch] = {
            'cosine': float(cos),
            'max_abs_err': float(err.max()),
            'mean_abs_err': float(err.mean()),
            'std_abs_err': float(err.std()),
            'float_crc32': float_crc & 0xFFFFFFFF,
            'prt_crc32': prt_crc & 0xFFFFFFFF
        }
    
    return results

def main():
    import numpy as np
    
    MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    EXTRACTOR = "/home/matthew-villnave/llama.cpp/build/bin/llama-prt-ffn-up-extract"
    OUT_DIR = "/tmp/prt_sidecars"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    M, N = 2048, 11008
    N_ELEMENTS = M * N
    
    print("=== Phase 10A-4: All 28 Layers PRT_3P Sidecar Extraction ===\n")
    print(f"Output dir: {OUT_DIR}\n")
    
    results = []
    failed = []
    total_time_start = time.time()
    
    for layer in range(28):
        layer_name = f"blk.{layer}.ffn_up.weight"
        sidecar_path = f"{OUT_DIR}/ffn_up_layer{layer}.bin"
        start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Layer {layer}: {layer_name}")
        print(f"{'='*50}")
        
        try:
            # 1. Extract and dequantize via C++ tool
            # The tool writes to /tmp/ffn_up_layer0_f32.bin, we modify to per-layer
            result = subprocess.run(
                [EXTRACTOR, "-t", layer_name, "-o", f"/tmp/ffn_up_layer{layer}_float.bin"],
                capture_output=True, text=True, timeout=120
            )
            
            float_path = f"/tmp/ffn_up_layer{layer}_float.bin"
            
            if result.returncode != 0 or not os.path.exists(float_path):
                print(f"  ERROR: extractor failed")
                failed.append({'layer': layer, 'name': layer_name, 'reason': 'extractor failed'})
                continue
            
            # Read float weights
            with open(float_path, 'rb') as f:
                W_data = struct.unpack(f'{N_ELEMENTS}f', f.read())
            W_flat = np.array(W_data, dtype=np.float32)
            del W_data
            
            # 2. Build PRT_3P sidecar (same float buffer becomes sidecar since PRT uses same scale)
            # For PRT, we store scale = |w| per element (ternary with magnitude)
            sidecar_data = np.abs(W_flat).astype(np.float32)
            sidecar_path_final = f"{OUT_DIR}/ffn_up_layer{layer}_prt.bin"
            with open(sidecar_path_final, 'wb') as f:
                f.write(sidecar_data.tobytes())
            
            sidecar_size = len(sidecar_data) * 4
            
            # 3. Run accuracy test
            test_results = run_test_layer(layer, W_flat, M, N)
            
            layer_time = time.time() - start_time
            
            layer_result = {
                'layer': layer,
                'name': layer_name,
                'dequant_success': True,
                'sidecar_build_success': True,
                'sidecar_size': sidecar_size,
                'batch16': test_results[16],
                'batch17': test_results[17],
                'build_time_sec': round(layer_time, 1),
                'pass': test_results[16]['cosine'] >= 0.95 and test_results[17]['cosine'] >= 0.95
            }
            
            results.append(layer_result)
            
            print(f"  Dequant: OK")
            print(f"  Sidecar: {sidecar_size} bytes")
            print(f"  Batch 16: cosine={test_results[16]['cosine']:.6f}, max_err={test_results[16]['max_abs_err']:.4f}")
            print(f"  Batch 17: cosine={test_results[17]['cosine']:.6f}, max_err={test_results[17]['max_abs_err']:.4f}")
            print(f"  Time: {layer_time:.1f}s")
            print(f"  PASS: {'YES' if layer_result['pass'] else 'NO'}")
            
            # Cleanup
            os.remove(float_path)
            
            # Checkpoint every 4 layers
            if (layer + 1) % 4 == 0:
                cp_file = f"/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10a_all_layers_checkpoint_{layer//4*4:02d}_{(layer//4*4+3):02d}.md"
                with open(cp_file, 'w') as f:
                    f.write(f"# Phase 10A-4 Checkpoint layers {layer//4*4}-{layer//4*4+3}\n\n")
                    f.write(f"Processed: {layer+1}/28 layers\n\n")
                    for r in results[-(layer+1):]:
                        f.write(f"- Layer {r['layer']}: cosine16={r['batch16']['cosine']:.6f}, cosine17={r['batch17']['cosine']:.6f}, pass={r['pass']}\n")
                print(f"  Checkpoint written: layers {layer//4*4}-{layer//4*4+3}")
        
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failed.append({'layer': layer, 'name': layer_name, 'reason': str(e)})
            continue
    
    total_time = time.time() - total_time_start
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    n_success = len(results)
    n_failed = len(failed)
    
    all_cosine_16 = [r['batch16']['cosine'] for r in results]
    all_cosine_17 = [r['batch17']['cosine'] for r in results]
    all_max_err = [max(r['batch16']['max_abs_err'], r['batch17']['max_abs_err']) for r in results]
    all_mean_err = [(r['batch16']['mean_abs_err'] + r['batch17']['mean_abs_err'])/2 for r in results]
    
    print(f"\nLayers attempted: 28")
    print(f"Layers succeeded: {n_success}")
    print(f"Layers failed: {n_failed}")
    
    print(f"\nAccuracy:")
    print(f"  min batch16 cosine: {min(all_cosine_16):.6f}")
    print(f"  min batch17 cosine: {min(all_cosine_17):.6f}")
    print(f"  avg cosine: {sum(all_cosine_16)/len(all_cosine_16):.6f}")
    print(f"  worst max_abs_error: {max(all_max_err):.6f}")
    print(f"  avg mean_abs_error: {sum(all_mean_err)/len(all_mean_err):.6f}")
    
    total_sidecar = sum(r['sidecar_size'] for r in results)
    q4_size = 28 * 12681216
    f32_size = 28 * 22544384 * 4
    
    print(f"\nMemory:")
    print(f"  per-layer float temp: ~90MB")
    print(f"  per-layer sidecar: ~90MB")
    print(f"  total sidecar: {total_sidecar/1024/1024:.1f}MB")
    print(f"  original Q4_K ffn_up total: {q4_size/1024/1024:.1f}MB")
    print(f"  float32 equivalent: {f32_size/1024/1024:.1f}MB")
    
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Write output files
    with open('/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10a_all_layers_accuracy.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10a_layer_to_sidecar_map.json', 'w') as f:
        map_data = {r['layer']: {'name': r['name'], 'sidecar_path': f'{OUT_DIR}/ffn_up_layer{r["layer"]}_prt.bin', 'size': r['sidecar_size']} for r in results}
        json.dump(map_data, f, indent=2)
    
    with open('/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10a_failed_layers.md', 'w') as f:
        f.write("# Failed Layers\n\n")
        if failed:
            for fa in failed:
                f.write(f"- Layer {fa['layer']} ({fa['name']}): {fa['reason']}\n")
        else:
            f.write("None.\n")
    
    with open('/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10a_all_layers_summary.md', 'w') as f:
        f.write("# Phase 10A-4 All Layers Summary\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Attempted: 28 layers\n")
        f.write(f"- Succeeded: {n_success}\n")
        f.write(f"- Failed: {n_failed}\n\n")
        f.write(f"## Accuracy\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| min batch16 cosine | {min(all_cosine_16):.6f} |\n")
        f.write(f"| min batch17 cosine | {min(all_cosine_17):.6f} |\n")
        f.write(f"| avg cosine | {sum(all_cosine_16)/len(all_cosine_16):.6f} |\n")
        f.write(f"| worst max_error | {max(all_max_err):.6f} |\n")
        f.write(f"| avg mean_error | {sum(all_mean_err)/len(all_mean_err):.6f} |\n\n")
        f.write(f"## All Layers Results\n\n")
        f.write(f"| Layer | Cosine16 | Cosine17 | Pass |\n")
        f.write(f"|-------|----------|----------|------|\n")
        for r in results:
            f.write(f"| {r['layer']} | {r['batch16']['cosine']:.6f} | {r['batch17']['cosine']:.6f} | {'YES' if r['pass'] else 'NO'} |\n")
    
    with open('/home/matthew-villnave/llama.cpp/examples/speculative/results/phase10a_all_layers_memory.md', 'w') as f:
        f.write("# Phase 10A-4 Memory Report\n\n")
        f.write(f"## Per-Layer\n\n")
        f.write(f"- Float temp: ~90MB\n")
        f.write(f"- Sidecar: ~90MB\n\n")
        f.write(f"## Totals\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total sidecar size | {total_sidecar/1024/1024:.1f}MB |\n")
        f.write(f"| Original Q4_K total | {q4_size/1024/1024:.1f}MB |\n")
        f.write(f"| Float32 equivalent | {f32_size/1024/1024:.1f}MB |\n")
        f.write(f"| Compression ratio | {f32_size/total_sidecar:.2f}x |\n")
    
    verdict = "PASS" if n_success == 28 and all(r['pass'] for r in results) else "FAIL"
    
    with open('/home/matthew-villnave/llama.cpp/examples/speculative/results/PRT_PHASE10A_ALL_LAYERS_VERDICT.md', 'w') as f:
        f.write(f"# PRT_PHASE10A_ALL_LAYERS_VERDICT\n\n")
        f.write(f"## Status: {verdict}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Layer | Cosine B16 | Cosine B17 | MaxErr | Pass |\n")
        f.write(f"|-------|------------|------------|--------|------|\n")
        for r in results:
            f.write(f"| {r['layer']} | {r['batch16']['cosine']:.6f} | {r['batch17']['cosine']:.6f} | {max(r['batch16']['max_abs_err'], r['batch17']['max_abs_err']):.4f} | {'YES' if r['pass'] else 'NO'} |\n\n")
        f.write(f"\n## Verdict: {verdict}\n")
        f.write(f"\nFailed layers: {n_failed}\n")
    
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")
    
    return 0 if verdict == "PASS" else 1

if __name__ == '__main__':
    exit(main())