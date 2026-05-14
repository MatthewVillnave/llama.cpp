#!/usr/bin/env python3
"""
PRT Phase 15B-B: Pure Python GGUF ffn_up extraction + INT8 parity.
Handles Q4_K_M dequantization without external dependencies.
Fixed: properly iterates all layers with correct tensor key per layer.
"""
import struct, os, json, sys, time, hashlib
import numpy as np

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
CURRENT_INT8 = "/tmp/prt_sidecars_7b_int8"
FRESH_INT8 = "/tmp/prt_sidecars_7b_int8_phase15b_fixed"
AUDIT = "/tmp/prt_phase15b_b_extraction_audit"
FFN, HIDDEN = 18944, 3584
N_LAYERS = 28

# Q4_K_M block params for 7B (large block = 256 elements, 144 bytes)
QK_K = 256   # elements per block
BPS = 144     # bytes per block

def read_gguf_tensors(path):
    """Read tensor metadata from GGUF file."""
    tensors = {}
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Not GGUF: {magic!r}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        alignment = struct.unpack('<Q', f.read(8))[0]
        
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            typ = struct.unpack('<I', f.read(4))[0]
            val_len = struct.unpack('<Q', f.read(8))[0]
            f.read(val_len)
        
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8', errors='replace')
            n_dim = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dim)]
            while len(dims) < 4:
                dims.append(1)
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {'dims': dims[:n_dim], 'dtype': dtype, 'offset': offset}
    
    return tensors

def dequant_q4_k_block(block_bytes):
    """Dequantize one Q4_K_M block (256 elements, 144 bytes) to float32."""
    result = np.empty(QK_K, dtype=np.float32)
    
    d_val = struct.unpack_from('<e', block_bytes, 0)[0]
    dmin_val = struct.unpack_from('<e', block_bytes, 2)[0]
    scales = memoryview(block_bytes)[4:20]
    qdata = memoryview(block_bytes)[20:148]
    
    qp = 0
    for j in range(QK_K // 64):
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
        base_qp = j * 128
        
        for l in range(32):
            q = qdata[base_qp + l] & 0xF
            result[qp] = d1 * q - m1v
            qp += 1
        for l in range(32):
            q = qdata[base_qp + l] >> 4
            result[qp] = d2 * q - m2v
            qp += 1
    
    return result

def dequant_ffn_up(path, layer_idx):
    """Extract and dequantize ffn_up.weight for layer from GGUF Q4_K_M."""
    tensor_name = f"blk.{layer_idx}.ffn_up.weight"
    
    tensors = read_gguf_tensors(path)
    if tensor_name not in tensors:
        raise ValueError(f"Tensor {tensor_name} not found. Keys: {list(tensors.keys())[:5]}")
    
    info = tensors[tensor_name]
    ne0, ne1 = info['dims'][:2]
    offset = info['offset']
    
    assert ne0 == FFN and ne1 == HIDDEN, f"Shape mismatch: {ne0}x{ne1} vs {FFN}x{HIDDEN}"
    
    n_elements = FFN * HIDDEN
    n_blocks = n_elements // QK_K
    
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read(n_blocks * BPS)
    
    W_rows = []
    for i in range(n_blocks):
        block = dequant_q4_k_block(data[i * BPS:(i+1) * BPS])
        W_rows.append(block)
    
    W = np.stack(W_rows, axis=0)
    return W  # [FFN, HIDDEN]

def quantize_int8(W):
    """Per-row INT8 quantization with float32 scales."""
    row_max = np.abs(W).max(axis=1)
    scales = np.where(row_max > 1e-10, row_max / 127.0, 1.0)
    W_q = np.clip(np.round(W / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    W_dq = W_q.astype(np.float32) * scales[:, np.newaxis]
    
    diff = W - W_dq
    norm_W = np.linalg.norm(W)
    norm_dq = np.linalg.norm(W_dq)
    weight_cosine = float(np.dot(W.flatten(), W_dq.flatten()) / (norm_W * norm_dq + 1e-8))
    
    return W_q, scales, {
        'weight_cosine': round(weight_cosine, 6),
        'max_abs_error': round(float(np.abs(diff).max()), 6),
        'mean_abs_error': round(float(np.abs(diff).mean()), 6),
        'rmse': round(float(np.sqrt((diff**2).mean())), 6),
    }

def matvec_parity(W, scales, seeds=[42, 123, 456, 789, 1011, 2022, 3033, 4044]):
    """Matvec parity test."""
    W_q = np.clip(np.round(W / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    W_dq = W_q.astype(np.float32) * scales[:, np.newaxis]
    
    results = []
    for seed in seeds:
        np.random.seed(seed)
        x = np.random.randn(HIDDEN).astype(np.float32)
        y_ref = W @ x
        y_dq = W_dq @ x
        
        cos = float(np.dot(y_ref, y_dq) / 
                    (np.linalg.norm(y_ref) * np.linalg.norm(y_dq) + 1e-8))
        rel_l2 = float(np.linalg.norm(y_ref - y_dq) / np.linalg.norm(y_ref))
        results.append({'seed': seed, 'cosine': round(cos, 6), 'rel_l2': round(rel_l2, 6)})
    
    coses = [r['cosine'] for r in results]
    return {
        'cosine_mean': round(float(np.mean(coses)), 6),
        'cosine_min': round(float(np.min(coses)), 6),
        'per_seed': results,
    }

def run_all_28_layers():
    """Extract all 28 layers, generate fresh INT8 sidecars, measure parity."""
    os.makedirs(AUDIT, exist_ok=True)
    
    print("=== GGUF Extraction + INT8 Parity (All 28 layers) ===")
    t_total = time.time()
    
    tensors = read_gguf_tensors(MODEL)
    print(f"GGUF loaded: {len(tensors)} tensors")
    
    all_results = {}
    
    for li in range(N_LAYERS):
        t0 = time.time()
        
        print(f"  Layer {li:2d}: dequantizing...", end=" ", flush=True)
        W = dequant_ffn_up(MODEL, li)
        finite = np.isfinite(W).all()
        print(f"OK finite={finite} ", end="", flush=True)
        
        W_q, scales, q_info = quantize_int8(W)
        print(f"WC={q_info['weight_cosine']} ", end="", flush=True)
        
        # Write fresh INT8 sidecar
        name = f"ffn_up_layer{li}_prt.int8"
        int8_path = os.path.join(FRESH_INT8, name)
        with open(int8_path, 'wb') as f:
            f.write(W_q.tobytes())
            f.write(scales.astype(np.float32).tobytes())
        
        h = hashlib.sha256(open(int8_path, 'rb').read()).hexdigest()
        
        # Matvec
        mv = matvec_parity(W, scales)
        print(f"MC={mv['cosine_min']} SHA={h[:12]}... ({time.time()-t0:.1f}s)")
        
        all_results[li] = {
            'layer': li,
            'int8_size': os.path.getsize(int8_path),
            'sha256': h,
            'weight_cosine': q_info['weight_cosine'],
            'max_abs_error': q_info['max_abs_error'],
            'mean_abs_error': q_info['mean_abs_error'],
            'rmse': q_info['rmse'],
            'matvec_cosine_mean': mv['cosine_mean'],
            'matvec_cosine_min': mv['cosine_min'],
            'matvec_per_seed': mv['per_seed'],
            'extract_time': round(time.time() - t0, 2),
        }
    
    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    
    hashes = [r['sha256'] for r in all_results.values()]
    unique_hashes = len(set(hashes))
    
    print(f"\n=== Uniqueness Audit ===")
    print(f"Layers processed: {len(all_results)}")
    print(f"Unique SHA256: {unique_hashes} / {N_LAYERS}")
    
    # Verify uniqueness
    if unique_hashes == N_LAYERS:
        print("RESULT: All 28 layers UNIQUE ✅")
    elif unique_hashes == 1:
        print("RESULT: All 28 layers IDENTICAL ❌ (bug still present)")
    else:
        print(f"RESULT: {unique_hashes} unique, {N_LAYERS - unique_hashes} duplicate")
    
    # Save
    out = {
        'all_28': True,
        'layers_processed': len(all_results),
        'unique_hashes': unique_hashes,
        'all_unique': unique_hashes == N_LAYERS,
        'min_weight_cosine': min(r['weight_cosine'] for r in all_results.values()),
        'min_matvec_cosine': min(r['matvec_cosine_min'] for r in all_results.values()),
        'results': all_results,
        'sha256s': {li: r['sha256'] for li, r in all_results.items()},
    }
    
    out_path = os.path.join(AUDIT, 'fresh_int8_all28_parity.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
    
    return out

if __name__ == '__main__':
    run_all_28_layers()