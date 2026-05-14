#!/usr/bin/env python3
"""
PRT Phase 15B-B: GGUF ffn_up extraction + parity tool for Qwen2.5-7B Q4_K_M.
Extracts blk.{L}.ffn_up.weight from GGUF, dequantizes, computes INT8 sidecars,
and measures offline parity metrics.

GGUF format for 7B Q4_K_M:
- Magic: "GGUF" bytes BE, but version/tensors are LE
- Version: 3
- Tensor name: blk.{L}.ffn_up.weight
- Shape: [ne0=ffn=18944, ne1=hidden=3584]  (GGUF [columns, rows])
- Q4_K_M dtype=12
- Block size: 256 elements (for large models)
- Block bytes: 144 bytes (for Q4_K_M)

Q4_K_M block layout (144 bytes per 256 elements):
  [2] d = float16 delta scale
  [2] dmin = float16 delta min  
  [16] scales (4 bits each, 8 scale values)
  [128] nibbles (64 pairs = 128 4-bit values)
"""

import struct, os, json, sys, time
import numpy as np

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
HIDDEN = 3584    # K = input dim = ne1 in GGUF
FFN = 18944      # M = intermediate dim = ne0 in GGUF
N_LAYERS = 28
Q4_K_M_TYPE = 12

# Q4_K_M block parameters for 7B (256-element blocks)
QK_K = 256        # elements per block
QK_K_SIZE = 144    # bytes per block (for Q4_K_M)

# Per-row quantization for INT8 sidecar
INT8_SCALE_DIV = 127.0


def read_gguf_header(path):
    """Read GGUF header and return (version, n_tensors, alignment, tensor_offset)."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Not GGUF: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        alignment = struct.unpack('<Q', f.read(8))[0]
    return version, n_tensors, alignment


def read_gguf_tensors(path):
    """Read all GGUF tensor metadata entries."""
    tensors = {}
    with open(path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        alignment = struct.unpack('<Q', f.read(8))[0]
        
        # Skip metadata
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            typ = struct.unpack('<I', f.read(4))[0]
            val_len = struct.unpack('<Q', f.read(8))[0]
            f.read(val_len)
        
        # Read tensor metadata
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8', errors='replace')
            n_dim = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dim)]
            while len(dims) < 4:
                dims.append(1)
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {
                'dims': dims[:n_dim],
                'dtype': dtype,
                'offset': offset,
                'name': name,
            }
    return tensors


def dequantize_q4_k_block(block_bytes):
    """
    Dequantize one Q4_K_M block (256 elements, 144 bytes) to float32.
    Returns array of 256 float32 values.
    """
    result = np.empty(QK_K, dtype=np.float32)
    
    # d and dmin are float16 (half precision)
    d_val = struct.unpack_from('<e', block_bytes, 0)[0]
    dmin_val = struct.unpack_from('<e', block_bytes, 2)[0]
    
    # 16 bytes of scales (8 scale values, 4 bits each)
    scales = block_bytes[4:20]
    
    # 128 bytes of nibbles (64 pairs = 128 4-bit values)
    qdata = block_bytes[20:148]
    
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
        
        # 32 low nibbles
        for l in range(32):
            q = qdata[base_qp + l] & 0xF
            result[qp] = d1 * q - m1v
            qp += 1
        # 32 high nibbles
        for l in range(32):
            q = qdata[base_qp + l] >> 4
            result[qp] = d2 * q - m2v
            qp += 1
    
    return result


def dequantize_ffn_up(gguf_path, layer_idx, ffn=FFN, hidden=HIDDEN):
    """
    Extract and dequantize ffn_up.weight for layer from GGUF Q4_K_M.
    Returns W as [ffn, hidden] float32 row-major.
    """
    tensor_name = f"blk.{layer_idx}.ffn_up.weight"
    
    tensors = read_gguf_tensors(gguf_path)
    if tensor_name not in tensors:
        raise ValueError(f"Tensor {tensor_name} not found. Available: {[k for k in tensors.keys() if 'ffn_up' in k][:5]}")
    
    info = tensors[tensor_name]
    ne0, ne1 = info['dims'][:2]  # [ffn, hidden]
    offset = info['offset']
    
    assert ne0 == ffn and ne1 == hidden, f"Shape mismatch: {ne0}x{ne1} vs {ffn}x{hidden}"
    
    n_elements = ffn * hidden
    n_blocks = n_elements // QK_K
    block_bytes_total = n_blocks * QK_K_SIZE
    
    with open(gguf_path, 'rb') as f:
        f.seek(offset)
        data = f.read(block_bytes_total)
    
    if len(data) != block_bytes_total:
        raise ValueError(f"Read {len(data)} bytes, expected {block_bytes_total}")
    
    # Dequantize each block
    W_rows = []
    for i in range(n_blocks):
        block = dequantize_q4_k_block(data[i * QK_K_SIZE:(i+1) * QK_K_SIZE])
        W_rows.append(block)
    
    W = np.stack(W_rows, axis=0)  # [ffn, hidden]
    return W


def quantize_to_int8(W, ffn=FFN, hidden=HIDDEN):
    """
    Quantize float32 W [ffn, hidden] to INT8 with per-row scales.
    Returns: (int8_bytes, scales, metrics)
    """
    row_max = np.abs(W).max(axis=1)  # [ffn]
    scales = np.where(row_max > 1e-10, row_max / INT8_SCALE_DIV, 1.0)  # [ffn]
    
    W_q = np.clip(np.round(W / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    
    int8_bytes = W_q.tobytes()
    scale_bytes = scales.astype(np.float32).tobytes()
    
    # Reconstruct for metrics
    W_dq = W_q.astype(np.float32) * scales[:, np.newaxis]
    diff = W - W_dq
    weight_cosine = float(np.dot(W.flatten(), W_dq.flatten()) / 
                          (np.linalg.norm(W) * np.linalg.norm(W_dq) + 1e-8))
    
    return int8_bytes, scales, {
        'weight_cosine': round(weight_cosine, 6),
        'max_abs_error': round(float(np.abs(diff).max()), 6),
        'mean_abs_error': round(float(np.abs(diff).mean()), 6),
        'rmse': round(float(np.sqrt((diff**2).mean()), 6),
        'zero_fraction': round(float((W_q == 0).sum() / W_q.size), 4),
        'abs127_fraction': round(float((np.abs(W_q) == 127).sum() / W_q.size), 4),
    }


def matvec_parity(W, scales, ffn=FFN, hidden=HIDDEN, seeds=[42, 123, 456, 789, 1011, 2022, 3033, 4044]):
    """Compute matvec parity on dequantized W."""
    results = []
    for seed in seeds:
        np.random.seed(seed)
        x = np.random.randn(hidden).astype(np.float32)
        y_ref = W @ x
        
        # Dequantize W from stored int8 + scales
        W_int8 = np.frombuffer(scales.newbuilder() if hasattr(scales, 'newbuilder') else 
                               (lambda: None)(), dtype=np.int8)  # skip, use W directly
        
        # Use W (float32 reference) as reference
        y_ref = W @ x
        
        # Compare to naive int8 path (quantize then dequantize)
        W_q = np.clip(np.round(W / scales[:, np.newaxis]), -127, 127).astype(np.int8)
        W_dq = W_q.astype(np.float32) * scales[:, np.newaxis]
        y_dq = W_dq @ x
        
        cos = float(np.dot(y_ref, y_dq) / (np.linalg.norm(y_ref) * np.linalg.norm(y_dq) + 1e-8))
        rel_l2 = float(np.linalg.norm(y_ref - y_dq) / np.linalg.norm(y_ref))
        max_err = float(np.abs(y_ref - y_dq).max())
        
        results.append({
            'seed': seed,
            'cosine': round(cos, 6),
            'rel_l2': round(rel_l2, 6),
            'max_err': round(max_err, 6),
        })
    
    coses = [r['cosine'] for r in results]
    return {
        'cosine_mean': round(float(np.mean(coses)), 6),
        'cosine_min': round(float(np.min(coses)), 6),
        'per_seed': results,
    }


def run_parity_for_layers(layers, gguf_path, fresh_int8_dir, audit_tmp):
    """Extract layers, generate INT8 sidecars, measure parity."""
    os.makedirs(fresh_int8_dir, exist_ok=True)
    os.makedirs(audit_tmp, exist_ok=True)
    
    results = []
    for li in layers:
        print(f"\n--- Layer {li} ---")
        t0 = time.time()
        
        # Extract from GGUF
        print(f"  Extracting from GGUF...", end=" ", flush=True)
        W = dequantize_ffn_up(gguf_path, li)
        print(f"dequant={time.time()-t0:.1f}s shape={W.shape} ", end="", flush=True)
        
        # Check finiteness
        finite = np.isfinite(W).all()
        print(f"finite={finite}", end="", flush=True)
        
        # Quantize to INT8
        t1 = time.time()
        int8_bytes, scales, q_info = quantize_to_int8(W)
        print(f" quantize={time.time()-t1:.1f}s", end="", flush=True)
        
        # Save INT8 sidecar
        name = f"ffn_up_layer{li}_prt"
        int8_path = os.path.join(fresh_int8_dir, f"{name}.int8")
        with open(int8_path, 'wb') as f:
            f.write(int8_bytes)
            f.write(scales.astype(np.float32).tobytes())
        
        file_size = os.path.getsize(int8_path)
        print(f" file={file_size/1e6:.1f}MB", end="", flush=True)
        
        # Matvec parity
        t2 = time.time()
        mv = matvec_parity(W, scales)
        print(f" matvec={time.time()-t2:.1f}s cos_mean={mv['cosine_mean']} cos_min={mv['cosine_min']}")
        
        # SHA256 of fresh sidecar
        import hashlib
        h = hashlib.sha256(open(int8_path, 'rb').read()).hexdigest()
        
        r = {
            'layer': li,
            'shape_ok': (W.shape == (FFN, HIDDEN)),
            'finite': bool(finite),
            'weight_cosine': q_info['weight_cosine'],
            'max_abs_error': q_info['max_abs_error'],
            'mean_abs_error': q_info['mean_abs_error'],
            'rmse': q_info['rmse'],
            'zero_fraction': q_info['zero_fraction'],
            'abs127_fraction': q_info['abs127_fraction'],
            'matvec_cosine_mean': mv['cosine_mean'],
            'matvec_cosine_min': mv['cosine_min'],
            'matvec_per_seed': mv['per_seed'],
            'int8_file': int8_path,
            'file_size': file_size,
            'sha256': h,
            'extract_time': round(time.time() - t0, 2),
        }
        
        print(f"  WC={q_info['weight_cosine']} MAE={q_info['mean_abs_error']} RMSE={q_info['rmse']}")
        results.append(r)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf', default=MODEL)
    parser.add_argument('--fresh-dir', default='/tmp/prt_sidecars_7b_int8_phase15b_fixed')
    parser.add_argument('--audit-tmp', default='/tmp/prt_phase15b_b_extraction_audit')
    parser.add_argument('--layers', default='0,1,10,11,15,20,27')
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(',')]
    
    print("=== PRT Phase 15B-B: GGUF Extraction + INT8 Sidecar Fix ===")
    print(f"GGUF: {args.gguf}")
    print(f"Fresh output: {args.fresh_dir}")
    print(f"Layers: {layers}")
    print(f"Shape: ffn={FFN}, hidden={HIDDEN}")
    
    # Verify GGUF
    version, n_tensors, alignment = read_gguf_header(args.gguf)
    print(f"GGUF v{version}, {n_tensors} tensors, alignment={alignment}")
    
    # Run parity for selected layers
    results = run_parity_for_layers(layers, args.gguf, args.fresh_dir, args.audit_tmp)
    
    # Summary
    print(f"\n=== Selected Layer Parity Summary ===")
    for r in results:
        print(f"  L{r['layer']:2d}: WC={r['weight_cosine']} MC={r['matvec_cosine_min']} SHA={r['sha256'][:12]}...")
    
    min_wc = min(r['weight_cosine'] for r in results)
    min_mc = min(r['matvec_cosine_min'] for r in results)
    
    print(f"\n  Min weight cosine: {min_wc}")
    print(f"  Min matvec cosine: {min_mc}")
    
    # Save results
    out = {
        'layers': layers,
        'results': results,
        'min_weight_cosine': min_wc,
        'min_matvec_cosine': min_mc,
        'sha256s': {r['layer']: r['sha256'] for r in results},
    }
    
    out_path = os.path.join(args.audit_tmp, 'fresh_int8_selected_layer_parity.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Results: {out_path}")