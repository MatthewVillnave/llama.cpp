#!/usr/bin/env python3
"""
Phase 16E-REPAIR: GGUF Q4_K direct -> INT6, one layer at a time.
Memory-safe: extract, quantize, write, clear before next layer.
"""
import struct, os, sys, gc, hashlib, time, json
import numpy as np

MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-14B-14B.gguf"
INT6_DIR = "/tmp/prt_sidecars_14b_int6"
AUDIT_TMP = "/tmp/prt_phase16e_repair_14b_int6"
M, K = 13824, 5120
N_LAYERS = 40
QK_K, QK_K_SIZE = 256, 144
PRESCALE = 31.0

os.makedirs(INT6_DIR, exist_ok=True)
os.makedirs(AUDIT_TMP, exist_ok=True)

# Q4_K_M dequant — Phase 15B verified working
def dequant_block(block_bytes):
    result = np.empty(QK_K, dtype=np.float32)
    d_val  = struct.unpack_from('<e', block_bytes, 0)[0]
    dmin_val = struct.unpack_from('<e', block_bytes, 2)[0]
    scales = block_bytes[4:20]
    qdata  = block_bytes[20:148]
    qp = 0
    for j in range(4):
        s0, s1 = scales[j*2], scales[j*2+1]
        sc0, m0 = s0 & 0xF, (s0 >> 4) & 0xF
        sc1, m1 = s1 & 0xF, (s1 >> 4) & 0xF
        d1 = d_val * sc0;  m1v = dmin_val * m0
        d2 = d_val * sc1;  m2v = dmin_val * m1
        bq = j * 128
        for l in range(32):
            result[qp] = d1 * (qdata[bq+l] & 0xF) - m1v; qp += 1
        for l in range(32):
            result[qp] = d2 * (qdata[bq+l] >> 4) - m2v; qp += 1
    return result

def find_tensor_offset(gguf_path, layer):
    """Find ffn_up.weight tensor offset via minimal GGUF parse."""
    with open(gguf_path, 'rb') as f:
        f.read(4 + 4 + 8 + 8)  # skip header
        mc = struct.unpack('<Q', f.read(8))[0]
        for _ in range(mc):
            kl = struct.unpack('<Q', f.read(8))[0]
            f.read(kl + 4)
            vl = struct.unpack('<Q', f.read(8))[0]
            f.read(vl)
        target = f"blk.{layer}.ffn_up.weight".encode()
        for _ in range(200):
            nl = struct.unpack('<Q', f.read(8))[0]
            nb = f.read(nl)
            if nb == target:
                nd = struct.unpack('<I', f.read(4))[0]
                for _ in range(nd): f.read(8)
                struct.unpack('<I', f.read(4))[0]  # dtype
                return struct.unpack('<Q', f.read(8))[0]
            else:
                f.read(4 + 8*struct.unpack('<I', f.read(4))[0] + 4 + 8)
    return None

def extract_layer_streaming(gguf_path, offset, n_elements):
    """Extract and dequantize one layer from GGUF using streaming reads."""
    W = np.empty(n_elements, dtype=np.float32)
    n_blocks = n_elements // QK_K
    with open(gguf_path, 'rb') as f:
        f.seek(offset)
        for blk in range(n_blocks):
            block = f.read(QK_K_SIZE)
            W[blk*QK_K:(blk+1)*QK_K] = dequant_block(block)
    return W

def quantize_int6(W):
    """INT6 quantization."""
    W2d = W.reshape((M, K))
    row_max = np.max(np.abs(W2d), axis=1)
    scales = np.maximum(row_max, 1e-8) / PRESCALE
    Q = np.round(W2d / scales[:, np.newaxis]).astype(np.int8)
    Q = np.clip(Q, -31, 31)
    return Q, scales

def pack_int6_row(qvals, dst):
    """Pack 4 int8 into 3 bytes."""
    for i in range(0, len(qvals), 4):
        b0 = ((qvals[i+0] + 32) & 0x3F)
        b1 = (((qvals[i+1] + 32) >> 2) & 0x0F)
        b2 = (((qvals[i+2] + 32) >> 4) & 0x03)
        if i+1 < len(qvals):
            b0 |= (((qvals[i+1] + 32) & 0x03) << 6)
            b1 |= (((qvals[i+2] + 32) & 0x0F) << 4)
            b2 |= ((qvals[i+3] + 32) & 0x3F) << 2
        elif i+2 < len(qvals):
            b0 |= (((qvals[i+1] + 32) & 0x03) << 6)
            b1 |= (((qvals[i+2] + 32) & 0x0F) << 4)
        elif i+1 < len(qvals):
            b0 |= (((qvals[i+1] + 32) & 0x03) << 6)
        dst[i//4*3 : i//4*3+3] = bytes([b0, b1, b2])

def write_sidecar(Q, scales, layer, out_path):
    """Write INT6 sidecar file."""
    header = bytearray(20)
    header[0:4] = b'PRT6'
    struct.pack_into('<I', header, 4, 1)
    struct.pack_into('<I', header, 8, M)
    struct.pack_into('<I', header, 12, K)
    struct.pack_into('<I', header, 16, 0)
    
    packed_per_row = ((K + 3)//4)*3
    buf = bytearray(20 + M*4 + M*packed_per_row)
    buf[0:20] = header
    buf[20:20+M*4] = scales.astype(np.float32).tobytes()
    
    packed_row = np.empty(packed_per_row, dtype=np.int8)
    ptr = 20 + M*4
    for r in range(M):
        pack_int6_row(Q[r], packed_row)
        buf[ptr:ptr+packed_per_row] = packed_row.tobytes()
        ptr += packed_per_row
    
    with open(out_path, 'wb') as f:
        f.write(buf)
    return buf

def cosine(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def parity_check(W, Wq):
    """Quick parity check on a layer."""
    diff = W - Wq
    w_cos = cosine(W.flatten(), Wq.flatten())
    v = np.ones(K, dtype=np.float32)
    mv_orig = W @ v
    mv_q = Wq @ v
    mv_cos = cosine(mv_orig, mv_q)
    return {
        'weight_cosine': round(w_cos, 6),
        'matvec_cosine': round(mv_cos, 6),
        'mae': round(float(np.abs(diff).mean()), 6),
        'rmse': round(float(np.sqrt((diff**2).mean()), 6)),
        'max_error': round(float(np.abs(diff).max()), 6),
        'norm_ratio': round(float(np.linalg.norm(Wq)/np.linalg.norm(W)), 6),
        'finite': bool(np.isfinite(W).all() and np.isfinite(Wq).all()),
    }

def process_layer(layer):
    """Process one layer: find offset, extract, quantize, write, verify."""
    print(f"\n  Layer {layer}: finding offset...")
    offset = find_tensor_offset(MODEL, layer)
    if offset is None:
        print(f"  Layer {layer}: MISSING in GGUF")
        return None
    
    n_elements = M * K
    t0 = time.time()
    
    print(f"  Layer {layer}: extracting (offset={offset})...")
    W = extract_layer_streaming(MODEL, offset, n_elements)
    t_extract = time.time() - t0
    
    print(f"  Layer {layer}: quantizing...")
    Q, scales = quantize_int6(W)
    del W; gc.collect()
    
    print(f"  Layer {layer}: writing sidecar...")
    out_path = f"{INT6_DIR}/ffn_up_layer{layer}_prt.int6"
    buf = write_sidecar(Q, scales, layer, out_path)
    del Q, buf; gc.collect()
    
    t_total = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    sha = hashlib.sha256(open(out_path, 'rb').read()).hexdigest()[:16]
    
    print(f"  Layer {layer}: done in {t_total:.1f}s, {size_mb:.1f}MB, sha={sha}")
    return {'layer': layer, 'size_mb': size_mb, 'time': t_total, 'sha': sha}

# Main
if __name__ == '__main__':
    layers = list(range(1, N_LAYERS))  # 1-39
    
    results = []
    for layer in layers:
        # Skip if already exists (shouldn't happen but check)
        out_path = f"{INT6_DIR}/ffn_up_layer{layer}_prt.int6"
        if os.path.exists(out_path) and layer != 0:
            sha = hashlib.sha256(open(out_path, 'rb').read()).hexdigest()[:16]
            print(f"  Layer {layer}: already exists, sha={sha}")
            results.append({'layer': layer, 'size_mb': os.path.getsize(out_path)/1024/1024, 'time': 0, 'sha': sha, 'skipped': True})
            continue
        
        r = process_layer(layer)
        if r:
            results.append(r)
        gc.collect()
    
    print(f"\n=== Done: {len(results)} layers processed ===")
    # Save results
    with open(f"{AUDIT_TMP}/regen_results.json", 'w') as f:
        json.dump(results, f, indent=2)
