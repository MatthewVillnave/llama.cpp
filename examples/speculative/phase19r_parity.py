#!/usr/bin/env python3
"""Phase 19R: 7B INT6 kernel matvec parity debug."""
import struct, json, sys, os, math, inspect
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
INT6_DIR = "/tmp/prt_sidecars_7b_int6_phase15b_packed"
HIDDEN = 3584
FFN = 18944
Q4_K_M_TYPE = 12
QK_K = 256
QK_K_SIZE = 144


def read_gguf_tensors(path):
    tensors = {}
    with open(path, 'rb') as f:
        magic = f.read(4)
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


def dequantize_q4_k_m_block(blk):
    d = struct.unpack('<e', blk[0:2])[0]
    dmin = struct.unpack('<e', blk[2:4])[0]
    scales_arr = np.frombuffer(blk[4:20], dtype=np.uint8)
    nibbles = np.frombuffer(blk[20:148], dtype=np.uint8)
    vals = np.zeros(QK_K, dtype=np.float32)
    for i in range(64):
        s0 = (scales_arr[i // 16] >> ((i % 16) // 8) * 4) & 0xF
        s1 = (scales_arr[i // 16] >> ((i % 16) // 8) * 4) & 0xF
        s0 = s0 if s0 < 8 else s0 - 16
        s1 = s1 if s1 < 8 else s1 - 16
        n0 = nibbles[i * 2] & 0xF
        n1 = nibbles[i * 2] >> 4
        n0 = n0 if n0 < 8 else n0 - 16
        n1 = n1 if n1 < 8 else n1 - 16
        vals[i * 4 + 0] = n0 * s0 * d + dmin
        vals[i * 4 + 1] = n1 * s0 * d + dmin
        n2 = nibbles[i * 2 + 1] & 0xF
        n3 = nibbles[i * 2 + 1] >> 4
        n2 = n2 if n2 < 8 else n2 - 16
        n3 = n3 if n3 < 8 else n3 - 16
        vals[i * 4 + 2] = n2 * s1 * d + dmin
        vals[i * 4 + 3] = n3 * s1 * d + dmin
    return vals


def extract_layer_weight(model_path, layer_idx):
    tensors = read_gguf_tensors(model_path)
    name = f"blk.{layer_idx}.ffn_up.weight"
    info = tensors[name]
    ne0, ne1 = info['dims'][0], info['dims'][1]
    with open(model_path, 'rb') as f:
        f.seek(info['offset'])
        data = f.read(ne0 * ne1 * QK_K_SIZE // QK_K)
    weight = np.zeros((ne0, ne1), dtype=np.float32)
    for row in range(ne0):
        blk_idx = row * QK_K_SIZE // QK_K
        blk_off = row * QK_K_SIZE % QK_K
        blk = data[blk_idx * QK_K_SIZE + blk_off:blk_idx * QK_K_SIZE + blk_off + QK_K_SIZE]
        weight[row, :] = dequantize_q4_k_m_block(blk)
    return weight


def unpack_int6_sidecar(path, M, K, scale_off=20):
    with open(path, 'rb') as f:
        data = f.read()
    scales = np.frombuffer(data[scale_off:scale_off + M * 4], dtype=np.float32).copy()
    packed_off = scale_off + M * 4
    packed_len = M * K * 3 // 4
    packed = data[packed_off:packed_off + packed_len]
    int8_w = np.zeros(M * K, dtype=np.int8)
    lut = np.array([(i - 32) for i in range(64)], dtype=np.int8)
    p = 0
    i = 0
    limit = M * K
    while i + 15 < limit:
        b0, b1, b2 = packed[p], packed[p + 1], packed[p + 2]
        int8_w[i] = lut[b0 & 0x3F]
        int8_w[i + 1] = lut[(b0 >> 6 | (b1 & 0x0F) << 2) & 0x3F]
        int8_w[i + 2] = lut[(b1 >> 4 | (b2 & 0x03) << 4) & 0x3F]
        int8_w[i + 3] = lut[(b2 >> 2) & 0x3F]
        b0, b1, b2 = packed[p + 3], packed[p + 4], packed[p + 5]
        int8_w[i + 4] = lut[b0 & 0x3F]
        int8_w[i + 5] = lut[(b0 >> 6 | (b1 & 0x0F) << 2) & 0x3F]
        int8_w[i + 6] = lut[(b1 >> 4 | (b2 & 0x03) << 4) & 0x3F]
        int8_w[i + 7] = lut[(b2 >> 2) & 0x3F]
        b0, b1, b2 = packed[p + 6], packed[p + 7], packed[p + 8]
        int8_w[i + 8] = lut[b0 & 0x3F]
        int8_w[i + 9] = lut[(b0 >> 6 | (b1 & 0x0F) << 2) & 0x3F]
        int8_w[i + 10] = lut[(b1 >> 4 | (b2 & 0x03) << 4) & 0x3F]
        int8_w[i + 11] = lut[(b2 >> 2) & 0x3F]
        b0, b1, b2 = packed[p + 9], packed[p + 10], packed[p + 11]
        int8_w[i + 12] = lut[b0 & 0x3F]
        int8_w[i + 13] = lut[(b0 >> 6 | (b1 & 0x0F) << 2) & 0x3F]
        int8_w[i + 14] = lut[(b1 >> 4 | (b2 & 0x03) << 4) & 0x3F]
        int8_w[i + 15] = lut[(b2 >> 2) & 0x3F]
        p += 12
        i += 16
    while i < limit:
        b0, b1, b2 = packed[p], packed[p + 1], packed[p + 2]
        int8_w[i] = lut[b0 & 0x3F]
        if i + 1 < limit:
            int8_w[i + 1] = lut[(b0 >> 6 | (b1 & 0x0F) << 2) & 0x3F]
        if i + 2 < limit:
            int8_w[i + 2] = lut[(b1 >> 4 | (b2 & 0x03) << 4) & 0x3F]
        if i + 3 < limit:
            int8_w[i + 3] = lut[(b2 >> 2) & 0x3F]
        p += 3
        i += 4
    W_int6 = int8_w.astype(np.float32).reshape(M, K)
    for j in range(M):
        W_int6[j, :] *= scales[j]
    return W_int6, scales


def cosine(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def matvec_parity(W_ref, W_test, x, label=""):
    y_ref = W_ref @ x
    y_test = W_test @ x
    cos = cosine(y_ref, y_test)
    mae = float(np.mean(np.abs(y_ref - y_test)))
    rmse = float(np.sqrt(np.mean((y_ref - y_test) ** 2)))
    max_err = float(np.max(np.abs(y_ref - y_test)))
    norm_ref = float(np.linalg.norm(y_ref))
    norm_ratio = float(np.linalg.norm(y_test) / (norm_ref + 1e-10))
    return {
        'label': label,
        'cosine': cos,
        'mae': mae,
        'rmse': rmse,
        'max_err': max_err,
        'norm_ratio': norm_ratio,
        'y_ref_norm': norm_ref,
        'y_test_norm': float(np.linalg.norm(y_test)),
    }


layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
AUDIT_TMP = SCRIPT_DIR

print("=== Phase 19R: Layer {} INT6 Matvec Parity ===".format(layer))
print("Extracting GGUF reference...")
W_ref = extract_layer_weight(MODEL, layer)
print("  shape: {}, finite: {}, mean: {:.6f}".format(
    W_ref.shape, bool(np.isfinite(W_ref).all()), float(W_ref.mean())))

sidecar_path = "{}/ffn_up_layer{}_prt.int6".format(INT6_DIR, layer)
print("Loading INT6 sidecar: {}".format(sidecar_path))
W_int6, scales = unpack_int6_sidecar(sidecar_path, FFN, HIDDEN, scale_off=20)
print("  shape: {}, finite: {}, mean: {:.6f}".format(
    W_int6.shape, bool(np.isfinite(W_int6).all()), float(W_int6.mean())))
print("  scale[0]: {:.8f}".format(scales[0]))

print("\n=== Weight Parity ===")
wcos = cosine(W_ref, W_int6)
wmae = float(np.mean(np.abs(W_ref - W_int6)))
wrmse = float(np.sqrt(np.mean((W_ref - W_int6) ** 2)))
wmax = float(np.max(np.abs(W_ref - W_int6)))
print("  weight cosine: {:.8f}".format(wcos))
print("  weight MAE: {:.8f}".format(wmae))
print("  weight RMSE: {:.8f}".format(wrmse))
print("  weight max_err: {:.8f}".format(wmax))

print("\n=== Row Samples ===")
for row in [0, 1, 10, 100, FFN // 2, FFN - 1]:
    rcos = cosine(W_ref[row, :], W_int6[row, :])
    rmae = float(np.mean(np.abs(W_ref[row, :] - W_int6[row, :])))
    print("  row {:5d}: cos={:.6f} mae={:.8f}".format(row, rcos, rmae))

print("\n=== Matvec Parity ===")
np.random.seed(42)
x_rand = np.random.randn(HIDDEN).astype(np.float32)
x_ones = np.ones(HIDDEN, dtype=np.float32)
x_sparse0 = np.zeros(HIDDEN, dtype=np.float32)
x_sparse0[0] = 1.0
x_sparseK2 = np.zeros(HIDDEN, dtype=np.float32)
x_sparseK2[HIDDEN // 2] = 1.0

matvec_results = {}
for label, x in [('random', x_rand), ('ones', x_ones), ('sparse_k0', x_sparse0), ('sparse_kK2', x_sparseK2)]:
    result = matvec_parity(W_ref, W_int6, x, label)
    matvec_results[label] = result
    print("  {}: cosine={:.6f} mae={:.6e} norm_ratio={:.6f}".format(
        label, result['cosine'], result['mae'], result['norm_ratio']))

print("\n=== Top Error Rows (random x) ===")
y_ref = W_ref @ x_rand
y_int6 = W_int6 @ x_rand
row_errors = np.array([abs(y_ref[j] - y_int6[j]) for j in range(FFN)])
top_idx = np.argsort(row_errors)[-10:][::-1]
for idx in top_idx:
    print("  row {:5d}: err={:.6e} ref={:.6e}".format(idx, row_errors[idx], abs(y_ref[idx])))

results = {
    'layer': layer,
    'weight_cosine': wcos,
    'weight_mae': wmae,
    'weight_rmse': wrmse,
    'weight_max_err': wmax,
    'scale_0': float(scales[0]),
    'finite': bool(np.isfinite(W_int6).all()),
    'matvec': matvec_results,
}
with open("{}/layer{}_parity.json".format(AUDIT_TMP, layer), 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved.")
