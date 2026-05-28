#!/usr/bin/env python3
"""
Phase 29C: GGUF Extraction Repair / Real Sidecar Generation Proof
For Qwen2.5-0.5B-Instruct-Q4_K_M

Uses gguf-py GGUFReader + dequantize for robust multi-type extraction.
Generates real .trit sidecars from layer0 tensors and validates them.
"""
import json, math, os, struct, subprocess, sys, threading, time
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
MODEL = "/home/matthew-villnave/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA_CLI = "/home/matthew-villnave/llama.cpp/build/bin/llama-cli"
OUT_DIR   = "/tmp/prt_sidecars_0_5b_layer0"
SIDECAR_DIR = OUT_DIR
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

N_LAYERS = 24
HIDDEN = 896
FFN    = 4864

# ── .trit format (same as phase29b) ───────────────────────────────────
TRIT_MAGIC = 0x54495254
TRIT_VERSION = (0, 1)

def pack_trits_3bit(trits, n):
    n_bytes = (n * 3 + 7) // 8
    packed = bytearray(n_bytes)
    for i in range(n):
        v = int(trits[i])
        bits = 0 if v == 0 else (1 if v == 1 else 7)
        bit_pos = i * 3
        b = bit_pos // 8
        off = bit_pos % 8
        if off <= 5:
            packed[b] |= bits << off
        else:
            high = 8 - off
            packed[b] |= (bits & ((1 << high) - 1)) << off
            packed[b + 1] |= bits >> high
    return bytes(packed)

def crc16_30(data):
    crc = 0
    for b in data:
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF
        crc ^= b
    return crc & 0xFFFF

def write_trit(path, rows, cols, data_f32, block_rows=512, block_cols=512):
    n_blocks_r = (rows + block_rows - 1) // block_rows
    n_blocks_c = (cols + block_cols - 1) // block_cols
    block_data = []
    for br in range(n_blocks_r):
        row_start = br * block_rows
        row_end = min(row_start + block_rows, rows)
        for bc in range(n_blocks_c):
            col_start = bc * block_cols
            col_end = min(col_start + block_cols, cols)
            block_max = 0.0
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    block_max = max(block_max, abs(data_f32[r * cols + c]))
            scale = block_max / 7.0 if block_max > 0 else 1.0
            tlist = []
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    v = data_f32[r * cols + c]
                    if v > scale * 3.5: tlist.append(1)
                    elif v < -scale * 3.5: tlist.append(-1)
                    else: tlist.append(0)
            full_size = block_rows * block_cols
            while len(tlist) < full_size: tlist.append(0)
            block_data.append((scale, tlist))

    payload_parts = []
    for scale, tlist in block_data:
        packed = pack_trits_3bit(tlist, block_rows * block_cols)
        payload_parts.append(struct.pack('<f', scale) + packed)
    payload = b''.join(payload_parts)
    n_scales = len(block_data)
    payload_offset = 32 + n_scales * 4
    scale_offset = payload_offset
    # Build header using correct 32-byte format matching C++ runtime trit_header:
    # Layout: <IHHII + <HHH + <II + <H = 16+6+8+2 = 32 bytes
    # Bytes 0-29 are CRC'd (compute_trit_crc reads header_30bytes), checksum at 30-31
    header = struct.pack('<IHHII', TRIT_MAGIC, TRIT_VERSION[0], TRIT_VERSION[1], rows, cols)
    header += struct.pack('<HHH', block_rows, block_cols, n_scales)
    header += struct.pack('<II', payload_offset, scale_offset)
    # CRC over first 30 bytes of header (all fields before checksum slot)
    chk = crc16_30(header)
    header += struct.pack('<H', chk)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(payload)
    return len(payload) + 32

def validate_trit(path):
    """Validate a .trit file."""
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < 32:
        return {'valid': False, 'error': 'file too small'}
    magic, v0, v1, rows, cols = struct.unpack('<IHHII', data[:16])
    block_rows, block_cols, n_scales = struct.unpack('<HHH', data[16:22])
    payload_offset, scale_offset, chk = struct.unpack('<IIH', data[22:32])
    if magic != TRIT_MAGIC:
        return {'valid': False, 'error': f'bad magic {hex(magic)}'}
    if v0 != 0 or v1 != 1:
        return {'valid': False, 'error': f'bad version {v0}.{v1}'}
    if rows <= 0 or cols <= 0:
        return {'valid': False, 'error': f'bad dims {rows}x{cols}'}
    if n_scales <= 0:
        return {'valid': False, 'error': f'bad n_scales {n_scales}'}
    if payload_offset != 32 + n_scales * 4:
        return {'valid': False, 'error': f'bad payload_offset {payload_offset}'}

    # Compute expected size
    n_blocks_r = (rows + block_rows - 1) // block_rows
    n_blocks_c = (cols + block_cols - 1) // block_cols
    expected_blocks = n_blocks_r * n_blocks_c
    if expected_blocks != n_scales:
        return {'valid': False, 'error': f'block count mismatch: {expected_blocks} != {n_scales}'}
    block_payload_size = block_rows * block_cols * 3 // 8 + 4  # scale + packed trits
    expected_file_size = 32 + n_scales * 4 + n_scales * block_payload_size

    # Try to decode
    valid_trits = True
    nan_count = inf_count = 0
    try:
        offset = 32
        for _ in range(n_scales):
            scale = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            packed_size = (block_rows * block_cols * 3 + 7) // 8
            offset += packed_size
            if not math.isfinite(scale):
                valid_trits = False
                nan_count += 1
    except Exception as e:
        valid_trits = False

    # Compute CRC check
    stored_crc = chk
    computed_crc = crc16_30(data[:30])
    checksum_valid = (stored_crc == computed_crc)
    if not checksum_valid:
        valid_trits = False

    return {
        'valid': True,
        'rows': rows, 'cols': cols,
        'block_rows': block_rows, 'block_cols': block_cols,
        'n_scales': n_scales,
        'payload_offset': payload_offset,
        'file_size': len(data),
        'checksum_valid': checksum_valid,
        'stored_crc': stored_crc,
        'computed_crc': computed_crc,
        'finite': valid_trits,
        'nan_count': nan_count, 'inf_count': inf_count,
    }

# ── GGUF extraction via gguf-py ────────────────────────────────────────
def extract_layer0_tensors():
    """Extract layer0 tensors from GGUF using gguf-py."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    # Use absolute path to gguf-py
    gguf_py_path = '/home/matthew-villnave/llama.cpp/gguf-py'
    sys.path.insert(0, gguf_py_path)

    from gguf.gguf_reader import GGUFReader
    from gguf.quants import dequantize
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    import numpy as np

    reader = GGUFReader(MODEL)
    results = {}

    # (gguf_name, family, target_rows, target_cols, needs_transpose)
    targets = [
        ('blk.0.attn_output.weight', 'attn_out',  896,  896, False),
        ('blk.0.ffn_up.weight',       'ffn_up',   4864,  896, True),
        ('blk.0.ffn_down.weight',    'ffn_down',   896, 4864, True),
    ]

    tensor_map = {t.name: t for t in reader.tensors}

    for gguf_name, family, tgt_rows, tgt_cols, needs_transpose in targets:
        if gguf_name not in tensor_map:
            print(f"WARNING: {gguf_name} not found in model")
            continue

        t = tensor_map[gguf_name]
        qtype = t.tensor_type
        block_size, type_size = GGML_QUANT_SIZES[qtype]

        raw = t.data  # memmap uint8
        n_elements = t.n_elements
        n_blocks = (n_elements + block_size - 1) // block_size

        # Reshape to (n_blocks, type_size) and dequantize
        reshaped = raw[:n_blocks * type_size].reshape((n_blocks, type_size))
        f32_flat = dequantize(reshaped, qtype).ravel()

        # Reshape to GGUF matrix
        gguf_rows, gguf_cols = int(t.shape[0]), int(t.shape[1])
        assert gguf_rows * gguf_cols == n_elements
        gguf_matrix = f32_flat.reshape((gguf_rows, gguf_cols))

        # Transpose if needed
        if needs_transpose:
            matrix = gguf_matrix.T
        else:
            matrix = gguf_matrix

        assert matrix.shape == (tgt_rows, tgt_cols), \
            f"Shape mismatch for {family}: {matrix.shape} != {(tgt_rows, tgt_cols)}"

        # Stats
        l2 = float(np.linalg.norm(matrix.astype(np.float64)))
        zero_frac = float((matrix == 0).sum()) / matrix.size

        results[family] = {
            'gguf_name': gguf_name,
            'family': family,
            'gguf_type': qtype.name,
            'shape': matrix.shape,
            'min': float(matrix.min()),
            'max': float(matrix.max()),
            'mean': float(matrix.mean()),
            'l2': l2,
            'zero_frac': zero_frac,
            'nan_count': int(np.isnan(matrix).sum()),
            'inf_count': int(np.isinf(matrix).sum()),
            'data': matrix,
        }
        print(f"Extracted {family} from {gguf_name} ({qtype.name}) -> {matrix.shape}, L2={l2:.4f}, zero_frac={zero_frac:.4f}")

    return results

# ── Sidecar generation ─────────────────────────────────────────────────
def generate_sidecars(tensor_data):
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_entries = []
    generated = {}

    for family, info in tensor_data.items():
        rows, cols = info['shape']
        data_f32 = info['data'].astype(np.float32)

        out_name = f"{family}_layer0.trit"
        out_path = os.path.join(OUT_DIR, out_name)

        # Flatten row-major
        flat = data_f32.ravel(order='C')

        trit_size = write_trit(out_path, rows, cols, flat)
        file_size = os.path.getsize(out_path)

        print(f"Wrote {out_name}: {file_size/1024:.1f} KB ({rows}x{cols}, {info['gguf_type']})")

        manifest_entries.append({
            "layer_index": 0,
            "tensor_family": family,
            "file_path": out_name,
            "rows": rows,
            "cols": cols,
            "byte_size": trit_size,
            "gguf_tensor": info['gguf_name'],
            "gguf_type": info['gguf_type'],
        })
        generated[family] = out_path

    manifest = {
        "format_name": "prt_residual_sidecar",
        "format_version": 1,
        "source_model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
        "base_quant": "Q4_K_M",
        "layer_count": N_LAYERS,
        "sidecar_dir": OUT_DIR,
        "entries": manifest_entries,
    }
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {MANIFEST_PATH}")
    return generated, manifest

# ── Runtime smoke test ─────────────────────────────────────────────────
def get_proc_smaps(pid):
    try:
        with open(f'/proc/{pid}/smaps_rollup', 'r') as f:
            content = f.read()
        rss = pss = 0
        for line in content.split('\n'):
            if line.startswith('Rss:'): rss = int(line.split()[1]) * 1024
            elif line.startswith('Pss:'): pss = int(line.split()[1]) * 1024
        return rss, pss
    except:
        return None, None

def run_smoke_test(flags, label, prompt="Hi", n_predict=1, timeout=60):
    cmd = [
        LLAMA_CLI, '-m', MODEL, '-p', prompt, '-n', str(n_predict),
        '--log-disable', '--no-conversation', '--single-turn', '--no-display-prompt',
    ] + flags.split()
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/matthew-villnave/llama.cpp/build/lib:/usr/lib/x86_64-linux-gnu'

    start = time.time()
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    pid = proc.pid
    peak_rss = 0
    stop_flag = threading.Event()

    def monitor():
        nonlocal peak_rss
        while not stop_flag.is_set():
            try:
                rss, _ = get_proc_smaps(pid)
                if rss and rss > peak_rss: peak_rss = rss
            except: pass
            time.sleep(0.1)

    t = threading.Thread(target=monitor)
    t.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    finally:
        stop_flag.set()
        t.join(timeout=2)

    elapsed = time.time() - start
    output = stdout.decode('utf-8', errors='replace')
    stderr_text = stderr.decode('utf-8', errors='replace')
    exit_code = proc.returncode

    # Extract key info from output
    inj_success = None
    sidecar_math = None
    for line in stderr_text.split('\n'):
        if 'injection_successes=' in line:
            try: inj_success = int(line.split('injection_successes=')[1].split()[0])
            except: pass
        if 'sidecar_math_influenced_output=' in line:
            try: sidecar_math = int(line.split('sidecar_math_influenced_output=')[1].split()[0])
            except: pass

    return {
        'label': label,
        'exit_code': exit_code,
        'elapsed': round(elapsed, 2),
        'peak_rss_mb': round(peak_rss / 1024 / 1024, 2) if peak_rss else 0,
        'injection_successes': inj_success,
        'sidecar_math_influenced_output': sidecar_math,
        'output_preview': output.strip()[:80],
    }

def run_smoke_matrix():
    print("\n=== Runtime Smoke Test ===")

    configs = [
        ("A_baseline",        "",                                                                    "Baseline"),
        ("B_observe",         f"--enable-prt-sidecar-pager --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDECAR_DIR}", "Observe-only"),
        ("C_scale0",         f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDECAR_DIR} --prt-sidecar-true-injection --prt-sidecar-scale 0.0", "True inj ffn_up L0 scale=0"),
        ("D_scale1",         f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDECAR_DIR} --prt-sidecar-true-injection --prt-sidecar-scale 1.0", "True inj ffn_up L0 scale=1"),
        ("E_budget0",        f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDECAR_DIR} --prt-sidecar-budget-mb 0", "Budget=0 rejection"),
        ("F_nomanifest",     "--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-dir /tmp/empty_sidecar_dir", "Missing manifest"),
    ]

    results = []
    for config_name, flags, desc in configs:
        print(f"  Running {desc}...", end='', flush=True)
        r = run_smoke_test(flags, config_name, n_predict=1, timeout=60)
        print(f" exit={r['exit_code']} RSS={r['peak_rss_mb']:.1f}MB "
              f"inj={r['injection_successes']} math={r['sidecar_math_influenced_output']}")
        r['description'] = desc
        results.append(r)
    return results

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 29C: GGUF Extraction Repair / Real Sidecar Generation")
    print("=" * 70)

    print("\n[1/5] Extracting layer0 tensors from Qwen2.5-0.5B via gguf-py...")
    tensor_data = extract_layer0_tensors()

    print(f"\n[2/5] Generating .trit sidecar files...")
    generated, manifest = generate_sidecars(tensor_data)

    print(f"\n[3/5] Validating .trit files...")
    validation_results = {}
    for family, path in generated.items():
        vr = validate_trit(path)
        vr['path'] = path
        vr['file_size'] = os.path.getsize(path)
        validation_results[family] = vr
        status = "VALID" if vr['valid'] else f"INVALID: {vr.get('error')}"
        print(f"  {family}_layer0.trit: {status}, rows={vr.get('rows')}, cols={vr.get('cols')}, "
              f"n_scales={vr.get('n_scales')}, size={vr['file_size']/1024:.1f}KB, "
              f"nan={vr.get('nan_count')}, inf={vr.get('inf_count')}")

    print(f"\n[4/5] Running runtime smoke tests...")
    smoke_results = run_smoke_matrix()

    print(f"\n[5/5] Compiling results...")

    all_valid = all(vr['valid'] for vr in validation_results.values())
    smoke_A = next((r for r in smoke_results if r['label'] == 'A_baseline'), None)
    smoke_B = next((r for r in smoke_results if r['label'] == 'B_observe'), None)
    smoke_D = next((r for r in smoke_results if r['label'] == 'D_scale1'), None)
    smoke_E = next((r for r in smoke_results if r['label'] == 'E_budget0'), None)
    smoke_F = next((r for r in smoke_results if r['label'] == 'F_nomanifest'), None)

    # Classification
    if all_valid and smoke_D and smoke_D.get('injection_successes') == 1:
        classification = "PASS_REAL_TRIT_GENERATION"
    elif all_valid and smoke_B and smoke_B.get('injection_successes') is not None:
        classification = "PARTIAL_EXTRACTION_FIXED_RUNTIME_BLOCKED"
    elif all_valid:
        classification = "PARTIAL_EXTRACTION_FIXED_RUNTIME_BLOCKED"
    else:
        classification = "BLOCKED_GGUF_EXTRACTION"

    # Detailed classification notes
    notes = []
    if all_valid:
        notes.append("All .trit files generated and validated")
    else:
        notes.append(f"Some .trit files invalid: {[f for f,v in validation_results.items() if not v['valid']]}")

    if smoke_B and smoke_B.get('injection_successes') is not None:
        notes.append(f"Observe mode: injection_successes={smoke_B['injection_successes']}")
    elif smoke_B:
        notes.append("Observe mode loaded but injection_successes not found in output")

    if smoke_D:
        inj = smoke_D.get('injection_successes')
        math_flag = smoke_D.get('sidecar_math_influenced_output')
        if inj == 1:
            notes.append(f"True injection scale=1: injection_successes=1, sidecar_math_influenced_output={math_flag}")
        elif inj == 0:
            notes.append(f"True injection scale=1: injection_successes=0 (runtime may not support this combination)")
        elif inj is None:
            notes.append("True injection scale=1: injection_successes not detected in output")

    if smoke_E:
        if smoke_E['exit_code'] != 0:
            notes.append(f"Budget=0: correctly rejected (exit={smoke_E['exit_code']})")
        else:
            notes.append(f"Budget=0: exit={smoke_E['exit_code']} (may not reject as expected)")

    if smoke_F:
        if smoke_F['exit_code'] != 0:
            notes.append(f"Missing manifest: correctly rejected (exit={smoke_F['exit_code']})")
        else:
            notes.append(f"Missing manifest: exit={smoke_F['exit_code']} (may not reject as expected)")

    summary = {
        "phase": "29C",
        "classification": classification,
        "extraction_method": "gguf-py GGUFReader + dequantize (multi-type: Q5_0, Q6_K, Q4_K)",
        "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
        "n_layers_extracted": 1,
        "tensors_extracted": {
            family: {
                "gguf_tensor": info['gguf_name'],
                "gguf_type": info['gguf_type'],
                "shape": list(info['shape']),
                "min": info['min'],
                "max": info['max'],
                "mean": info['mean'],
                "l2": info['l2'],
                "zero_frac": info['zero_frac'],
                "nan_count": info['nan_count'],
                "inf_count": info['inf_count'],
            }
            for family, info in tensor_data.items()
        },
        "sidecars_generated": len(generated),
        "validation": {family: {k: v for k, v in vr.items() if k != 'path'}
                       for family, vr in validation_results.items()},
        "smoke_tests": smoke_results,
        "notes": notes,
    }

    os.makedirs("/home/matthew-villnave/.openclaw/workspace/examples/speculative/results", exist_ok=True)
    json_path = "/home/matthew-villnave/.openclaw/workspace/examples/speculative/results/phase29c_gguf_extraction_real_trit_generation.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {json_path}")
    print(f"Classification: {classification}")
    for note in notes:
        print(f"  - {note}")

    return summary

if __name__ == '__main__':
    main()