#!/usr/bin/env python3
"""
Phase 29B: Real Sidecar Materialization + Budget Enforcement Memory Audit
For Qwen2.5-0.5B-Instruct-Q4_K_M

Extracts real PRT sidecars from the 0.5B model and measures memory behavior
under various pager/budget configurations.
"""
import json, os, struct, subprocess, sys, threading, time

# ── Paths ────────────────────────────────────────────────────────────────
MODEL = "/home/matthew-villnave/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA_CLI = "/home/matthew-villnave/llama.cpp/build/bin/llama-cli"
OUT_DIR = "/tmp/prt_sidecars_0.5b"
SIDEKAR_DIR = OUT_DIR
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

N_LAYERS = 24
HIDDEN = 896
FFN = 4864

FFN_UP_SIZE = FFN * HIDDEN * 4
FFN_DOWN_SIZE = FFN * HIDDEN * 4
ATTN_OUT_SIZE = HIDDEN * HIDDEN * 4

PROMPTS = {"short": "Hi", "math": "2+2="}

# ── .trit format ─────────────────────────────────────────────────────────
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
    header = struct.pack('<IHHII', TRIT_MAGIC, TRIT_VERSION[0], TRIT_VERSION[1], rows, cols)
    header += struct.pack('<HHI', block_rows, block_cols, n_scales)
    chk = crc16_30(payload[:30]) if len(payload) >= 30 else crc16_30(payload)
    header += struct.pack('<IH', payload_offset, chk)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(payload)
    return len(payload) + 32

# ── GGUF reading via llama model loader ──────────────────────────────────
def get_tensor_info():
    """Use llama-cli --verbose to get all tensor names and types."""
    import subprocess
    result = subprocess.run(
        [LLAMA_CLI, '-m', MODEL, '-n', '1', '--log-disable', '-v'],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, 'LD_LIBRARY_PATH': '/home/matthew-villnave/llama.cpp/build/lib:/usr/lib/x86_64-linux-gnu'}
    )
    tensors = []
    for line in result.stderr.split('\n'):
        if 'create_tensor: loading tensor' in line:
            parts = line.strip().split('tensor ')[1]
            tensors.append(parts.strip())
    return tensors

def discover_tensors():
    """Get tensor metadata via llama model params."""
    import subprocess
    cmd = [
        LLAMA_CLI, '-m', MODEL, '-n', '1', '--log-disable', '-v',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
        env={**os.environ, 'LD_LIBRARY_PATH': '/home/matthew-villnave/llama.cpp/build/lib:/usr/lib/x86_64-linux-gnu'})
    out = result.stderr
    tensors = {}
    for line in out.split('\n'):
        if 'create_tensor: loading tensor' in line:
            name = line.strip().split('tensor ')[1]
            tensors[name] = {'name': name}
    return tensors

# ── Binary GGUF parser ────────────────────────────────────────────────────
def parse_gguf_tensor_table(path):
    """Parse GGUF v3 tensor table from raw file bytes."""
    with open(path, 'rb') as f:
        data = f.read()
    fs = len(data)

    magic = struct.unpack('<I', data[0:4])[0]
    version = struct.unpack('<I', data[4:8])[0]
    n_tensors = struct.unpack('<Q', data[8:16])[0]
    n_kv = struct.unpack('<Q', data[16:24])[0]

    if magic != 0x46554747:
        raise ValueError(f"Not GGUF: {hex(magic)}")

    offset = 24

    # Skip KV pairs using safe bounds checking
    for i in range(n_kv):
        if offset + 8 > fs:
            break
        kl = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8
        if offset + kl > fs: break
        offset += kl
        offset += (8 - (kl % 8)) % 8
        if offset + 4 > fs: break
        vt = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        if vt == 8:
            if offset + 8 > fs: break
            sl = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8
            offset += sl; offset += (8 - (sl % 8)) % 8
        elif vt in (0, 1, 7):
            offset += 1
        elif vt in (2, 3, 4, 5, 6):
            offset += 4
        elif vt == 9:
            offset += 12
        else:
            offset += 8

    # Alignment
    if offset + 8 <= fs:
        alignment = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8

    # Tensor table
    tensors = []
    for i in range(n_tensors):
        if offset + 8 > fs: break
        nl = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8
        if offset + nl > fs: break
        name = data[offset:offset+nl].decode('utf-8', 'replace'); offset += nl
        offset += (8 - (nl % 8)) % 8
        if offset + 4 > fs: break
        nd = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        if offset + nd * 8 > fs: break
        dims = [struct.unpack('<Q', data[offset+j*8:offset+j*8+8])[0] for j in range(nd)]
        offset += nd * 8
        if offset + 4 > fs: break
        tt = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        if offset + 8 > fs: break
        toff = struct.unpack('<Q', data[offset:offset+8])[0]; offset += 8
        tensors.append({'name': name, 'dims': dims, 'type': tt, 'offset': toff})

    return tensors

def read_tensor_f32(data, tensor_info):
    """Read a Q4_K_M tensor as float32."""
    t = tensor_info
    t_rows, t_cols = t['dims'][1], t['dims'][0]  # GGUF: [cols, rows]
    n_elements = t_rows * t_cols

    if t['type'] == 12:  # Q4_K_M
        block_size = 128
        n_blocks = (n_elements + block_size - 1) // block_size

        result = bytearray(n_elements * 4)
        tensor_start = t['offset']
        src_offset = tensor_start

        for bi in range(n_blocks):
            if src_offset + 2 > len(data):
                break
            scale_arr = struct.unpack_from('<e', data, src_offset)
            if len(scale_arr) == 0: break
            scale = scale_arr[0]; src_offset += 2

            offset_val_arr = struct.unpack_from('<e', data, src_offset)
            if len(offset_val_arr) == 0: break
            offset_val = offset_val_arr[0]; src_offset += 2

            n_q = min(block_size, n_elements - bi * block_size)
            n_q_bytes = (n_q + 1) // 2

            for qi in range(n_q):
                byte_idx = src_offset + qi // 2
                if byte_idx >= len(data): break
                qval = (data[byte_idx] >> 4) if (qi % 2 == 0) else (data[byte_idx] & 0x0F)
                qval_s = qval - 8
                fval = scale * qval_s
                elem_idx = bi * block_size + qi
                if elem_idx < n_elements:
                    struct.pack_into('<f', result, elem_idx * 4, fval)

            src_offset += n_q_bytes

        return list(struct.unpack(f'<{n_elements}f', result))
    else:
        raise ValueError(f"Unsupported type {t['type']}")

# ── Sidecar generation ────────────────────────────────────────────────────
def generate_sidecars(layers=None, families=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    if layers is None:
        layers = list(range(N_LAYERS))
    if families is None:
        families = ['ffn_up', 'ffn_down', 'attn_output']

    print("Discovering tensor names...")
    tensors = parse_gguf_tensor_table(MODEL)
    print(f"Found {len(tensors)} tensors total")

    tensor_map = {}
    for t in tensors:
        parts = t['name'].split('.')
        if len(parts) >= 4 and parts[0] == 'blk':
            layer = int(parts[1])
            if layer not in layers:
                continue
            tensor_map[t['name']] = (layer, parts[2])

    print(f"Found {len(tensor_map)} relevant tensors")
    for name in sorted(tensor_map.keys())[:5]:
        t = next(x for x in tensors if x['name'] == name)
        print(f"  {name}: dims={t['dims']}")

    manifest_entries = []
    generated = {}

    data_cache = {}

    for name, (layer, family) in sorted(tensor_map.items()):
        if family not in families:
            continue
        t = next(x for x in tensors if x['name'] == name)
        rows, cols = t['dims'][1], t['dims'][0]

        # Map family name
        if family == 'attn_output':
            out_family = 'attn_out'
        elif family in ('ffn_up', 'ffn_down'):
            out_family = family
        else:
            continue

        out_name = f"{out_family}_layer{layer}.trit"
        out_path = os.path.join(OUT_DIR, out_name)

        print(f"  Extracting {name} (layer={layer}, family={family}, shape={rows}x{cols})...")
        try:
            # Load data once
            if name not in data_cache:
                data_cache[name] = read_tensor_f32(open(MODEL, 'rb').read(), t)
            data_f32 = data_cache[name]

            trit_size = write_trit(out_path, rows, cols, data_f32)
            file_size = os.path.getsize(out_path)
            print(f"    -> {out_name}: {file_size/1024:.1f} KB, raw F32: {rows*cols*4/1024:.1f} KB")
            manifest_entries.append({
                "layer": layer, "family": out_family, "filename": out_name,
                "rows": rows, "cols": cols, "trit_bytes": trit_size,
                "raw_f32_bytes": rows * cols * 4
            })
            generated[(layer, out_family)] = out_path
        except Exception as e:
            print(f"    ERROR: {e}")

    manifest = {
        "format_version": 1,
        "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
        "architecture": "qwen2", "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN, "ffn_dim": FFN,
        "sidecar_dir": OUT_DIR, "files": manifest_entries
    }
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {MANIFEST_PATH}")
    return generated, manifest

# ── Memory measurement ────────────────────────────────────────────────────
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

def run_memory_test(flags, prompt, n_predict, timeout=60):
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
    peak_pss = 0
    sample_times = []
    stop_flag = threading.Event()

    def monitor():
        nonlocal peak_rss, peak_pss
        while not stop_flag.is_set():
            try:
                rss, pss = get_proc_smaps(pid)
                if rss and rss > peak_rss: peak_rss = rss
                if pss and pss > peak_pss: peak_pss = pss
                sample_times.append((time.time() - start, rss or 0, pss or 0))
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
    exit_code = proc.returncode
    output = stdout.decode('utf-8', errors='replace')
    lines = output.split('\n')
    token_output = ""
    skip_prefixes = ['[', 'llama_model_loader', 'create_tensor', 'Loading model',
                     'model', 'print_info', 'srv', 'load_model', 'common_init',
                     'llama_params_fit', '- kv']
    for line in lines:
        stripped = line.strip()
        if stripped and not any(stripped.startswith(p) for p in skip_prefixes):
            token_output += stripped + '\n'

    return {
        'elapsed': round(elapsed, 2), 'exit_code': exit_code,
        'peak_rss': peak_rss, 'peak_pss': peak_pss,
        'token_output': token_output.strip()[:100],
        'n_predict': n_predict, 'timeout': elapsed >= timeout - 1,
        'rss_samples': [(round(t, 1), r, ps) for t, r, ps in sample_times]
    }

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 29B: Real Sidecar Materialization + Budget Enforcement Audit")
    print("=" * 70)

    print("\n[1/5] Generating real .trit sidecars for Qwen2.5-0.5B...")
    generated, manifest = generate_sidecars()

    print(f"\nGenerated {len(generated)} sidecar files")
    for (layer, family), path in sorted(generated.items()):
        sz = os.path.getsize(path)
        print(f"  {family}_layer{layer}.trit: {sz/1024:.1f} KB")

    print(f"\nModel: Qwen2.5-0.5B-Instruct-Q4_K_M")
    print(f"  Layers: {N_LAYERS}, Hidden: {HIDDEN}, FFN: {FFN}")
    print(f"  F32 size per ffn_up layer: {FFN_UP_SIZE/1024:.1f} KB")
    print(f"  F32 size per ffn_down layer: {FFN_DOWN_SIZE/1024:.1f} KB")
    print(f"  F32 size per attn_out layer: {ATTN_OUT_SIZE/1024:.1f} KB")

    n_predict_values = [1, 8, 32]
    results = []

    print("\n[2/5] Running memory measurement matrix...")

    configs = [
        ("A: Baseline native (no pager)", ""),
        ("B: Pager observe-only", f"--enable-prt-sidecar-pager --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR}"),
        ("C: Single ffn_up L0", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR}"),
        ("D: Single ffn_down L0", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_down --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR}"),
        ("E: Combined L0 all families", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR}"),
        ("F: Multi-layer L0+L1", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-layer 1 --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR}"),
    ]
    budget_configs = [
        ("C1: budget=0MB", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR} --prt-sidecar-budget-mb 0"),
        ("C2: budget=1MB", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR} --prt-sidecar-budget-mb 1"),
        ("C3: budget=8MB", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR} --prt-sidecar-budget-mb 8"),
        ("C4: budget=32MB", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR} --prt-sidecar-budget-mb 32"),
        ("C5: budget=512MB", f"--enable-prt-sidecar-pager --prt-sidecar-apply --prt-sidecar-apply-layer 0 --prt-sidecar-apply-family ffn_up --prt-sidecar-manifest {MANIFEST_PATH} --prt-sidecar-dir {SIDEKAR_DIR} --prt-sidecar-budget-mb 512"),
    ]
    all_configs = configs + budget_configs

    for config_name, flags in all_configs:
        print(f"\n  --- {config_name} ---")
        for np_val in n_predict_values:
            prompt_key = "math" if np_val > 1 else "short"
            prompt = PROMPTS[prompt_key]
            print(f"    n_predict={np_val}, prompt=\"{prompt}\"", end='', flush=True)
            result = run_memory_test(flags, prompt, np_val, timeout=90)
            rss_mb = result['peak_rss'] / 1024 / 1024 if result['peak_rss'] else 0
            pss_mb = result['peak_pss'] / 1024 / 1024 if result['peak_pss'] else 0
            print(f" -> RSS={rss_mb:.1f}MB PSS={pss_mb:.1f}MB, exit={result['exit_code']}, elapsed={result['elapsed']:.1f}s")
            results.append({
                'config': config_name, 'flags': flags, 'n_predict': np_val,
                'prompt': prompt, 'elapsed_s': result['elapsed'],
                'exit_code': result['exit_code'],
                'peak_rss_bytes': result['peak_rss'],
                'peak_rss_mb': round(rss_mb, 2),
                'peak_pss_bytes': result['peak_pss'],
                'peak_pss_mb': round(pss_mb, 2),
                'token_output': result['token_output'],
                'timeout': result['timeout'],
                'rss_growth': result['rss_samples']
            })

    print("\n[3/5] Analyzing memory patterns...")
    baseline_rss = {}
    sidecar_rss = {}
    for r in results:
        if r['config'].startswith('A:'):
            baseline_rss[r['n_predict']] = r['peak_rss_mb']
        else:
            if r['config'] not in sidecar_rss:
                sidecar_rss[r['config']] = {}
            sidecar_rss[r['config']][r['n_predict']] = r['peak_rss_mb']

    print("\nBaseline RSS (no pager) by n_predict:")
    for np_val, rss in sorted(baseline_rss.items()):
        print(f"  n={np_val}: {rss:.1f} MB")

    print("\nSidecar RSS delta over baseline:")
    for cfg, np_dict in sorted(sidecar_rss.items()):
        print(f"  {cfg}:")
        for np_val, rss in sorted(np_dict.items()):
            if np_val in baseline_rss:
                delta = rss - baseline_rss[np_val]
                print(f"    n={np_val}: {rss:.1f}MB (Δ={delta:+.1f}MB)")

    print("\n[4/5] Budget enforcement check:")
    budget_rows = [r for r in results if 'budget=' in r['config']]
    for r in budget_rows:
        np_val = r['n_predict']
        baseline = baseline_rss.get(np_val, 0)
        delta = r['peak_rss_mb'] - baseline
        print(f"  {r['config']}, n={np_val}: RSS={r['peak_rss_mb']:.1f}MB (Δ={delta:+.1f}MB over baseline)")

    print("\n[5/5] Classification:")
    classification = "INCONCLUSIVE"
    notes = []
    has_delta = any(
        r['peak_rss_mb'] > baseline_rss.get(r['n_predict'], 0) + 5
        for r in results if not r['config'].startswith('A:')
    )
    rss_growth = {}
    for cfg in ['C: Single ffn_up L0', 'D: Single ffn_down L0', 'E: Combined L0 all families']:
        cfg_rows = [r for r in results if r['config'] == cfg]
        if len(cfg_rows) >= 2:
            np1, np2 = cfg_rows[0]['n_predict'], cfg_rows[-1]['n_predict']
            rss1, rss2 = cfg_rows[0]['peak_rss_mb'], cfg_rows[-1]['peak_rss_mb']
            if np2 > np1:
                growth = rss2 - rss1
                rss_growth[cfg] = (np1, np2, rss1, rss2, growth)
    if rss_growth:
        print("  RSS growth over n_predict:")
        for cfg, (np1, np2, rss1, rss2, growth) in rss_growth.items():
            print(f"    {cfg}: n={np1}->{np2}, RSS={rss1:.1f}MB->{rss2:.1f}MB (Δ={growth:+.1f}MB)")

    if has_delta:
        classification = "ADDITIVE_OVERHEAD_CONFIRMED"
        notes.append("Sidecar configs show measurable RSS increase over baseline")
        notes.append("Pager architecture is currently additive (full model + sidecars)")
    else:
        classification = "LAZY_PAGER_CANDIDATE"
        notes.append("No significant RSS delta observed; may indicate lazy loading")

    budget_effects = []
    for r in budget_rows:
        np_val = r['n_predict']
        baseline = baseline_rss.get(np_val, 0)
        delta = r['peak_rss_mb'] - baseline
        budget_mb = int(r['config'].split('=')[1].replace('MB',''))
        if delta > budget_mb + 5:
            budget_effects.append(f"budget={budget_mb}MB exceeded (delta={delta:.1f}MB)")
    if budget_effects:
        classification = "BUDGET_ENFORCEMENT_BROKEN"
        notes.extend(budget_effects)
    elif budget_rows:
        notes.append("Budget enforcement: no clear rejection evidence from memory")

    summary = {
        "phase": "29B",
        "classification": classification,
        "notes": notes,
        "model": {"name": "Qwen2.5-0.5B-Instruct-Q4_K_M", "n_layers": N_LAYERS, "hidden_dim": HIDDEN, "ffn_dim": FFN},
        "sidecar_info": {"sidecar_dir": OUT_DIR, "manifest_path": MANIFEST_PATH,
            "sidecars_generated": len(generated), "per_layer_ffn_up_f32_bytes": FFN_UP_SIZE,
            "per_layer_ffn_down_f32_bytes": FFN_DOWN_SIZE, "per_layer_attn_out_f32_bytes": ATTN_OUT_SIZE},
        "results": results, "baseline_rss": baseline_rss, "sidecar_rss": sidecar_rss,
        "rss_growth": {k: list(v) for k, v in rss_growth.items()},
    }

    os.makedirs("/home/matthew-villnave/.openclaw/workspace/examples/speculative/results", exist_ok=True)
    json_path = "/home/matthew-villnave/.openclaw/workspace/examples/speculative/results/phase29b_real_sidecar_memory_audit.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults written to {json_path}")
    print(f"Classification: {classification}")
    for note in notes:
        print(f"  - {note}")
    return summary

if __name__ == '__main__':
    main()