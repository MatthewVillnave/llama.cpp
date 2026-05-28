#!/usr/bin/env python3
"""
Phase 28BR-AV: deterministic fixture generator for layer1 and layer2.
Mirrors the Phase 28BR-O layer-0 generator but with layer-specific seeds.
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
from prt_trit_io import (
    make_synthetic_scales,
    make_synthetic_ternary,
    read_trit,
    validate_trit,
    write_trit,
)

BLOCK_ROWS = 32
BLOCK_COLS = 48

FAMILY_SPEC = {
    'attn_out': {'rows': 896,  'cols': 896,  'seed_l0': 4244712744},
    'ffn_up':   {'rows': 4864, 'cols': 896,  'seed_l0': 1990868534},
    'ffn_down': {'rows': 896,  'cols': 4864, 'seed_l0': 1792446221},
}

LAYER_SEEDS = {
    1: lambda base: base ^ 0xAAAA_AAAA,
    2: lambda base: base ^ 0x5555_5555,
}

LAYER_OUT = {
    1: Path('/tmp/phase28br_av_layer1_multifamily_trit'),
    2: Path('/tmp/phase28br_av_layer2_multifamily_trit'),
}

def expected_scale_count(rows, cols):
    return ((rows + BLOCK_ROWS - 1) // BLOCK_ROWS) * ((cols + BLOCK_COLS - 1) // BLOCK_COLS)

def generate_layer(layer, families, out_dir):
    layer_str = 'layer_{:03d}'.format(layer)
    layer_dir = out_dir / 'layers' / layer_str
    entries = []
    files = []

    for family, spec in families.items():
        rows = spec['rows']
        cols = spec['cols']
        seed = LAYER_SEEDS[layer](spec['seed_l0'])
        rel_path = Path('layers') / layer_str / (family + '.trit')
        trit_path = out_dir / rel_path
        n_scales = expected_scale_count(rows, cols)

        entry = {
            'layer_index': layer,
            'tensor_name': 'blk.{}.{}.weight'.format(layer, family),
            'tensor_family': family,
            'shape': [rows, cols],
            'rows': rows,
            'cols': cols,
            'dtype': 'ternary',
            'block_rows': BLOCK_ROWS,
            'block_cols': BLOCK_COLS,
            'scale_count': n_scales,
            'seed': seed,
            'file_path': str(rel_path),
            'status': 'active',
            'required': True,
        }
        entries.append(entry)

        layer_dir.mkdir(parents=True, exist_ok=True)
        ternary = make_synthetic_ternary(rows, cols, seed)
        scales = make_synthetic_scales(ternary, BLOCK_ROWS, BLOCK_COLS)
        write_info = write_trit(trit_path, ternary, scales, BLOCK_ROWS, BLOCK_COLS)

        valid, msg = validate_trit(trit_path)
        if not valid:
            raise ValueError('{} layer{}: validate_trit failed: {}'.format(family, layer, msg))

        dec_t, dec_s, meta = read_trit(trit_path)
        actual_bytes = trit_path.stat().st_size
        sha256 = hashlib.sha256(trit_path.read_bytes()).hexdigest()
        entry['byte_size'] = actual_bytes
        entry['checksum'] = write_info['checksum']
        entry['sha256'] = sha256

        files.append({
            'family': family,
            'layer': layer,
            'path': str(trit_path),
            'bytes': actual_bytes,
            'validate': msg,
            'rows': rows,
            'cols': cols,
            'n_scales': len(dec_s),
            'finite': bool(np.all(np.isfinite(dec_s))),
            'trit_unique': sorted(set(dec_t.flat)),
            'payload_offset': meta['payload_offset'],
            'scale_offset': meta['scale_offset'],
            'header_crc16': write_info['checksum'],
            'sha256': sha256,
        })

    manifest = {
        'format_name': 'prt_residual_sidecar',
        'format_version': 1,
        'source_model': 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
        'source_model_hash': 'unknown_phase28bo_fixture',
        'base_quant': 'Q4_K_M',
        'residual_format': 'ternary',
        'generator': 'examples/speculative/phase28br_av_generator.py',
        'block_rows': BLOCK_ROWS,
        'block_cols': BLOCK_COLS,
        'layer_count': 1,
        'tensor_families': list(families.keys()),
        'entries': entries,
    }

    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    return {'manifest': manifest, 'files': files, 'manifest_path': str(manifest_path)}

def main():
    results = {}
    for layer in [1, 2]:
        out_dir = LAYER_OUT[layer]
        print('Generating layer {} fixtures in {}...'.format(layer, out_dir))
        r = generate_layer(layer, FAMILY_SPEC, out_dir)
        results[layer] = r
        fam_list = [f['family'] for f in r['files']]
        print('  Generated: {}'.format(fam_list))
        for f in r['files']:
            print('  {}: {}x{} scales={} finite={} bytes={}'.format(
                f['family'], f['rows'], f['cols'], f['n_scales'], f['finite'], f['bytes']))

    print('')
    print('Done. Manifests at:')
    for layer in [1, 2]:
        print('  Layer {}: {}'.format(layer, results[layer]['manifest_path']))

if __name__ == '__main__':
    main()
