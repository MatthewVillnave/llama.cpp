#!/usr/bin/env python3
"""
Phase 28BR-AV: Multi-Layer True Injection Sweep Harness
Runs A/B/C single-layer + D/E multi-layer + controls using llama-cli.
"""

import json
import subprocess
import time
from pathlib import Path

WORKSPACE   = Path('/home/matthew-villnave/llama.cpp')
MODEL       = '/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'
LLAMA_CLI   = WORKSPACE / 'build/bin/llama-cli'
RESULTS_DIR = WORKSPACE / 'examples' / 'speculative' / 'results'

L0_MANIFEST = '/tmp/phase28br_o_layer0_multifamily_trit/manifest.json'
L1_MANIFEST = '/tmp/phase28br_av_layer1_multifamily_trit/manifest.json'
L2_MANIFEST = '/tmp/phase28br_av_layer2_multifamily_trit/manifest.json'
COMBINED_DIR = '/tmp/phase28br_av_combined_trit'

PROMPTS = ['Hi', '2+2=']
BUDGET  = 512
N_PRED  = 1
FAMILY  = 'attn_out'

def build_combined_manifest(layers_included, out_path):
    """Create a manifest covering all three layers, but only include entries
    for the specified layers."""
    combined = {
        'format_name': 'prt_residual_sidecar',
        'format_version': 1,
        'source_model': 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
        'base_quant': 'Q4_K_M',
        'residual_format': 'ternary',
        'generator': 'phase28br_av_sweep.py',
        'tensor_families': ['attn_out', 'ffn_up', 'ffn_down'],
        'entries': [],
    }
    path_map = {
        0: L0_MANIFEST,
        1: L1_MANIFEST,
        2: L2_MANIFEST,
    }
    manifest_base = {
        0: '/tmp/phase28br_o_layer0_multifamily_trit',
        1: '/tmp/phase28br_av_layer1_multifamily_trit',
        2: '/tmp/phase28br_av_layer2_multifamily_trit',
    }
    for layer in layers_included:
        m = json.load(open(path_map[layer]))
        base = Path(manifest_base[layer])
        for e in m.get('entries', []):
            new_e = dict(e)
            new_e['file_path'] = str(base / e['file_path'])
            combined['entries'].append(new_e)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    return out_path

def run_inference(prompt, mode, layer=None, family=FAMILY, scale=1.0, budget=BUDGET, manifest=None):
    """Run llama-cli inference. modes: baseline, true_inj, scale0, wrong_layer, budget0, missing_manifest"""
    cmd = [
        str(LLAMA_CLI),
        '-m', MODEL,
        '-p', prompt,
        '--enable-prt-sidecar-pager',
        '--prt-mode', '5700',
        '--prt-sidecar-budget-mb', str(budget),
        '--single-turn',
        '-n', str(N_PRED),
        '--log-disable',
    ]

    if mode == 'baseline':
        pass  # no PRT flags

    elif mode == 'true_inj':
        cmd += [
            '--prt-sidecar-apply',
            '--prt-sidecar-true-injection',
            '--prt-sidecar-apply-layer', str(layer),
            '--prt-sidecar-apply-family', family,
            '--prt-sidecar-scale', str(scale),
            '--prt-sidecar-manifest', manifest,
        ]

    elif mode == 'scale0':
        cmd += [
            '--prt-sidecar-apply',
            '--prt-sidecar-true-injection',
            '--prt-sidecar-apply-layer', str(layer),
            '--prt-sidecar-apply-family', family,
            '--prt-sidecar-scale', '0.0',
            '--prt-sidecar-manifest', manifest,
        ]

    elif mode == 'wrong_layer':
        cmd += [
            '--prt-sidecar-apply',
            '--prt-sidecar-true-injection',
            '--prt-sidecar-apply-layer', '99',
            '--prt-sidecar-apply-family', family,
            '--prt-sidecar-scale', str(scale),
            '--prt-sidecar-manifest', manifest,
        ]

    elif mode == 'budget0':
        cmd += [
            '--prt-sidecar-apply',
            '--prt-sidecar-true-injection',
            '--prt-sidecar-apply-layer', str(layer),
            '--prt-sidecar-apply-family', family,
            '--prt-sidecar-scale', str(scale),
            '--prt-sidecar-manifest', manifest,
        ]
        cmd[cmd.index('--prt-sidecar-budget-mb') + 1] = '0'

    elif mode == 'missing_manifest':
        cmd += [
            '--prt-sidecar-apply',
            '--prt-sidecar-true-injection',
            '--prt-sidecar-apply-layer', str(layer),
            '--prt-sidecar-apply-family', family,
            '--prt-sidecar-scale', str(scale),
            '--prt-sidecar-manifest', '/nonexistent/manifest.json',
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + '\n' + result.stderr
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', 124

def parse_prt_output(output):
    """Parse PRT output for token, logit, top-k, and NaN/Inf."""
    info = {
        'selected_token': None,
        'selected_logit': None,
        'top10_ids': [],
        'top10_logits': [],
        'prt_flags_found': False,
        'nan_detected': False,
        'inf_detected': False,
    }
    for line in output.splitlines():
        line = line.strip()
        if 'PRT-FLAGS-SET' in line:
            info['prt_flags_found'] = True
        if 'nan' in line.lower() and 'token' in line.lower():
            info['nan_detected'] = True
        if 'inf' in line.lower() and ('token' in line.lower() or 'logit' in line.lower()):
            info['inf_detected'] = True
        if '[TOKEN]' in line:
            parts = line.split()
            for p in parts:
                if p.startswith('id='):
                    try: info['selected_token'] = int(p[3:])
                    except: pass
                elif p.startswith('logit='):
                    try: info['selected_logit'] = float(p[6:])
                    except: pass
                elif p.startswith('top_ids='):
                    info['top10_ids'] = [int(x) for x in p[8:].split(',') if x]
                elif p.startswith('top_logits='):
                    info['top10_logits'] = [float(x) for x in p[11:].split(',') if x]
    return info

def jaccard(a, b, k=10):
    if not a or not b: return 0.0
    s_a = set(a[:k])
    s_b = set(b[:k])
    if not s_a or not s_b: return 0.0
    return len(s_a & s_b) / len(s_a | s_b)

def inject_success(info, baseline_info):
    """Injection succeeded if token changed or logit delta detected, and no NaN/Inf."""
    if info.get('nan_detected') or info.get('inf_detected'):
        return False, 'NaN_Inf_detected'
    if info.get('selected_token') is None:
        return False, 'no_token'
    if baseline_info.get('selected_token') is None:
        return False, 'baseline_no_token'
    if info['selected_token'] != baseline_info['selected_token']:
        return True, 'token_changed'
    if info.get('top10_ids') and baseline_info.get('top10_ids'):
        j = jaccard(info['top10_ids'], baseline_info['top10_ids'])
        if j < 0.95:
            return True, 'topk_diverged_jaccard={:.3f}'.format(j)
    return False, 'no_effect_detected'

def run_phase():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline runs (no PRT)
    print('=== BASELINE ===')
    baseline = {}
    for prompt in PROMPTS:
        out, rc = run_inference(prompt, 'baseline')
        info = parse_prt_output(out)
        baseline[prompt] = info
        print('  {}: token={} logit={} rc={}'.format(prompt, info['selected_token'], info['selected_logit'], rc))

    # Build combined manifests
    combined_01_path = '/tmp/phase28br_av_combined_l01/manifest.json'
    combined_012_path = '/tmp/phase28br_av_combined_l012/manifest.json'
    build_combined_manifest([0, 1], combined_01_path)
    build_combined_manifest([0, 1, 2], combined_012_path)

    manifest_map = {
        0: L0_MANIFEST,
        1: L1_MANIFEST,
        2: L2_MANIFEST,
    }

    layer_combinations = [
        ('A', 'single_layer0',    0, L0_MANIFEST),
        ('B', 'single_layer1',    1, L1_MANIFEST),
        ('C', 'single_layer2',    2, L2_MANIFEST),
        ('D', 'multi_layer01',    None, combined_01_path),
        ('E', 'multi_layer012',   None, combined_012_path),
    ]

    results = {'baseline': {}, 'tests': {}}

    for prompt in PROMPTS:
        results['baseline'][prompt] = {
            'token': baseline[prompt]['selected_token'],
            'logit': baseline[prompt]['selected_logit'],
        }

    for test_letter, test_name, layer, manifest in layer_combinations:
        print('')
        print('=== TEST {}: {} ==='.format(test_letter, test_name))
        results['tests'][test_letter] = {}

        for prompt in PROMPTS:
            effective_layer = layer if layer is not None else 'all'
            out, rc = run_inference(prompt, 'true_inj',
                                   layer=layer if layer is not None else 0,
                                   family=FAMILY, scale=1.0,
                                   manifest=manifest)
            info = parse_prt_output(out)
            success, reason = inject_success(info, baseline[prompt])
            results['tests'][test_letter][prompt] = {
                'token': info['selected_token'],
                'logit': info['selected_logit'],
                'top10_ids': info['top10_ids'],
                'top10_logits': info['top10_logits'],
                'prt_flags': info['prt_flags_found'],
                'nan': info['nan_detected'],
                'inf': info['inf_detected'],
                'rc': rc,
                'injection_success': success,
                'reason': reason,
            }
            print('  {}: token={} inj_success={} reason={} prt={} rc={}'.format(
                prompt, info['selected_token'], success, reason, info['prt_flags_found'], rc))

    # Controls
    print('')
    print('=== CONTROLS ===')
    controls = {}

    # Scale=0 control
    for prompt in PROMPTS:
        out, rc = run_inference(prompt, 'scale0', layer=0, family=FAMILY, manifest=L0_MANIFEST)
        info = parse_prt_output(out)
        token_match = info['selected_token'] == baseline[prompt]['selected_token'] if info['selected_token'] else False
        key = 'scale0_' + prompt
        controls[key] = {
            'token': info['selected_token'],
            'baseline_token': baseline[prompt]['selected_token'],
            'matches_baseline': token_match,
            'rc': rc,
        }
        print('  scale0 {}: match={} token={} vs base={}'.format(
            prompt, token_match, info['selected_token'], baseline[prompt]['selected_token']))

    # Wrong layer control
    for prompt in PROMPTS:
        out, rc = run_inference(prompt, 'wrong_layer', layer=0, family=FAMILY, manifest=L0_MANIFEST)
        info = parse_prt_output(out)
        # Wrong layer should bypass (manifest only has layer 0)
        key = 'wrong_layer_' + prompt
        controls[key] = {
            'token': info['selected_token'],
            'rc': rc,
        }
        print('  wrong_layer {}: token={} rc={}'.format(prompt, info['selected_token'], rc))

    results['controls'] = controls

    # Write results JSON
    results_path = RESULTS_DIR / 'phase28br_av_multilayer_true_injection_sweep.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('')
    print('Results written to: {}'.format(results_path))

    return results

if __name__ == '__main__':
    run_phase()
