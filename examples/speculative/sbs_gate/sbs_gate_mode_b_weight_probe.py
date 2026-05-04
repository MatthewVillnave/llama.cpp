#!/usr/bin/env python3
"""
SBS-Gate Mode B: Real Weight Oracle Probe
Uses pre-extracted Qwen2.5-3B FFN weights to test whether 
trained model weights have more structured block contributions than random.
"""
import json, math, time
import numpy as np

def silu(x): return x / (1.0 + np.exp(-np.clip(x, -15, 15)))

def cosim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

def rel2(a, b):
    d = np.linalg.norm(a)
    return 0.0 if d == 0 else float(np.linalg.norm(a - b) / d)

def gini(c):
    s = np.sort(np.abs(c).flatten()); n = len(s); tot = float(np.sum(s))
    if tot == 0: return 0.0
    return float((2 * np.sum((np.arange(n)+1) * s)) / (n * tot) - (n+1)/n)

def oracle(hidden, W_down, block_size):
    """Oracle block-skip analysis for one activation sample."""
    ffn = len(hidden)
    hidden_dim = W_down.shape[0]  # Output dimension (first dim of W_down)
    nb = math.ceil(ffn / block_size)
    full = W_down @ hidden  # [hidden_dim]
    fn = np.linalg.norm(full)
    if fn < 1e-10: return None

    bouts = np.zeros((nb, hidden_dim), dtype=np.float32)  # [n_blocks, hidden_dim]
    for b in range(nb):
        s, e = b*block_size, min((b+1)*block_size, ffn)
        if s >= ffn: continue
        bouts[b] = W_down[:, s:e] @ hidden[s:e]
    contrib = np.linalg.norm(bouts, axis=1)
    te = float(np.sum(contrib))
    if te == 0: return None
    normed = contrib / te
    order = np.argsort(contrib)

    def find_skip(ct, lt):
        best = 0.0
        for nk in range(1, nb+1):
            kept = np.sum(bouts[order[-nk:]], axis=0)
            if cosim(full, kept) >= ct and rel2(full, kept) <= lt:
                best = max(best, 1.0 - nk/nb)
        return best

    return {
        "n_blocks": nb,
        "skip_cos098": find_skip(0.98, 1.0),
        "skip_cos095": find_skip(0.95, 1.0),
        "skip_cos090": find_skip(0.90, 1.0),
        "skip_cos099": find_skip(0.99, 1.0),
        "skip_strict": find_skip(0.98, 0.10),
        "skip_l2_010": find_skip(1.0, 0.10),
        "gini": gini(contrib),
        "eff_bk": float(1.0 / float(np.sum(normed**2))) if np.sum(normed**2) > 0 else nb,
        "top10": float(np.sum(contrib[-max(1, int(nb*0.10)):]) / te),
        "top25": float(np.sum(contrib[-max(1, int(nb*0.25)):]) / te),
    }

def main():
    # Load pre-extracted weights from PRT runs
    # Files are at /tmp/PRT_LITERAL_REMOVED_layer{N}.bin
    # Each is float32 [2048, 11008] = 90MB
    
    # Test on 3 layers: early (layer 1), mid (layer 15), late (layer 35)
    test_layers = [1, 15, 35]
    
    # Block size for oracle probe
    BS = 256
    N_SAMPLES = 32
    
    results = {}
    h_dim, ffn_dim = 2048, 11008
    
    print(f"Mode B: Using pre-extracted Qwen2.5-3B FFN weights", flush=True)
    print(f"  hidden={h_dim}, ffn={ffn_dim}, block_size={BS}", flush=True)
    print(f"  Testing {N_SAMPLES} samples per layer...", flush=True)
    
    for lid in test_layers:
        weight_path = f"/tmp/PRT_LITERAL_REMOVED_layer{lid}.bin"
        
        try:
            # Load the weight matrix (float32 [hidden, ffn])
            W = np.fromfile(weight_path, dtype=np.float32).reshape((ffn_dim, h_dim))
            W_down = W.T  # Transpose to [ffn, hidden] -> [hidden, ffn] for our computation
            print(f"  Loaded layer {lid}: {W_down.shape}", flush=True)
        except Exception as e:
            print(f"  Layer {lid}: could not load: {e}", flush=True)
            continue
        
        # Run oracle on synthetic-but-realistic-hidden vectors
        cos98, cos95, cos90, strict, l210, ginis, top10s, effs = [], [], [], [], [], [], [], []
        nb_val = 0
        
        for s in range(N_SAMPLES):
            rng = np.random.default_rng(42 + s + lid*100)
            # Hidden vector: sparse plus burst pattern
            h = rng.standard_normal(ffn_dim).astype(np.float32)
            h[rng.uniform(0,1,ffn_dim) < 0.70] = 0.0
            h[rng.uniform(0,1,ffn_dim) < 0.05] *= 5.0
            
            r = oracle(h, W_down, BS)
            if r:
                cos98.append(r["skip_cos098"]); cos95.append(r["skip_cos095"])
                cos90.append(r["skip_cos090"]); strict.append(r["skip_strict"])
                l210.append(r["skip_l2_010"]); ginis.append(r["gini"])
                top10s.append(r["top10"]); effs.append(r["eff_bk"])
                nb_val = r["n_blocks"]
        
        av = lambda lst: float(np.mean(lst)) if lst else 0.0
        results[lid] = {
            "skip_cos098": av(cos98), "skip_cos095": av(cos95),
            "skip_cos090": av(cos90), "skip_strict": av(strict),
            "skip_l2_010": av(l210), "gini": av(ginis),
            "top10_energy": av(top10s), "eff_block_count": av(effs),
            "n_blocks": nb_val,
        }
        
        r = results[lid]
        g, t10, sk, st = r["gini"], r["top10_energy"], r["skip_cos098"], r["skip_strict"]
        
        if g >= 0.3 and t10 >= 0.30 and sk >= 0.20:
            verdict = "PROMISING"
        elif g >= 0.15 or t10 >= 0.15 or sk >= 0.05:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        
        print(f"    @0.98: {sk*100:.1f}%  strict: {st*100:.1f}%  "
              f"gini: {g:.3f}  top10: {t10*100:.1f}%  "
              f"effbk: {r['eff_block_count']:.1f}/{nb_val} [{verdict}]", flush=True)
    
    # Overall verdict
    verdicts = []
    for lid in test_layers:
        if lid in results:
            r = results[lid]
            g, t10, sk = r["gini"], r["top10_energy"], r["skip_cos098"]
            if g >= 0.3 and t10 >= 0.30 and sk >= 0.20: v = "PROMISING"
            elif g >= 0.15 or t10 >= 0.15 or sk >= 0.05: v = "WEAK"
            else: v = "FAIL"
            verdicts.append(v)
    
    overall = "PROMISING" if all(v=="PROMISING" for v in verdicts) else \
              "WEAK" if any(v=="WEAK" for v in verdicts) else "FAIL"
    
    print(f"\nOverall verdict: {overall}", flush=True)
    
    # Save results
    out = {
        "method": "real_weight_driven",
        "model": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "source": "pre-extracted PRT weights",
        "block_size": BS,
        "n_samples": N_SAMPLES,
        "results": results,
        "overall_verdict": overall,
        "comparison_vs_synthetic": {
            "synthetic_0b": "skip@0.98=5.2%, gini=0.07",
            "mode_b_real": f"skip@0.98={results.get(15,{}).get('skip_cos098',0)*100:.1f}%, gini={results.get(15,{}).get('gini',0):.3f}",
        }
    }
    
    with open("/tmp/sbs_mode_b_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved /tmp/sbs_mode_b_results.json", flush=True)

if __name__ == "__main__":
    main()