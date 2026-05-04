#!/usr/bin/env python3
"""
SBS-Gate Mode B: Real Activation Oracle Probe
Uses actual Qwen2.5 model weights — computes oracle on FFN block contributions
driven by real weight matrices (not pure random).
"""
import json, math, time
import gguf
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
    ffn = len(hidden); hd = W_down.shape[1]
    nb = math.ceil(ffn / block_size)
    full = W_down @ hidden; fn = np.linalg.norm(full)
    if fn < 1e-10: return None

    # Block outputs precomputed
    bouts = np.zeros((nb, hd), dtype=np.float32)
    for b in range(nb):
        s, e = b*block_size, min((b+1)*block_size, ffn)
        bouts[b] = W_down[s:e, :] @ hidden[s:e]
    contrib = np.linalg.norm(bouts, axis=1)
    te = float(np.sum(contrib))
    if te == 0: return None
    normed = contrib / te
    order = np.argsort(contrib)  # ascending: least important first

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
        "skip_l2_015": find_skip(1.0, 0.15),
        "gini": gini(contrib),
        "eff_bk": float(1.0 / float(np.sum(normed**2))),
        "top10": float(np.sum(contrib[-max(1, int(nb*0.10)):]) / te),
        "top25": float(np.sum(contrib[-max(1, int(nb*0.25)):]) / te),
        "top50": float(np.sum(contrib[-max(1, int(nb*0.50)):]) / te),
    }

def main():
    model_path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    print(f"Mode B: Loading {model_path}...", flush=True)
    t0 = time.time()
    reader = gguf.GGUFReader(model_path, 'r')
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # Get dims
    n_layers = h_dim = ffn_dim = 0
    for t in reader.tensors:
        if t.name == "token_embd.weight": h_dim = t.shape[-1]
        if t.name == "blk.0.ffn_up.weight": ffn_dim = t.shape[-1]
        if t.name == "blk.0.ffn_down.weight": h_dim = t.shape[-1]
        if "ffn_up.weight" in t.name:
            parts = t.name.split('.')
            n_layers = max(n_layers, int(parts[1]) + 1)
    print(f"  Model: {n_layers} layers, hidden={h_dim}, ffn={ffn_dim}", flush=True)

    # Load weights: gate, up, down for 3 layers
    layers = [1, 10, 23]  # early, mid, late
    weights = {}
    for t in reader.tensors:
        for lid in layers:
            if f"blk.{lid}.ffn_gate.weight" in t.name:
                weights[(lid, 'gate')] = np.array(t.data, dtype=np.float32)
            if f"blk.{lid}.ffn_up.weight" in t.name:
                weights[(lid, 'up')] = np.array(t.data, dtype=np.float32)
            if f"blk.{lid}.ffn_down.weight" in t.name:
                weights[(lid, 'down')] = np.array(t.data, dtype=np.float32)

    BS = 256
    N = 32
    results = {}
    all_verdicts = []

    for lid in layers:
        gate = weights.get((lid, 'gate'))
        up = weights.get((lid, 'up'))
        down = weights.get((lid, 'down'))
        if gate is None: continue
        print(f"\n  Layer {lid}: gate={gate.shape}, up={up.shape}, down={down.shape}", flush=True)

        cos98 = []; cos95 = []; cos90 = []; cos99 = []; strict = []; l210 = []
        ginis = []; top10s = []; effs = []; nb_val = 0

        for s in range(N):
            rng = np.random.default_rng(42 + s + lid*100)
            h = rng.standard_normal(h_dim).astype(np.float32)
            h[rng.uniform(0,1,h_dim) < 0.70] = 0.0
            h[rng.uniform(0,1,h_dim) < 0.05] *= 5.0

            # gate/up: [hidden, ffn], so gate.T @ h = [ffn, hidden] @ [hidden] = [ffn]
            gp = silu(gate.T @ h)
            up_o = up.T @ h
            ffn_h = gp * up_o  # [ffn_dim]

            r = oracle(ffn_h, down, BS)
            if r:
                cos98.append(r["skip_cos098"]); cos95.append(r["skip_cos095"])
                cos90.append(r["skip_cos090"]); cos99.append(r["skip_cos099"])
                strict.append(r["skip_strict"]); l210.append(r["skip_l2_010"])
                ginis.append(r["gini"]); top10s.append(r["top10"])
                effs.append(r["eff_bk"]); nb_val = r["n_blocks"]

        # Small sample for cos≥0.995
        cos0995_vals = []
        for s in range(8):
            rng = np.random.default_rng(999 + s + lid*100)
            h = rng.standard_normal(h_dim).astype(np.float32)
            h[rng.uniform(0,1,h_dim) < 0.70] = 0.0
            h[rng.uniform(0,1,h_dim) < 0.05] *= 5.0
            gp = silu(gate.T @ h); up_o = up.T @ h
            r = oracle(gp * up_o, down, BS)
            if r: cos0995_vals.append(r["skip_cos098"])

        av = lambda lst: float(np.mean(lst)) if lst else 0.0
        results[lid] = {
            "skip_cos098": av(cos98), "skip_cos095": av(cos95),
            "skip_cos090": av(cos90), "skip_cos099": av(cos99),
            "skip_cos0995": av(cos0995_vals),
            "skip_strict": av(strict), "skip_l2_010": av(l210),
            "gini": av(ginis), "top10_energy": av(top10s),
            "eff_block_count": av(effs), "n_blocks": nb_val,
        }
        r = results[lid]
        g = r["gini"]; t10 = r["top10_energy"]
        sk = r["skip_cos098"]; st = r["skip_strict"]
        if g >= 0.3 and t10 >= 0.30 and sk >= 0.20: verdict = "PROMISING"
        elif g >= 0.15 or t10 >= 0.15 or sk >= 0.05: verdict = "WEAK"
        else: verdict = "FAIL"
        all_verdicts.append(verdict)
        print(f"    @0.98:{sk*100:.1f}% @0.995:{r['skip_cos0995']*100:.1f}% "
              f"strict:{st*100:.1f}% gini:{g:.3f} top10:{t10*100:.1f}% "
              f"effbk:{r['eff_block_count']:.1f}/{nb_val} [{verdict}]", flush=True)

    overall = "PROMISING" if all(v=="PROMISING" for v in all_verdicts) else \
              "WEAK" if any(v=="WEAK" for v in all_verdicts) else "FAIL"
    print(f"\nOverall: {overall}", flush=True)

    out = {"model": model_path, "n_layers": n_layers, "h_dim": h_dim, "ffn_dim": ffn_dim,
           "block_size": BS, "n_samples": N, "results": results,
           "overall_verdict": overall}
    with open("/tmp/sbs_mode_b_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved /tmp/sbs_mode_b_results.json", flush=True)

if __name__ == "__main__":
    main()