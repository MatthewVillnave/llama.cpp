#!/usr/bin/env python3
"""
SBS-Gate Oracle Probe 0B: Threshold Sweep + Rescale Diagnostic
Tests structured FFN block skipping across threshold grid, block sizes,
ranking methods, and includes rescale + concentration diagnostics.
"""
import argparse, json, math, time
import numpy as np

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def cosim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def rel2(a, b):
    d = np.linalg.norm(a)
    return 0.0 if d == 0 else float(np.linalg.norm(a - b) / d)

def rescale_correct(full, skip):
    """Best scalar alpha: out_rescaled = alpha * out_skip"""
    num = float(np.dot(full, skip))
    den = float(np.dot(skip, skip))
    if den < 1e-15: return 1.0, np.zeros_like(full)
    alpha = num / den
    return alpha, alpha * skip

def gini_from_contrib(contrib_sorted):
    """Gini coefficient of sorted contributions (ascending). 0=equal, 1=max unequal."""
    n = len(contrib_sorted)
    if n < 2: return 0.0
    cumsum = np.cumsum(contrib_sorted)
    return float((2 * np.sum((np.arange(n) + 1) * contrib_sorted)) / (n * cumsum[-1]) - (n + 1) / n)

# ---------------------------------------------------------------------------
# Oracle analysis
# ---------------------------------------------------------------------------
def oracle_analysis(hidden, W_down, block_size, cos_thresholds, l2_thresholds,
                    rank_method="contribution_norm"):
    """
    Run oracle block skip analysis for one sample.

    rank_method:
      contribution_norm  — rank blocks by ||W[:,block] @ h_block||_2
      hidden_norm        — rank blocks by ||h_block||_2
      random             — random keep order (null baseline)
      reverse_contrib    — worst case: drop highest-contribution blocks first
    """
    n_blocks = math.ceil(len(hidden) / block_size)

    # Full FFN output
    full = W_down @ hidden
    full_norm = np.linalg.norm(full)

    # Compute block-level contributions based on ranking method
    if rank_method == "random":
        order = np.random.default_rng(42).permutation(n_blocks)
    elif rank_method == "reverse_contribution":
        contrib = np.array([np.linalg.norm(W_down[:, b*block_size:min((b+1)*block_size, len(hidden))] @
                                        hidden[b*block_size:min((b+1)*block_size, len(hidden))])
                          for b in range(n_blocks)])
        order = np.argsort(contrib)  # ascending: least important first
        order = order[::-1]  # reverse: drop most important first (worst case)
    else:
        contrib = np.zeros(n_blocks)
        for b in range(n_blocks):
            s, e = b*block_size, min((b+1)*block_size, len(hidden))
            bh = hidden[s:e]
            bW = W_down[:, s:e]
            if rank_method == "contribution_norm":
                contrib[b] = np.linalg.norm(bW @ bh)
            elif rank_method == "hidden_norm":
                contrib[b] = np.linalg.norm(bh)
            else:
                raise ValueError(f"Unknown rank_method: {rank_method}")
        order = np.argsort(contrib)  # ascending: least important first

    # Concentration metrics
    total_energy = contrib.sum()
    if total_energy > 0:
        normed = contrib / total_energy
        top10 = contrib[-max(1, int(n_blocks*0.10)):]
        top25 = contrib[-max(1, int(n_blocks*0.25)):]
        top50 = contrib[-max(1, int(n_blocks*0.50)):]
        concentration = {
            "top_10pct_energy": float(top10.sum() / total_energy),
            "top_25pct_energy": float(top25.sum() / total_energy),
            "top_50pct_energy": float(top50.sum() / total_energy),
        }
        # Effective block count (1/sum(p_i^2))
        eff = 1.0 / float(np.sum(normed**2)) if np.sum(normed**2) > 0 else n_blocks
        concentration["effective_block_count"] = eff
        concentration["normalized_eff_count"] = eff / n_blocks
        concentration["gini"] = gini_from_contrib(np.sort(contrib))
    else:
        concentration = {}

    # Skip rate curve: try keeping k most-important blocks
    curve = []  # (n_keep, skip_rate, cos_raw, rel_l2_raw, cos_rescaled, rel_l2_rescaled, alpha)
    for n_keep in range(1, n_blocks + 1):
        keep = order[-n_keep:]
        kept_sum = np.zeros_like(full)
        for b in keep:
            s, e = b*block_size, min((b+1)*block_size, len(hidden))
            kept_sum += W_down[:, s:e] @ hidden[s:e]

        sr = 1.0 - n_keep / n_blocks
        c_raw = cosim(full, kept_sum)
        l2_raw = rel2(full, kept_sum)
        alpha, rescaled = rescale_correct(full, kept_sum)
        c_resc = cosim(full, rescaled)
        l2_resc = rel2(full, rescaled)
        curve.append((n_keep, sr, c_raw, l2_raw, c_resc, l2_resc, alpha))

    # For each threshold pair, find max skip rate passing BOTH cos AND l2_raw
    results = {}
    for ct in cos_thresholds:
        for lt in l2_thresholds:
            passing = [(nk, sr, cr, lr) for nk, sr, cr, lr, _, l2_r, _ in curve
                       if cr >= ct and lr <= lt]
            if passing:
                best = max(passing, key=lambda x: x[1])
                results[f"cos{int(ct*1000)}_l2{int(lt*100)}"] = {
                    "skip_rate": best[1], "n_keep": best[0],
                    "cos_raw": best[2], "rel_l2_raw": best[3]}
            else:
                results[f"cos{int(ct*1000)}_l2{int(lt*100)}"] = {"skip_rate": 0.0}

    # Per-threshold standalone (cos only, l2 only)
    for ct in cos_thresholds:
        passing = [(nk, sr, cr, lr) for nk, sr, cr, lr, _, l2_r, _ in curve if cr >= ct]
        if passing:
            best = max(passing, key=lambda x: x[1])
            results[f"cos_only_{int(ct*1000)}"] = {"skip_rate": best[1], "cos_raw": best[2]}
        else:
            results[f"cos_only_{int(ct*1000)}"] = {"skip_rate": 0.0}

    for lt in l2_thresholds:
        passing = [(nk, sr, cr, lr) for nk, sr, cr, lr, _, l2_r, _ in curve if lr <= lt]
        if passing:
            best = max(passing, key=lambda x: x[1])
            results[f"l2_only_{int(lt*100)}"] = {"skip_rate": best[1], "rel_l2_raw": best[3]}
        else:
            results[f"l2_only_{int(lt*100)}"] = {"skip_rate": 0.0}

    # Max skip at any quality
    if curve:
        best_any = max(curve, key=lambda x: x[1])
        results["max_skip_any"] = {
            "skip_rate": best_any[1], "cos_raw": best_any[2], "rel_l2_raw": best_any[3],
            "cos_rescaled": best_any[4], "rel_l2_rescaled": best_any[5], "alpha": best_any[6]}

    return results, concentration, contrib, order

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def gen_sample(h_dim, ffn_dim, rng, sparsity_pct=40, burst_pct=10, burst_mult=5.0):
    """Generate synthetic FFN hidden vector and down weight matrix."""
    W_down = rng.standard_normal((h_dim, ffn_dim)).astype(np.float32)
    h_raw = rng.standard_normal(ffn_dim).astype(np.float32)
    mag = np.abs(h_raw)
    thr = np.percentile(mag, sparsity_pct)
    h = np.where(mag < thr, np.float32(0.0), h_raw)
    burst_idx = np.where(rng.uniform(0, 1, ffn_dim) < burst_pct)[0]
    h[burst_idx] *= np.float32(burst_mult)
    return h.astype(np.float32), W_down.astype(np.float32)

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
COS_THRESHOLDS = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
L2_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
BLOCK_SIZES = [64, 128, 256, 512, 1024]
RANK_METHODS = ["contribution_norm", "hidden_norm", "random", "reverse_contribution"]
HIDDEN_DIM = 2048  # Reduced for memory efficiency
FFN_DIM = 11008
N_SAMPLES = 64     # Moderate sample count
SEED = 1234

def run_sweep():
    results_by_bs = {}
    t0 = time.time()
    for bs in BLOCK_SIZES:
        nb = math.ceil(FFN_DIM / bs)
        print(f"  block_size={bs:4d} n_blocks={nb:3d}...", end="", flush=True)
        by_method = {}
        for method in RANK_METHODS:
            cos_0998 = []; cos_0995 = []; l2_010_resc = []; n_pass_strict = []
            concentration_acc = {
                "top_10": [], "top_25": [], "top_50": [], "eff": [], "gini": []}
            for s in range(N_SAMPLES):
                rng = np.random.default_rng(SEED + s)
                h, W = gen_sample(HIDDEN_DIM, FFN_DIM, rng)
                oracle_res, conc, _, _ = oracle_analysis(
                    h, W, bs, COS_THRESHOLDS, L2_THRESHOLDS, rank_method=method)

                # Collect key metrics
                cos_0998.append(oracle_res.get(f"cos_only_{int(0.98*1000)}", {}).get("skip_rate", 0.0))
                cos_0995.append(oracle_res.get(f"cos_only_{int(0.995*1000)}", {}).get("skip_rate", 0.0))
                strict = oracle_res.get(f"cos{int(0.98*1000)}_l2{int(0.10*100)}", {})
                l2_resc_entry = oracle_res.get("max_skip_any", {})
                l2_010_resc.append(l2_resc_entry.get("rel_l2_rescaled", 1.0) if
                                  l2_resc_entry.get("rel_l2_rescaled", 1.0) <= 0.10 else 1.0)
                n_pass_strict.append(strict.get("skip_rate", 0.0))

                for key, lst in [("top_10", concentration_acc["top_10"]),
                                 ("top_25", concentration_acc["top_25"]),
                                 ("top_50", concentration_acc["top_50"]),
                                 ("eff", concentration_acc["eff"]),
                                 ("gini", concentration_acc["gini"])]:
                    ckey = "top_10pct_energy" if key == "top_10" else \
                           "top_25pct_energy" if key == "top_25" else \
                           "top_50pct_energy" if key == "top_50" else \
                           "effective_block_count" if key == "eff" else "gini"
                    if ckey in conc:
                        lst.append(conc[ckey])

            by_method[method] = {
                "avg_skip_cos0.98": float(np.mean(cos_0998)),
                "avg_skip_cos0.995": float(np.mean(cos_0995)),
                "avg_l2_rescaled_le0.10": float(np.mean(l2_010_resc)),
                "avg_strict_pass_rate": float(np.mean(n_pass_strict)),
                "concentration": {k: float(np.mean(v)) if v else 0.0
                                  for k, v in concentration_acc.items()}}
        results_by_bs[bs] = by_method
        print(" done", flush=True)
    elapsed = time.time() - t0
    return results_by_bs, elapsed

def main():
    print(f"SBS-Gate Oracle 0B — hidden={HIDDEN_DIM} ffn={FFN_DIM}", flush=True)
    print(f"  samples={N_SAMPLES} seed={SEED}", flush=True)
    print(f"  block_sizes={BLOCK_SIZES}", flush=True)
    print(f"  rank_methods={RANK_METHODS}", flush=True)
    print(f"  cos_thresholds={COS_THRESHOLDS}", flush=True)
    print(f"  l2_thresholds={L2_THRESHOLDS}", flush=True)

    results, elapsed = run_sweep()

    print(f"\n=== Results === (elapsed: {elapsed:.1f}s)", flush=True)
    print(f"\n{'blk_sz':>7} {'n_blk':>6} {'method':>20} {'skip@0.98':>9} {'skip@0.995':>10} "
          f"{'strict_pass':>11} {'top10%':>8} {'eff_bks':>9} {'gini':>8}", flush=True)
    print("-" * 105, flush=True)

    for bs in BLOCK_SIZES:
        nb = math.ceil(FFN_DIM / bs)
        for method in RANK_METHODS:
            m = results[bs][method]
            c = m["concentration"]
            strict_pass = m["avg_strict_pass_rate"]
            print(f"{bs:>7} {nb:>6} {method:>20} "
                  f"{m['avg_skip_cos0.98']*100:>8.1f}% "
                  f"{m['avg_skip_cos0.995']*100:>9.1f}% "
                  f"{strict_pass*100:>10.1f}% "
                  f"{c.get('top_10pct_energy',0)*100:>7.1f}% "
                  f"{c.get('effective_block_count',0):>9.1f} "
                  f"{c.get('gini',0):>8.3f}", flush=True)
        print()

    # Rescale analysis: compare raw vs rescaled at different skip levels
    print("=== Rescale Diagnostic ===", flush=True)
    print("Testing whether residual error is mostly norm (rescaleable) vs directional.", flush=True)
    rng = np.random.default_rng(SEED)
    h, W = gen_sample(HIDDEN_DIM, FFN_DIM, rng)
    bs = 256; nb = math.ceil(FFN_DIM / bs)
    _, _, _, order = oracle_analysis(h, W, bs, COS_THRESHOLDS, L2_THRESHOLDS, "contribution_norm")
    full = W @ h
    print(f"{'n_keep':>8} {'skip%':>7} {'cos_raw':>9} {'l2_raw':>9} {'alpha':>8} "
          f"{'cos_resc':>9} {'l2_resc':>9}", flush=True)
    print("-" * 70, flush=True)
    for nk in [43, 40, 35, 30, 25, 20, 15, 10, 5]:
        keep = order[-nk:]
        kept = sum([W[:, b*bs:min((b+1)*bs,FFN_DIM)] @ h[b*bs:min((b+1)*bs,FFN_DIM)] for b in keep])
        sr = 1 - nk/nb
        cr = cosim(full, kept); lr = rel2(full, kept)
        alpha, resc = rescale_correct(full, kept)
        cr_r = cosim(full, resc); lr_r = rel2(full, resc)
        print(f"{nk:>8} {sr*100:>6.1f}% {cr:>9.5f} {lr:>9.5f} {alpha:>8.3f} "
              f"{cr_r:>9.5f} {lr_r:>9.5f}", flush=True)

    output = {
        "hidden_dim": HIDDEN_DIM, "ffn_dim": FFN_DIM,
        "n_samples": N_SAMPLES, "seed": SEED,
        "block_sizes": BLOCK_SIZES, "rank_methods": RANK_METHODS,
        "cos_thresholds": COS_THRESHOLDS, "l2_thresholds": L2_THRESHOLDS,
        "elapsed_s": elapsed, "results": results}
    with open("/tmp/sbs_gate_oracle_0b_sweep.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: /tmp/sbs_gate_oracle_0b_sweep.json", flush=True)

if __name__ == "__main__":
    main()
