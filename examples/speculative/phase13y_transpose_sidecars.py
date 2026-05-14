#!/usr/bin/env python3
"""
PRT Phase 13Y: Transpose sidecar generator.
Reads original [ffn,hidden] sidecars and writes transposed [hidden,ffn] sidecars.
Only for experimental layout comparison — NOT for production use.
"""
import struct, os, json, sys, argparse
import numpy as np

def transpose_sidecar(src_path, dst_path, M, N, layer):
    """Read [N,M]=[ffn,hidden] file, write [M,N]=[hidden,ffn] transposed."""
    with open(src_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
    
    expected = N * M
    if len(data) != expected:
        print(f"  ERROR: layer {layer}: expected {expected} floats, got {len(data)}")
        return None
    
    orig = data.reshape((N, M))       # [ffn, hidden]
    transposed = orig.T               # [hidden, ffn]
    
    assert transposed.shape == (M, N), f"Shape mismatch: {transposed.shape} vs ({M},{N})"
    
    with open(dst_path, 'wb') as f:
        f.write(transposed.astype(np.float32).tobytes())
    
    # Stats
    s = transposed.flatten()
    stats = {
        "layer": layer,
        "size_bytes": os.path.getsize(dst_path),
        "checksum": float(s.sum()),
        "sum_abs": float(np.abs(s).sum()),
        "min": float(s.min()), "max": float(s.max()),
        "mean": float(s.mean()), "std": float(s.std()),
        "zero_frac": float((s == 0).sum() / len(s)),
        "neg_frac": float((s < 0).sum() / len(s)),
        "finite": int(np.isfinite(s).sum()),
    }
    
    # Verify
    with open(dst_path, 'rb') as f:
        verify = np.frombuffer(f.read(), dtype=np.float32)
    assert np.allclose(verify, s), f"Layer {layer}: verify failed"
    
    return stats

def main():
    ap = argparse.ArgumentParser(description="PRT Phase 13Y: transpose sidecar files")
    ap.add_argument("--src-dir", required=True, help="Original sidecar dir")
    ap.add_argument("--dst-dir", required=True, help="Output transposed dir")
    ap.add_argument("--hidden", type=int, required=True, help="M = hidden dim")
    ap.add_argument("--ffn", type=int, required=True, help="N = ffn dim")
    ap.add_argument("--layers", type=int, default=24, help="Number of layers")
    args = ap.parse_args()
    
    os.makedirs(args.dst_dir, exist_ok=True)
    
    all_stats = {}
    for layer in range(args.layers):
        src = f"{args.src_dir}/ffn_up_layer{layer}_prt.bin"
        dst = f"{args.dst_dir}/ffn_up_layer{layer}_prt.bin"
        
        if not os.path.exists(src):
            print(f"  SKIP layer {layer}: {src} not found")
            continue
        
        print(f"Layer {layer:2d}: {args.hidden}x{args.ffn} -> {args.ffn}x{args.hidden}...", end=" ", flush=True)
        stats = transpose_sidecar(src, dst, args.hidden, args.ffn, layer)
        all_stats[layer] = stats
        print(f"OK  size={stats['size_bytes']//1024}KB  sum={stats['checksum']:.4f}")
    
    # Write stats
    stats_path = os.path.join(args.dst_dir, "stats.json")
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    n = len(all_stats)
    print(f"\nDone. {n}/{args.layers} layers transposed.")
    print(f"Stats: {stats_path}")
    return 0

if __name__ == '__main__':
    sys.exit(main())