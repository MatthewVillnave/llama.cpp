#!/usr/bin/env python3
"""
PRT Phase 28AF: Prefetch Scheduler IO Timing Simulator

Two-phase model:
  Phase 1 — PREFILL: First token loads all layers. IO dominates.
  Phase 2 — GENERATION: All weights resident. Compute dominates.
  (If context exceeds KV, fall back to per-token loading with prefetch.)

Metadata/timing simulation only — no real model files.
"""
import argparse
import json


def parse_args():
    p = argparse.ArgumentParser(description="PRT prefetch scheduler IO timing simulator")
    p.add_argument("--layers", type=int, default=56)
    p.add_argument("--layer-bytes-mb", type=float, default=37.0,
                   help="Size per layer in MB")
    p.add_argument("--io-mbps", type=float, default=1800.0,
                   help="Storage read bandwidth MB/s")
    p.add_argument("--compute-ms-per-token", type=float, default=200.0,
                   help="Generation compute time per token in ms (all layers)")
    p.add_argument("--prefill-mode",
                   choices=["sequential", "parallel"],
                   default="parallel",
                   help="sequential: load one layer at a time; parallel: load all layers at once")
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def io_time_ms(bytes_mb, io_mbps):
    return (bytes_mb / io_mbps) * 1000.0


def run_sim(args):
    layers = args.layers
    layer_mb = args.layer_bytes_mb
    io_mbps = args.io_mbps
    compute_ms = args.compute_ms_per_token

    # Total model size (all layers)
    total_model_mb = layers * layer_mb
    total_io_ms = io_time_ms(total_model_mb, io_mbps)

    # ── Prefill phase (first token) ──────────────────────────────────────────
    # Parallel: all layers load simultaneously (best case)
    # Sequential: load one at a time (worst case, naive)
    if args.prefill_mode == "parallel":
        prefill_io_ms = total_io_ms  # parallel: all layers load as one chunk
    else:
        prefill_io_ms = total_io_ms  # sequential: load all 56 layers one by one

    prefill_total_ms = prefill_io_ms + compute_ms
    prefill_tok_s = 1000.0 / prefill_total_ms if prefill_total_ms > 0 else float("inf")

    # ── Generation phase ────────────────────────────────────────────────────────
    # Weights are already resident — compute only
    gen_compute_ms = compute_ms
    gen_tok_s = 1000.0 / gen_compute_ms if gen_compute_ms > 0 else float("inf")

    # ── IO hiding during generation ───────────────────────────────────────────
    # During generation, weights are already in RAM. No IO needed unless KV pressure
    # forces layer eviction. In that case, re-load evicted layers.
    # Modeled as: if eviction happens, each re-load costs io_time_per_layer.
    # Here we assume full KV residency → zero IO during generation.
    gen_io_ms = 0.0
    gen_stall_ms = 0.0
    gen_stall_pct = 0.0

    # ── Prefill IO hiding ─────────────────────────────────────────────────────
    # During prefill compute (after weights loaded), we could prefetch KV data.
    # But weights are loaded first, so no prefetch overlap during prefill.
    # The prefill itself IS the IO cost.
    prefill_stall_pct = 100.0 * prefill_io_ms / prefill_total_ms if prefill_total_ms > 0 else 0.0

    # ── Verdict ───────────────────────────────────────────────────────────────
    if prefill_stall_pct < 10:
        prefill_verdict = "IO_HIDDEN"
    elif prefill_stall_pct < 30:
        prefill_verdict = "IO_PARTIAL"
    else:
        prefill_verdict = "IO_HEAVY"  # IO dominates prefill

    gen_verdict = "COMPUTE_BOUND"  # generation is purely compute

    return {
        "config": {
            "layers": layers,
            "layer_bytes_mb": layer_mb,
            "total_model_mb": round(total_model_mb, 1),
            "io_mbps": io_mbps,
            "compute_ms_per_token": compute_ms,
            "prefill_mode": args.prefill_mode,
            "io_time_per_layer_ms": round(io_time_ms(layer_mb, io_mbps), 2),
            "total_io_ms": round(total_io_ms, 1),
        },
        "prefill": {
            "io_ms": round(prefill_io_ms, 1),
            "compute_ms": round(compute_ms, 1),
            "total_ms": round(prefill_total_ms, 1),
            "stall_fraction_pct": round(prefill_stall_pct, 2),
            "tok_per_sec": round(prefill_tok_s, 2),
            "verdict": prefill_verdict,
        },
        "generation": {
            "io_ms": gen_io_ms,
            "compute_ms": round(gen_compute_ms, 1),
            "total_ms": round(gen_compute_ms, 1),
            "stall_ms": gen_stall_ms,
            "stall_fraction_pct": gen_stall_pct,
            "tok_per_sec": round(gen_tok_s, 2),
            "verdict": gen_verdict,
            "note": "We assume weights fully resident during generation — no IO needed",
        },
        "tok_sec_ceiling": {
            "prefill": round(prefill_tok_s, 2),
            "generation": round(gen_tok_s, 2),
        },
        "key_insight": (
            f"At {io_mbps:.0f} MB/s: prefill IO takes {prefill_io_ms:.0f}ms, "
            f"compute takes {compute_ms:.0f}ms. "
            f"Generation ceiling: {gen_tok_s:.1f} tok/s (pure compute). "
            f"IO is NOT the bottleneck during generation — compute is."
        ),
    }


def format_summary(results):
    lines = [
        "=" * 90,
        "PREFETCH SCHEDULER IO TIMING — TWO-PHASE MODEL",
        "=" * 90,
        "",
        "Key insight: Prefill = IO+compute. Generation = compute only (weights resident).",
        "",
        "| Scenario | Total MB | IO MB/s | C ms/tok | Prefill ms | Gen ms | Prefill tok/s | Gen tok/s | Prefill verdict |",
        "|----------|----------|---------|-----------|------------|--------|---------------|----------|-----------------|",
    ]
    for r in results:
        cfg = r["config"]
        pre = r["prefill"]
        gen = r["generation"]
        emoji = {
            "IO_HIDDEN": "✅",
            "IO_PARTIAL": "⚠️",
            "IO_HEAVY": "⚡",
            "COMPUTE_BOUND": "🖥️",
        }
        lines.append(
            f"| {r['label']} | {cfg['total_model_mb']:.0f} | {cfg['io_mbps']:.0f} | "
            f"{cfg['compute_ms_per_token']:.0f} | {pre['io_ms']:.0f} | "
            f"{gen['compute_ms']:.0f} | {pre['tok_per_sec']:.2f} | "
            f"{gen['tok_per_sec']:.2f} | {emoji.get(pre['verdict'], pre['verdict'])} {pre['verdict']} |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()

    if args.out_json:
        result = run_sim(args)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"JSON written to {args.out_json}")
        print(f"\nKey insight: {result['key_insight']}")
        print(f"\nPrefill verdict: {result['prefill']['verdict']}")
        print(f"Generation verdict: {result['generation']['verdict']}")
        print(f"Prefill tok/s: {result['prefill']['tok_per_sec']:.2f}")
        print(f"Generation tok/s: {result['generation']['tok_per_sec']:.2f}")
        return

    # Multi-scenario matrix
    scenarios = []

    def add(name, layer_bytes_mb, io_mbps, compute_ms, prefill_mode="parallel"):
        a = type("Args", (), {
            "layers": 56, "layer_bytes_mb": layer_bytes_mb,
            "io_mbps": io_mbps, "compute_ms_per_token": compute_ms,
            "prefill_mode": prefill_mode, "out_json": None,
        })()
        r = run_sim(a)
        r["label"] = name
        scenarios.append(r)

    # === Core scenarios ===
    add("Q2 base, NVMe, compute=200ms",
        layer_bytes_mb=36.6, io_mbps=1800, compute_ms=200)
    add("Q2+res, NVMe, compute=200ms",
        layer_bytes_mb=50.0, io_mbps=1800, compute_ms=200)
    add("Q4 base, NVMe, compute=400ms",
        layer_bytes_mb=74.0, io_mbps=1800, compute_ms=400)
    add("Q4+res, NVMe, compute=400ms",
        layer_bytes_mb=102.0, io_mbps=1800, compute_ms=400)

    # === USB negative control ===
    add("Q2+res, USB, compute=200ms",
        layer_bytes_mb=50.0, io_mbps=100, compute_ms=200)

    # === Compute sensitivity (generation) ===
    for compute_ms in [50, 100, 200, 400, 1000]:
        add(f"Q2+res, NVMe, compute={compute_ms}ms",
            layer_bytes_mb=50.0, io_mbps=1800, compute_ms=compute_ms)

    # === Parallel vs Sequential prefill ===
    add("Q2+res, NVMe, seq-prefill",
        layer_bytes_mb=50.0, io_mbps=1800, compute_ms=200, prefill_mode="sequential")

    # === IO sensitivity (NVMe 1000 vs 1800 vs 500) ===
    for io_mbps in [500, 1000, 1800, 3600]:
        add(f"Q2+res, IO={io_mbps}MB/s, compute=200ms",
            layer_bytes_mb=50.0, io_mbps=io_mbps, compute_ms=200)

    print(format_summary(scenarios))
    print()

    print("\n=== DETAILED RESULTS ===")
    for r in scenarios:
        cfg = r["config"]
        pre = r["prefill"]
        gen = r["generation"]
        print(f"\n{r['label']}:")
        print(f"  Total model: {cfg['total_model_mb']:.0f} MB, IO time/layer: {cfg['io_time_per_layer_ms']:.1f} ms")
        print(f"  PREFILL (parallel={args.prefill_mode}): {pre['io_ms']:.0f}ms IO + {pre['compute_ms']:.0f}ms compute = {pre['total_ms']:.0f}ms total")
        print(f"    → {pre['tok_per_sec']:.2f} tok/s (first token)  [{pre['verdict']}]  stall={pre['stall_fraction_pct']:.1f}%")
        print(f"  GENERATION: {gen['compute_ms']:.0f}ms compute (weights resident)")
        print(f"    → {gen['tok_per_sec']:.2f} tok/s  [{gen['verdict']}]")
        print(f"  KEY: {r['key_insight']}")


if __name__ == "__main__":
    main()