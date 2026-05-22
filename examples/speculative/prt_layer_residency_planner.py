#!/usr/bin/env python3
"""
PRT Phase 28AD: Simulated Layer Residency Planner
Simulates autoregressive layer traversal to estimate peak resident memory.
Metadata-only simulation — no real model files.
"""
import argparse
import json
import math
import sys
import os


def parse_args():
    p = argparse.ArgumentParser(description="PRT simulated layer residency planner")
    p.add_argument("--layers", type=int, default=56,
                   help="Total number of layers (default 56 for 30B)")
    p.add_argument("--layer-bytes-mb", type=float, default=37.0,
                   help="Q2 base bytes per layer in MB (default 37.0 for 30B)")
    p.add_argument("--residual-bytes-mb", type=float, default=0.0,
                   help="Residual bytes per layer in MB (default 0)")
    p.add_argument("--residual-per-family-mb", type=float, default=None,
                   help="Per-family residual MB (for MLP-specific simulation)")
    p.add_argument("--window-size", type=int, default=4,
                   help="Max layers resident at once (default 4)")
    p.add_argument("--prefetch-distance", type=int, default=1,
                   help="How many layers ahead to prefetch (default 1)")
    p.add_argument("--kv-mb", type=float, default=2.0,
                   help="KV context bytes in MB (default 2.0)")
    p.add_argument("--context-size", type=int, default=1024,
                   help="Context length in tokens (affects KV)")
    p.add_argument("--runtime-buffer-mb", type=float, default=1024.0,
                   help="Runtime buffer MB (default 1024)")
    p.add_argument("--os-headroom-mb", type=float, default=2048.0,
                   help="OS headroom MB (default 2048)")
    p.add_argument("--ram-gb", type=float, default=16.0,
                   help="Available RAM GB (default 16.0)")
    p.add_argument("--policy",
                   choices=["base_only", "mlp_all", "attention_partial",
                            "all_residuals", "budget_greedy"],
                   default="base_only",
                   help="Residual loading policy")
    p.add_argument("--residual-budget-mb", type=float, default=None,
                   help="Max residual budget in MB (budget_greedy)")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Number of tokens to simulate (default 128)")
    p.add_argument("--out-json", default=None, help="Output JSON")
    p.add_argument("--out-md", default=None, help="Output markdown table")
    return p.parse_args()


class ResidencySimulator:
    """
    Simulates autoregressive layer traversal with paging.
    Tracks per-token and peak resident memory.
    """

    def __init__(self, args):
        self.layers = args.layers
        self.layer_bytes = args.layer_bytes_mb * 1024 * 1024  # bytes
        self.residual_bytes = args.residual_bytes_mb * 1024 * 1024
        self.window_size = args.window_size
        self.prefetch_distance = args.prefetch_distance
        self.kv_bytes = args.kv_mb * 1024 * 1024
        self.runtime_buffer = args.runtime_buffer_mb * 1024 * 1024
        self.os_headroom = args.os_headroom_mb * 1024 * 1024
        self.ram_budget = args.ram_gb * 1024 * 1024 * 1024
        self.policy = args.policy
        self.residual_budget = (args.residual_budget_mb * 1024 * 1024
                                if args.residual_budget_mb else None)
        self.max_tokens = args.max_tokens
        self.args = args

        # State
        self.resident = {}  # layer_idx -> bytes resident
        self.prefetched = {}  # layer_idx -> bytes (prefetched, not yet active)
        self.eviction_count = 0
        self.prefetch_count = 0
        self.token_history = []  # per-token snapshots
        self.peak_resident = 0
        self.peak_residual = 0
        self.peak_total = 0
        self.swap_violation = False

    def resident_weight_bytes(self):
        return sum(self.resident.values()) + sum(self.prefetched.values())

    def resident_residual_bytes(self):
        return sum(v for k, v in self.resident.items() if isinstance(k, str) and k.startswith("_res_")) + \
               sum(v for k, v in self.prefetched.items() if isinstance(k, str) and k.startswith("_res_"))

    def total_resident(self):
        return (self.resident_weight_bytes() +
                self.kv_bytes +
                self.runtime_buffer +
                self.os_headroom)

    def available_for_weights(self):
        return self.ram_budget - self.kv_bytes - self.runtime_buffer - self.os_headroom

    def _select_residual_for_layer(self, layer_idx, current_resident_bytes):
        """Select residual bytes based on policy. Returns residual bytes to load."""
        if self.policy == "base_only":
            return 0

        elif self.policy == "all_residuals":
            return self.residual_bytes

        elif self.policy == "mlp_all":
            # MLP = ffn_up + ffn_down + ffn_gate = ~36.6 MB per layer in Q2+ternary
            # MLP residual ~13 MB/layer (from Phase 28AC)
            return self.residual_bytes * 0.88  # MLP = 88% of memory

        elif self.policy == "attention_partial":
            # Attention = attn_q + attn_output = ~101 MB per layer in Q4
            # Attention residual ~101 MB... no wait
            # attn residual ~3.6 MB/layer (from Phase 28AC: attn_q + attn_output)
            return self.residual_bytes * 0.12  # attention = 12% of memory

        elif self.policy == "budget_greedy":
            if self.residual_budget is None:
                return self.residual_bytes
            # Budget greedy: fill until residual budget exhausted
            # Use resident_residual_bytes() — tracks ACTUAL residual in memory
            current_residual = self.resident_residual_bytes()
            remaining = self.residual_budget - current_residual
            # Allocate: cap at remaining budget, never exceed residual budget
            # If remaining >= full residual: allocate full residual
            # If 0 < remaining < full residual: allocate min(remaining, residual_bytes)
            #   (this is a genuine partial allocation up to available budget)
            # If remaining <= 0: allocate nothing
            if remaining <= 0:
                return 0
            if remaining >= self.residual_bytes:
                return self.residual_bytes
            # remaining < residual_bytes: partial budget available
            return min(remaining, self.residual_bytes)

        return 0

    def _evict_if_needed(self, current_layer):
        """Evict base layers and residuals independently.
        
        Base layers: maintain window_size active layers (IRAM constraint).
        Residuals: maintain residual_budget ceiling (memory constraint).
        
        Both evictions happen in the same call to keep them in sync.
        """
        # Evict oldest base layers when window exceeds size
        int_keys = [k for k in self.resident.keys() if isinstance(k, int)]
        while len(int_keys) > self.window_size:
            evict_candidates = [k for k in int_keys if k < current_layer]
            if not evict_candidates:
                break
            oldest = min(evict_candidates)
            del self.resident[oldest]
            res_key = f"_res_{oldest}"
            if res_key in self.resident:
                del self.resident[res_key]
            self.eviction_count += 1
            int_keys = [k for k in self.resident.keys() if isinstance(k, int)]

        # Evict oldest residuals when residual budget exceeded
        if self.residual_budget is not None:
            while self.resident_residual_bytes() > self.residual_budget:
                res_keys = sorted([k for k in self.resident.keys()
                                   if isinstance(k, str) and k.startswith("_res_")],
                                  key=lambda x: int(x.split("_")[2]))
                if not res_keys:
                    break
                oldest_res = res_keys[0]
                del self.resident[oldest_res]
                self.eviction_count += 1

    def _prefetch_ahead(self, current_layer, token_idx):
        """Prefetch next N layers."""
        for offset in range(1, self.prefetch_distance + 1):
            next_layer = current_layer + offset
            if next_layer < self.layers and next_layer not in self.resident:
                self.resident[next_layer] = self.layer_bytes
                self.prefetch_count += 1

    def simulate_token(self, token_idx, start_layer=0):
        """Simulate one token's layer traversal. Returns peak for this token."""
        token_resident = {}

        for layer_idx in range(start_layer, self.layers):
            # Ensure layer is resident
            if layer_idx not in self.resident:
                self.resident[layer_idx] = self.layer_bytes
                self.prefetch_count += 1

            # Evict if needed BEFORE adding new residuals
            self._evict_if_needed(layer_idx)

            # Also evict residuals independently if budget is exceeded
            # (do this BEFORE adding the new residual so budget never exceeds)
            if self.residual_budget is not None:
                while self.resident_residual_bytes() > self.residual_budget:
                    res_keys = sorted([k for k in self.resident.keys()
                                       if isinstance(k, str) and k.startswith("_res_")],
                                      key=lambda x: int(x.split("_")[2]))
                    if not res_keys:
                        break
                    oldest_res = res_keys[0]
                    del self.resident[oldest_res]
                    self.eviction_count += 1

            # Select residual based on policy
            current_weight = self.resident_weight_bytes()
            residual_to_load = self._select_residual_for_layer(layer_idx, current_weight)

            if residual_to_load > 0:
                res_key = f"_res_{layer_idx}"
                self.resident[res_key] = residual_to_load

            # Record peak for this layer
            total_now = self.total_resident()
            if total_now > self.peak_total:
                self.peak_total = total_now
            self.peak_resident = max(self.peak_resident, self.resident_weight_bytes())

            # Prefetch ahead
            self._prefetch_ahead(layer_idx, token_idx)

            token_resident[layer_idx] = {
                "resident_bytes": self.resident_weight_bytes(),
                "total_resident": total_now,
                "layers_in_memory": len([k for k in self.resident.keys() if isinstance(k, int)]),
            }

        # Record snapshot
        self.token_history.append({
            "token": token_idx,
            "peak_layer_resident": self.resident_weight_bytes(),
            "peak_total": self.peak_total,
            "active_layers": len([k for k in self.resident.keys() if isinstance(k, int)]),
        })

        return token_resident

    def run(self):
        """Run full simulation across max_tokens."""
        for token_idx in range(self.max_tokens):
            self.simulate_token(token_idx)

        return self.summarize()

    def summarize(self):
        """Compute summary statistics."""
        total_resident_peak = self.peak_resident + self.kv_bytes + self.runtime_buffer + self.os_headroom
        safe = total_resident_peak < self.ram_budget
        remaining = self.ram_budget - total_resident_peak
        io_per_token = (self.layer_bytes + self.residual_bytes) / (1024 * 1024)

        return {
            "scenario": {
                "layers": self.layers,
                "layer_bytes_mb": self.layer_bytes / (1024 * 1024),
                "residual_bytes_mb": self.residual_bytes / (1024 * 1024),
                "window_size": self.window_size,
                "prefetch_distance": self.prefetch_distance,
                "policy": self.policy,
                "context_size": self.args.context_size,
                "kv_mb": self.args.kv_mb,
                "ram_gb": self.args.ram_gb,
            },
            "results": {
                "peak_layer_resident_mb": self.peak_resident / (1024 * 1024),
                "peak_total_resident_mb": self.peak_total / (1024 * 1024),
                "total_ram_budget_mb": self.ram_budget / (1024 * 1024),
                "headroom_mb": (self.ram_budget - self.peak_total) / (1024 * 1024),
                "safe": safe,
                "eviction_count": self.eviction_count,
                "prefetch_count": self.prefetch_count,
                "max_active_layers": max(t["active_layers"] for t in self.token_history) if self.token_history else 0,
                "io_per_token_mb": io_per_token,
            },
            "memory_breakdown": {
                "peak_layer_weights_mb": self.peak_resident / (1024 * 1024),
                "kv_mb": self.args.kv_mb,
                "runtime_buffer_mb": self.args.runtime_buffer_mb,
                "os_headroom_mb": self.args.os_headroom_mb,
                "peak_total_mb": self.peak_total / (1024 * 1024),
            },
            "available_for_weights_mb": self.available_for_weights() / (1024 * 1024),
        }


def run_scenario(args_dict, label):
    """Run one scenario and return result summary."""
    args = type("Args", (), args_dict)()
    sim = ResidencySimulator(args)
    result = sim.run()
    result["label"] = label
    return result


def format_md_table(results):
    """Format results as markdown table."""
    header = "| Scenario | Layers | Layer MB | Residual MB | Window | Context | Peak Resident MB | Headroom MB | SAFE | IO/token MB |"
    sep    = "|----------|--------|----------|------------|--------|--------|-----------------|-------------|------|-------------|"
    rows = []
    for r in results:
        s = r["scenario"]
        res = r["results"]
        mem = r["memory_breakdown"]
        rows.append(
            f"| {r['label']} | {s['layers']} | {s['layer_bytes_mb']:.1f} | "
            f"{s['residual_bytes_mb']:.1f} | {s['window_size']} | "
            f"{s['context_size']} | {mem['peak_total_mb']:.1f} | "
            f"{res['headroom_mb']:.1f} | "
            f"{'✅' if res['safe'] else '❌'} | "
            f"{res['io_per_token_mb']:.1f} |"
        )
    return "\n".join([header, sep] + rows)


def main():
    args = parse_args()

    # If --policy is set, run single scenario
    if args.policy:
        sim = ResidencySimulator(args)
        result = sim.run()
        print(f"Scenario: {args.policy}")
        print(f"  Peak layer resident: {result['results']['peak_layer_resident_mb']:.2f} MB")
        print(f"  Peak total resident: {result['results']['peak_total_resident_mb']:.2f} MB")
        print(f"  RAM budget: {result['scenario']['ram_gb']:.1f} GB")
        print(f"  Headroom: {result['results']['headroom_mb']:.2f} MB")
        print(f"  SAFE: {result['results']['safe']}")
        print(f"  Evictions: {result['results']['eviction_count']}")
        print(f"  Prefetches: {result['results']['prefetch_count']}")
        print(f"  IO/token: {result['results']['io_per_token_mb']:.1f} MB")
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(result, f, indent=2)
        if args.out_md:
            with open(args.out_md, "w") as f:
                f.write(format_md_table([result]))
        return

    # Run scenario matrix
    scenarios = []

    # 30B Q2 base, no residuals
    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 36.6, "residual_bytes_mb": 0.0,
        "window_size": 4, "prefetch_distance": 1,
        "kv_mb": 2.0, "context_size": 1024,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "base_only", "max_tokens": 128, "residual_budget_mb": None,
        "residual_per_family_mb": None,
    }, "30B Q2 base, c=1024, win=4"))

    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 36.6, "residual_bytes_mb": 0.0,
        "window_size": 4, "prefetch_distance": 1,
        "kv_mb": 4.0, "context_size": 2048,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "base_only", "max_tokens": 128, "residual_budget_mb": None,
        "residual_per_family_mb": None,
    }, "30B Q2 base, c=2048, win=4"))

    # 30B Q2 + all residuals
    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 36.6, "residual_bytes_mb": 13.4,
        "window_size": 4, "prefetch_distance": 1,
        "kv_mb": 2.0, "context_size": 1024,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "all_residuals", "max_tokens": 128, "residual_budget_mb": None,
        "residual_per_family_mb": None,
    }, "30B Q2+res, c=1024, win=4"))

    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 36.6, "residual_bytes_mb": 13.4,
        "window_size": 4, "prefetch_distance": 1,
        "kv_mb": 4.0, "context_size": 2048,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "all_residuals", "max_tokens": 128, "residual_budget_mb": None,
        "residual_per_family_mb": None,
    }, "30B Q2+res, c=2048, win=4"))

    # 30B Q4 baseline (Q4 base, no residuals — no residual savings with Q4 base)
    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 74.0, "residual_bytes_mb": 0.0,
        "window_size": 4, "prefetch_distance": 1,
        "kv_mb": 4.0, "context_size": 2048,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "base_only", "max_tokens": 128, "residual_budget_mb": None,
        "residual_per_family_mb": None,
    }, "30B Q4 base, c=2048, win=4"))

    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 74.0, "residual_bytes_mb": 0.0,
        "window_size": 4, "prefetch_distance": 1,
        "kv_mb": 8.0, "context_size": 4096,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "base_only", "max_tokens": 128, "residual_budget_mb": None,
        "residual_per_family_mb": None,
    }, "30B Q4 base, c=4096, win=4"))

    # Stress: larger window
    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 36.6, "residual_bytes_mb": 13.4,
        "window_size": 8, "prefetch_distance": 2,
        "kv_mb": 4.0, "context_size": 2048,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "all_residuals", "max_tokens": 128, "residual_budget_mb": None,
        "residual_per_family_mb": None,
    }, "30B Q2+res, c=2048, win=8, pref=2"))

    # 30B Q4+residuals at c=2048
    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 74.0, "residual_bytes_mb": 28.0,
        "window_size": 4, "prefetch_distance": 1,
        "kv_mb": 4.0, "context_size": 2048,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "all_residuals", "max_tokens": 128, "residual_budget_mb": None,
        "residual_per_family_mb": None,
    }, "30B Q4+res, c=2048, win=4"))

    # budget_greedy at 512MB residual budget
    scenarios.append(run_scenario({
        "layers": 56, "layer_bytes_mb": 36.6, "residual_bytes_mb": 13.4,
        "window_size": 4, "prefetch_distance": 1,
        "kv_mb": 4.0, "context_size": 2048,
        "runtime_buffer_mb": 1024.0, "os_headroom_mb": 2048.0, "ram_gb": 16.0,
        "policy": "budget_greedy", "max_tokens": 128, "residual_budget_mb": 512.0,
        "residual_per_family_mb": None,
    }, "30B Q2+res, c=2048, win=4, budget=512MB"))

    # Print results
    print("\n" + "=" * 80)
    print("LAYER RESIDENCY PLANNER — 30B SIMULATION RESULTS")
    print("=" * 80)
    print()
    print(format_md_table(scenarios))
    print()

    # Detailed summary
    print("\n=== DETAILED RESULTS ===")
    for r in scenarios:
        s = r["scenario"]
        res = r["results"]
        mem = r["memory_breakdown"]
        status = "✅ SAFE" if res["safe"] else "❌ UNSAFE"
        print(f"\n{r['label']}: {status}")
        print(f"  Peak resident: {mem['peak_total_mb']:.1f} MB "
              f"(layers={mem['peak_layer_weights_mb']:.1f}MB + "
              f"kv={mem['kv_mb']:.1f}MB + "
              f"buffer={mem['runtime_buffer_mb']:.1f}MB + "
              f"os={mem['os_headroom_mb']:.1f}MB)")
        print(f"  Headroom: {res['headroom_mb']:.1f} MB")
        print(f"  Evictions: {res['eviction_count']}, Prefetches: {res['prefetch_count']}")
        print(f"  IO/token: {res['io_per_token_mb']:.1f} MB")

    # Save JSON
    if args.out_json:
        output = {"scenarios": scenarios}
        with open(args.out_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON written to {args.out_json}")

    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write("# Phase 28AD: Simulated Layer Residency Planner\n\n")
            f.write(format_md_table(scenarios))
        print(f"Markdown written to {args.out_md}")


if __name__ == "__main__":
    main()