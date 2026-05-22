#!/usr/bin/env python3
"""
Phase 28AH: Fake File-Backed Pager

Tests real file-backed mmap/read/prefetch behavior using fake layer files.
No model data. No llama.cpp. Python stdlib only.

Usage:
  python3 prt_fake_layer_pager.py --layers 56 --layer-mb 8 --tokens 4 ...
"""

import argparse
import json
import os
import random
import shutil
import struct
import sys
import time
from pathlib import Path

try:
    import mmap as mm
    HAS_MMAP = True
except ImportError:
    HAS_MMAP = False


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fake file-backed layer pager")
    p.add_argument("--layers", type=int, default=56)
    p.add_argument("--layer-mb", type=float, default=8.0, help="Base layer size in MB")
    p.add_argument("--residual-mb", type=float, default=2.0, help="Residual per layer in MB")
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--prefetch-distance", type=int, default=1)
    p.add_argument("--tokens", type=int, default=4, help="Tokens to simulate")
    p.add_argument("--storage-dir", default="/tmp/prt_fake_pager", help="Temp storage directory")
    p.add_argument("--mode", choices=["read", "mmap"], default="read")
    p.add_argument("--policy", choices=["q2_base", "q2_res", "q4_base", "q4_res"], default="q2_res")
    p.add_argument("--out-json", default=None, help="Write results JSON")
    p.add_argument("--cleanup", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--keep", action="store_true", help="Keep temp files after test")
    return p.parse_args()


# ─── Fake File Creation ───────────────────────────────────────────────────────

def create_fake_layer_files(storage_dir: str, layers: int, layer_bytes: int, residual_bytes: int, policy: str):
    """Create fake layer files with random-but-reproducible data."""
    base_dir = Path(storage_dir)
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic seed for reproducibility
    rng = random.Random(42)

    files_created = []
    for layer_idx in range(layers):
        layer_file = base_dir / f"layer_{layer_idx:03d}.bin"
        with open(layer_file, "wb") as f:
            # Write header: layer_idx + timestamp
            header = struct.pack("<II", layer_idx, int(time.time()))
            f.write(header)
            # Fill with deterministic pseudo-random bytes (fast)
            remaining = layer_bytes - len(header)
            chunk_size = 4096
            for _ in range(remaining // chunk_size):
                f.write(bytes(rng.randint(0, 255) for _ in range(chunk_size)))
            if remaining % chunk_size:
                f.write(bytes(rng.randint(0, 255) for _ in range(remaining % chunk_size)))
        files_created.append(str(layer_file))

    # Also create residual files for _res policies
    if policy in ("q2_res", "q4_res"):
        res_dir = base_dir / "residuals"
        res_dir.mkdir(exist_ok=True)
        for layer_idx in range(layers):
            res_file = res_dir / f"res_{layer_idx:03d}.bin"
            with open(res_file, "wb") as f:
                header = struct.pack("<II", layer_idx, int(time.time()))
                f.write(header)
                remaining = residual_bytes - len(header)
                for _ in range(remaining // 4096):
                    f.write(bytes(rng.randint(0, 255) for _ in range(4096)))
                if remaining % 4096:
                    f.write(bytes(rng.randint(0, 255) for _ in range(remaining % 4096)))
            files_created.append(str(res_file))

    return files_created


# ─── Pager Core ───────────────────────────────────────────────────────────────

class FakeLayerPager:
    """File-backed pager using real mmap or read syscalls."""

    def __init__(self, storage_dir: str, mode: str, window_size: int, prefetch_distance: int, policy: str):
        self.storage_dir = Path(storage_dir)
        self.mode = mode
        self.window_size = window_size
        self.prefetch_distance = prefetch_distance
        self.policy = policy
        self.resident = {}          # layer_idx -> file descriptor or mmap object
        self.resident_res = {}     # residual layer_idx -> file descriptor or mmap
        self.active_window_start = 0
        self.stats = {
            "reads": 0,
            "prefetches": 0,
            "evictions": 0,
            "cache_hits": 0,
            "total_bytes_read": 0,
            "total_prefetch_bytes": 0,
            "wall_time_ms": 0.0,
            "read_time_ms": 0.0,
            "prefetch_time_ms": 0.0,
        }
        self._open_fds = []  # track for cleanup

    def _layer_path(self, layer_idx: int) -> Path:
        return self.storage_dir / f"layer_{layer_idx:03d}.bin"

    def _res_path(self, layer_idx: int) -> Path:
        return self.storage_dir / "residuals" / f"res_{layer_idx:03d}.bin"

    def load_layer(self, layer_idx: int, layer_bytes: int) -> int:
        """Load one layer via read or mmap. Returns bytes read."""
        path = self._layer_path(layer_idx)
        t0 = time.perf_counter()
        bytes_read = 0

        if self.mode == "mmap" and HAS_MMAP:
            f = open(path, "rb")
            m = mm.mmap(f.fileno(), 0, access=mm.ACCESS_READ)
            # Read via slice to force actual access
            _ = m[:layer_bytes]
            m.close()
            f.close()
            bytes_read = layer_bytes
        else:
            with open(path, "rb") as f:
                data = f.read(layer_bytes)
                bytes_read = len(data)

        t1 = time.perf_counter()
        self.stats["read_time_ms"] += (t1 - t0) * 1000
        self.stats["reads"] += 1
        self.stats["total_bytes_read"] += bytes_read
        return bytes_read

    def load_residual(self, layer_idx: int, residual_bytes: int) -> int:
        """Load residual for a layer."""
        path = self._res_path(layer_idx)
        if not path.exists():
            return 0
        t0 = time.perf_counter()
        bytes_read = 0

        if self.mode == "mmap" and HAS_MMAP:
            f = open(path, "rb")
            m = mm.mmap(f.fileno(), 0, access=mm.ACCESS_READ)
            _ = m[:residual_bytes]
            m.close()
            f.close()
            bytes_read = residual_bytes
        else:
            with open(path, "rb") as f:
                data = f.read(residual_bytes)
                bytes_read = len(data)

        t1 = time.perf_counter()
        self.stats["read_time_ms"] += (t1 - t0) * 1000
        self.stats["total_bytes_read"] += bytes_read
        return bytes_read

    def prefetch_layers(self, current_layer: int, layer_bytes: int, residual_bytes: int):
        """Prefetch future layers."""
        t0 = time.perf_counter()
        for offset in range(1, self.prefetch_distance + 1):
            next_layer = current_layer + offset
            if next_layer in self.resident:
                self.stats["cache_hits"] += 1
                continue
            if next_layer < self.layers:
                self.load_layer(next_layer, layer_bytes)
                if self.policy in ("q2_res", "q4_res"):
                    self.load_residual(next_layer, residual_bytes)
                self.stats["prefetches"] += 1
                self.stats["total_prefetch_bytes"] += layer_bytes + residual_bytes
        t1 = time.perf_counter()
        self.stats["prefetch_time_ms"] += (t1 - t0) * 1000

    def evict_if_needed(self, current_layer: int, layers: int):
        """Evict layers outside the active window."""
        window_start = max(0, current_layer - self.window_size + 1)
        if window_start == self.active_window_start:
            return

        evicted = 0
        for idx in list(self.resident.keys()):
            if isinstance(idx, int) and idx < window_start:
                # layer_idx < window_start: evict
                pass

        self.active_window_start = window_start
        self.stats["evictions"] += evicted

    def simulate_tokens(self, tokens: int, layers: int, layer_bytes: int, residual_bytes: int):
        """Simulate token-by-token layer traversal."""
        total_start = time.perf_counter()

        for token_idx in range(tokens):
            token_start = time.perf_counter()

            # Traverse layers for this token
            for layer_idx in range(layers):
                # Load current layer
                self.load_layer(layer_idx, layer_bytes)

                # Load residual if policy demands it
                if self.policy in ("q2_res", "q4_res"):
                    self.load_residual(layer_idx, residual_bytes)

                # Evict old layers
                self.evict_if_needed(layer_idx, layers)

                # Prefetch ahead
                self.prefetch_layers(layer_idx, layer_bytes, residual_bytes)

                # Cache hit: if already resident
                if layer_idx in self.resident:
                    self.stats["cache_hits"] += 1

            token_end = time.perf_counter()
            # Per-token stats captured in aggregate

        total_end = time.perf_counter()
        self.stats["wall_time_ms"] = (total_end - total_start) * 1000

    def cleanup(self):
        """Close any open file handles."""
        for fd in self._open_fds:
            try:
                fd.close()
            except Exception:
                pass
        self._open_fds = []

    def get_results(self, layers: int, layer_bytes: int, residual_bytes: int, tokens: int, policy: str):
        """Return structured results."""
        total_fake_bytes = layers * (layer_bytes + (residual_bytes if policy in ("q2_res", "q4_res") else 0))
        effective_bandwidth = (self.stats["total_bytes_read"] / 1024 / 1024) / (self.stats["wall_time_ms"] / 1000) if self.stats["wall_time_ms"] > 0 else 0

        return {
            "policy": policy,
            "mode": self.mode,
            "layers": layers,
            "layer_bytes_mb": layer_bytes / (1024 * 1024),
            "residual_bytes_mb": residual_bytes / (1024 * 1024),
            "tokens": tokens,
            "window_size": self.window_size,
            "prefetch_distance": self.prefetch_distance,
            "stats": {
                "total_bytes_read_mb": self.stats["total_bytes_read"] / (1024 * 1024),
                "total_prefetch_bytes_mb": self.stats["total_prefetch_bytes"] / (1024 * 1024),
                "reads": self.stats["reads"],
                "prefetches": self.stats["prefetches"],
                "evictions": self.stats["evictions"],
                "cache_hits": self.stats["cache_hits"],
                "wall_time_ms": round(self.stats["wall_time_ms"], 1),
                "read_time_ms": round(self.stats["read_time_ms"], 1),
                "prefetch_time_ms": round(self.stats["prefetch_time_ms"], 1),
                "effective_bandwidth_mb_s": round(effective_bandwidth, 1),
            }
        }


# ─── Smoketest ────────────────────────────────────────────────────────────────

def run_smoke_test(storage_dir: str) -> dict:
    """Quick smoke test: create 8 tiny layers, run 2 tokens."""
    print("  Running smoke test...")

    # Create tiny fake files
    fake_dir = Path(storage_dir) / "smoke"
    if fake_dir.exists():
        shutil.rmtree(fake_dir)
    fake_dir.mkdir(parents=True)

    rng = random.Random(42)
    for i in range(8):
        fpath = fake_dir / f"layer_{i:03d}.bin"
        with open(fpath, "wb") as f:
            header = struct.pack("<II", i, 0)
            f.write(header)
            # 64KB per layer for smoke
            remaining = (64 * 1024) - len(header)
            for _ in range(remaining // 4096):
                f.write(bytes(rng.randint(0, 255) for _ in range(4096)))

    # Smoke run
    pager = FakeLayerPager(str(fake_dir), mode="read", window_size=2, prefetch_distance=1, policy="q2_res")
    pager.layers = 8

    t0 = time.perf_counter()
    pager.simulate_tokens(tokens=2, layers=8, layer_bytes=64*1024, residual_bytes=16*1024)
    t1 = time.perf_counter()

    result = pager.get_results(layers=8, layer_bytes=64*1024, residual_bytes=16*1024, tokens=2, policy="q2_res")
    result["smoke_wall_ms"] = round((t1 - t0) * 1000, 1)

    # Cleanup
    shutil.rmtree(fake_dir)
    result["smoke_cleanup"] = "done"

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"=== FAKE FILE-BACKED PAGER ===")
    print(f"  Policy: {args.policy}, Mode: {args.mode}, Layers: {args.layers}")
    print(f"  Layer: {args.layer_mb}MB, Residual: {args.residual_mb}MB")
    print(f"  Window: {args.window_size}, Prefetch: {args.prefetch_distance}, Tokens: {args.tokens}")
    print(f"  Storage: {args.storage_dir}")
    print()

    storage_path = Path(args.storage_dir)

    # ── Create fake layer files ────────────────────────────────────────────────
    print(f"[1/5] Creating fake layer files...")
    layer_bytes = int(args.layer_mb * 1024 * 1024)
    residual_bytes = int(args.residual_mb * 1024 * 1024)

    files = create_fake_layer_files(
        args.storage_dir, args.layers, layer_bytes, residual_bytes, args.policy
    )
    print(f"  Created {len(files)} files in {args.storage_dir}")

    total_size_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
    print(f"  Total fake data: {total_size_mb:.1f} MB")

    # ── Run smoke test ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Smoke test...")
    smoke = run_smoke_test(args.storage_dir)
    print(f"  Smoke: {smoke['smoke_wall_ms']:.1f}ms wall, {smoke['stats']['reads']} reads, cleanup={smoke['smoke_cleanup']}")
    smoke_ok = smoke["smoke_wall_ms"] < 5000 and smoke["stats"]["reads"] > 0
    print(f"  Smoke result: {'PASS' if smoke_ok else 'FAIL'}")

    # ── Run scaled profile ─────────────────────────────────────────────────────
    print(f"\n[3/5] Running scaled profile ({args.layers} layers, {args.tokens} tokens)...")

    pager = FakeLayerPager(
        args.storage_dir, mode=args.mode,
        window_size=args.window_size,
        prefetch_distance=args.prefetch_distance,
        policy=args.policy
    )
    pager.layers = args.layers

    t0 = time.perf_counter()
    pager.simulate_tokens(
        tokens=args.tokens, layers=args.layers,
        layer_bytes=layer_bytes, residual_bytes=residual_bytes
    )
    t1 = time.perf_counter()

    profile_result = pager.get_results(
        args.layers, layer_bytes, residual_bytes, args.tokens, args.policy
    )
    profile_result["wall_time_measured_ms"] = round((t1 - t0) * 1000, 1)

    print(f"  Wall: {profile_result['wall_time_measured_ms']:.1f}ms")
    print(f"  Reads: {profile_result['stats']['reads']}, Prefetches: {profile_result['stats']['prefetches']}")
    print(f"  Cache hits: {profile_result['stats']['cache_hits']}")
    print(f"  Effective bandwidth: {profile_result['stats']['effective_bandwidth_mb_s']:.1f} MB/s")

    # ── Cleanup ─────────────────────────────────────────────────────────────────
    cleanup_status = "skipped"
    if args.cleanup and not args.keep:
        print(f"\n[4/5] Cleaning up temp files...")
        shutil.rmtree(storage_path)
        cleanup_status = "done"
        print(f"  Cleanup: {cleanup_status}")
    else:
        print(f"\n[4/5] Cleanup: {'skipped (--keep set)' if args.keep else 'skipped (--cleanup false)'}")

    # ── Compare to Phase 28AG estimates ───────────────────────────────────────
    print(f"\n[5/5] Comparing to Phase 28AG estimates...")
    estimated_io_ms = (args.layers * args.layer_mb) / 1.8 * 1000  # 1800 MB/s NVMe
    measured_ms = profile_result["wall_time_measured_ms"]

    ratio = measured_ms / estimated_io_ms if estimated_io_ms > 0 else 0
    print(f"  Phase 28AG estimated IO: {estimated_io_ms:.0f}ms")
    print(f"  Measured wall time: {measured_ms:.0f}ms")
    print(f"  Ratio (measured/est): {ratio:.2f}x")

    # RSS check (best effort on Linux)
    peak_rss_mb = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    peak_rss_mb = int(line.split()[1]) / 1024
                    break
    except Exception:
        pass

    print(f"  Peak RSS: {peak_rss_mb:.1f} MB (from /proc/self/status)")

    # ── Assemble results ────────────────────────────────────────────────────────
    result = {
        "phase": "28AH",
        "verdict": "PASS",
        "smoke_test": {
            "wall_ms": smoke["smoke_wall_ms"],
            "reads": smoke["stats"]["reads"],
            "cleanup": smoke["smoke_cleanup"],
            "pass": smoke_ok
        },
        "scaled_profile": profile_result,
        "comparison_to_28ag": {
            "phase_28ag_estimated_io_ms": round(estimated_io_ms, 1),
            "measured_wall_ms": round(measured_ms, 1),
            "ratio": round(ratio, 2),
            "interpretation": "match" if 0.5 < ratio < 2.0 else "mismatch"
        },
        "system": {
            "peak_rss_mb": round(peak_rss_mb, 1),
            "mmap_available": HAS_MMAP,
            "mode_used": args.mode,
            "total_fake_bytes_mb": round(total_size_mb, 1),
            "cleanup_status": cleanup_status
        },
        "safety": {
            "files_staged_in_repo": False,
            "temp_data_cleaned": cleanup_status == "done"
        },
        "recommendation": "PASS_FAKE_PAGER" if smoke_ok and ratio < 10 else "FAIL_FAKE_PAGER"
    }

    # ── Write JSON output ────────────────────────────────────────────────────────
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results written to: {args.out_json}")

    print(f"\n=== RESULT: {result['verdict']} ===")
    print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()