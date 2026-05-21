#!/usr/bin/env python3
"""
Phase 26M: Standalone SDI Memory Residency Guard.

This guard is intentionally runtime-agnostic. It reads host memory/swap state,
reports likely stale local inference processes, and returns a conservative JSON
decision for whether the caller should run local model inference or compress.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any


PROC_PATTERN = re.compile(
    r"(ollama|llama-cli|llama-completion|llama-server|litert|phase26|\bprt\b|nohup|\bscript\b)",
    re.IGNORECASE,
)

MODEL_RAM_MB = {
    "qwen2.5:0.5b": 512,
    "qwen2.5-0.5b": 512,
    "qwen2.5:3b": 2300,
    "qwen2.5-3b": 2300,
    "qwen2.5:7b": 5200,
    "qwen2.5-7b": 5200,
    "llama3.2:latest": 2600,
}


@dataclass
class MemoryState:
    ram_available_mb: int
    swap_total_mb: int
    swap_used_mb: int
    swap_free_mb: int


def _parse_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            key, rest = line.split(":", 1)
            match = re.search(r"(\d+)", rest)
            if match:
                values[key] = int(match.group(1)) // 1024
    return values


def read_memory_state() -> MemoryState:
    meminfo = _parse_meminfo()
    swap_total = meminfo.get("SwapTotal", 0)
    swap_free = meminfo.get("SwapFree", 0)
    return MemoryState(
        ram_available_mb=meminfo.get("MemAvailable", 0),
        swap_total_mb=swap_total,
        swap_used_mb=max(0, swap_total - swap_free),
        swap_free_mb=swap_free,
    )


def list_stale_processes() -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,etimes,comm,args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return processes

    this_pid = os.getpid()
    for line in out.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or not PROC_PATTERN.search(stripped):
            continue
        parts = stripped.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            age_s = int(parts[1])
        except ValueError:
            continue
        if pid == this_pid:
            continue
        processes.append(
            {
                "pid": pid,
                "age_s": age_s,
                "command": parts[2],
                "args": parts[3][:240],
            }
        )
    return processes


def estimate_context_mb(active_context_target: int, model: str) -> int:
    model_lower = model.lower()
    if "0.5b" in model_lower:
        per_8192 = 160
    elif "3b" in model_lower:
        per_8192 = 520
    elif "7b" in model_lower:
        per_8192 = 1200
    else:
        per_8192 = 700
    return max(64, int(per_8192 * (active_context_target / 8192)))


def evaluate_guard(
    model: str,
    active_context_target: int,
    force_tier: int | None = None,
) -> dict[str, Any]:
    state = read_memory_state()
    stale = list_stale_processes()
    warnings: list[str] = []
    safe = True
    recommended_tier = 2
    reason = "memory state is within local inference guard limits"

    if stale:
        warnings.append("local inference/runtime processes detected; reported only, not killed")

    if state.swap_used_mb > 2048:
        safe = False
        recommended_tier = 0
        reason = "swap used > 2GB; block all heavy local inference"
    elif state.swap_used_mb > 1024:
        safe = False
        recommended_tier = 0
        reason = "swap used > 1GB; block local 7B and unsafe heavy inference"
    elif active_context_target > 8192 and state.swap_used_mb > 0:
        recommended_tier = 1
        reason = "active context target > 8192 with swap in use; compress to Tier 1"
        warnings.append("context target exceeds 8192 while swap is nonzero")
    elif state.swap_used_mb > 0:
        recommended_tier = 1
        warnings.append("swap is already in use; keep context compressed")

    model_ram = MODEL_RAM_MB.get(model.lower(), 2600)
    context_ram = estimate_context_mb(active_context_target, model)
    estimated_required = model_ram + context_ram + 512
    if estimated_required > int(state.ram_available_mb * 0.85):
        safe = False
        recommended_tier = 0
        reason = (
            f"estimated model+context requirement {estimated_required}MB exceeds "
            f"85% of available RAM {state.ram_available_mb}MB"
        )

    if force_tier is not None:
        recommended_tier = force_tier

    return {
        "safe": safe,
        "swap_used_mb": state.swap_used_mb,
        "swap_total_mb": state.swap_total_mb,
        "ram_available_mb": state.ram_available_mb,
        "active_context_target": active_context_target,
        "model": model,
        "estimated_model_context_mb": estimated_required,
        "recommended_tier": recommended_tier,
        "reason": reason,
        "warnings": warnings,
        "reported_processes": stale[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone SDI memory guard")
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--active-context-target", type=int, default=4096)
    parser.add_argument("--tier", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument("--out")
    args = parser.parse_args()

    result = evaluate_guard(
        model=args.model,
        active_context_target=args.active_context_target,
        force_tier=args.tier,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    print(payload)
    return 0 if result["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
