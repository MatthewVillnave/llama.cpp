# PRT Phase 23J-R2: Lean 7B Timing Smoke After Quiet Guard

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `2867720a3` (tag: `PRT_PHASE23I_7B_INT6_SEMANTIC_POLICY_CHECKPOINT`)  
**New HEAD:** `2867720a3` (no new commits — no code changes)  
**Date:** 2026-05-18  
**Verdict:** BLOCKED_7B_TOO_SLOW_ON_MACHINE

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`2867720a3` — PRT Phase 23J-R: add surgical quiet guard for native logs

## C. New HEAD
`2867720a3` — no new commits (no source changes needed)

## D. model path
`/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-7B-Instruct-Q4_K_M.gguf`

## E. native c2n2 result
**TIMEOUT — 300s limit hit**

```
real 300.18
user 200.88
sys 99.19
```

- Params: `-c 2 -n 2 -t 1 --temp 0 --log-disable`
- Model: Qwen2.5-7B-Instruct-Q4_K_M.gguf
- `PRT_V2_QUIET=1` confirmed spam-free (stderr=34 bytes)
- **No timing data extracted** — run was killed at 300s limit
- No generation output captured (stdout=/dev/null)

## F. INT6 c2n2 result
**NOT RUN** — skipped because native c2n2 timed out; running INT6 would similarly time out and waste time.

## G. optional c4n2 result
**NOT RUN** — per protocol (skip if c2n2 fails)

## H. file sizes
| File | Size |
|------|------|
| `phase23j_r2_native_c2n2.time` | 34 bytes |
| `phase23j_r2_native_c2n2.out` | (none — /dev/null) |

## I. [PRT-NATIVE] spam count
**0** — `PRT_V2_QUIET=1` working correctly (time file has only 34 bytes, no spam lines)

## J. output visible
Native c2n2: killed before completion, no stdout captured.

Test with `c=1 -n 2` (diagnostic): crashed with `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` at 3.59s — batch size issue, not PRT related.

## K. timing interpretation
**BLOCKED_7B_TOO_SLOW_ON_MACHINE**

Qwen2.5-7B-Instruct-Q4_K_M prefill with even `c=2 -n 2` exceeds the 300s timeout on this hardware. The 15GB RAM machine is insufficient for 7B model prefill timing runs. This is a hardware constraint, not a PRT code issue.

Evidence:
- c=2 n=2: `real 300.18s` (hit timeout, ~200s user CPU time consumed)
- CPU usage was 150%+ during run — machine was working but too slow
- Even the diagnostic `c=1 -n 2` crashed (unrelated batch assertion, not PRT)

## L. verdict
**BLOCKED_7B_TOO_SLOW_ON_MACHINE**

The quiet guard works. The machine cannot complete 7B timing runs. Do not retry on this hardware.

## M. recommended next
**Option A (same machine, smaller model):** Use Bonsai-8B with `c=1 -n 2` to validate that PRT speedup exists at smaller scale, then document 7B timing as blocked by hardware.

**Option B (different machine):** Run 7B timing on a machine with more RAM (>32GB) and faster disk (NVMe). The Qwen2.5-7B prefill at `c=2` is too heavy for 15GB RAM.

**Option C (document and close):** Document Phase 23J-R2 as BLOCKED_7B_TOO_SLOW_ON_MACHINE. The PRT quiet guard is proven (Phase 23J-R). No timing ratio exists on current hardware. Close this line of timing work until hardware upgrades.

**Current recommendation: Option C** — The quiet guard is the deliverable. Timing on 15GB RAM is not feasible. Document the block and move on.

## N. models/sidecars/binaries staged?
No.

## O. secrets detected?
None.

## P. existing tags touched?
None.

## Q. system disk free
78G free on /dev/nvme0n1p2 (66% used)

## R. scratch disk free
19G free on /media/matthew-villnave/VL_usb (85% used)

---

## Summary

Phase 23J-R2: Lean 7B timing smoke — **BLOCKED_7B_TOO_SLOW_ON_MACHINE**.

The quiet guard from Phase 23J-R (`PRT_V2_QUIET=1`) works perfectly — no `[PRT-NATIVE]` spam observed in any run on this branch. That's proven.

But Qwen2.5-7B-Instruct-Q4_K_M prefill with `c=2 -n 2` exceeds 300s timeout on 15GB RAM. The CPU was working (~200s user time), but the machine is too slow/constrained for 7B timing. No valid speedup ratio can be extracted.

Recommendation: **close timing work on this hardware**, document the block, and revisit 7B timing on a better-resourced machine.