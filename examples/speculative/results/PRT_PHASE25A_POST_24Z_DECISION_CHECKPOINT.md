# PRT Phase 25A: Post-24Z Decision Checkpoint

**Verdict:** `PASS_PHASE25A_DECISION_CHECKPOINT`

**Date:** Wed 2026-05-20 22:40 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Current HEAD:** `db42bb202`

---

## A. Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

## B. Current HEAD
```
db42bb202 ("PRT Phase 24Z: redo 3B timing after scale fix")
```

## C. Latest Correctness Fix
| Commit | Description |
|--------|-------------|
| `8de123d54` | PRT Phase 24X: fix INT8 dequant scale factor — broken C decode `f32 = int8 * scale` corrected to `f32 = int8 * scale / 127.0f` |

**Python quantization (reference):**
```python
int8 = round(f32 / scale * 127)
```
**Broken C decode:**
```c
f32 = int8 * scale                          // WRONG
```
**Correct C decode:**
```c
f32 = int8 * scale / 127.0f                 // RIGHT
```

---

## D. Phase 24Z Clean Timing
> First trustworthy timing result. All prior timing data is void.

**Conditions:** Qwen2.5-3B, layer0 only, c=4, n=8, single-thread, custom op active, AVX2 backend active

| Run | Native | PRT INT8 |
|-----|--------|----------|
| 1   | 2.90s  | 3.44s    |
| 2   | 2.39s  | 3.48s    |
| 3   | 2.34s  | 3.46s    |
| **avg** | **2.54s** | **3.46s** |

**Ratio:** `native / PRT = 0.73`
**Interpretation:** PRT is ~36% slower than native in this narrow smoke test.

**Correctness during timing runs:**
- n=4: native and PRT outputs identical ✓
- n=8: native and PRT outputs identical ✓
- custom op active ✓
- AVX2 backend active ✓
- native fallback = 0 ✓
- scalar fallback = 0 ✓
- valid numeric output ✓

---

## E. Correctness Status

| Item | Status |
|------|--------|
| 3B PRT INT8 layer0 correctness after /127 fix | ✅ RESTORED |
| 3B native vs PRT output match (n=4, n=8) | ✅ CONFIRMED |
| 7B regression pass after /127 fix | ✅ CONFIRMED |
| Canonical INT8 layout | ✅ VALID |
| Fallback counts (native=0, scalar=0) | ✅ CLEAN |

---

## F. Invalidated Claims

All timing and performance data from before commit `8de123d54` (Phase 24X) is **VOID**, including:

- Any prior "correct but slow" characterizations made before the /127 fix
- All pre-24X speedup claims
- All pre-24X performance interpretations
- Any Phase 24W "Paris-like" correctness claims that relied on the broken op

---

## G. Decision Options

### Option 1 — Pause PRT Speed Work
**Rationale:** Current custom-op path is correct but slower. Native Q4_K_M is hard to beat. More speed work risks rabbit holes.

| Pros | Cons |
|------|------|
| Stops wheel-spinning | Does not pursue possible optimizations |
| Preserves clean findings | |
| Frees effort for other CPU inference paths | |

### Option 2 — Profile Corrected-Op Overhead From Scratch (Later Phase)
**Rationale:** Previous overhead profiles were run before the /127 fix and are suspect. Now correctness is restored, a new profile could identify the real bottleneck.

**Permitted later scope (if explicitly approved):**
- 3B only, layer0 only
- No 7B timing
- No claims
- Split execution: one run per response or background/polling
- Profile: sidecar decode, custom op graph overhead, AVX2 kernel, eval wall time

| Pros | Cons |
|------|------|
| Gives true bottleneck map post-correctness-fix | Could burn time and still conclude native path is better |

### Option 3 — Tiny Broader Timing Sanity (Later Phase)
**Do NOT run in Phase 25A.**

Possible later scope:
- One extra prompt
- Maybe c=32
- 1 native and 1 PRT each (single run each, no repeat)
- No repeated benchmark
- Split into one run per response or background/polling

Purpose: See whether c=4 n=8 is an unusually bad/best case.

### Option 4 — Pivot Away from PRT Speed Work
**Possible pivots:**
- Speculative decoding CPU route
- SDI routing / selective compute
- ContextOS / MemoryOS productization
- Local inference packaging
- Native-layout sidecar theory only
- Full backend op only if justified later

---

## H. Recommendation

**Recommended action: Pause broad PRT speed work for now.**

Preserve the current branch as:
- A correctness-restored PRT INT8 sidecar research artifact
- A canonical INT8 layout / reference implementation
- A documented negative result for the current custom-op speed path

**Optional later work (only if explicitly requested by Matt):**
1. One tightly scoped corrected-overhead profile
2. One tiny c=32 / second-prompt sanity test
3. Native-layout / full-backend design only

**Core principle:** Do not keep optimizing blindly.

---

## I. Allowed Claims

- ✅ PRT INT8 layer0 correctness restored for 3B after /127 fix
- ✅ 7B regression passed after /127 fix
- ✅ First clean 3B timing shows PRT slower in narrow c=4 n=8 smoke
- ✅ Current implementation is not a speed win
- ✅ All pre-24X timing data is void

## J. Forbidden Claims

- ❌ Speedup
- ❌ Production readiness
- ❌ Broad slowdown conclusion
- ❌ Broad performance conclusion
- ❌ 7B timing
- ❌ All-layer support
- ❌ Multi-layer support
- ❌ 14B support
- ❌ 0.5B canonical INT8 pass
- ❌ Performance claims from pre-24X data

---

## K. Optional Sanity Run Status

**Skipped.**

**Reason:** Phase 25A is docs-only to avoid OpenClaw timeout/session instability and to prevent mixing decision-making with new timing data.

---

## L. Recommended Next

| Priority | Action |
|----------|--------|
| 1 | Pause PRT speed work. Preserve branch as correctness artifact. |
| 2 | (Optional) One corrected overhead profile if explicitly approved |
| 3 | (Optional) One tiny c=32 sanity run if explicitly approved |
| 4 | (Optional) Native-layout / full-backend design only |

---

## M. Models / Sidecars / F32 Refs Staged?

**No models, sidecars, or F32 refs staged.**

Only unstaged change: `ggml/src/ggml-cpu/ops.cpp` (12 lines, /127 fix).

---

## N. Secrets Detected?

**No.**

Grep scan of `examples/speculative/`, `common/`, `tools/`, `src/` returned no API keys, passwords, GitHub PATs, or Bearer tokens. All hits were checklist/documentation false positives.

---

## O. Tags Touched?

**No.**

No tags created, modified, or deleted in this phase.

---

## P. System Disk Free

| Filesystem | Size | Used | Avail | Use% |
|------------|------|------|-------|------|
| /dev/nvme0n1p2 (root) | 233G | 76G | 145G | 35% |

---

## Q. Scratch Disk Free

| Filesystem | Size | Used | Avail | Use% |
|------------|------|------|-------|------|
| /dev/sda1 (VL_usb) | 115G | 70G | 45G | 61% |

---

## Safety Scan

```
git status --short
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json

git diff --stat
 ggml/src/ggml-cpu/ops.cpp | 13 ++++++++++++-
 1 file changed, 12 insertions(+), 1 deletion(-)

find . -type f -size +20M (truncated listing):
./build-asan/src/CMakeFiles/llama.dir/llama-model-loader.cpp.o
./build-asan/src/CMakeFiles/llama.dir/llama-model.cpp.o
./build-asan/src/CMakeFiles/llama.dir/unicode.cpp.o
./build-asan/bin/libggml-cpu.so.0.9.11
./build-asan/bin/libllama.so.0.0.8720
./models/Bonsai-8B.gguf
./build/tools/server/bundle.js.hpp
```

No secrets, no GGUF/Bonsai files staged, no unexpected large files.

---

## Verdicts

| Verdict | Value |
|---------|-------|
| `PASS_PHASE25A_DECISION_CHECKPOINT` | ✅ |
| `DECISION_PAUSE_PRT_SPEED_WORK` | ✅ Recommended |
| `DECISION_PROFILE_CORRECTED_OVERHEAD_LATER` | ✅ Optional |
| `DECISION_RUN_TINY_BROADER_SANITY_LATER` | ✅ Optional |
| `DECISION_PIVOT_CPU_INFERENCE_PATH` | ✅ Optional |
| `OPTIONAL_TIMING_SKIPPED` | ✅ |
| `BLOCKED_MACHINE_STATE` | ✅ No block |

---

*Phase 25A complete. Decision checkpoint documented. Awaiting Matt's direction.*