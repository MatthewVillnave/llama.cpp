# PRT Phase 25B: Correctness-Restored Freeze

**Verdict:** `PASS_PHASE25B_FREEZE`

**Date:** Wed 2026-05-20 22:43 EDT
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Previous HEAD (25A):** `5ff5278c6`
**New HEAD:** (same after docs-only commit)

---

## 1. Frozen State

### Branch
```
experimental/prt-phase19a-alt-sidecar-backed
```

### Current HEAD
```
5ff5278c6 ("PRT Phase 25A: post-24Z decision checkpoint")
```

### Key Commits Frozen

| Commit | Phase | Description |
|--------|-------|-------------|
| `8de123d54` | 24X | INT8 dequant scale factor fix — broken decode `f32 = int8 * scale` corrected to `f32 = int8 * scale / 127.0f` |
| `8a8030e62` | 24Y | 3B and 7B regression pass after /127 fix |
| `db42bb202` | 24Z | First clean 3B timing after correctness restoration |
| `5ff5278c6` | 25A | Post-24Z decision checkpoint — pause PRT speed work |

### Tag
```
PRT_PHASE25B_CORRECTNESS_RESTORED_CHECKPOINT
```

Annotated tag message: *"PRT Phase 25B: correctness-restored INT8 checkpoint; no speedup claim"*

---

## 2. Validated Claims

The following claims are supported by test evidence on this branch:

- ✅ **Canonical INT8 layout exists** — canonical INT8 layout fully defined and validated
- ✅ **3B layer0 PRT INT8 correctness restored** — native vs PRT outputs identical for n=4 and n=8 after /127 fix
- ✅ **7B regression passed** after /127 fix
- ✅ **Native and PRT outputs match** for all tested 3B prompts after fix
- ✅ **Current custom-op speed path is not a speed win** in narrow 3B c=4 n=8 single-thread smoke (PRT ~36% slower)
- ✅ **Pre-24X timing data is invalid** — all timing before commit `8de123d54` was measured with the broken op

---

## 3. Forbidden Claims

The following claims are **not supported** by evidence on this branch:

- ❌ Speedup (no speedup measured or demonstrated)
- ❌ Production readiness
- ❌ Multi-layer support
- ❌ All-layer support
- ❌ 14B support
- ❌ 7B timing (only regression pass done; no full 7B timing)
- ❌ Broad semantic equivalence (only layer0 tested)
- ❌ Broad performance conclusion (only 3B layer0 c=4 n=8 tested)
- ❌ 0.5B canonical INT8 pass
- ❌ Anything based on pre-24X timing (all void)

---

## 4. Final Technical Conclusion

**The current PRT custom-op path is correct after the /127 decode fix but slower than native Q4_K_M in the tested 3B layer0 single-thread smoke.**

| Metric | Value |
|--------|-------|
| 3B layer0 correctness | ✅ Restored |
| 7B regression | ✅ Passed |
| First valid PRT vs native ratio | 0.73 (native faster) |
| PRT overhead interpretation | ~36% slower in narrow smoke |

**Branch purpose:** This branch is preserved as a correctness-restored INT8 sidecar / PRT research artifact. It is **not** an active speed path. It documents:
1. The canonical INT8 layout and decode formula
2. The /127 fix and its rationale
3. The first clean timing result (negative for speed)
4. The decision to pause speed work

---

## 5. Recommended Next Project Direction

| Priority | Direction |
|----------|-----------|
| 1 | Pause PRT speed work entirely |
| 2 | Speculative decoding router improvements |
| 3 | SDI architecture / design |
| 4 | Native-layout sidecar theory only |
| 5 | Smart Agent Router / local inference productization |
| 6 | ContextOS / MemoryOS productization |

**Core principle:** Do not continue PRT speed optimization without a clear new hypothesis backed by profiling data from the corrected op.

---

## 6. Safety Scan

```
git status --short
 M ggml/src/ggml-cpu/ops.cpp
?? examples/speculative/results/PRT_PHASE24R_NATIVE_MULMAT_PROBE.md
?? examples/speculative/results/phase24r_native_mulmat_probe.json
?? examples/speculative/results/PRT_PHASE25A_POST_24Z_DECISION_CHECKPOINT.md
?? examples/speculative/results/phase25a_post_24z_decision_checkpoint.json

git diff --stat
 ggml/src/ggml-cpu/ops.cpp | 13 ++++++++++++-
 1 file changed, 12 insertions(+), 1 deletion(-)

No GGUF files staged.
No large unexpected files.
No secrets found.
```

---

## Verdicts

| Verdict | Value |
|---------|-------|
| `PASS_PHASE25B_FREEZE` | ✅ |
| `CORRECTNESS_RESTORED` | ✅ |
| `NO_SPEEDUP_CLAIM` | ✅ |
| `PRE_24X_VOID` | ✅ |
| `BRANCH_FROZEN_AS_RESEARCH_ARTIFACT` | ✅ |

---

*Phase 25B freeze complete. Branch tagged and preserved.*