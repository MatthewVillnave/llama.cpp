# PRT_ROUTE_A_RC1 — TAGGED ✓

**Tag:** `PRT_ROUTE_A_RC1`
**Date:** 2026-05-02 16:25 EDT
**Commit:** `aaa5f290240dbaacfe355ca073bcbce49b18fde7`
**Branch:** `master` (ahead of origin by 2 commits)

---

## Tag Message

> PRT Route A RC1: true FFN_UP graph replacement with L12/L15 native anchors, ~1.84x controlled-suite speedup, clean counters, stable memory, and loud missing-sidecar validation.

---

## Latest Commit Summary

```
aaa5f2902 PRT Phase 11BP: Route A core — build_ffn hook, force-native API, sidecar validation, custom op
3763be222 PRT Phase 11BM-11BP: RC1 docs — Route A + L12/L15, ~1.84x speedup, sidecar validation
d12cc3d1c CUDA: also store `node->src->data` ptrs for equality check (#21635)
2dcb7f74e fix: free ctx_copy in ggml_opt_free to plug per-training-session leak (#21592)
660600081 server: respect the ignore eos flag (#21203)
```

---

## Phase 11BN Mini-Suite Results

| Prompt | n | Native | L12+L15 | Speedup | Status |
|--------|---|--------|---------|---------|--------|
| "Once upon a time in a" | 100 | 1m22.8s | 45.1s | **1.83x** | ✓ PASS |
| "distant galaxy" | 100 | 1m24.9s | 46.0s | **1.85x** | ✓ PASS |
| "Python reverse list" | 100 | 1m25.2s | 46.3s | **1.84x** | ✓ PASS |
| "capital of France?" | 100 | 1m23.9s | 45.4s | **1.85x** | ✓ PASS |
| JSON object | 50 | 46.6s | 25.6s | **1.82x** | ✓ PASS |
| "company is a large" | 100 | 1m23.3s | 44.9s | **1.86x** | ✓ PASS |

**Average speedup: ~1.84x**

### Counters (L12+L15 mode, all runs)

| Counter | Value | Status |
|---------|-------|--------|
| callback_overwrites | **0** | ✓ Clean |
| native_fallback_calls | **16** | ✓ 2×8 correct |
| identity_fallback_calls | **0** | ✓ Clean |
| prt_true_replacement_calls | 3706 (n=100) / 2006 (n=50) | ✓ Valid |

---

## Phase 11BP Sidecar Validation

| Test | Result |
|------|--------|
| All sidecars present → proceed | ✓ |
| Missing L5 (non-force-native) → `[PRT-ERROR] FATAL: 1 required sidecar(s) missing. Exiting.` | ✓ |
| Force-native layers (L12, L15) sidecars unused → not checked | ✓ |
| Native mode (mode 0) → validation skipped | ✓ |
| Checksums L0/L12/L15/L35 printed on success | ✓ |

---

## Commits in Tag

| Commit | Description |
|--------|-------------|
| `aaa5f2902` | Route A core: build_ffn hook, force-native API, sidecar validation, custom op |
| `3763be222` | RC1 docs: Route A + L12/L15 policy, results, claims, code review, merge checklist |
| `d12cc3d1c` | (pre-existing) CUDA equality check |
| `2dcb7f74e` | (pre-existing) ggml_opt free leak fix |
| `660600081` | (pre-existing) server eos flag |

---

## What RC1 Is

- **Experimental RC1** — PRT Route A with L12/L15 native fallback
- **Production-candidate** — validated speedup + quality, not production-ready
- **~1.84x speedup** — confirmed on 6 prompts
- **Clean counters** — callback_overwrites=0, no hidden overwrite paths
- **Loud missing-sidecar validation** — fatal startup error if required sidecar missing

## What RC1 Is Not

- ~~Production-ready~~
- ~~Universal 2x speedup~~
- ~~Validated on all prompts~~
- ~~Pure all36 Route A~~ (requires L12+L15 fallback for known-failure prompts)

---

*Tagged 2026-05-02 16:25 EDT*
*Commit: aaa5f290240dbaacfe355ca073bcbce49b18fde7*
