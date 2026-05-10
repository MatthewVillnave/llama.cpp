# PRT Phase 19A-ALT: 0.5B Sidecar-Backed Loader Feasibility

## Phase 19A-ALT Summary

**Goal:** Test if PRT sidecars can become primary tensor backing format vs overlay

**Status:** PHASE D INSTRUMENTATION BLOCKED — runtime visibility issue

---

## A. Metadata Grounding ✅

- **0.5B model:** `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (380MB)
- **Layers:** 24 (blk.0 through blk.23)
- **FFN_UP shape:** K=896, M=4864 (tensor shape {896, 4864})
- **Type:** q5_0 (GGML type 6)
- **FFN_UP per layer:** Q5_0 ~8MB / FP16 ~8.7MB

## B. Source-of-Truth Sidecar Schema ✅

From CLI runtime (`tools/cli/cli.cpp` lines ~720-780):
- **Magic:** 'P','R','T','6' (PRT6, NOT 'PRTF')
- **Header:** 16 bytes at offset 0
  - Offset 0-3: Magic "PRT6"
  - Offset 4-7: version (uint32 LE)
  - Offset 8-11: rows/M (uint32 LE)  
  - Offset 12-15: cols/K (uint32 LE)
- **Scales:** float32[M] at offset 16
- **Packed payload:** INT6[M*K*4/3] starting at offset 16+M*4
- **Expected sidecar size:** 16 + 4864*4 + 3268608 = 3,288,080 bytes

**Important:** Runtime uses SIZE-BASED lookup (NOT header fields) for 7B/14B:
```cpp
if (raw_bytes == 50997268) { M = 18944; K = 3584; }  // 7B
else if (raw_bytes == 53139472) { M = 13824; K = 5120; }  // 14B
```
**0.5B would NOT match these sizes** → runtime would skip loading sidecar!

## C. Layer0 Sidecar Generation ✅

**Generated:** `ffn_up_layer0_prt.int6` with PRT6 magic
- **Weight cosine:** 0.999432 (>= 0.995 threshold: PASS)
- **Matvec cosine:** 0.999444 (>= 0.995 threshold: PASS)

## D. Loader Instrumentation ❌ BLOCKED

**Attempted:** Modified `llama_model_loader::load_data_for()` to log FFN_UP tensors
**Issue:** Instrument log NOT appearing at runtime - tensor loading path different than expected or buffered

**What WAS learned from source analysis:**
- Graph construction in `build_ffn()` checks `g_prt_sidecar_data[il]`
- PRT true replacement path: `build_prt_ffn_up(ctx0, cur, il)`
- Native fallback: `build_lora_mm(up, cur)`
- Tensor data (`up`) still loaded from GGUF, but PRT custom op ignores it when active

**Key insight:** The FFN_UP tensor IS loaded even when PRT is active - PRT reads from its own sidecar buffer (`g_prt_sidecar_data[il]`) not from the GGUF tensor. This confirms sidecar-backing IS the compute path - but we still pay RAM to load GGUF FFN_UP.

## E. Safe Layer0 Probe ❌ NOT ATTEMPTED

**Reason:** Phase D instrumentation didn't produce visibility

## F. Runtime Canary ❌ NOT RUN

## G. Report

| Field | Value |
|-------|-------|
| A. Branch | `experimental/prt-phase14a-packed-sidecars` |
| B. Previous HEAD | `045aa6039` |
| C. New HEAD | `045aa6039` (no change - instrumentation reverted) |
| D. 0.5B model found | YES |
| E. Metadata | K=896, M=4864, layers=24, type=q5_0 |
| F. Runtime sidecar schema | 'PRT6', 16-byte header, size-based M/K lookup |
| G. Layer0 sidecar cosine | 0.999432 PASS |
| H. Instrumentation | BLOCKED - no runtime visibility |
| I. Probe attempted | NO |
| J. Runtime result | N/A |
| K. RAM/RSS estimate | FFN_UP=8MB per layer × 24 |
| L. Main blocker | Runtime loader hook not firing |
| M. Recommended | Alternative: probe via PRT log at compute time |

## Verdict

**PARTIAL_INSTRUMENTATION_ONLY** — Blocked at Phase D

## Key Learnings

1. **Runtime sidecar requires size-matching:** 0.5B sidecar (3.2MB) won't load because runtime only matches 7B (50MB) or 14B (53MB) sizes

2. **Sidecar as format concept is conceptually sound:** Even if FFN_UP tensor loads from GGUF, PRT path uses sidecar buffer during compute - but RAM is still consumed

3. **Would need loader modification:** To truly skip GGUF FFN_UP loading, would need to modify `load_data_for()` to redirect tensor data pointer to sidecar buffer

## Next Steps

- Alternative approach: Add PRT log output at `build_prt_ffn_up()` call time to verify sidecar is used
- Or: Modify runtime size lookup to accept 0.5B sidecar by header fields (not just size-based)
- Probe with 7B/14B where loader visibility works

---

*Commit: 045aa6039 (no new commits - instrumentation reverted)*
*Date: 2026-05-10*