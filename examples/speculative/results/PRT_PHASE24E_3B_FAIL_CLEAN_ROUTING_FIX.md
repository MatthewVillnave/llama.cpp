# PRT Phase 24E: 3B Missing-Sidecar Fail-Clean Routing Fix

## Summary

**Phase 24E-R2** — Early sidecar compatibility gate prevents 3B (and any unknown-shape model) from entering the PRT sidecar loading path when no compatible sidecar exists. 3B with missing sidecar now routes native immediately instead of hanging for 90s.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`457b9c173` — "PRT Phase 24A: close 7B timing lane and plan 3B timing"

## C. New HEAD
`457b9c173` + uncommitted fix (to be committed as this phase)

## D. 3B Model Shape
- K = 2048 (hidden dim)
- M = 11008 (FFN dim)
- layers = 36
- n_head = 16
- n_head_kv = 2
- Model: `Qwen2.5-3B-Instruct-Q4_K_M.gguf`

## E. Sidecar Inventory
- Zero compatible 3B sidecars exist
- Existing sidecars: 0.5B (K=896, M=4864) and 7B (K=3584, M=18944)
- No sidecar path valid for K=2048 / M=11008

## F. Early Gate Fix

**Location:** `src/llama-graph.cpp` lines ~1260–1281

**Mechanism:**
- Module-level flags `g_prt_sidecar_compat_checked` and `g_prt_sidecar_compat_ok`
- Gate fires ONCE at `il=0` (first FFN call), sets module-level flag
- All subsequent layers check flag and skip sidecar loading entirely
- Uses `up->ne[0]` (gate tensor K) and `up->ne[1]` (gate tensor M) to detect model shape
- Supports: K=896/M=4864 (0.5B), K=3584/M=18944 (7B)
- Unknown shapes → `g_prt_sidecar_compat_ok = false` → immediate `build_lora_mm` fallback

**Code:**
```cpp
// Phase 24E-R2: Early sidecar compatibility gate
static bool g_prt_sidecar_compat_checked = false;
static bool g_prt_sidecar_compat_ok = false;

if (!g_prt_sidecar_compat_checked) {
    const int gate_K = (int)up->ne[0];
    const int gate_M = (int)up->ne[1];
    g_prt_sidecar_compat_checked = true;
    const bool model_is_05b = (gate_K == 896 && gate_M == 4864);
    const bool model_is_7b = (gate_K == 3584 && gate_M == 18944);
    g_prt_sidecar_compat_ok = model_is_05b || model_is_7b;
    prt_logf("[PRT_V2_MODEL_SHAPE] K=%d M=%d layers=%d\n", gate_K, gate_M, ...);
    prt_logf("[PRT_V2_SIDECAR_COMPAT] model_K=%d model_M=%d compatible=%d reason=%s\n", ...);
    if (!g_prt_sidecar_compat_ok) {
        prt_logf("[PRT_V2_ROUTE] action=native_no_sidecar reason=no_compatible_sidecar\n");
    }
}
if (!g_prt_sidecar_compat_ok) {
    return this->build_lora_mm(up, cur);  // native fallback immediately
}
```

## G. Native Smoke Result
✅ PASS — exit 0, sane output (Paris), ~30s load, stderr ~3KB

## H. INT8 Missing-Sidecar Result
✅ PASS — exit 0, no timeout, ~30s, native fallback used

## I. INT6 Missing-Sidecar Result
✅ PASS — exit 0, no timeout, ~30s, native fallback used

## J. AUTO Missing-Sidecar Result
✅ PASS — exit 0, no timeout, ~30s, native fallback used

## K. Wrong-Sidecar Prevention
✅ Gate fires for 3B (K=2048), blocks 0.5B/7B sidecar path
✅ No attempt to load from `prt_phase22e_05b_int8_from_f32` (0.5B path)
✅ No attempt to load from `prt_sidecars_7b_int6_phase15b_packed` (7B path)
✅ `f32_weight_loaded[]` never set to true for 3B

## L. Load Overhead Eliminated?
✅ YES — load time reduced from ~90s (timeout) to ~30s (native-level)
✅ Sidecar loading section completely skipped for unknown shapes

## M. Verdict
✅ **PASS** — Phase 24E-R2 complete

### Verdicts:
- ✅ PASS_3B_MISSING_SIDECAR_FAILS_CLEANLY
- ✅ PASS_3B_AUTO_ROUTES_NATIVE_WITHOUT_SIDECAR
- ✅ PASS_3B_WRONG_SIDECAR_PREVENTED
- ✅ PASS_3B_PRT_HANG_MISSING_SIDECAR (was FAIL before fix)
- ✅ PASS_3B_WRONG_SIDECAR_SELECTED (was FAIL before fix)
- ✅ PASS_BUILD

## N. Recommended Next
**Phase 24F:** Investigate `c=1` GGML_ASSERT(`n_tokens_all <= cparams.n_batch`) crash — separate issue from missing-sidecar routing.

## O. Models/Sidecars/Binaries Staged?
NO — no model files, sidecars, captures, logs, or binaries staged.

## P. Secrets Detected?
NO

## Q. Tags Touched?
NO

## R. System Disk Free
`/dev/nvme0n1p2` — 42% used

## S. Scratch Disk Free
`/media/matthew-villnave/VL_usb` (`/dev/sda1`) — 37% used

---

## Change Log

### Phase 24E-R2 (this phase)
- Added module-level `g_prt_sidecar_compat_checked` / `g_prt_sidecar_compat_ok` flags
- Early gate at build_ffn entry — fires once, blocks unknown shapes from sidecar loading
- New logs: `[PRT_V2_MODEL_SHAPE]`, `[PRT_V2_SIDECAR_COMPAT]`, `[PRT_V2_ROUTE] action=native_no_sidecar`
- 3B with missing sidecar now completes in ~30s (native) instead of 90s timeout

### Phase 24E (prior partial fix)
- Late K/M check at line ~1611 — prevented wrong-sidecar crash after attempted load
- Phase 24E partial fix was not committed; superseded by Phase 24E-R2 early gate