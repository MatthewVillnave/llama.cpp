# PRT Phase 24G-R2: 3B INT8 Selector Debug Markers

## Date
2026-05-20

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Status: PARTIAL_DEBUG_MARKERS

### Files Verified
- f32 ref: 86MB ✅
- INT8 sidecar: 22MB ✅

### Parity
- True cosine: **0.9999614671** ✅

### Markers Added
- [PRT24G_R2] build_reached=1 at line 1307
- [PRT24G_R2_SELECTOR_ENTRY] at line ~1331
- [PRT24G_R2_POINTER_SET] at line ~1378

### Build
- Build passes ✅
- Binary timestamps updated

### Runtime Test
- PRT logs still show: sidecar=(nil)
- PRT24G_R2 markers NOT visible in stderr

### Analysis
Markers should appears but are not visible. Possible causes:
1. Code path not reached (build_lora_mm bypass skipped)
2. Log suppression still active
3. Debug logs need enable

### Runtime Output
- Model loads and runs ✅
- Output generated ✅

### Verdict
PARTIAL_DEBUG - Markers added, build passes, runtime works,
but selector path logs not captured.
