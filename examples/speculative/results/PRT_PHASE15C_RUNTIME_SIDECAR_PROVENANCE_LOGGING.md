# PRT Phase 15C — Runtime Sidecar SHA / Provenance Logging

## Verdict

**PASS_PROVENANCE_LOGGING** ✅

## Context

Phase 15B identified and fixed a major sidecar provenance issue:
- Old `/tmp/prt_sidecars_7b_int8/` had 28 files but only 1 unique SHA
- Fresh INT8 regeneration fixed this with 28/28 unique sidecars
- The duplicated-sidecar scare proved that "28/28 loaded" is not enough evidence
- Future runtime validation should record sidecar provenance directly

## Implementation

**Files changed:**
- `tools/cli/cli.cpp` — Added provenance logging to PRT sidecar loading
- `src/llama-graph.cpp` — Added `g_prt_force_native_count` global (used by later phases)
- `src/llama.cpp` — Track count in `llama_set_prt_force_native_layers`

**Hash strategy:** Uses `sha256sum` via popen() for zero linking dependency

**Log format:**
```
[PRT_PROVENANCE_BEGIN]
[PRT_PROVENANCE] sidecar_dir=...
[PRT_PROVENANCE] sidecar_format=int8
[PRT_PROVENANCE] expected_layers=28
[PRT_PROVENANCE] force_native_count=2
[PRT_PROVENANCE] force_native_layer=11
[PRT_PROVENANCE] force_native_layer=15
  [PRT_SIDECAR_LAYER] layer=0 file=ffn_up_layer0_prt.int8 size=67971072 sha256=f02a4673... status=loaded fallback=false
[PRT_PROVENANCE] loaded_count=28 unique_sha_count=28
[PRT_PROVENANCE_WARNING] duplicate_sidecar_hashes=true unique_sha_count=X loaded_count=Y
[PRT_PROVENANCE_END]
```

**Overhead:** Adds SHA256 hash per sidecar file loaded (one-time at load, ~50ms per file on NVMe)

## INT8 Smoke Result

- Run: Exit 0 ✅
- Output: "Paris." ✅
- Provenance begin/end: ✅
- sidecar dir: `/tmp/prt_sidecars_7b_int8_phase15b_fixed` ✅
- format=int8 ✅
- 28/28 loaded ✅
- unique SHA count = 28 ✅
- layers 11 and 15: fallback=true ✅

## INT6 Smoke Result

- Run: Exit 0 ✅
- Provenance begin/end: ✅
- sidecar dir: `/tmp/prt_sidecars_7b_int6_phase15b_packed` ✅
- format=int6 ✅
- 28/28 loaded ✅
- unique SHA count = 28 ✅
- layers 11 and 15: fallback=true ✅

## Duplicate Warning Control

**Test against known-bad dir:** `/tmp/prt_sidecars_7b_int8/` (28 files, 1 unique SHA)

- Result: ✅ PASS
- loaded_count=28
- unique_sha_count=1
- Warning triggered: `[PRT_PROVENANCE_WARNING] duplicate_sidecar_hashes=true unique_sha_count=1 loaded_count=28`

## Interpretation

- **Q: Does runtime now prove distinct sidecars were loaded?** Yes — unique SHA count logged
- **Q: Does this prevent another "28/28 loaded but duplicated" blind spot?** Yes — duplicate warning
- **Q: Did INT8 and INT6 paths remain functional?** Yes — both pass smoke tests
- **Q: Should future validations require provenance logs?** Recommended

## Allowed Claims

- Runtime provenance logging works ✅
- INT8 and INT6 smoke tests run ✅
- Distinct sidecar loading can now be verified through logs ✅

## Forbidden Claims

- Do NOT claim: new quality validation
- Do NOT claim: new speed validation
- Do NOT claim: production readiness
- Do NOT claim: universal speedup
- Do NOT claim: INT6 replaces INT8
- Do NOT claim: larger-than-7B support

## Recommended Next Phase

Phase 15D: INT6 timing repeatability (re-test INT6 with provenance log enabled to verify timing)

OR

Phase 15D: runtime sidecar SHA/provenance logging integration with benchmark harness

---

**Tested:**
- Branch: `experimental/prt-phase14a-packed-sidecars`
- HEAD: `f02dceb17` (was `2feb9f0ad` before build)
- Model: Qwen2.5-7B-Instruct-Q4_K_M (SHA 1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49)
- INT8 sidecar dir: `/tmp/prt_sidecars_7b_int8_phase15b_fixed/` (28 files, 28 unique SHA)
- INT6 sidecar dir: `/tmp/prt_sidecars_7b_int6_phase15b_packed/` (28 files, 28 unique SHA)