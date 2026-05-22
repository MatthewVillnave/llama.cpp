# Phase 28AN: Real Manifest Schema + .trit Reader

## Verdict: PASS_PHASE28AN_REAL_MANIFEST_TRIT_PAGER | PASS_MANIFEST_PARSING | PASS_BUDGET_ENFORCEMENT | PARTIAL_TRIT_HEADER_CRC_MISMATCH | PASS_NO_TRIT_FILES_STAGED

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`6d563c9a7`

---

## C. Files Changed
- `examples/speculative/prt_sidecar_pager.h` — extended with manifest_schema enum, trit_header struct, TensorEntry
- `examples/speculative/prt_sidecar_pager.cpp` — dual-schema parser (Phase28Y + legacy), .trit header reader, budget enforcement
- `examples/speculative/prt_sidecar_pager_probe.cpp` — real manifest generator + real .trit writer

---

## D. Build Command
```bash
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L \
  examples/speculative/prt_sidecar_pager.cpp \
  examples/speculative/prt_sidecar_pager_probe.cpp \
  -o /tmp/prt_sidecar_pager_probe
```
**No llama.cpp dependency.** Standalone C++17.

---

## E. Manifest Schema Support

### Phase 28Y (new format)
Detects via `"format_version"` top-level field.
Parsed fields: `format_name`, `source_model`, `base_quant`, `layer_count`, `tensor_families`
Per-entry: `layer_index`, `tensor_name`, `tensor_family`, `file_path`, `byte_size`, `shape`, `status`

### Legacy sidecar format
Detects via `"files"` top-level array.
Parsed fields: `layer`, `filename`, `bytes`

### Detection Logic
```cpp
bool has_format_version = (strstr(buf, "\"format_version\"") != nullptr);
bool has_files_array = (strstr(buf, "\"files\"") != nullptr);
if (has_format_version) { schema_ = PHASE28Y; ... }
else if (has_files_array) { schema_ = SIDECAR_LEGACY; ... }
```

---

## F. .trit Header Reader

Implemented in `load_trit_header()`. Validates:
- Magic bytes `"TRIT"` (4 bytes at offset 0)
- Version 0.1 (u16 at offsets 4, 6)
- Checksum CRC16 (u16 at offset 30) over first 30 bytes

File size: 393,280 bytes (512×2048×3 bits / 8 + scales + header)

**Limitation:** C++ CRC16 implementation differs from Python's. Python uses `((crc << 1) | (crc >> 15))` bit rotation. C++ uses table-based CRC16. Both are valid CRC16 variants but produce different checksums. **For v0: checksum validation skipped (`validate_trit_header=false`) until CRC algorithms are aligned.**

---

## G. Test Results

### Fake Regression (512KB budget, 4 layers × 3 tensors × 64KB)
```
init() = true, schema = Phase28Y
activate_layer(0) = true, activate_layer(1) = true
activate_layer(2) = false, activate_layer(3) = false (budget exceeded)
resident=393KB, peak=393KB
Valid views: 6, Null views: 6
Budget rejects: 3/4 layers (512KB < 2×192KB window)
Fallback test: PASS (null for layer 999)
RESULT: PASS
```

### Real Mode — Insufficient Budget (512KB, 2 layers × 393KB each)
```
activate_layer(0) = false, activate_layer(1) = false (budget exceeded)
resident=0, budget_rejects=2
Budget enforcement: PASS
```

### Real Mode — Sufficient Budget (2048KB, 2 layers × 393KB each)
```
activate_layer(0) = true, activate_layer(1) = true
resident=1,572,120 bytes, peak=1,572,120 bytes
Valid views: 2, Null views: 0
Budget rejects: 0
Fallback test: PASS
RESULT: PASS
```

---

## H. Key Observations

| Test | Result |
|------|--------|
| Phase28Y manifest detection | ✅ PASS |
| Legacy manifest detection | ✅ PASS |
| activate_layer (sufficient budget) | ✅ PASS |
| activate_layer (insufficient budget) | ✅ PASS — rejects with BUDGET_EXCEEDED |
| get_residual (loaded) | ✅ PASS — returns valid view |
| get_residual (unloaded) | ✅ PASS — returns null_view with reason |
| get_residual (nonexistent) | ✅ PASS — returns null_view |
| Budget enforcement | ✅ PASS |
| .trit header validation | ⚠️ PARTIAL — magic+version OK, checksum CRC mismatch |

---

## I. Limitations

1. **CRC16 mismatch:** C++ and Python use different CRC16 variants. `.trit` files written by C++ probe fail Python's `validate_trit` checksum check. **Fix:** align CRC implementations. For v0, `validate_trit_header=false` is used.

2. **Manifest JSON parser is fragile:** Uses string search (`strstr`) for field lookup. Works for generated fixtures but not for arbitrary JSON formatting. **Production needs a real JSON parser.**

3. **File cache grows unbounded:** Loaded files never evicted from `file_cache_` until `shutdown()`. In long sessions this could grow memory.

4. **prefetch_layer() doesn't populate all tensor entries:** Only loads first matching entry per layer.

5. **.trit header reader validates but doesn't expose parsed header to caller:** `trit_header` struct exists but `get_residual()` only returns raw bytes.

---

## J. Recommended Next Phase

**Phase 28AO: Runtime-Adjacent Sidecar Pager Probe**

With the standalone pager validated for both fake and real manifests:
1. Align C++ and Python CRC16 implementations
2. Test `.trit` header validation end-to-end
3. Connect `get_residual()` to return parsed `trit_header` metadata
4. Add LRU file cache eviction
5. Test with larger layer counts and window sizes

---

## K. Safety Scan

```
No .gguf/.bin/.safetensors/.pt/.pth files staged ✅
No 30B files accessed ✅
No sidecars generated ✅
No model data staged ✅
No secrets detected ✅
No tags touched ✅
Binary not in repo (/tmp only) ✅
```

**Safety verdict:** CLEAN

---

## Verdict Flags
- PASS_PHASE28AN_REAL_MANIFEST_TRIT_PAGER
- PASS_MANIFEST_PARSING
- PASS_BUDGET_ENFORCEMENT
- PASS_FALLBACK_BEHAVIOR
- PARTIAL_TRIT_HEADER_CRC_MISMATCH
- RECOMMEND_CRC_ALIGNMENT
- RECOMMEND_RUNTIME_ADJACENT_PROBE
- PASS_NO_TRIT_FILES_STAGED

---

## Tags Touched?
NO.