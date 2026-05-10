# PRT Phase 19L — Format Forensic: 0.5B Sidecar Format Resolution

## Verdict: **CONFIRMED_05B_PACKED_INT6**

## Discovery: Format Ambiguity Corrected

Previous Phase 19L report was INCORRECTLY labeled as testing INT8.
The correct interpretation:

| Directory | Size | Format | Content |
|-----------|------|--------|---------|
| `/tmp/prt_sidecars_05b_int8/` | 4,377,600 | unpacked_int8 | int8_data + float32 scales |
| `/tmp/prt_phase19b/` | 3,288,080 | **packed_int6** | PRT6 header + packed payload + float32 scales |
| Phase 19L ran with: | int8 format | ❌ NOT INT6 |
| **Correct INT6 path**: | int6 format with `/tmp/prt_phase19b/` | ✅ CORRECT |

## All 0.5B Sidecar Directories Inspected

### 1. `/tmp/prt_sidecars_05b_int8/`
- **Files**: 24
- **Extension**: `.int8`
- **Layer0 size**: 4,377,600 bytes
- **Format by size**: UNPACKED_INT8
- **Expected**: M*K=4,358,144 int8 + M*4=19,456 scales = 4,377,600 ✅
- **Header**: None (raw int8 + appended scales)
- **Loader path**: `--prt-sidecar-format int8` → `llama_set_prt_sidecar_int8()` → format=1
- **Status**: Unpacked INT8 format (NOT INT6)

### 2. `/tmp/prt_phase19b/`
- **Files**: 24 (matching 0.5B model)
- **Extension**: `.int6`
- **Layer0 size**: 3,288,080 bytes
- **Format by size**: PACKED_INT6
- **Expected packed**: 16 header + 19,456 scales + 3,268,608 packed = 3,288,080 ✅
- **Header magic**: `PRT6` (0x5052 5436)
- **Header M**: 4864
- **Header K**: 896
- **Payload**: 3,268,608 bytes (packed INT6, 4 values per 3 bytes)
- **Loader path**: `--prt-sidecar-format int6` → mmap → unpack → `llama_set_prt_sidecar_int6()` → format=2
- **Status**: PACKED_INT6 format (confirmed)

### 3. `/tmp/prt_phase19e_05b_sidecar_loader_substitution/`
- **Files**: 0 (empty directory)
- **Status**: Not usable

## Phase 19L Correction

Original Phase 19L ran with:
- `--prt-sidecar-format int8`
- `--prt-sidecar-dir /tmp/prt_sidecars_05b_int8/`
- **Result**: UNPACKED_INT8, NOT INT6

## Re-Run with Correct INT6 Path: `/tmp/prt_phase19b/`

### Timing Results (n=64, c=256, t=4, 3 runs)

| Mode | Run 1 | Run 2 | Run 3 | Avg | Mode logged |
|------|-------|-------|-------|-----|-------------|
| Native | 94.9 | - | - | **94.9** | none |
| **INT6 scalar** | 18.8 | 18.9 | 19.0 | **18.9** | `mode=int6` ✅ |
| **INT6+predecode** | 19.2 | 19.2 | 19.2 | **19.2** | `mode=fp32` ✅ |

### Corrected Analysis

- INT6 scalar: 18.9 tok/s (format=int6, LUT-based scalar unpack)
- INT6+predecode: 19.2 tok/s (format=fp32, AVX2 path)
- Speedup: 1.6% (19.2/18.9 - 1)
- Root cause: Memory bandwidth bounded; AVX2 compute improvement doesn't manifest

## PRT Compute Activation (Verified)

- INT6 scalar: `[PRT_COMPUTE] layer=N mode=int6 hit=1` ✅
- INT6+predecode: `[PRT_COMPUTE] layer=N mode=fp32 hit=1` ✅

## Conclusion

Phase 19L tested UNPACKED_INT8 (not INT6).
Corrected re-run tested PACKED_INT6.
Both formats confirm: AVX2 activates but no meaningful speedup on 0.5B.
Memory bandwidth is the bottleneck.
