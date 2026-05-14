# PRT Phase 19D: 0.5B INT6 Kernel Zero-Output Fix

## Status: PASS

**Root Cause:** The INT6 format (format=2) had NO handler in the custom op kernel. It fell through to the float32 scalar fallback which reads from `ud->sidecar` — nullptr for INT6. Result: all zeros output.

## Fix Applied

In `prt_graph_replace.h`, change:
```cpp
if (ud->format == 1 && ud->int8_data && ud->int8_scales)
```

To:
```cpp
if ((ud->format == 1 || ud->format == 2) && ud->int8_data && ud->int8_scales)
```

This adds INT6 (format=2) alongside INT8 (format=1) to use the same dequantization path.

## Test Results

- **Native 0.5B:** 97.4 t/s
- **PRT 0.5B:** 17.7 t/s (scalar kernel, expected slower)
- **24/24 sidecars loaded:** ✅
- **PRT compute hit count:** 24/24 layers ✅
- **Output semantic match:** ✅
- **No crash/hang:** ✅

## Files Modified
- `examples/speculative/prt_graph_replace.h` — INT6 format handler fix
- `tools/cli/cli.cpp` — loader audit log (for debugging)

## Commit
1e2b50616

*Date: 2026-05-10*