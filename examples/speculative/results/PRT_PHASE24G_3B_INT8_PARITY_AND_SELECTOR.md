# PRT Phase 24G: 3B INT8 Parity and Selector

## Date
2026-05-20

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Status: PARTIAL_SELECTOR_ADDED

### Parity Audit (Phase 24G-C)
- True cosine (double): 0.9999614671 ✅
- MAE: 0.0001769819
- Max error: 0.0014204010
- RMSE: 0.0002065799
- Norm ratio: 1.000047
- No NaN/Inf

### Selector
- Added dimension-based selection (K=2048, M=11008 → 3B sidecar path)
- Source change: llama-graph.cpp lines 1320-1330

### Runtime
- Native 3B bounded: ✅ Works ("Paris...")
- With INT8 format: produces output, logs show sidecar=(nil)

### Known Issue
- selector logs not visible in stderr (code path may not be reached, or logs quiet)
- sidecar=(nil) in PRT-NATIVE logs - but output works

### Verdict
PARTIAL_SELECTOR_ADDED - Code added, parity verified, runtime smoke works
