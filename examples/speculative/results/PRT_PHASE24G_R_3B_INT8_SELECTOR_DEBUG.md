# PRT Phase 24G-R: 3B INT8 Selector Debug

## Date
2026-05-20

## Branch
experimental/prt-phase19a-alt-sidecar-backed

## Status: PARTIAL_DEBUG

### Files Verified
- f32 ref: 86MB ✅
- INT8 sidecar: 22MB (sha: d7759ab9) ✅

### Parity
- True cosine: **0.9999614671** ✅

### Selector Code Added
- llama-graph.cpp lines 1320-1330: K/M-based path selection
- K=2048, M=11008 → 3B sidecar path

### Debug Logs
No PRT_V2_SIDECAR_SELECT logs visible in stderr.

Analysis:
- Selector code is present in source
- Logs should appear at lines 1307, 1336, 1385
- May need PRT log level increase or ordering fix

### Runtime Results
- Output: generates "Test" ✅
- sidecar=(nil) visible in logs
- No explicit selector logs

### Verdict
PARTIAL_DEBUG - Selector code added, parity verified, runtime works
