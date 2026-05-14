# results/phase10e2_risk_register.md
# Phase 10E-2 Risk Register

## Risks from Custom Op Implementation

### Risk 1: Memory Pool Corruption
- **Severity**: HIGH
- **Impact**: Custom op writes to wrong memory location
- **Likelihood**: LOW (custom op runs correctly, data is valid)
- **Mitigation**: Use `ggml_map_custom1_inplace` which operates on matmul result buffer

### Risk 2: Zero-Fill Breaking Model
- **Severity**: HIGH (but expected for smoke test)
- **Impact**: Output garbage because layer 0 FFN is zeroed
- **Likelihood**: CERTAIN (by design)
- **Mitigation**: None needed - this is the smoke test goal

### Risk 3: Custom Op Not Firing
- **Severity**: LOW
- **Impact**: Replacement count stays 0
- **Likelihood**: RESOLVED (count = 6 confirmed)

### Risk 4: Thread Safety in Custom Op
- **Severity**: MEDIUM
- **Impact**: Data races when n_tasks > 1
- **Likelihood**: LOW (using n_tasks=1)
- **Mitigation**: Use n_tasks=1 for current implementation

### Risk 5: Sidecar Loading for Full PRT
- **Severity**: MEDIUM
- **Impact**: Need to load sidecar in harness and pass via userdata
- **Likelihood**: LOW (straightforward)
- **Mitigation**: Load at startup, pass pointer via userdata

---

## Risks for Full PRT Implementation (Phase 10E-3)

### Risk 6: Activation Buffer Access
- **Severity**: LOW
- **Impact**: Need to read activation from dst->data after matmul computes
- **Likelihood**: LOW (dst->data is valid after matmul)
- **Mitigation**: Verified with zero-fill test

### Risk 7: Sidecar Shape Mismatch
- **Severity**: MEDIUM
- **Impact**: Wrong PRT computation if dimensions don't match
- **Likelihood**: LOW (sidecar was validated in Phase 10A)
- **Mitigation**: Verify M=2048, N=11008 before compute

### Risk 8: Thread Partitioning
- **Severity**: LOW
- **Impact**: Need to partition PRT computation across threads
- **Likelihood**: LOW (using n_tasks=1)
- **Mitigation**: Keep single-threaded for now

### Risk 9: Output Zero/NaN
- **Severity**: MEDIUM
- **Impact**: PRT computation produces 0 or NaN
- **Likelihood**: LOW (threshold filters should prevent)
- **Mitigation**: Verify output before writing

---

## Overall Risk Assessment

| Phase | Risk Level | Notes |
|-------|------------|-------|
| Phase 10E-2 Smoke | LOW | Integration proven |
| Phase 10E-3 PRT | MEDIUM | Sidecar loading + compute |

**Recommendation**: Proceed to Phase 10E-3 with PRT implementation.