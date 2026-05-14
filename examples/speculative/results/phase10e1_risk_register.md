# results/phase10e1_risk_register.md
# Phase 10E-1 Risk Register

## Risks from Graph-Level Interception

### Risk 1: Performance Impact from Logging
- **Severity**: LOW
- **Impact**: fprintf to stderr on every ffn_up matmul (hundreds per decode)
- **Likelihood**: HIGH (observed in test)
- **Mitigation**: Remove or guard with debug flag before production

### Risk 2: Build Contamination
- **Severity**: LOW
- **Impact**: Static counters persist across runs
- **Likelihood**: LOW (static vars reset on process restart)
- **Mitigation**: None needed for testing

### Risk 3: Wrong Layer Detection
- **Severity**: HIGH
- **Impact**: Replacement fires on wrong layer
- **Likelihood**: LOW (il parameter correctly passed from caller)
- **Mitigation**: Verified by interception count for layer=0 being count=1,2

### Risk 4: Downstream Reads Stale Float Data
- **Severity**: HIGH
- **Impact**: If substitution attempted but fails, downstream gets wrong output
- **Likelihood**: N/A (no substitution implemented)
- **Mitigation**: Not implemented yet

---

## Risks for Actual PRT Substitution (Next Phase)

### Risk 5: Custom Op Registration
- **Severity**: MEDIUM
- **Impact**: Cannot register custom op without ggml-core patch
- **Likelihood**: HIGH (ggml custom ops require compute function registration)
- **Mitigation**: Use GGML_OP_MAP_CUSTOM1 if pre-registered in build

### Risk 6: Quantization Handling
- **Severity**: HIGH
- **Impact**: Q4_K weights must be dequantized inside custom op
- **Likelihood**: HIGH (weights are Q4_K in our model)
- **Mitigation**: Load float sidecars (already done in Phase 10A) — use those

### Risk 7: Graph Topology Breakage
- **Severity**: CRITICAL
- **Impact**: Wrong tensor pointer breaks downstream ops
- **Likelihood**: MEDIUM (if using precomputed tensor approach)
- **Mitigation**: Use custom op (preserves topology)

### Risk 8: Sidecar Dimension Mismatch
- **Severity**: MEDIUM
- **Impact**: PRT matmul produces wrong shape output
- **Likelihood**: LOW (validated in Phase 10A: M=2048, N=11008)
- **Mitigation**: Cross-check tensor ne[] dimensions before compute

### Risk 9: Memory Allocation in Custom Op
- **Severity**: MEDIUM
- **Impact**: dst buffer may not be allocated when custom op fires
- **Likelihood**: LOW (ggml allocates dst before compute)
- **Mitigation**: Use ggml_backend_tensor_get/set for cross-backend safety

---

## Residual Risks After Phase 10E-1

| Risk | Severity | Mitigation |
|------|----------|------------|
| Logging performance impact | LOW | Guard with PRT_DEBUG flag |
| Static counter persistence | LOW | Reset between runs |
| Wrong layer detection | HIGH | Already verified with interception |
| Custom op registration | MEDIUM | Investigate GGML_OP_MAP_CUSTOM1 availability |

---

## Overall Risk Assessment

**Phase 10E-1**: LOW RISK ✅
- Interception is safe — no substitution, just logging
- Generation runs normally
- No fragile layers touched

**Phase 10E (substitution)**: HIGH RISK ⚠️
- Custom op approach needed (complex)
- Quantization handling required
- Graph topology must be preserved