# results/phase10e0_risk_register.md
# Phase 10E-0 Risk Register

## Identified Risks

### Risk 1: Eval Callback Not Firing
- **Severity**: HIGH
- **Impact**: Cannot implement PRT replacement
- **Likelihood**: HIGH (observed in test)
- **Mitigation**: Use graph-level approach instead

### Risk 2: Downstream Reads Stale Data
- **Severity**: HIGH
- **Impact**: Wrong output if PRT replaces but downstream reads old float
- **Likelihood**: LOW (if replacement happens AFTER compute, writes to buffer before downstream reads)
- **Mitigation**: Verify with small test

### Risk 3: Graph Modification Complexity
- **Severity**: MEDIUM
- **Impact**: Larger patch to llama-graph.cpp
- **Likelihood**: MEDIUM (graph-level approach required)
- **Mitigation**: Complete hook trace documentation

### Risk 4: Sidecar Wrong Layer
- **Severity**: LOW
- **Impact**: Wrong PRT for wrong layer
- **Likelihood**: LOW (uses layer index from tensor name)
- **Mitigation**: LAYER_SCOPE=0 only

### Risk 5: Dimension Mismatch
- **Severity**: LOW
- **Impact**: PRT computation fails
- **Likelihood**: LOW (validated at Phase 10A)
- **Mitigation**: Check M=2048, N=11008 matches

---

## Residual Risks (If Replacement Works)

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Accuracy loss from PRT | MEDIUM | Verify cosine > 0.95 |
| Downstream coherence | LOW | Test generation quality |
| Fragile layer corruption | HIGH | Only layer 0 touched |

---

## Overall Risk Assessment

**Phase 10E-0**: Implementation risk HIGH
- Need larger patch to get callback working OR
- Need graph-level integration

**Risk-Adjusted Recommendation**: Proceed with graph-level approach in Phase 10E-1