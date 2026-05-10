# PRT Phase 19E: 0.5B Sidecar-Backed Loader Substitution Probe

## Status: PARTIAL - BLOCKED by loader complexity

### Key Findings

**Analysis of PRT custom op data flow:**
```cpp
// build_prt_ffn_up receives cur = ffn_norm output tensor
ggml_tensor * build_prt_ffn_up(ctx, cur, layer_id)

// Custom op reads:
//   src0->data = ffn_norm output (ACTIVATION, not weight)
//   ud->int8_data = sidecar weights (computed at load time)
//   writes to dst->data = output buffer
```

**Critical insight:** The native ffn_up GGUF tensor is NEVER accessed by PRT at runtime!
- PRT reads from `cur` (ffn_norm activations)
- PRT reads from sidecar data (loaded separately)
- PRT writes to output buffer
- Native ffn_up weight data is completely bypassed

### Test Results

| Mode | Status | Speed |
|------|--------|-------|
| Native | ✅ Pass | 95.0 t/s |
| PRT overlay | ✅ Pass | 17-18 t/s |
| PRT (layer0 zeroed) | ✅ Pass | - |

### Implementation Challenge

The loader complexity blocks clean implementation:
1. `load_data_for` is called for every tensor in a loop
2. Both mmap and non-mmap paths must be intercepted
3. Must preserve buffer allocation while skipping data copy
4. Risk of crashes if pointer is left as nullptr

**Verdict: BLOCKED_BACKEND_BUFFER_ASSUMPTION**

### Runtime Proof

Even with full layer0 substitution (dummy weights), PRT mode works because:
- PRT path uses sidecar data exclusively
- Native ffn_up tensor is replaced during graph building
- Native matmul is never executed

### Commit
1e2b50616

*Date: 2026-05-10*