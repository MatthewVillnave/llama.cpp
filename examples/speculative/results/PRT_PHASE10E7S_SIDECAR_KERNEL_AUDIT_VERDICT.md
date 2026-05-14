# PRT_PHASE10E7S_SIDECAR_KERNEL_AUDIT_VERDICT.md

## Phase 10E-7S: PRT Custom Op Integration Verdict

### Summary
The PRT sidecar kernel produces CORRECT output in standalone mode (outside llama.cpp) but GARBAGE when integrated via ggml_map_custom2.

### Root Cause: ggml Custom Op Integration Bug

**When sidecars are loaded**, the ggml computation graph produces garbage output via path string fragmentation, regardless of what the custom op does inside.

### Evidence

| Test | Sidecar Loaded | Custom Op Logic | Output |
|------|--------------|---------------|--------|
| Standalone PRT compute | ✅ Yes | PRT math | ✅ Correct |
| llama-completion (no sidecar) | ❌ No | Identity fallback | ✅ Correct |
| llama-phase10e0 (sidecar loaded) | ✅ Yes | Identity copy | ❌ GARBAGE |

### Root Cause Details

1. The matmul result tensor (`src0`) arrives at the custom op already containing garbage path strings
2. Even replacing the op with identity copy (no sidecar access) produces garbage output
3. The corruption is upstream in the ggml graph computation BEFORE the custom op fires

### Where Corruption Occurs

The path string "ffn_up_layer35_prt.bin" appears to be leaking from the sidecar storage into the tensor computation graph. When `llama_set_prt_sidecar` stores sidecar pointers and the ggml_map_custom2 runs, something about this combination corrupts the ggml tensor buffers.

### Recommendation

This is a deeper ggml integration bug. The sidecar loading and custom op registration mechanism is incompatible with llama.cpp's graph computation in a way that produces memory corruption. Further debugging of ggml_map_custom2 is required to identify the exact corruption point.

### Status
- Phase 10E-7S: **FAIL** ❌
- Root cause: In ggml custom op integration (not in custom op code)