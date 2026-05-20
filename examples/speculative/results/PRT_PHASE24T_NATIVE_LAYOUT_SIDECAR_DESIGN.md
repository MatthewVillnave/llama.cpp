# PRT Phase 24T: Native-Layout Sidecar Feasibility Design Doc

## Status: COMPLETE ✅

**Date:** 2026-05-20
**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Previous HEAD:** `dedcb4b0e` (Phase 24S)
**New HEAD:** `TBD` (docs-only commit)
**Scope:** Design-only feasibility doc — no implementation, no new runs

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD (Phase 24S)
`dedcb4b0e`

## C. New HEAD
`TBD` — docs-only commit

## D. Native Layout Feasibility

### D.1 What Layout Does GGML Native Matmul Actually Want?

From `ggml/src/ggml.c`:

```cpp
// ggml_can_mul_mat check:
return (t0->ne[0] == t1->ne[0])  &&   // K must match
       (t1->ne[2] % t0->ne[2] == 0)    &&   // broadcast
       (t1->ne[3] % t0->ne[3] == 0);

// ggml_mul_mat output shape:
result->ne = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
// = [M, n, 1, 1] for weight [K,M] × activation [K,n]
```

**Tensor shape expectation for weight W:**
- W.ne[0] = K (inner/dim-most dimension)
- W.ne[1] = M (intermediate size)
- **This is [K,M] shape** — same as current canonical sidecar output

**GGML matmul formula:** result[j,n] = sum_k a[k,j] * b[k,n]
- Where a is weight with shape [K,M] and a[k,j] = W at row k, col j
- Where b is activation with shape [K,n]

**Our PRT formula:** Y[j,n] = sum_k X[k,n] * W[k,j]
- Same math — GGML's a[k,j] * b[k,n] equals W[k,j] * X[k,n]

**Key finding:** The [K,M] shape is already correct for ggml_mul_mat. The issue is not the shape — it's that ggml_mul_mat checks !ggml_is_transposed(a). If W has op == GGML_OP_TRANSPOSE, the check fails.

**Why ggml_is_transposed fails:**
- ggml_is_transposed checks: (a)->op == GGML_OP_TRANSPOSE || (a)->op == GGML_OP_PERMUTE
- If W was created via ggml_transpose or other view ops, it inherits the TRANSPOSE op flag
- Memory layout (row vs column major) does NOT matter — GGML doesn't check memory, only ne dimensions and op flag

**So the actual requirements are:**
1. Tensor shape [K,M] with ne[0]=K, ne[1]=M
2. !ggml_is_transposed(tensor) — tensor must NOT have the TRANSPOSE op flag
3. Memory-contiguous or broadcast-compatible stride layout

### D.2 Would Current [K,M] Sidecar Work Directly?

**Analysis:** If the decoded f32 W tensor from our sidecar has:
- ne[0]=K, ne[1]=M ✓ (correct shape)
- op is NOT GGML_OP_TRANSPOSE ✓ (it's a data tensor, not a transpose view)
- nb strides are contiguous ✓ (row-major memory from our explicit copy)

Then ggml_mul_mat(W, cur) **should work** without any reshape.

**The Phase 24R crash was at ggml_reshape_2d, NOT at ggml_mul_mat itself.** We never actually tested whether ggml_mul_mat(W, cur) with raw [K,M] tensors works.

**This is the key unanswered question from Phase 24R.**

### D.3 Why Phase 24R Never Tested the Direct Path

Phase 24R jumped to reshape approaches because of an incorrect assumption about GGML's layout requirements. The actual crash path was:

1. **Attempt 1 (v1):** Direct ggml_mul_mat(W, cur) → crashed at ggml_can_mul_mat assertion
2. **Attempt 2 (v2):** Added explicit transpose + copy → crashed at ggml_can_mul_mat with transposed W
3. **Attempt 3 (v3):** Tried ggml_transpose + ggml_cont → GGML_ASSERT(!ggml_is_transposed) on result
4. **Attempt 4 (v4):** Reshape to 4D → ggml_mul_mat(4D_W, 4D_cur) succeeded BUT ggml_reshape_2d(result) failed

The crash in Attempt 1 (ggml_can_mul_mat failed) happened BEFORE we added any reshape code. But we then assumed the issue was the tensor shape and jumped to reshape solutions, never fixing Attempt 1.

**Root cause of Attempt 1 crash:** Still unclear. Possible causes:
- W tensor from ggml_new_tensor has op == GGML_OP_TRANSPOSE? (unlikely for data tensor)
- W tensor has non-contiguous stride? (possible if ggml_new_tensor uses exotic allocation)
- Some other property of W triggers ggml_can_mul_mat failure

**The correct next step should have been:** Add debug logging to Attempt 1 to see exactly which ggml_can_mul_mat condition failed.

---

## E. Proposed Native-Layout Sidecar Design

### E.1 Current Canonical INT8 Layout

```
File structure (K + M bytes total):
  Offset 0 to K-1:          int8 q[K]       (quantized values)
  Offset K to K+M-1:        float32 scale[M] (dequantization scales)

Memory layout (row-major, K rows × M cols):
  row k contains: q[k*M + 0], q[k*M + 1], ..., q[k*M + M-1] for k in [0,K)

Decoded f32 tensor W with shape [K,M] in row-major memory:
  W[k*M + j] = (float)q[k*M + j] * scale[j]   // for k in [0,K), j in [0,M)
```

### E.2 GGML-Native-Compatible Layout (Hypothetical)

For ggml_mul_mat(W, cur) to work directly without reshape:
- W tensor: ne[0]=K, ne[1]=M, contiguous row-major memory, op != TRANSPOSE
- Memory layout: W[k,j] at offset k*M + j (same as current)

**No layout change needed from current canonical.**

The only requirements are:
1. W tensor must be properly constructed with correct ne, contiguous memory, no transpose op
2. The exact same [K,M] shape and memory layout as current canonical

### E.3 What Would a "Native-Layout Sidecar" Actually Mean?

If the issue is that current sidecar produces a tensor that fails ggml_can_mul_mat for some subtle reason, a "native-layout sidecar" might need to:
- Store data already laid out in a way that survives GGML tensor construction
- OR store [M,K] layout (transposed) to compensate for something

But since [K,M] is the correct shape for ggml_mul_mat, and current canonical produces [K,M], the issue is likely in HOW the tensor is constructed, not WHAT is stored.

**Conclusion:** There may be no need for a different sidecar layout at all. The fix might be in how we construct the W tensor in llama-graph.cpp.

### E.4 Expected 3B Sidecar Size

Current canonical INT8 (K=2048, M=11008):
```
Size = K bytes (int8) + M * 4 bytes (float32 scales)
     = 2048 + 11008 * 4
     = 2048 + 44032
     = 46080 bytes ≈ 45KB
```

Decoded f32 (K=2048, M=11008):
```
Size = K * M * 4 bytes
     = 2048 * 11008 * 4
     = 90,177,536 bytes ≈ 86MB per layer
```

For reference:
- Qwen2.5-3B-Q4_K_M model: ~1.9GB
- Qwen2.5-3B native FFN_UP Q4_K_M: ~45MB (estimated)

### E.5 Decoded f32 vs Native Q4_K_M Comparison

| Weight Type | Size for 3B FFN_UP |
|-------------|-------------------|
| Native Q4_K_M (model weight) | ~45MB estimated |
| Decoded f32 (PRT sidecar) | ~86MB |

**The decoded f32 PRT weight is ~2x larger than native Q4_K_M.** This is expected — INT8 decoding doesn't compress, it just shifts precision. The PRT sidecar uses INT8 for canonicity and storage efficiency, but decoded f32 is full precision and larger.

This means: even if we fix the native matmul path, the decoded f32 weights are significantly larger than native quantized weights. The memory bandwidth advantage of PRT over native Q4 would be negative.

**This is the core argument against the native-layout path.**

---

## F. Minimal Proof Design

### F.1 Step 1: Debug Attempt 1 (Direct ggml_mul_mat)

**Goal:** Determine why Attempt 1 ggml_mul_mat(W, cur) crashed.

**Method:**
1. Add debug logging before ggml_mul_mat(W, cur):
   - W->ne[0], W->ne[1], W->op, W->nb[0], W->nb[1]
   - cur->ne[0], cur->ne[1], cur->op
   - ggml_is_contiguous(W), ggml_is_transposed(W)
2. Run with the debug logging
3. Check which condition causes the crash

**Expected outcomes:**
- If ggml_is_transposed(W) == true → W has transpose op flag, fix how W is created
- If W->ne[0] != cur->ne[0] → shape mismatch, check tensor construction
- If W->nb[0] is unexpected → stride issue, fix tensor construction
- If all conditions pass but it still fails → deeper GGML issue

**If the fix is in W construction:** We can use ggml_mul_mat(W, cur) without any layout change or sidecar regeneration. This would be a **GO** signal.

### F.2 Step 2: If Step 1 Succeeds, Measure Performance

**Minimal test:**
1. 3B layer0 only
2. PRT_V2_USE_NATIVE_MULMAT=1 (fixed W construction) + ggml_mul_mat(W, cur)
3. Compare output to custom-op PRT and native
4. Compare timing to custom-op PRT (Mode D) and native (Mode A)

**Success criteria:**
- Output matches custom-op PRT (within FP32 tolerance)
- Timing < Mode D (custom-op PRT at 3.47s)
- Ideally, timing < Mode A (native at 2.35s) — but this is unlikely given 2x memory size

**If timing is between native and custom-op PRT:** The graph overhead savings (~470ms) may be real, but kernel compute is the same. Net benefit may be modest.

**If timing is worse than custom-op PRT:** Native matmul doesn't help because the kernel is the bottleneck, not the graph overhead.

### F.3 Step 3: If Steps 1+2 Are Promising, Consider Sidecar Format Change

Only if Step 1+2 reveals that a layout change would genuinely help AND the performance benefit is significant.

---

## G. Main Risks

### G.1 Risk: f32 Memory Footprint
- Decoded f32 PRT weights are ~86MB for 3B layer0 vs ~45MB native Q4
- Memory bandwidth for f32 = 4 bytes/element vs Q4 = ~0.5 bytes/element
- **PRT would use ~4-8x more memory bandwidth than native Q4** even with perfect matmul
- This makes PRT strictly slower than native for memory-bound inference workloads

### G.2 Risk: Losing Compression Advantage
- Current PRT sidecar is INT8 (2 bytes per element for q+scale, ~1 byte effective after scale)
- Native Q4_K_M is ~4 bits per element
- Neither achieves actual compression in decoded form
- But: PRT's value is in the canonical sidecar representation, not in runtime speed

### G.3 Risk: Native Q4 Still Faster
- Even if we fix the GGML tensor construction issue, the f32 matmul is memory-bound
- Native Q4_K_M matmul uses bitwise operations + quantization math that is much faster per byte
- Expected outcome: PRT with native matmul is slower than native Q4 inference

### G.4 Risk: GGML Tensor Ownership/Layout Complexity
- The Phase 24R crash in ggml_reshape_2d suggests GGML graph building has constraints we don't fully understand
- Before committing to a native-layout path, we'd need to understand why reshape fails
- This requires careful debugging, not just design

### G.5 Risk: Sidecar Format Fragmentation
- If we add a "native-layout" sidecar variant, we now have 2+ formats to maintain
- Canonical [K,M] for custom kernel, native [K,M] for GGML (if different)
- This increases tooling complexity

---

## H. Decision Matrix

| Condition | Recommendation |
|-----------|----------------|
| Step 1: Direct ggml_mul_mat works with debug fix | **GO** — minimal implementation, test performance |
| Step 1: W construction issue, fixable | **GO** — fix W construction, test performance |
| Step 1: GGML issue deeper, requires reshape | **MAYBE** — requires more investigation |
| Step 2: Performance better than custom-op PRT | **GO** — native path has graph overhead benefit |
| Step 2: Performance same/worse than custom-op PRT | **NO-GO** — graph overhead savings don't outweigh kernel |
| Step 2: Performance worse than native | **NO-GO** — PRT is not a speed path |
| Sidecar format change required | **NO-GO** — format fragmentation cost not justified |

---

## I. Recommendation

### GO/NO-GO Framework:

**GO if:** Step 1 debug reveals a simple tensor construction fix (e.g., W needs ggml_set_op or contiguous flag). Proceed with minimal implementation probe.

**MAYBE if:** Step 1 debug reveals the issue is in GGML graph building constraints (reshape or similar). Requires deeper investigation before committing.

**NO-GO if:**
- Step 1 reveals the issue is fundamental to how GGML handles custom tensors (not fixable without major GGML work)
- Step 2 shows decoded f32 is not faster than custom-op PRT (because f32 bandwidth >> Q4 bandwidth advantage)
- Performance is worse than native Q4 (PRT is not a speed path)

### Recommended Path:

**Phase 24U:** One-session debug probe. Add tensor inspection logging to Attempt 1. Run once. If simple fix found, implement + test timing. If deeper issue, document and stop.

**Do NOT:**
- Generate new sidecar formats without confirming the path is viable
- Claim speedup before Step 2 confirms it
- Proceed to multi-layer or 7B timing without clean Step 1+2 result

---

## J. Safety Scan

git status: Clean (docs only)
Large files: None in this commit
Secrets: None detected
Tags: None touched

## K. Files in This Commit

```
A examples/speculative/results/PRT_PHASE24T_NATIVE_LAYOUT_SIDECAR_DESIGN.md
A examples/speculative/results/phase24t_native_layout_sidecar_design.json
```

## L. Phase Series Summary

| Phase | Subject | Key Result |
|-------|---------|------------|
| 24H | Canonical INT8 layout | Canonical format defined |
| 24K | State forensics | State machine understood |
| 24L/24O | Timing cleanup | Clean baseline timing |
| 24P | Overhead isolation | ~1.19s overhead identified |
| 24Q | Custom-op vs kernel | ~470ms graph + ~610ms kernel |
| 24R | Native matmul probe | Blocked by GGML layout |
| 24S | Architecture decision | Decision checkpoint created |
| **24T** | **Native-layout design** | **Design doc + GO/NO-GO framework** |