# PRT Phase 20K: ggml-Native PRT Backend / Operator Design

**Verdict:** `RECOMMEND_GGML_NATIVE_PRT_OP`

---

## Phase 20K-A: Current Architecture Summary

### CLI Flags
- `--prt-mode` → `g_prt_debug_mode` (int)
- `--prt-only-layers` → `g_prt_only_layers_set[]` (bool[36])
- `--prt-force-native` → `g_prt_force_native_layer[]` (bool[36])
- `--prt-sidecar-dir` → `g_prt_sidecar_dir` (string)
- `--prt-sidecar-format` → `int6|int8|float32`
- `--prt-predecode-f32` → format=2→format=0 predecode at load
- `--prt-log-level` → `g_prt_log_level` (int)
- `--prt-log-file` → `g_prt_log_file` (FILE*)

### Sidecar Loading
- Harness loads sidecar files at startup
- `llama_set_prt_sidecar_int6()` → sets `g_prt_int8_data[]` + `g_prt_int8_scales[]` + `g_prt_sidecar_format[]=2`
- INT6 stored as int8 range [-31,+31], per-row scales, row-major
- **No per-token decode/unpack** — weights already in memory as int8
- Format 2 = scalar path, Format 0 = AVX2 path (via predecode)

### Graph Integration
- `llama-graph.cpp:1213`: checks `g_prt_only_layers_set[il]` + sidecar presence
- If active → `build_prt_ffn_up()` (custom op)
- If not → `build_lora_mm()` (native matmul)
- Native path: `ggml_mul_mat(w, cur)` → `GGML_OP_MUL_MAT`
- PRT path: `ggml_custom_4d()` → `prt_ffn_up_custom_op()` scalar loop

### Compute Paths

| Format | Storage | Kernel | Speed |
|--------|---------|--------|-------|
| 0 (float32) | f32 | AVX2 | Fast |
| 1 (int8) | int8 + scales | Scalar | Medium |
| 2 (INT6) | int8 + scales | **Scalar** | **SLOW** ← current |

### Why Native is Faster
Native `ggml_mul_mat` with Q4_K_M weights → dispatches to `vec_dot_q4_K_q8_K()`:
- AVX2 vectorized dot products
- Bit-level dequantization inline
- Fused multiply-add
- Optimized cache access

PRT `ggml_custom_4d` with format=2:
- Generic callback dispatch overhead
- `n_tasks=1` threading
- Triple-nested scalar loop (token, ffn, hidden)
- No vectorization, no SIMD

---

## Phase 20K-C: Architecture Options Evaluated

### Option 1: Optimize scalar INT6 (keep custom op)
**Verdict: Not recommended**
- Pros: Minimal code change
- Cons: Ceiling limited — scalar loop can't beat AVX2
- Risk: Even optimized scalar can't close gap with native

### Option 2: INT8 resident compute buffer
**Verdict: Backup option**
- Pros: Uses existing AVX2 float32 kernel path
- Cons: 2x RAM cost (INT8 weights + f32 decoded)
- Status: Already exists as `--prt-predecode-f32` path

### Option 3: Direct packed INT6 AVX2 kernel
**Verdict: Complex, risky**
- Pros: Memory-efficient
- Cons: Unpack overhead in inner loop; complex; likely bugs
- Risk: High

### Option 4: ggml-native PRT op ← **PRIMARY RECOMMENDATION**
**Verdict: Best long-term**
- Pros: Clean integration, proper SIMD, no custom_op overhead
- Cons: Needs new op enum + kernel registration
- Risk: Medium (well-defined scope)
- **Estimated: 80-90% native speed feasible**

### Option 5: CPU backend kernel extension
**Verdict: Similar to Option 4 but deeper**
- Pros: Best performance potential
- Cons: More invasive to ggml upstream
- Risk: Higher (upstream compatibility concern)

### Option 6: Clean PRT-v2 branch
**Verdict: Good hygiene, not primary architecture**
- Pros: Removes accumulated debt
- Cons: Doesn't solve the performance problem by itself

---

## Phase 20K-E: Recommended PRT-v2 Architecture

### Proposed: GGML_OP_PRT_FFN_UP

A new first-class ggml operation that replaces `ggml_mul_mat` for selected layers.

#### Operator Shape

```
ggml_tensor * ggml_prt_ffn_up(
    struct ggml_context * ctx,
    ggml_tensor * activation,      // [hidden, n_tokens] input
    const float * weights,          // [hidden, ffn] sidecar (f32 or predecoded)
    const float * scales,           // [ffn] per-row scales
    int M, int K, int N,            // M=hidden, K=ffn, N=n_tokens
    int layer_id,                   // for logging/debug
    enum ggml_prt_format format     // PRT_F32, PRT_INT8, PRT_INT6
);
```

#### Sidecar Representation
- **Default: f32 sidecar** (from `--prt-predecode-f32` path)
  - Enables full AVX2 vectorization immediately
  - RAM: ~M×K×4 bytes per layer (for Qwen2.5-7B: ~272MB per layer)
- **INT8 resident** (future): Keep INT8 in RAM, decode to f32 in kernel
- **INT6 on disk** (future): Decode once to INT8 buffer at load

#### Compute Kernel (CPU backend)
```
// Reference kernel (AVX2 target):
void ggml_compute_forward_prt_ffn_up(params, dst) {
    // Input: [M, N] activations (f32)
    // Weights: [M, K] f32 sidecar
    // Output: [K, N] ffn_up (f32)
    // For each token t:
    //   for each output col j:
    //     dot = 0
    //     for k in 0..M-1 (vectorized):
    //       dot += X[t,k] * W[j,k]  // AVX2 FMA
    //     Y[t,j] = dot * scales[j]
}
```

#### Graph Integration
- In `llama-graph.cpp`: replace `build_lora_mm(up, cur)` with:
  ```cpp
  if (g_prt_only_layers_set[il] && has_sidecar) {
      tmp = ggml_prt_ffn_up(ctx0, cur, sidecar, scales, hidden, ffn, il, format);
  } else {
      tmp = this->build_lora_mm(up, cur);  // native
  }
  ```
- No more `ggml_custom_4d` → eliminates custom-op overhead
- ggml scheduler sees PRT as standard op → proper fusion/scheduling

#### Validation Ladder
1. **Phase 21A**: Add `GGML_OP_PRT_FFN_UP` to ggml.h op enum + name function
2. **Phase 21B**: Scalar reference kernel in ggml-cpu/ops.cpp
3. **Phase 21C**: Synthetic kernel test (known inputs → known outputs)
4. **Phase 21D**: 0.5B runtime canary (single layer PRT vs native)
5. **Phase 21E**: 7B single-layer → verify timing split
6. **Phase 21F**: AVX2 kernel prototype (vectorized dot)
7. **Phase 21G**: Sparse policy (10,20) timing comparison
8. **Phase 21H**: Checkpoint or abandon based on metrics

---

## Phase 20K-F: Migration Plan

### Phase 21A — Skeleton (1-2 days)
- Add `GGML_OP_PRT_FFN_UP` to `ggml/include/ggml.h` enum
- Add `ggml_prt_ffn_up()` declaration
- Implement stub returning nullptr in ggml-backend.c
- **Pass criteria**: Compiles, llama-cli still works
- **Rollback**: Revert ggml.h changes

### Phase 21B — Scalar Reference Kernel (2-3 days)
- Add `ggml_compute_forward_prt_ffn_up()` in `ggml-cpu/ops.cpp`
- Scalar triple-loop (token, ffn, hidden)
- Register in `ggml-cpu` ops dispatch table
- **Pass criteria**: Graph builds without error
- **Rollback**: Remove from dispatch table

### Phase 21C — Synthetic Test (1 day)
- Create unit test: known weights × known activations → known output
- Compare against numpy reference
- **Pass criteria**: 100% numerical match
- **Rollback**: Disable op dispatch

### Phase 21D — 0.5B Canary (1-2 days)
- Test on tiny model with single PRT layer
- Compare output and timing against native
- **Pass criteria**: Output matches, no crash
- **Rollback**: Revert to custom_4d path

### Phase 21E — 7B Single-Layer (2 days)
- Test (10) on Qwen2.5-7B
- Verify: clean output, timing captured
- **Pass criteria**: Clean, timing available
- **Rollback**: Keep custom_4d as fallback

### Phase 21F — AVX2 Kernel (3-5 days)
- Vectorized dot: `__m256` FMA loops over hidden dimension
- Row-parallel over ffn dimension
- **Pass criteria**: AVX2 kernel produces same output as scalar reference
- **Rollback**: Fall back to scalar

### Phase 21G — Sparse Policy Timing (1-2 days)
- Compare (10,20) timing: old custom_op vs new GGML_OP
- Compare (10,20) output: must match exactly
- **Pass criteria**: Faster than custom_op AND output matches
- **Rollback**: Keep custom_op as fallback

### Phase 21H — Checkpoint (1 day)
- If (10,20) reaches ≥80% native speed: merge and publish
- If <80%: classify as "needs more work" and pause
- **Falsification**: If AVX2 kernel can't beat scalar by >20%, abandon path

---

## Phase 20K-G: Report

### A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

### B. Previous HEAD
`e3f59447a` (Phase 20J)

### C. New HEAD
Pending commit

### D. Bottleneck Summary
Format=2 scalar loop in custom_op vs AVX2 in native; custom_op overhead; only 2/28 layers replaced

### E. Architecture Options Compared
All 6 evaluated (see Section C)

### F. Recommended Architecture
**Option 4: GGML_OP_PRT_FFN_UP** — first-class ggml operation with AVX2 CPU backend kernel

### G. Why Current Custom-Op Overlay is Insufficient
- `ggml_custom_4d` is generic callback → scheduler can't optimize
- `n_tasks=1` forces single-thread (no parallelism)
- No vectorization possible through generic interface
- Custom op prevents graph fusion around PRT

### H. Integration Points
1. `ggml/include/ggml.h` — add `GGML_OP_PRT_FFN_UP` enum + `ggml_prt_ffn_up()` declaration
2. `ggml/src/ggml-cpu/ops.cpp` — add `ggml_compute_forward_prt_ffn_up()`
3. `ggml/src/ggml-backend.c` — register op dispatch
4. `src/llama-graph.cpp` — replace `build_prt_ffn_up()` with `ggml_prt_ffn_up()`
5. `src/llama.cpp` — keep sidecar loading API unchanged
6. `examples/speculative/prt_graph_replace.h` — deprecate

### I. Sidecar Representation
- **Phase 21A-D**: Use existing `--prt-predecode-f32` path (f32 sidecar in RAM)
  - Simplest integration, enables AVX2 immediately
  - RAM cost acceptable for 2-3 sparse layers
- **Phase 21E+**: INT8 resident if memory becomes bottleneck

### J. Compute Kernel
1. **Scalar reference**: triple-loop, correctness baseline
2. **AVX2 target**: `__m256` FMA over hidden dimension, row-parallel over ffn
3. **ARM NEON** (future): `float32x4_t` dot products
4. **AVX512** (future): `__m512` for newer hardware

### K. Validation Ladder
21A → 21B → 21C → 21D → 21E → 21F → 21G → 21H (see Section E)

### L. Phase 21 Implementation Plan
See Section F (8 phases with pass/fail criteria)

### M. Risks
- ggml upstream compatibility: mitigated by keeping GGML_OP_MUL_MAT path as default
- New op enum causes conflicts: mitigated by checking latest ggml master
- AVX2 kernel complexity: mitigated by scalar-first, vectorize second

### N. Expected Benefits
- Eliminates custom_op overhead (5-10% improvement)
- Enables AVX2 vectorization (30-50% improvement over scalar)
- Proper graph scheduling (5-10% improvement)
- **Target: 80-90% native speed** (vs current 65%)

### O. What Would Falsify This Path
- If scalar reference kernel can't even match current custom_op speed → stop
- If AVX2 kernel can't beat scalar by >20% → stop
- If new op breaks upstream ggml compatibility → stop

### P. Models/Sidecars Staged? NO
### Q. Secrets Detected? NONE
### R. Existing Tags Touched? NO

---

## Recommendation

**Primary: Option 4 — GGML_OP_PRT_FFN_UP**

**Backup: Option 5 — CPU backend kernel** (if Option 4 proves too invasive to ggml upstream)

**Why not Option 1 (optimize scalar):** Because scalar has a hard ceiling. The gap between scalar and AVX2 is 3-5x, and no amount of scalar optimization closes that.

**Why not Option 2 (INT8 resident):** It's a workaround, not a solution. It uses more memory and still has custom_op overhead.

**Why not Option 3 (direct INT6 AVX2):** Too complex for initial implementation. The unpack overhead inside a vectorized dot is non-trivial.

**Why Option 4:** It's the cleanest long-term path. It fixes the root cause (custom_op overhead + scalar kernel) and aligns with ggml's design principles.

**Next step: Phase 21A** — Add `GGML_OP_PRT_FFN_UP` to ggml.h op enum.

---

*Phase 20K Complete*
*HEAD: e3f59447a*