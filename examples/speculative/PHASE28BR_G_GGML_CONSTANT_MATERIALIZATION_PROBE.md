# Phase 28BR-G: GGML Constant Materialization Probe

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Base HEAD:** `ae9e701c5` (Phase 28BR-F: add guarded true injection canary)
**Timestamp:** `2026-05-24T13:00 EDT`
**Verdict:** `PASS`

---

## Goal

Understand how to materialize decoded F32 residual data into a GGML tensor so it can participate in graph compute safely. This is the blocker that prevented true injection in Phase 28BR-F.

---

## GGML Orientation Findings

**Critical discoveries:**

1. **`ggml_new_tensor_2d(type, ne0, ne1)`**: `ne[0]=ne0` (cols), `ne[1]=ne1` (rows). Storage is col-major: `data[col*ne[1]+row] = element(row, col)`.

2. **`ggml_mul_mat(a, b)`**: Computes `Y = a @ b^T` (TRANSPOSE matmul convention), NOT `a @ b`. 
   - `Y_GGML[m,n] = sum_k a[m,k] * b[n,k]`
   - Requires `a->ne[0] == b->ne[0]`
   - Result shape: `[a->ne[1], b->ne[0]]` = [rows of a, cols of b]

3. **Result tensor buffer is `nil`** after `ggml_backend_graph_compute()`. This is normal GGML behavior for context-allocated tensors. `ggml_backend_tensor_get()` fails with `GGML_ASSERT(buf != NULL)` on result tensors. **But `tensor->data` is still valid** — direct memcpy readback works fine.

4. **Square residual required**: For `X[M,K] @ R[K,N]` to work with `ggml_mul_mat`, we need `X.ne[0]==R.ne[0]` which means `K==N`. Non-square residual needs reshape or transpose approach.

---

## Materialization Method: MethodA (PASS)

**Approach:** `no_alloc=false + direct memcpy`

- Create GGML context with `no_alloc=false` (immediate tensor allocation)
- Tensors get valid `data` pointer immediately upon creation
- Copy decoded F32 residual data via `memcpy` into `tensor->data`
- Build graph and compute
- Read back result via direct memcpy from `tensor->data` (col-major → row-major conversion)

```cpp
struct ggml_init_params params = { .mem_size = 512*1024*1024, .mem_buffer = NULL, .no_alloc = false };
struct ggml_context * ctx = ggml_init(params);
ggml_backend_t cpu = ggml_backend_cpu_init();

// R[K,N] -> ne[0]=N, ne[1]=K; X[M,K] -> ne[0]=K, ne[1]=M
struct ggml_tensor * R_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, K);
struct ggml_tensor * X_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);

// Copy data
memcpy(R_t->data, R_data.data(), K * N * sizeof(float));
memcpy(X_t->data, X_data.data(), M * K * sizeof(float));

// mul_mat: Y = X @ R^T, Y shape [M, N]
struct ggml_tensor * Y_t = ggml_mul_mat(ctx, X_t, R_t);

struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, Y_t);
ggml_backend_graph_compute(cpu, gf);

// Readback: col-major → row-major
for (int m=0; m<M; m++) for (int n=0; n<N; n++)
    Y_out[m*N+n] = ((float*)Y_t->data)[n*M + m];
```

---

## Probe Results

| Test | K | M | N | max_abs_err | finite | compute |
|------|---|---|---|-------------|--------|---------|
| TINY | 4 | 5 | 4 | **0.0** | YES | SUCCESS |
| MEDIUM | 32 | 48 | 32 | **6.7e-8** | YES | SUCCESS |
| LARGE | 64 | 96 | 64 | **1.31e-7** | YES | SUCCESS |

All within `1e-5` threshold. Compute is clean: `nan=0`, `inf=0`.

---

## Negative Tests

| Test | Expected | Actual |
|------|----------|--------|
| `ggml_backend_tensor_get()` on result tensor | Should work | Fails: `GGML_ASSERT(buf != NULL)` — result tensor has `buffer=nil` |
| `no_alloc=true + ggml_backend_alloc_ctx_tensors()` then `tensor_set()` then compute | Works for inputs, result readback fails | Same as above |
| `memcpy` into null `tensor->data` | Crashes | N/A — `no_alloc=false` gives valid data immediately |

---

## llama-graph.cpp Integration Recommendation

**MethodA is safe for injection.** Here's why:

1. **Tensor data is valid immediately**: No two-stage allocation needed.
2. **Result tensor `buffer=nil` is fine for in-graph compute**: GGML reads `tensor->data` directly during `ggml_backend_graph_compute()`. The result doesn't need to be read back to verify correctness — it just needs to participate in the next compute step.
3. **Shape requirements**: Square residual (K==N) or reshape/transpose for non-square.
4. **No `ggml_backend_tensor_set` required**: `memcpy` into `tensor->data` works fine.

**Integration path:**
```cpp
// In llama-graph.cpp at attn_out injection point
// decoded R is in decoded_f32_buffer (size K*N*sizeof(float))

// Create tensor for residual R[K,N] (ne[0]=N, ne[1]=K)
struct ggml_tensor * R_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, K);
memcpy(R_t->data, decoded_f32_buffer, K * N * sizeof(float));

// Create X[M,K] tensor for attn_out_input
struct ggml_tensor * X_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
memcpy(X_t->data, attn_out_input_data, M * K * sizeof(float));

// delta = X @ R^T (transpose matmul)
struct ggml_tensor * delta_t = ggml_mul_mat(ctx, X_t, R_t);

// Add delta to residual stream via ggml_add
struct ggml_tensor * updated_residual = ggml_add(ctx, residual_tensor, delta_t);
```

---

## Risks & Limitations

- **Square residual required**: K==N for `ggml_mul_mat`. Non-square residual needs reshape/transpose.
- **`ggml_backend_tensor_get` unusable on result tensors**: Direct memcpy is the only readback path.
- **Result tensor `buffer=nil`**: GGML internal behavior, not a bug — but requires awareness when debugging.
- **Only tested on CPU backend**: GPU/metal backends may have different lifecycle behavior.
- **No clamping/NaN handling**: True injection could still produce NaN/Inf if input data is bad — requires separate guards.

---

## Next Recommended Phase

**Phase 28BR-H: True injection integration — apply MethodA to llama-graph.cpp**

1. At `layer=0, family=attn_out` injection point, materialize decoded F32 residual into `ggml_tensor` via `no_alloc=false + memcpy`
2. Use `ggml_mul_mat(ctx, attn_out_input, R_tensor)` to compute residual delta (transpose matmul)
3. Add delta to residual stream via `ggml_add`
4. Test on actual model inference with `--prt-sidecar-true-injection`
5. Verify: finite, nan=0, inf=0, quality not degraded

---

## Files Changed

- `examples/speculative/prt_ggml_constant_materialization_probe.cpp` — standalone probe
- `examples/speculative/PHASE28BR_G_GGML_CONSTANT_MATERIALIZATION_PROBE.md` — this report
- `examples/speculative/results/phase28br_g_ggml_constant_materialization_probe.json` — structured results

**No model files, no sidecars, no binaries staged.**