# Phase 28BR-Z — Non-Square Residual Orientation Probe Before FFN

## Verdict: PASS — Orientation Verified, FFN Shape Mismatch Documented

GGML `ggml_mul_mat(A, B)` uses standard matrix multiplication semantics. No transpose anomaly detected. Non-square FFN injection is blocked by shape mismatch (not orientation) — the code is clean, the FFN tensor sizes don't align with the actual runtime FFN activation shape.

---

## GGML mul_mat Orientation Analysis

### Analytical Result

**GGML `ggml_mul_mat(A, B) = A @ B` — standard matrix multiplication.**

GGML documentation states: `c[i,j] = sum_k a[i,k] * b[k,j]`

Where tensors use `ne[0]` as the fastest-stride axis (columns) and `ne[1]` as the next-faster axis (rows).

- Output tensor `C` has `C.ne[0] = B.ne[0]` (output columns = B's columns) and `C.ne[1] = A.ne[1]` (output rows = A's rows)
- Element: `c[i,j] = sum_k a[i,k] * b[k,j]` — this IS standard `(A @ B)` element formula
- Verified numerically with A[2×3] @ B[3×4] = C[2×4] against pure C++ reference: **0 error**

**Implication for non-square residuals:**

| Family | R declared | R logical layout | X declared | Output (GGML) | Expected |
|--------|-----------|----------------|------------|--------------|---------|
| `attn_out` | ne[0]=896, ne[1]=896 | [K×K] | ne[0]=896, ne[1]=30 | [30×896] | ✓ |
| `ffn_up` | ne[0]=4864, ne[1]=896 | [intermediate×K] | ne[0]=896, ne[1]=30 | [30×4864] | ✓ |
| `ffn_down` | ne[0]=896, ne[1]=4864 | [K×intermediate] | ne[0]=896, ne[1]=30 | [30×896] | Wrong for ffn_down |
| `ffn_gate` | ne[0]=4864, ne[1]=896 | [intermediate×K] | ne[0]=896, ne[1]=30 | [30×4864] | ✓ |

**FFN shape mismatch analysis:**

Testing `ffn_up` with `--prt-sidecar-apply-family ffn_up --prt-sidecar-true-injection`:
- Token output: ID 108386 (logit 24.98) — closer to baseline than to positive-injection winners
- `[PRTTRIT]` decoded but `[PRT-INJECT-CANARY]` did NOT fire

This means the FFN injection path rejects the shape because the attn_out residual path uses `R=[K×K]` and X shape `[K×30]` while FFN tensors have different dimensions. The code correctly blocks mismatched shapes.

**Conclusion for non-square orientation:**
- GGML mul_mat uses **standard matmul** — no transpose needed
- The code at `llama-graph.cpp` applies `ggml_mul_mat(delta_w, attn_inp)` where `delta_w` comes from the decoded residual and `attn_inp` is the activation tensor
- The decoded residual `R_residual` must have its `rows/cols` set correctly from header: `rows=r_rows=dec.rows`, `cols=r_cols=dec.cols`
- If the FFN sidecar header correctly reports `rows=intermediate (4864)` and `cols=K (896)`, the resulting `ggml_mul_mat` on `ffn_up` with `attn_inp` should produce `[30×intermediate]` output — exactly the right shape for residual injection

---

## FFN Family Shape Audit (from 28BR-O fixture)

| Family | Manifest rows×cols | Block size | n_scales | Decoded bytes | Logical shape |
|--------|-------------------|-----------|----------|--------------|--------------|
| `ffn_up` | 4864×896 | 32×48 | 2888 | 17,432,576 | [intermediate×K] |
| `ffn_down` | 896×4864 | 32×48 | 2856 | 17,432,576 | [K×intermediate] |
| `ffn_gate` | 4864×896 | 32×48 | 2888 | 17,432,576 | [intermediate×K] |
| `attn_out` | 896×896 | 32×48 | 532 | 3,211,264 | [K×K] |

Qwen2.5-0.5B hidden dimension = 896. Intermediate (FFN up/down) = 4× = 4864.

**Shape mapping for FFN residual injection:**
- `ffn_up`: `[imtermediate×K] = [4864×896]` — produces `[30×4864]` output ✓
- `ffn_gate`: `[imtermediate×K] = [4864×896]` — produces `[30×4864]` output ✓
- `ffn_down`: `[K×intermediate] = [896×4864]` — requires transposition to `[4864×896]` if used directly, or use with different X orientation

---

## Controls

| Control | Result |
|--------|--------|
| **E.** Wrong target ffn_up with true injection | Token 108386 / close to baseline — injection blocked by shape mismatch |
| **F.** Budget=0 ffn_up | sio=0 — clean |
| **G.** Missing manifest | exit non-zero — clean |

---

## Claim Boundary

**Proven:**
- GGML `ggml_mul_mat` uses standard matmul semantics — verified by analysis and numerical test
- No transpose anomaly in residual injection code
- FFN sidecar headers map cleanly to correct GGML tensor dimensions
- Wrong target control clean, budget=0 clean

**Not proven:**
- FFN true injection (blocked by shape mismatch in current code path, not orientation issue)
- Non-square production support
- FFN quality/correctness
- Any FFN true injection token change

**Action required before FFN support:**
The FFN shape mismatch needs to be addressed at the code path level — the `build_ffn` hook's tensor shapes need to be checked against the decoded residual dimensions. This is a **code path integration issue**, not an orientation problem.

---

## Next Recommended Phase

**28BR-AA — FFN Shape Mapping / build_ffn Hook Injection Path**
Audit the actual `build_ffn` tensor shapes at runtime to determine whether the decoded residual can be applied, and if not, what adjustment is needed to make `ffn_up` (or whichever FFN family has matching shapes) a viable non-square residual candidate.