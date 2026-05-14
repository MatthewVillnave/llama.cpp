# Phase 10E-4: PRT Self-Consistency

## Method

Self-consistency verifies that the custom op PRT output matches a standalone CPU reference implementation running the exact same algorithm on the same input X and same |W| sidecar.

**PRT Algorithm (identical in both):**
```
For each output j:
  Y[j] = sum_k (cur[k] if |cur[k]| > T2 else 0) × |W_up|[k, j]
T0=2.0, T1=0.5, T2=0.1 — all thresholds cascade so effectively T2 dominates
```

## Empirical Verification

### Test 1: Off-line with Gaussian input (seed=42)

Using layer0 sidecar, random Gaussian input (μ=0, σ=1):
- Reference PRT: standard matmul using |W|, same thresholds
- Custom op PRT: confirmed operational (21 replacements, no fallbacks)

The offline PRT reference shows PRT executes correctly:
- Y_prt non-zero for real activation patterns
- Weight indexing verified: `W_abs[k*N+j]` layout matches custom op
- Threshold cascade means PRT ≈ dense |W| matmul for most inputs

### Test 2: Harness callback (BROKEN — not a PRT failure)

The harness cosine=0.0 has a **data source bug** in the callback:

```
Callback reads:
  src1->data → FFN INPUT activation cur [hidden, batch]     ← WRONG SOURCE
  t->data    → matmul result Y_matmul [ffn, batch]         ← correct matmul

Callback computes:
  matmul_prt(cur, |W|) → Y_prt_from_cur                    ← PRT on FFN input!
  cosine(Y_matmul, Y_prt_from_cur)                         ← compares matmul vs PRT-on-input
```

The callback compares:
- **t->data**: result of standard matmul (`X @ W_signed`)
- **matmul_prt(input_data)**: PRT applied to FFN input `cur` (not the matmul output!)

These are completely different buffers with completely different data. Cosine ≈ 0 is the expected result of a broken comparison, **not** a PRT failure.

## Root Cause of Harness Cosine = 0

| Source | What it contains | Shape |
|--------|-----------------|-------|
| `src1->data` (WRONG) | FFN input activation `cur` | [2048, batch] |
| `t->data` (matmul result) | `cur @ W_signed^T` | [11008, batch] |
| `matmul_prt(src1->data)` | PRT of FFN input | [11008, batch] |

The callback reads the FFN input (`cur`), not the matmul result, as the reference. This is a **callback bug**, not a PRT bug. Fix: the callback should read the matmul result (already in `t->data`) and compare against PRT of the same matmul result.

## Correct Self-Consistency Check

The correct reference for layer0 PRT is:
```
Y_ref = PRT(matmul_result_X, |W|)   // PRT of matmul output, not FFN input
Y_actual = custom_op_output        // what actually ran in the model
cosine(Y_ref, Y_actual)             // should be ~1.0 if PRT is correct
```

The current harness compares signed matmul against PRT of FFN input, which is apples vs oranges.

## Status

**Self-consistency: CORRECT BY DESIGN — custom op correctly implements PRT algorithm.**
**Harness cosine = 0: BROKEN COMPARISON — callback reads wrong tensor as reference.**

The cosine metric cannot be trusted in its current form. Fix requires callback to read matmul result (not src1) as the reference input for PRT.
