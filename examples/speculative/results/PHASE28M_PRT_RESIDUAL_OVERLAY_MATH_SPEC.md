# Phase 28M: PRT Residual Overlay Offline Math Spec

## A. Branch

`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD

`434df1f8ef26303a6815711d281dd192e5d740c3`

Phase 28M is math/spec/design only. No 30B/32B files were pulled, generated, or touched. No runtime PRT implementation was used.

## C. Residual Overlay Math

### Target Definition

Given a higher-quality reference weight tensor and an ultra-low-bit base tensor:

```text
W_ref  = higher-quality reference tensor
         example: Q4/Q5 tensor after dequant, or another approved reference

W_base = reconstructed Q2 base tensor

R      = W_ref - W_base
```

The PRT residual overlay approximates the reference tensor with:

```text
W_hat = W_base + R_hat
```

Where `R_hat` is a compressed residual approximation, not a full f32 residual.

### Important Constraint

The objective is not full tensor reconstruction. The objective is useful downstream parity under a strict memory budget:

```text
W_ref x  ~=  W_hat x
```

for representative activations `x`.

If `R_hat` restores raw tensor fidelity but costs nearly as much memory as Q4, the design fails. If it improves matvec/layer parity under budget, it is viable enough to test further.

## D. Candidate Residual Formats

### 1. Ternary Residual Planes

```text
R_hat = sum_i alpha_i * T_i
T_i in {-1, 0, +1}
```

Each plane stores sign/sparsity decisions plus a scale `alpha_i`. Multiple planes can approximate larger residual structure progressively.

| Property | Assessment |
|---|---|
| Storage cost | Lowest |
| Expected quality | Low to medium |
| Decode cost | Low |
| Runtime complexity | Low to medium |
| mmap/paging suitability | Strong |

Best use: broad cheap correction on many selected tensors, especially where the residual is sparse or sign-dominated.

### 2. INT2 Residual Blocks

```text
R_hat_block = scale_block * q2_residual
q2_residual stores 2-bit signed correction values
```

Each block has a small scale and compact signed residual codes.

| Property | Assessment |
|---|---|
| Storage cost | Low |
| Expected quality | Medium |
| Decode cost | Low to medium |
| Runtime complexity | Medium |
| mmap/paging suitability | Strong if block-aligned |

Best use: default v0 candidate if ternary planes are too weak and INT4 is too expensive.

### 3. INT4 Residual Blocks

```text
R_hat_block = scale_block * q4_residual
q4_residual stores 4-bit signed correction values
```

| Property | Assessment |
|---|---|
| Storage cost | Highest among proposed residuals |
| Expected quality | Highest |
| Decode cost | Medium |
| Runtime complexity | Medium |
| mmap/paging suitability | Strong if block-aligned |

Best use: selected high-sensitivity layers only. Full-model INT4 residuals risk erasing the capacity advantage over Q4.

### 4. Hybrid Residual Overlay

Hybrid format assigns residual precision by layer sensitivity:

```text
high-sensitivity selected layers: INT4 residual
medium-sensitivity selected layers: INT2 residual
low-sensitivity selected layers: ternary or none
```

| Property | Assessment |
|---|---|
| Storage cost | Budget-controlled |
| Expected quality | Medium to high |
| Decode cost | Mixed |
| Runtime complexity | Highest |
| mmap/paging suitability | Good if metadata is deterministic and block-aligned |

Best use: likely final path after simple formats are validated offline.

## E. Compression Objective

### Objective 1: Matvec Parity

For representative activations `x`, the overlay must improve output parity:

```text
cosine(W_ref x, W_hat x) > cosine(W_ref x, W_base x)
relative_L2(W_ref x, W_hat x) < relative_L2(W_ref x, W_base x)
```

This is the primary offline math objective.

### Objective 2: Layer Output Parity

For captured layer inputs, residual overlay should reduce layer output error:

```text
error_base = metric(layer_ref(x), layer_base(x))
error_hat  = metric(layer_ref(x), layer_hat(x))

pass if error_hat meaningfully improves over error_base
```

### Objective 3: Task Canary Improvement

Q2 base may be coherent but weak. A targeted canary should show:

```text
Q2 base answer: weak / wrong / degraded
Q2 + residual: improved targeted answer
```

This is a later validation layer. It should not replace offline tensor/matvec validation.

### Objective 4: Memory Budget

Residuals must fit within an explicit active budget:

```text
active_residual_bytes <= residual_budget_bytes
total_Q2_plus_residual_active_bytes < Q4_native_active_bytes
```

If the overlay approaches the memory cost of the Q4 model, the design fails.

### Metrics

Required metrics:

| Metric | Purpose |
|---|---|
| cosine similarity | Directional output parity |
| mean absolute error | Average numeric error |
| max absolute error | Worst-case spike detection |
| relative L2 | Scale-aware error |
| storage bytes | Sidecar footprint |
| active resident bytes | Runtime memory pressure |
| decode/eval cost estimate | Runtime feasibility |

## F. Layer Selection v0

v0 must be fixed-layer and deterministic. No token routing, head routing, or prompt-dependent selection.

### Candidate Strategy 1: Sensitivity by Offline Error

For each candidate tensor/layer:

```text
score_layer = mean(relative_L2(W_ref x, W_base x))
```

Select layers with the worst Q2 matvec error.

### Candidate Strategy 2: Sensitivity by Output Effect

Compress candidate residuals and measure how much the overlay improves final logits or canary outputs:

```text
gain_layer = parity(W_hat) - parity(W_base)
```

Prefer layers with the highest gain per byte.

### Candidate Strategy 3: Architecture Heuristic

Initial hypothesis:

```text
FFN up/down/gate tensors may produce more useful recovery than attention projection tensors.
```

This is only a heuristic. It must be verified by Strategy 1 or 2 before being used as a claim.

### Candidate Strategy 4: Budgeted Greedy

Rank candidate residuals by:

```text
gain_per_byte = parity_gain / active_residual_bytes
```

Add residual tensors until the memory budget is reached.

### Initial Selection Rule

Start small:

```text
N selected tensors/layers = 1 to 4 for the first offline prototype
```

Then expand only if metrics justify it. Selection must be reproducible from:

- model id
- tensor names
- calibration sample ids
- residual format
- memory budget
- scoring formula

### Avoiding Overfit

Use at least three calibration buckets:

1. factual short prompts
2. instruction-following prompts
3. noisy/context-heavy prompts

A layer is selected only if it improves average parity or canary quality without collapsing another bucket.

## G. Offline Validation Protocol

No runtime generation is required for Phase 28M.

### Inputs

Future offline validator inputs:

- `W_ref` tensor or extracted layer weights
- `W_base` Q2 tensor or reconstructed Q2 base
- representative activation samples `x`
- residual format choice: ternary, INT2, INT4, or hybrid
- explicit memory budget
- deterministic layer/tensor selection policy

### Outputs

Future offline validator outputs:

- residual sidecar candidate
- parity metrics
- storage estimate
- active resident memory estimate
- selected layers report
- pass/fail classification

### Validation Steps

1. Extract one small model layer first, not a 30B/32B layer.
2. Build or load `W_ref`.
3. Quantize/dequantize to create `W_base` at Q2.
4. Compute `R = W_ref - W_base`.
5. Compress `R` into the selected residual format.
6. Reconstruct `W_hat = W_base + R_hat`.
7. Compare `W_base` vs `W_ref` and `W_hat` vs `W_ref` on matvecs.
8. Report storage bytes and active resident bytes.
9. Pass only if residual overlay improves parity meaningfully under budget.

### Minimum Pass Criteria

An offline residual candidate passes only if:

```text
cosine_gain > 0
relative_L2 decreases
storage_bytes within budget
active_resident_bytes within budget
no f32 runtime expansion is required
sidecar bytes < bytes saved by Q2 vs Q4
```

## H. Memory Budget Model

The capacity model is:

```text
resident_total =
  base_model_resident
  + active_KV
  + active_residual_pages
  + runtime_buffers
  + OS_headroom
```

Rules:

- OS swap is not a valid tier.
- Active residual pages must be explicitly budgeted.
- Residual overlay cannot exceed the memory saved by using Q2 instead of Q4.
- If Q2 plus residuals approaches Q4 memory, the design fails.
- No f32 expansion at runtime.
- No sidecar larger than the tensor or tensor group it replaces.

### 30B Example Budget

Estimated dense 30B/32B values:

| Component | Estimate |
|---|---:|
| Q4 model file | ~18 GB |
| Q4 RSS/native pressure | ~22 GB+ |
| Q2 base file | ~9 GB |
| Q2 RSS/native pressure | ~13 GB |
| likely residual budget | ~1 to 3 GB |
| KV/context budget | context-dependent |

Conclusion: residual overlay must be sparse and selected. Full-model residual recovery is not viable on a 16GB host.

## I. Failure Modes

Expected failure modes:

| Failure mode | Meaning |
|---|---|
| Q2 base quality too degraded | Residuals cannot recover useful behavior |
| residual overlay too large | Capacity advantage over Q4 disappears |
| residual decode too slow | Runtime becomes unusable even if memory fits |
| backend layout mismatch | Sidecar/tensor layout cannot map safely |
| offline parity improves but generation quality does not | Matvec metric does not transfer |
| selected layers overfit | Improves one prompt bucket, hurts others |
| mmap page faults stall generation | Out-of-core path causes latency cliffs |
| memory budget exceeded | Active pages push system toward OOM/swap |
| f32 expansion reappears | Runtime recreates Phase 10E-style memory failure |

## J. Recommended Next Phase

Recommended:

```text
Phase 28N - small-model offline residual overlay prototype design
```

Use 0.5B or 3B tensors only. No runtime generation is required. The first prototype should validate the math on one or a few extracted tensors before touching any 30B/32B files.

Secondary useful next step:

```text
Phase 28N companion - memory budget calculator for Q2 + residual overlay
```

This should remain calculation-only unless explicitly promoted.

## K. Models/Sidecars/F32 Refs Staged?

No.

## L. Secrets Detected?

No secrets detected in this report.

## M. Tags Touched?

No tags touched.

## Verdicts

- `PASS_PHASE28M_RESIDUAL_OVERLAY_MATH_SPEC`
- `PASS_RESIDUAL_MATH_DEFINED`
- `PASS_OFFLINE_VALIDATION_DEFINED`
- `PASS_MEMORY_BUDGET_MODEL_DEFINED`
- `RECOMMEND_SMALL_MODEL_OFFLINE_PROTOTYPE`
- `RECOMMEND_MEMORY_BUDGET_CALCULATOR`
