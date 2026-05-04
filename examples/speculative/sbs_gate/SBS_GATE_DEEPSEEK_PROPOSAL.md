# CPU-Native LLM Inference Acceleration: Structured FFN Block-Skip with Per-Token Gate (SBS-Gate)

---

# 1. Executive Thesis

The highest-leverage non-PRT path is **structured FFN block-skip with per-token dynamic gating**—conditionally skip contiguous blocks of FFN matrix-vector multiply based on a lightweight runtime gate that runs on attention output, reducing DRAM reads proportionally to block-sparsity rate, the only variable that actually moves the memory-bandwidth bottleneck in batch=1 CPU inference.

---

# 2. Why CPU Inference Is Slow

### 2.1 Memory Bandwidth: The Hard Wall

Consumer DDR4/DDR5 runs at 40–80 GB/s real-world. A 4-bit 7B model occupies ~3.5 GB. Batch=1 generation requires reading **every weight byte for every token**. Arithmetic intensity for the dominant GEMV operation is:

```
AI = MACs / bytes_read
 = (dim_in × dim_out) / (dim_in × dim_out × bytes_per_weight)
 ≈ 1 FLOP / 0.5 bytes
 = 2 FLOPs per byte
```

Consumer CPUs deliver ~50–200 GOPS (SIMD). At 50 GB/s bandwidth and 2 FLOPs/byte, the **compute roof is 100 GOPS effective**. The actual compute roof of the hardware is irrelevant—you are walled by DRAM. The only lever is **read fewer bytes**.

### 2.2 GEMV Shape Penalty

LLM FFN layers are GEMV, not GEMM. GEMV has no data reuse—each weight byte is touched exactly once. No tiling helps. No cache blocking helps. The output activation vector is tiny (~4096 floats) and stays in L1. The input activation is similarly small. The weight matrix is 11008×4096 = 45M elements per projection, of which 3 projections per layer, 32 layers, totaling ~65% of model parameters. **FFN is the bottleneck.**

### 2.3 KV Cache Growth

Attention KV cache grows linearly with sequence length. At 4K context, a 7B model KV cache is ~2 GB at FP16. For long contexts, KV cache reads become a secondary bandwidth consumer. However, for typical batch=1 with < 2K context, KV cache bandwidth is < 5% of total—**not the primary bottleneck**.

### 2.4 Attention Cost (Batch=1)

Self-attention at batch=1 is a series of small GEMV operations—one per head. For LLaMA 32-head, each head computes a 128×4096 GEMV for Q/K/V and a 128-dim softmax + 128×128 GEMV for the attention output. These are cached well (output vectors are tiny) and represent ~3% of total compute. **Attention is not the bottleneck for typical-length batch=1 inference.**

### 2.5 Cache Locality

Model weights are 3–4 orders of magnitude larger than L3 cache (30 MB L3 vs 3500 MB weights). No prefetcher can hide DRAM latency when the working set is this large and read sequentially exactly once. Every access misses.

### 2.6 Quantization Decode Overhead

Q4_K_M dequantization requires:
- Reading packed nibbles + scales + mins
- Unpacking nibbles to int8
- Integer dot product accumulation
- Scale application

The unpack + scale step adds ~20–30% instruction overhead over raw SIMD multiply-add. This is measurable but secondary to bandwidth.

### 2.7 Branch/Scheduler Overhead

llama.cpp's inner loop (`ggml_compute_forward_mul_mat`) is a single giant GEMV with very few branches. Branch mispredict is not the bottleneck. However, the CPU scheduler's inability to overlap compute with memory fetch (because GEMV has zero data parallelism in the outer loop) means the pipeline is always stalled on DRAM.

### 2.8 Batch=1 Limitations

With batch > 1, GEMM replaces GEMV. GEMM reuses weights across batch items, recovering arithmetic intensity. A batch of 8 increases intensity 8×—you become compute-bound, not memory-bound. But real interactive use is batch=1. **We target batch=1.**

---

# 3. Candidate Solution Directions

### 3.1 Structured FFN Block-Skip with Per-Token Gate (SBS-Gate) ← SELECTED

- **Core mechanism:** A tiny gate MLP (~200K params) runs on each layer's attention output and produces a binary mask selecting which contiguous 256-dimension FFN blocks to compute. Skipped blocks are not read from memory.
- **What it removes:** DRAM reads for skipped FFN weight blocks (up to 65% of model weights if 100% skip, realistically 40–60%).
- **Why CPU-native:** Contiguous block skip = contiguous memory skip = real DRAM savings. Small gate runs in < 5 μs on CPU. No GPU-style warp divergence.
- **Expected speedup:** 1.3–1.8× at 50% block skip rate, proportional to bandwidth saved.
- **Quality risk:** Gate must not skip blocks needed for token quality. False negatives cause degradation.
- **Implementation difficulty:** Medium. Requires gate training (or heuristic gate), kernel modification in ggml, and validation harness.
- **Falsification test:** If gate accuracy (precision of "should not skip") is below 95% for random tokens, the approach fails.

### 3.2 Integer-Only Fused GEMV with VNNI and Cache-Line-Packed Layout

- **Core mechanism:** Keep all weights in int4 (packed 2/byte), expand to int8 in-register during matmul using VNNI `VPDPBUSD`, fuse scale+shift into a single post-accumulation step. Memory layout is column-major with 256-column blocks that align with L1 cache lines.
- **What it removes:** Float dequantization overhead, half of the memory traffic (int4 vs int8), instruction-level dequant scatter.
- **Why CPU-native:** VNNI is Intel consumer-available (AVX-VNNI on Alder Lake+, AVX-512-VNNI on Zen 4+). Integer throughput is 2–4× float.
- **Expected speedup:** 1.2–1.6×, limited because you're still reading all weights.
- **Quality risk:** int4 quantization noise is slightly higher than Q4_K_M (no importance-aware compensation).
- **Implementation difficulty:** High. Requires custom VNNI kernels per matrix shape, integer calibration, careful overflow handling.
- **Falsification test:** If VNNI GEMV throughput is < 90% of theoretical peak (bytes/second), implementation overhead dominates.

### 3.3 Token-Level Speculative Decoding with On-Chip Draft Model

- **Core mechanism:** Use a tiny 10M-param draft model (fits entirely in L3 cache, never touches DRAM) to predict 3–5 tokens ahead. Verify with the full model in one pass. Accepted tokens amortize the bandwidth cost across multiple tokens.
- **What it removes:** DRAM reads per accepted token—N tokens verified in one full-model pass.
- **Why CPU-native:** Tiny model lives in cache, runs at cache bandwidth (300+ GB/s vs 50 GB/s DRAM). Verification is batched GEMV with N>1, recovering arithmetic intensity.
- **Expected speedup:** 1.5–2.5×, depending on acceptance rate.
- **Quality risk:** Draft-vs-verify divergence wastes compute. Acceptance rate must be > 60% to beat baseline.
- **Implementation difficulty:** High. Requires training a draft model per base model, implementing batched verification in llama.cpp, handling draft token feeding.
- **Falsification test:** If acceptance rate is < 2.0 tokens/batch on average for creative text generation, speedup vanishes.

### 3.4 KV-Cache Dynamic Pruning via Attention-Score Gating

- **Core mechanism:** During autoregressive generation, prune KV cache entries whose attention scores (averaged across heads and recent tokens) fall below an adaptive threshold. Reclaim memory and avoid reading pruned entries in subsequent attention computations.
- **What it removes:** KV-cache memory reads and attention computation for pruned positions.
- **Why CPU-native:** Pruning is a simple threshold on pre-computed scores. Contiguous pruning (pruning runs of low-score positions) enables memmove-based compaction.
- **Expected speedup:** 1.05–1.2× at moderate context lengths (2K–4K); more at very long contexts.
- **Quality risk:** Aggressive pruning loses long-range dependencies; "needle-in-haystack" failures.
- **Implementation difficulty:** Medium. KV cache structure modification in llama.cpp is well-understood but invasive.
- **Falsification test:** If any pruned position's score is in the top 10% of scores for any future head, pruning is too aggressive for quality.

### 3.5 Early-Exit Decoding with Layerwise Confidence Estimator

- **Core mechanism:** At each layer's output, compute a confidence score (norm of residual stream delta, or entropy of token prediction from an attached LM head). If confidence exceeds threshold, exit early and decode without computing remaining layers.
- **What it removes:** Entire transformer layers for confident ("easy") tokens.
- **Why CPU-native:** The confidence estimator is a single vector norm or a small linear projection—negligible cost. Skipping layers skips all their DRAM reads.
- **Expected speedup:** 1.1–1.4× if 30% of tokens exit at layer 20/32.
- **Quality risk:** Confidence estimators are unreliable at layer boundaries; premature exits cause token quality collapse.
- **Implementation difficulty:** Medium. Requires attaching LM heads at multiple layers, calibrating thresholds, and modifying the forward loop.
- **Falsification test:** If confidence score has correlation < 0.3 with final-token correctness, early exit is no better than random.

---

# 4. Choose the Best Direction

**Structured FFN Block-Skip with Per-Token Gate (SBS-Gate).**

It is better than the other four because:

1. **vs Integer-Only Fused GEMV (3.2):** Integer-only still reads all weights. The math is unambiguous: if you read all bytes, you're hard-capped by bandwidth. SBS-Gate reads fewer bytes and can combine with integer-only for multiplicative gains.

2. **vs Speculative Decoding (3.3):** Speculative decoding has a thermodynamic limit: you must run the full model periodically, and draft accuracy for creative/unpredictable text drops sharply. SBS-Gate provides perpetual savings on *every* token, not probabilistic batch amortization.

3. **vs KV-Cache Pruning (3.4):** KV cache is < 5% of bandwidth cost at typical context lengths. Attacking FFN (65% of cost) has 13× more headroom.

4. **vs Early-Exit (3.5):** Early-exit throws away coarse layers but computes full-width FFN on remaining layers. SBS-Gate preserves depth (all layers attend, preserving global coherence) while reducing width (FFN dimension per layer). Width reduction is more quality-preserving than depth reduction for language modeling—evidence from width-vs-depth scaling laws.

SBS-Gate attacks the dominant cost (FFN DRAM reads), preserves model depth (attention layers see full context), is additive with quantization improvements, and the gate can be bootstrapped from existing activation statistics without expensive training.

---

# 5. Full System Design

## 5.1 Architecture

```
Token → Embed → [for each layer]:
 1. RMSNorm
 2. Self-Attention (unchanged, full precision, full KV cache)
 3. RMSNorm
 4. SBS-Gate → binary block mask (43 bits for 11008-dim FFN with 256-dim blocks)
 5. FFN compute (only masked-in blocks):
  a. gate_proj[selected_blocks] × residual → silu_input
  b. up_proj[selected_blocks] × residual → up_output
  c. silu(gate_output) * up_output → gated_activation
  d. down_proj[:, selected_blocks] × gated_activation → output
 6. Residual add
→ Output LM head
```

Key: step 4 runs on the layer's attention output (the residual stream). It predicts which FFN blocks will produce near-zero activations and skips them entirely.

## 5.2 Gate Architecture

```
SBS-Gate:
 Input: residual vector x ∈ R^{4096}
 Linear: W_gate ∈ R^{4096 × 43} (43 = ceil(11008 / 256))
 Bias: b_gate ∈ R^{43}
 Sigmoid threshold at 0.5 → binary mask m ∈ {0,1}^{43}
```

Total gate parameters per layer: 4096 × 43 + 43 = 176,171 ≈ 0.18M params.
Across 32 layers: 5.6M params = 0.08% of 7B model.

Gate cost: one 4096×43 GEMV per layer. At 50 GB/s, this reads 4096×43=176K bytes = 3.5 μs. Negligible.

## 5.3 Data Structures

```c
// Per-layer block-skip metadata (added to llama_layer)
struct llama_ffn_blockskip {
 int n_blocks; // 43
 int block_size; // 256
 uint64_t active_mask; // 43-bit bitmap

 // Gate weights (quantized int8)
 int8_t *gate_weight; // [n_embd × n_blocks] col-major, int8
 float *gate_bias; // [n_blocks] float
 float gate_threshold; // scalar, default 0.5

 // FFN weights restructured in block-major order
 // Instead of [n_ffn × n_embd], stored as [n_blocks × block_size × n_embd]
 // where blocks not in active_mask are simply not read
 block_q4_0_t *gate_proj_blocks; // [n_blocks] pointers or offset array
 block_q4_0_t *up_proj_blocks;
 block_q4_0_t *down_proj_blocks; // down is [n_embd × n_ffn], blocked on n_ffn dim
};

// Global block-skip config
struct llama_blockskip_config {
 bool enabled;
 float base_threshold; // gate threshold
 float threshold_adapt; // adaptive threshold adjustment rate
 int min_active_blocks; // safety floor, default 8
 bool log_counters; // emit skip statistics
};
```

## 5.4 Memory Layout

**Before (standard):**
```
gate_proj: [n_ffn=11008 × n_embd=4096] — 45M weights, 22.5 MB at Q4_0
 up_proj: [n_ffn=11008 × n_embd=4096] — same
down_proj: [n_embd=4096 × n_ffn=11008] — same
```
All contiguous, read sequentially per GEMV.

**After (block-major with skip capability):**
```
gate_proj_blocks[0]: [256 × 4096] — block 0
gate_proj_blocks[1]: [256 × 4096] — block 1
...
gate_proj_blocks[42]: [256 × 4096] — block 42 (last, dims=256)
```

Each block is contiguous in memory and independently addressable. The kernel iterates over blocks, checks `active_mask`, and skips the block entirely (no memory read, no compute) if the bit is 0.

Down-projection is column-blocked: `down_proj_blocks[i] = [4096 × 256]` so we can skip columns corresponding to inactive FFN activation dimensions.

## 5.5 Kernel Path (C pseudocode)

```c
void ggml_compute_forward_mul_mat_ffn_blockskip(
 const struct ggml_tensor *src0, // weights [n_blocks][block_size][n_embd]
 const struct ggml_tensor *src1, // input vector [n_embd]
 struct ggml_tensor *dst, // output vector [n_ffn] or [n_embd]
 uint64_t active_mask,
 int block_size,
 int n_blocks
) {
 const int n_embd = src1->ne[0];
 float *output = dst->data;
 memset(output, 0, n_blocks * block_size * sizeof(float));

 // For gate_proj and up_proj (output is [n_ffn], block-major)
 for (int b = 0; b < n_blocks; b++) {
 if (!(active_mask & (1ULL << b))) continue;

 // This block IS active — read weights from DRAM
 const block_q4_0_t *block_w = &((block_q4_0_t*)src0->data)[b * block_size * n_embd / QK4_0];
 float *block_out = &output[b * block_size];

 // Standard Q4_0 GEMV for this block (vectorized, unrolled)
 #if defined(__AVX2__)
 // ... AVX2 Q4_0 GEMV for 256 rows x n_embd cols ...
 #elif defined(__ARM_NEON)
 // ... NEON Q4_0 GEMV ...
 #endif
 }

 // For down_proj (output is [n_embd], column-blocked)
 // Iterate over FFN dimensions, skip inactive columns
 // ...
}
```

Key property: the loop body for a skipped block is a single `if` + `continue`. No memory access. No cache pollution. The branch is perfectly predictable (same mask for entire layer).

## 5.6 Runtime Decision Logic

**Gate execution (per layer, per token):**
```
1. Compute gate_logits = gate_weight^T × residual + gate_bias [43 floats]
2. Compute gate_probs = sigmoid(gate_logits) [43 floats]
3. active_mask = gate_probs > threshold [43 bits]
4. enforce min_active: while popcount(active_mask) < min_active_blocks:
 set lowest gate_prob above threshold to 1
5. Run FFN with active_mask
```

**Adaptive threshold (optional):**
```
if recent_tokens_quality_flag == DEGRADED:
 threshold *= 0.9 // be more conservative
elif block_skip_rate > target_skip_rate:
 threshold *= 1.05 // increase selectivity
```

## 5.7 Model Assumptions

- Model has GLU-style FFN (gate/up/down triple). Covers LLaMA, Mistral, Qwen, Phi.
- FFN intermediate dimension ≥ 2048 so block_size=256 yields ≥ 8 blocks—otherwise block_size is floor(n_ffn / 8).
- Model uses SiLU activation (GELU also works; block skip is activation-agnostic).
- Residual stream dimension is the attention/FFN hidden dimension (standard).
- Quantized weights in Q4_0, Q4_1, Q4_K_M, or similar. Block-skip works with any quantization format that supports sub-matrix addressing.

## 5.8 llama.cpp Integration Points

1. **Model loading** (`llama_model_load`): Detect SBS-Gate weights in GGUF. If present, allocate `llama_ffn_blockskip` per layer. If absent, fall back to normal FFN.

2. **Forward pass** (`llama_decode_internal` → `llama_build_graph`): After attention, insert gate node. Use gate output as mask input to modified FFN nodes.

3. **GGML graph** (`ggml_build_forward_expand`): New op `GGML_OP_MUL_MAT_BLOCK_SKIP` that takes weight tensor, activation vector, and mask tensor.

4. **Backend selection** (`ggml_compute_forward`): `mul_mat_blockskip` dispatches to AVX2, AVX-512, NEON, or scalar fallback based on CPU features and mask density.

5. **GGUF format**: New key-value pairs:
 - `llama.blockskip.enabled` (bool)
 - `llama.blockskip.block_size` (int)
 - `tensor: blk.{layer}.ffn_gate.gate_weight` (int8, [n_embd, n_blocks])
 - `tensor: blk.{layer}.ffn_gate.gate_bias` (float, [n_blocks])
 - FFN tensors restructured to block-major layout (or stored with block offset table)

## 5.9 CLI Flags

```
--blockskip Enable SBS-Gate FFN block-skip
--blockskip-threshold N Gate threshold (float, default 0.5, range 0.1-0.9)
--blockskip-min-blocks N Minimum active blocks per layer (int, default 8)
--blockskip-adaptive Enable adaptive threshold
--blockskip-log Log per-layer skip statistics to stderr
--blockskip-no-fallback Error if gate weights missing (default: fallback silently)
```

## 5.10 Fallback Behavior

If SBS-Gate weights are absent from the GGUF file:
1. Print warning: `SBS-Gate weights not found, using full FFN`
2. Load standard FFN weights (original layout)
3. Bypass gate entirely
4. Performance = baseline llama.cpp
5. Quality = baseline llama.cpp

If only some layers have gate weights:
1. Use SBS-Gate for layers with gates
2. Use full FFN for layers without
3. Mixed operation; counters report per-layer statistics

## 5.11 Validation Counters

```
struct llama_blockskip_counters {
 uint64_t tokens_processed;
 uint64_t total_blocks_available;
 uint64_t total_blocks_skipped;
 float avg_skip_rate; // rolling average

 // Per-layer statistics (for 32 layers)
 uint64_t layer_skipped_blocks[32];
 uint64_t layer_total_blocks[32];

 // Quality monitoring
 uint64_t min_block_violations; // times we hit min-blocks floor
 float gate_entropy_avg; // higher entropy = less confident gate
 float residual_norm_avg; // if residual explodes, gate may be failing

 // Debug
 uint64_t gate_compute_ns; // cumulative gate compute time
 uint64_t ffn_compute_ns; // cumulative FFN compute time
 uint64_t ffn_skipped_compute_ns; // estimated time saved
};
```

## 5.12 Logging

```
llama_blockskip: token=42 avg_skip=61.3% layer_skips=[18/43, 22/43, ...]
llama_blockskip: gate_time=4.2us ffn_time=1.83ms skipped_est=0.91ms
llama_blockskip: min_block_hit=0 gate_entropy=0.34 residual_norm=12.7
```

## 5.13 Failure Modes

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| Gate false negatives (skips needed block) | Token quality degradation, repetition, collapse | Increase `--blockskip-threshold`, increase `--blockskip-min-blocks` |
| Gate false positives (computes unnecessary block) | Skip rate lower than expected, speedup < target | Decrease threshold, retrain gate |
| Block-major layout memory fragmentation | Slower than baseline even with skipping | Store block-offset table instead of physical reordering |
| First-token latencies worse | Prefill reads all blocks regardless of mask | Disable blockskip during prefill (all blocks active for first token) |
| Min-block floor too high | Skip rate clamped, speedup limited | Reduce `--blockskip-min-blocks`, retrain gate for higher confidence |
| Numerical drift over long generations | Residual norm grows, values clip | Monitor residual norm, disable blockskip if norm exceeds threshold |
| Gate bias collapse | Gate outputs all-0 or all-1 | Gate batch-norm (included in gate architecture), threshold adaptation |

---

# 6. Minimal Prototype Plan

## Phase 1: Standalone Benchmark (Week 1)

**Goal:** Measure block-skip speedup on synthetic FFN GEMV without any model integration.

**Files to create:**
```
bench/ffn_blockskip_bench.c
bench/ffn_blockskip_bench.h
bench/CMakeLists.txt (add target)
```

**What it does:**
- Allocate synthetic FFN weights (Q4_0, 11008×4096, 11008×4096, 4096×11008) with random data
- Allocate random input vector (4096)
- Generate random active mask with configurable density (25%, 50%, 75% active)
- Time full GEMV (all blocks) vs block-skip GEMV (only masked-in blocks)
- Repeat 10,000 iterations, report min/mean/max and bytes-read

**Pass criteria:**
- Wall-time ratio (blockskip / full) ≤ active_frac + 0.05 (i.e., near-linear scaling with skip rate)
- Gate compute overhead < 10 μs (negligible)
- No segfaults, no NaN outputs

**Falsification:** If blockskip is slower than full at any skip rate → kernel-level failure.

---

## Phase 2: llama.cpp Integration Canary (Week 2–3)

**Goal:** Add SBS-Gate to one model, one layer, hardcoded active mask. Verify correctness.

**Pass criteria:**
- All 100 token outputs match within 1e-4 relative L2 error
- Forward pass completion without crash
- Per-layer timing counters populate correctly

**Falsification:** Mismatch > 1e-3 → indexing error, quantization interaction bug.

---

## Phase 3: Quality Validation (Week 3–4)

**Goal:** Train or bootstrap a heuristic gate, measure quality impact.

**Gate bootstrapping (no training needed):**
For each layer, run 1000 diverse tokens through the full model. Record per-block max abs activation. Fit logistic regression to predict active/inactive from residual stream. Produces W_gate and b_gate directly from statistics.

**Pass criteria:**
- WikiText-2 perplexity increase < 2.0
- HellaSwag accuracy drop < 3 percentage points
- No token generation collapses (repetition detection)

**Falsification:** PPL increase > 5.0 or task accuracy drop > 10%. Indicates activation sparsity is NOT structured.

---

## Phase 4: Speed Validation (Week 4–5)

**Goal:** Measure end-to-end wall-time speedup on real generation.

**Pass criteria:**
- tokens/sec increase ≥ 1.2× at batch=1
- Memory RSS increase < 5% (gate weights are tiny)
- P99 latency per token ≤ baseline P99
- Skip rate 40–60% achieved (validated from counters)

**Falsification:** Speedup < 1.1×.

---

## Phase 5: Broader Model Validation (Week 5–6)

**Goal:** Verify the approach generalizes across model sizes and architectures.

**Models to test:** Qwen2.5-1.5B, Qwen2.5-7B, LLaMA-3.1-8B, Mistral-7B, Phi-3-mini

**Pass criteria:**
- Speedup ≥ 1.15× for all models
- Quality degradation within acceptable bounds for all models
- No architecture-specific failures

**Falsification:** Speedup varies wildly by model (e.g., 1.5× for LLaMA but 1.01× for Phi).

---

# 7. Benchmark Plan

## 7.1 Benchmark Harness Design

**Tool:** Extend `llama-bench` (already in llama.cpp) with blockskip counters.

**Comparison points:**
1. `vanilla`: Standard llama.cpp with Q4_K_M, no blockskip
2. `blockskip`: SBS-Gate enabled with heuristic gate
3. `blockskip-fallback`: SBS-Gate with `--blockskip-min-blocks 43` (effectively disabled, all blocks active) — measures gate overhead alone
4. `blockskip-oracle`: SBS-Gate with perfect oracle mask (ground-truth activation sparsity) — measures maximum theoretical speedup

---

# 8. Claim Discipline

### Allowed Claims

- "SBS-Gate reduces FFN DRAM reads by X% as measured by hardware performance counters on CPU Y, resulting in a wall-time speedup of Z× for batch=1 generation."
- "On model M with heuristic gate, average skip rate is R% with perplexity increase on WikiText-2 of Δ PPL."
- "The gate adds G microseconds of latency per token per layer, representing < 0.5% of total per-token compute."
- "First-token accuracy (greedy decoding match) is preserved for P% of prompts when using SBS-Gate with threshold T."
- "SBS-Gate combined with Q4_K_M achieves S tokens/sec on hardware H, compared to B tokens/sec baseline."
- "The approach fails (speedup < 1.05×) on model architectures where FFN activation sparsity is uniformly distributed rather than block-structured."

### Forbidden Claims

- ~~"Production-ready"~~
- ~~"Universal CPU acceleration"~~
- ~~"Works for all models"~~
- ~~"Solved inference"~~
- ~~"Optimal block-skip strategy"~~
- ~~"Zero quality loss"~~
- ~~"PRT-killer" or "better than PRT"~~
- ~~"Near-linear speedup with sparsity"~~

---

# 9. The First Exact Experiment

**Smallest next test:** On Qwen2.5-1.5B, collect activation traces for 512 tokens, compute per-block max activation, and verify that ≥ 30% of blocks are below a reasonable activation threshold (e.g., 1% of max). If true, the approach has headroom. If < 10%, the approach has no headroom on this model and should be abandoned.

**Most likely reason it will fail:** Activation sparsity is not *structured* in practice—high-magnitude activations are scattered uniformly across all FFN blocks rather than clustered.

---

# 10. Final Recommendation

**Best non-PRT direction:** Structured FFN Block-Skip with Per-Token Gate (SBS-Gate).

**Why it has the highest chance of working:**
1. It attacks the first-order bottleneck (DRAM bandwidth for FFN weights) through the only mechanism that works for memory-bound kernels: reading fewer bytes.
2. Activation sparsity in GLU FFNs is a well-documented phenomenon—40–80% of activations are near-zero for typical tokens.
3. The gate can be bootstrapped from activation traces without gradient-based training, making the first experiment runnable in hours, not weeks.
4. The kernel modification is tractable—we add a conditional block loop to an existing GEMV, not a fundamentally new compute paradigm.
5. It composes additively with quantization improvements, integer kernels, and KV-cache optimizations.

**Fallback hierarchy if it fails:**
1. Dynamic block sizing (reorder FFN dims by activation co-activation)
2. Integer-only fused GEMV (VNNI kernels, 1.2–1.5×)
3. Speculative decoding (draft model, proven approach)
