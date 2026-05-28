# Phase 28BR-AL: Residual Structure / Singular Vector Analysis

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  

**Fixture:** `/tmp/phase28br_o_layer0_multifamily_trit/`  

**Goal:** Explain why all injection families produce the same 10-token override pool.

## 1. Fixture Provenance

- **Generator:** `examples/speculative/phase28br_o_reconstruct_fixture.py`
- **Format:** prt_residual_sidecar v1
- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M.gguf (Q4_K_M)
- **Residual format:** ternary
- **Block geometry:** 32 × 48

**Seeds per family:**
- `ffn_up`: seed=1990868534
- `ffn_down`: seed=1792446221
- `ffn_gate`: seed=895829913
- `attn_out`: seed=4244712744

**Shapes:**
- `ffn_up`: 4864 × 896
- `ffn_down`: 896 × 4864
- `ffn_gate`: 4864 × 896
- `attn_out`: 896 × 896

## 2. Residual Tensor Statistics

| Family | Shape | L2 Norm | Abs Sum | Mean | Std | Zero Frac |
|--------|-------|--------|---------|------|-----|-----------|
| `attn_out` | 896×896 | 633.2109 | 400956.0000 | 0.0000 | 0.7067 | 0.5006 |
| `ffn_up` | 4864×896 | 1476.3676 | 2179661.0000 | -0.0002 | 0.7072 | 0.4999 |
| `ffn_down` | 896×4864 | 1476.8757 | 2181162.0000 | -0.0001 | 0.7074 | 0.4995 |

**Row norms (top 5 per family):**
- `attn_out`: [22.3607, 22.1359, 22.0907, 22.0000, 21.9773]
- `ffn_up`: [22.3159, 22.3159, 22.2935, 22.2711, 22.2486]
- `ffn_down`: [50.6656, 50.3984, 50.3786, 50.2394, 50.2394]

**Column norms (top 5 per family):**
- `attn_out`: [22.3159, 22.2935, 22.2036, 22.1359, 22.1133]
- `ffn_up`: [50.5074, 50.2494, 50.2096, 50.1996, 50.1797]
- `ffn_down`: [22.6053, 22.3607, 22.3383, 22.3159, 22.2486]

## 3. Low-Rank / Singular Vector Analysis

**Method:** Full SVD for tensors ≤4M elements; randomized SVD (power-3) for larger.


### `attn_out` (896×896)

SVD method: `full`
Frobenius norm: **181.466690**
Effective rank @90% energy: **18**
Effective rank @99% energy: **20**

**Top-20 singular values:**
`[42.0694, 41.6861, 41.6043, 41.3485, 41.1443, 41.0933, 40.9846, 40.8586, 40.7256, 40.6163, 40.5238, 40.4342, 40.3094, 40.0989, 39.9195, 39.7975, 39.6568, 39.6109, 39.4819, 39.4368]`

**Frobenius fraction in top-k:**
- top-1: **0.053745** (5.37%)
- top-3: **0.159078** (15.91%)
- top-5: **0.262405** (26.24%)
- top-10: **0.515853** (51.59%)
- top-20: **1.000000** (100.00%)

**Cumulative explained variance:**
- top-1: **0.053745** (5.37%)
- top-3: **0.159078** (15.91%)
- top-5: **0.262405** (26.24%)
- top-10: **0.515853** (51.59%)
- top-20: **1.000000** (100.00%)

### `ffn_up` (4864×896)

SVD method: `randomized_power3`
Frobenius norm: **286.891693**
Effective rank @90% energy: **18**
Effective rank @99% energy: **20**

**Top-20 singular values:**
`[66.7554, 66.5579, 66.1242, 65.9551, 65.6080, 65.2730, 64.9831, 64.7127, 64.4425, 64.0217, 63.7445, 63.7152, 63.5465, 63.2810, 63.0214, 62.6085, 62.4906, 62.2452, 62.1595, 61.4044]`

**Frobenius fraction in top-k:**
- top-1: **0.054142** (5.41%)
- top-3: **0.161088** (16.11%)
- top-5: **0.266237** (26.62%)
- top-10: **0.520441** (52.04%)
- top-20: **1.000000** (100.00%)

**Cumulative explained variance:**
- top-1: **0.054142** (5.41%)
- top-3: **0.161088** (16.11%)
- top-5: **0.266237** (26.62%)
- top-10: **0.520441** (52.04%)
- top-20: **1.000000** (100.00%)

### `ffn_down` (896×4864)

SVD method: `randomized_power3`
Frobenius norm: **287.320404**
Effective rank @90% energy: **18**
Effective rank @99% energy: **20**

**Top-20 singular values:**
`[66.9387, 66.4038, 65.9775, 65.6464, 65.5401, 65.2600, 65.0260, 64.8968, 64.3853, 64.1936, 64.0496, 63.8363, 63.5819, 63.4817, 63.3563, 62.9512, 62.5961, 62.2851, 62.2140, 62.0040]`

**Frobenius fraction in top-k:**
- top-1: **0.054278** (5.43%)
- top-3: **0.160421** (16.04%)
- top-5: **0.264657** (26.47%)
- top-10: **0.518617** (51.86%)
- top-20: **1.000000** (100.00%)

**Cumulative explained variance:**
- top-1: **0.054278** (5.43%)
- top-3: **0.160421** (16.04%)
- top-5: **0.264657** (26.47%)
- top-10: **0.518616** (51.86%)
- top-20: **1.000000** (100.00%)

## 4. Cross-Family Singular Vector Alignment

Cosine similarity between top-k right singular vectors (rows of V^T):

| Pair | Max |Cos| | Top-1 Cos | Mean Top-5 |Cos|| Note |
|------|------|------|------|------|----|
| `attn_out_vs_ffn_up` | 0.1087 | -0.0082 | 0.0997 |  |
| `attn_out_vs_ffn_down` | N/A | N/A | N/A | incompatible column spaces (896 vs 4864) |
| `ffn_up_vs_ffn_down` | N/A | N/A | N/A | incompatible column spaces (896 vs 4864) |

**Row norm distribution correlations:**
- `attn_out_vs_ffn_up`: hist_corr=-0.0232 (truncated to 896)
- `attn_out_vs_ffn_down`: corr=0.0345
- `ffn_up_vs_ffn_down`: hist_corr=-0.0498 (truncated to 896)

**Interpretation:**
- `attn_out_vs_ffn_up`: Not aligned — independent singular structure
- `attn_out_vs_ffn_down`: INCOMPATIBLE — incompatible column spaces (896 vs 4864)
- `ffn_up_vs_ffn_down`: INCOMPATIBLE — incompatible column spaces (896 vs 4864)

## 5. Row/Column Norm Profiles


**`attn_out`** — row norm top-5: `[22.3607, 22.1359, 22.0907, 22.0000, 21.9773]`
          col norm top-5: `[22.3159, 22.2935, 22.2036, 22.1359, 22.1133]`

**`ffn_up`** — row norm top-5: `[22.3159, 22.3159, 22.2935, 22.2711, 22.2486]`
          col norm top-5: `[50.5074, 50.2494, 50.2096, 50.1996, 50.1797]`

**`ffn_down`** — row norm top-5: `[50.6656, 50.3984, 50.3786, 50.2394, 50.2394]`
          col norm top-5: `[22.6053, 22.3607, 22.3383, 22.3159, 22.2486]`

## 6. Artifact Controls (Structural)

Five structural controls created per tensor — all preserve the value distribution but destroy different structural properties:


| Control | What it destroys |
|---------|-----------------|
| `flat_shuffle` | All positional/row/col structure |
| `row_shuffle` | Row ordering (preserves row contents) |
| `col_shuffle` | Column ordering (preserves column contents) |
| `sign_random` | Sign structure (preserves magnitudes) |
| `scaled_random` | All structure, same L2 norm |

**Control statistics vs. original:**


### `attn_out`

| Control | L2 Norm | Mean | Std |
|---------|---------|------|-----|
| `flat_shuffle` | 633.2109 (+0.00%) | 0.0000 | 0.7067 |
| `row_shuffle` | 633.2109 (+0.00%) | 0.0000 | 0.7067 |
| `col_shuffle` | 633.2109 (+0.00%) | 0.0000 | 0.7067 |
| `sign_random` | 633.2109 (-0.00%) | -0.0003 | 0.7067 |
| `scaled_random` | 633.2104 (-0.00%) | 0.0011 | 0.7067 |

### `ffn_up`

| Control | L2 Norm | Mean | Std |
|---------|---------|------|-----|
| `flat_shuffle` | 1476.3676 (+0.00%) | -0.0002 | 0.7072 |
| `row_shuffle` | 1476.3676 (+0.00%) | -0.0002 | 0.7072 |
| `col_shuffle` | 1476.3676 (+0.00%) | -0.0002 | 0.7072 |
| `sign_random` | 1476.3675 (-0.00%) | -0.0000 | 0.7072 |
| `scaled_random` | 1476.3497 (-0.00%) | -0.0005 | 0.7072 |

### `ffn_down`

| Control | L2 Norm | Mean | Std |
|---------|---------|------|-----|
| `flat_shuffle` | 1476.8757 (+0.00%) | -0.0001 | 0.7074 |
| `row_shuffle` | 1476.8757 (+0.00%) | -0.0001 | 0.7074 |
| `col_shuffle` | 1476.8757 (+0.00%) | -0.0001 | 0.7074 |
| `sign_random` | 1476.8758 (+0.00%) | 0.0007 | 0.7074 |
| `scaled_random` | 1476.8580 (-0.00%) | -0.0005 | 0.7074 |

## 7. Summary & Key Findings


**Low-rank structure:**
- No family has eff_rank@90% ≤ 5 — residuals are not extremely low-rank
- `attn_out`: top-1 = 5.37% of Frobenius norm, top-3 = 15.91%
- `ffn_up`: top-1 = 5.41% of Frobenius norm, top-3 = 16.11%
- `ffn_down`: top-1 = 5.43% of Frobenius norm, top-3 = 16.04%

**Cross-family alignment:**
- No pair shows strong singular vector alignment

**Override pool hypothesis assessment:**
- All families produce the same 10-token override pool (from 28BR-AK)
- If residuals share aligned singular directions → geometry determines pool
- If residuals are low-rank with dominant top singular vector → single direction dominates
- If shuffling destroys the pool → structure matters (not just magnitude)

## 8. Runtime Canary: Shuffled Residual Test

**Config:** `--prt-sidecar-apply --prt-sidecar-apply-family ffn_down --prt-sidecar-true-injection --prt-sidecar-scale 1.0`

**Prompt:** "Hi" (short, unambiguous)

**n_predict:** 1

**Hypothesis:** If the override pool persists after shuffling → magnitude dominates (structure doesn't matter)

If the override pool disappears → structure matters


*Canary execution would require modifying the sidecar to inject shuffled values at runtime.*
*This is marked optional and guarded — skipped unless trivial to execute.*


## 9. Recommended Next Phase

**Phase 28BR-AM: Override Pool Causal Isolation**

- Run runtime canary: inject shuffled ffn_down residual for "Hi" → check if pool persists

- If pool persists with shuffled residual → magnitude mechanism (not structure)
- If pool disappears with shuffled residual → structural alignment drives pool selection
- Regardless: analyze whether the shared pool is deterministic (same tokens every run)
