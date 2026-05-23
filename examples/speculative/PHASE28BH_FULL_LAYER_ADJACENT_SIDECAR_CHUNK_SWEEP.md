# Phase 28BH: Full-Layer-Adjacent Sidecar Chunk Sweep

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## Goal
Test larger real Qwen2.5-0.5B chunks (256+, not just 70-144) with the proven shadow/pager path to expose scale-related issues.

## Result: ✅ PASS

**7/7 positive tests PASS · 5/5 controls PASS · OVERALL: PASS**

---

## Positive Test Results

| Case | Family | Layer | K×M | Blocks | Block Count | Sidecar Bytes | R_err | Y_err | Y_cosine | Pass |
|------|--------|-------|-----|--------|-------------|---------------|-------|-------|---------|------|
| ffn_up_l0_256x256 | ffn_up | 0 | 256×256 | 8×6 | 48 | 24800 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |
| ffn_gate_l0_384x256 | ffn_gate | 0 | 384×256 | 12×6 | 72 | 37184 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |
| ffn_down_l0_256x384 | ffn_down | 0 | 256×384 | 8×8 | 64 | 37152 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |
| attn_out_l0_512 | attn_output | 0 | 512×512 | 16×11 | 176 | 99040 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |
| ffn_up_l1_257x389 | ffn_up | 1 | 257×389 | 9×9 | 81 | 37846 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |
| ffn_down_l1_256x256 | ffn_down | 1 | 256×256 | 8×6 | 48 | 24800 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |
| attn_out_l5_384x256 | attn_output | 5 | 384×256 | 12×6 | 72 | 37184 | 0.00e+00 | 0.00e+00 | 1.00000000 | ✅ |

### Tensor Orientation
- **ffn_up/down/gate**: K = intermediate_size = 4864, M = hidden_dim = 896 (safetensors: [K, M])
- **attn_output**: K = num_heads × head_dim = 896, M = hidden_dim = 896 (safetensors: [K, M])

### Block Grids Covered
- **Row-heavy**: 8×6 (256×256 ffn_up/down), 9×9 (257×389 ffn_up)
- **Col-heavy**: 12×6 (384×256 ffn_gate, 384×256 attn_out)
- **Grid-heavy**: 8×8 (256×384 ffn_down)
- **Partial-edge**: 16×11 (512×512 attn_out — most blocks at 176)
- **25+ blocks**: attn_out_l0_512 (176 blocks) ✅

---

## Control Test Results

| Control | Result | Detail |
|---------|-------|--------|
| DISABLED_MODE | ✅ PASS | `pager_disabled_mode`: view correctly null when pager disabled |
| MISSING_SIDECAR | ✅ PASS | `init_correctly_rejected_missing_root`: init fails with bad root |
| BAD_TENSOR_KEY | ✅ PASS | `correctly_returned_null`: nonexistent tensor key → null view |
| BUDGET_REJECT | ✅ PASS | `correctly_rejected_tiny_budget`: 4-byte budget causes rejection |
| REPEATED_LOAD | ✅ PASS | `3x_decode_no_crash`: same chunk decoded 3× sequentially, no crash |

---

## Methodology

### Fixture Generation
- Source: Qwen2.5-0.5B-Instruct safetensors
- Python: `phase28bh_generate_fixtures.py` using `safetensors.safe_open` + `torch` for bf16→f32 conversion
- Process per case:
  1. Load tensor slice `[0:K, 0:M]` as float32
  2. Deterministic rounding: `W_base = round(W_slice / 0.25) * 0.25`
  3. Residual: `R = W_slice - W_base`
  4. Block-scales: mean-nonzero-abs per block
  5. Ternary: `sign(R)`, zeros → +1
  6. Encode `.trit` with `block_rows=32`, `block_cols=48`
  7. Write manifest + `.trit` to `/tmp/phase28bh_pkg_/<name>/`

### C++ Harness
- `phase28bh_full_layer_adjacent_chunk_sweep.cpp`
- Reference decode: `prt_trit_decoder::decode_file()` directly from `.trit`
- Pager path: `prt_init_pager()` → `activate_layer(0)` → `run_shadow_test()` → `prt_get_residual_view()`
- Shadow stats: `prt_reset_shadow_stats()` at start of each case, `prt_get_shadow_stats()` after pager operations
- Metrics: R_max_abs_err, W_max_abs_err, Y_max_abs_err, Y_cosine_sim

### Pass Criteria
- `pager_hits > 0` AND `legacy_hits == 0`
- R_max_abs_err < 1e-4
- W_max_abs_err < 1e-3
- Y_max_abs_err < 1.0 (large tolerance for large N=4-6)

---

## Files Changed

```
examples/speculative/phase28bh_full_layer_adjacent_chunk_sweep.cpp
examples/speculative/results/phase28bh_full_layer_adjacent_sidecar_chunk_sweep.json
examples/speculative/PHASE28BH_FULL_LAYER_ADJACENT_SIDECAR_CHUNK_SWEEP.md
```

---

## Build & Run

```bash
cd /home/matthew-villnave/llama.cpp
python3 /tmp/phase28bh_generate_fixtures.py

g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
    -I. -Iggml/include -Iinclude -c examples/speculative/prt_sidecar_pager.cpp \
    -o /tmp/prt_pager_28bh.o
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
    -I. -Iggml/include -Iinclude -c examples/speculative/prt_trit_decode.cpp \
    -o /tmp/prt_decode_28bh.o
g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL \
    -I. -Iggml/include -Iinclude examples/speculative/phase28bh_full_layer_adjacent_chunk_sweep.cpp \
    /tmp/prt_pager_28bh.o /tmp/prt_decode_28bh.o -o /tmp/phase28bh_chunk_sweep

/tmp/phase28bh_chunk_sweep
```

---

## Key Findings

1. **Large chunks work correctly**: 256×256, 384×256, 512×512, 257×389 — all decode identically via reference and pager paths
2. **Scale correctness**: R_err = 0.00 for all cases (perfect trit round-trip parity)
3. **Pager correctness**: pager_hits = 1, legacy_hits = 0 for every positive test (shadow path correctly exercised)
4. **Repeated load stress**: Same chunk decoded 3× sequentially without crash or memory corruption
5. **Budget rejection**: 4-byte budget correctly causes init rejection (before any file access)
6. **Edge shapes**: 257×389 (awkward prime-ish K) and 512×512 (M=512 out-of-range for ffn_up/gate/down) both pass

## Previous Phase Reference
Phase 28BG: 8/8 PASS, 4/4 controls — established coverage sweep baseline. Phase 28BH confirms results hold at larger chunk sizes (48-176 blocks vs 4-9 blocks in 28BG).