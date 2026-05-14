# PRT Phase 21N: 0.5B PRT-v2 Three-Path Baseline Checkpoint

## Status: PASS_05B_PRT_V2_THREE_PATH_BASELINE_CHECKPOINT

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Previous HEAD
`ee29e208f` (Phase 21M — INT6 runtime path)

## C. New HEAD
`ee29e208f` (no code changes — checkpoint only)

## D. Checkpoint File
`examples/speculative/results/PRT_PHASE21N_05B_PRT_V2_THREE_PATH_CHECKPOINT.md`

## E. JSON File
`examples/speculative/results/phase21n_05b_prt_v2_three_path_checkpoint.json`

## F. Tag Created
`PRT_PHASE21N_05B_PRT_V2_THREE_PATH_BASELINE_CHECKPOINT` at `ee29e208f` (pushed)

---

## 1. Executive Summary

PRT-v2 now has a **trusted 0.5B layer0 baseline across three paths**:

| Path | Status | offline cosine |
|------|--------|----------------|
| **f32** | ✅ Baseline | 1.0 |
| **INT8** | ✅ Regenerated | 0.99996084 |
| **INT6** | ✅ Regenerated | 0.99936563 |

All three paths run through `GGML_OP_PRT_FFN_UP`. INT8 and INT6 sidecars decode correctly into f32 W for the native ggml op. File prompts (`-f`) are required for stable validation. This is a **correctness/plumbing baseline**, not a speed baseline.

---

## 2. Frozen Technical Baseline

### Common Parameters
| Parameter | Value |
|-----------|-------|
| **Branch** | `experimental/prt-phase19a-alt-sidecar-backed` |
| **HEAD** | `ee29e208fcb98f2521966c93de3b31cc0743989c` |
| **Model** | Qwen2.5-0.5B-Instruct-Q4_K_M.gguf |
| **Selected layer** | layer 0 only |
| **PRT-v2 op** | GGML_OP_PRT_FFN_UP |
| **Prompt method** | File-based prompts (`-f`) only |
| **Native layers** | 1-23 (native routing) |
| **Canary result** | 4/4 prompts exit 0 |

### f32 Path
| Parameter | Value |
|-----------|-------|
| **Source** | `/tmp/prt_phase21f_layer0_W_f32.bin` |
| **Layout** | [K,M] row-major, native f32 |
| **output_abs_sum** | 3.842286 |

### INT8 Path
| Parameter | Value |
|-----------|-------|
| **Source** | `/tmp/prt_phase21h_u_int8_from_f32/ffn_up_layer0_prt.int8` |
| **Storage** | [K,M] row-major (raw int8 + scales) |
| **Decode formula** | `W[k,j] = int8[k*M + j] * scale[j]` |
| **Offline cosine** | 0.99996084 |
| **output_abs_sum** | 3.849652 (+0.2% vs f32) |

### INT6 Path
| Parameter | Value |
|-----------|-------|
| **Source** | `/tmp/prt_phase21h_v_int6_from_f32/ffn_up_layer0_prt.int6` |
| **Storage** | [M,K] column-major (packed 6-bit) |
| **Header** | 16 bytes: BE magic, LE M/K |
| **Decode formula** | `W[k,j] = q_flat[j*K + k] * scale[j]` |
| **Sign** | 6-bit value - 32 |
| **Offline cosine** | 0.99936563 |
| **output_abs_sum** | 3.795631 (-1.2% vs f32) |

### Three-Path Kernel Comparison
| Path | output_abs_sum | vs f32 baseline |
|------|----------------|----------------|
| f32 | 3.842286 | baseline |
| INT8 | 3.849652 | +0.2% |
| INT6 | 3.795631 | -1.2% |

---

## 3. Format Schemas

### f32 Format
- **Layout**: [K,M] row-major, direct f32 tensor
- **Storage**: raw float32 values
- **No decode needed**: W is the native f32 tensor

### INT8 Format (Phase 21H-U regenerated)
- **File structure**: `[K*M bytes int8] + [M*4 bytes float32 scales]`
- **Storage**: [K,M] row-major (k-th row = features for row k)
- **Decode formula**: `W[k,j] = int8[k*M + j] * scale[j]`
- **Phase 21H-T correction**: previous wrong formula assumed [M,K]; corrected to [K,M] row-major
- **Offline cosine**: 0.99996084

### INT6 Format (Phase 21H-V regenerated)
- **Header** (16 bytes, mixed-endian):
  - Offset 0-3: BE magic `0x50525436` ("PRT6")
  - Offset 4-7: LE version (= 1)
  - Offset 8-11: LE M (= 4864)
  - Offset 12-15: LE K (= 896)
- **Packed data**: offset 16, `ceil(M*K*6/8)` bytes
- **Scales**: at end of file, M × float32
- **Unpack**: 4 × 6-bit → 3 bytes, `q[i] = (packed & 0x3F) - 32`
- **Storage**: [M,K] column-major (`q[j*K + k]` = value at column j, row k)
- **Decode formula**: `W[k,j] = q_flat[j*K + k] * scale[j]`
- **Offline cosine**: 0.99936563

---

## 4. Corrected Claims / Lessons

### Phase 21G INT8 Semantic Proof — INVALIDATED
Phase 21G claimed semantic divergence between INT8 and f32 paths. **This was wrong** — caused by using the **wrong INT8 layout** (treating [K,M] as [M,K]). The correct INT8 formula `int8[k*M+j]*scale[j]` has cosine 0.99996.

### INT6 Header Required Mixed-EndIan Parsing
Phase 21H-V wrote a scrambled header with big-endian offset fields that don't match the physical file layout. The runtime decode correctly handles this by:
1. Reading BE magic at offset 0
2. Reading LE M/K at offsets 8 and 12
3. Placing packed data at fixed offset 16
4. Placing scales at end of file

### Inline Prompts Remain Unstable
Inline `-p` prompts with PRT-v2 trigger SIGKILL. File-based prompts (`-f`) are the only validated stable method for PRT-v2 inference across all three paths.

### Capture Limitations Persist
Output text appears only in TTY. Route/op/kernel logs confirm correct behavior, but exact/token output comparison is not yet possible via pipe capture.

---

## 5. Allowed Claims

- ✅ 0.5B layer0 PRT-v2 works through GGML_OP_PRT_FFN_UP
- ✅ f32, INT8, and INT6 paths all run 4-prompt file canary with exit 0
- ✅ INT8 offline cosine 0.99996084 (strong parity)
- ✅ INT6 offline cosine 0.99936563 (acceptable parity)
- ✅ All three paths are trusted correctness baselines
- ✅ This is a plumbing/correctness baseline only

---

## 6. Forbidden Claims

- ❌ Speedup over native path
- ❌ Production readiness
- ❌ 7B/14B support
- ❌ Multi-layer support (layer0 only validated)
- ❌ Inline prompt stability
- ❌ Exact token match (capture limited)
- ❌ AVX2 performance
- ❌ All-layer replacement
- ❌ Model-general support

---

## 7. Recommended Next: Phase 21O — 7B Layer0 PRT-v2 Sidecar Plumbing

**Recommended path**: Option B — 7B layer0 PRT-v2 sidecar plumbing

**Justification**:
- 0.5B layer0 baseline is now clean and documented
- The same disciplined ladder (f32 → INT8 → INT6) should be validated on 7B
- 7B introduces realistic quantization behavior (larger matrices, more layers)
- The file-prompt method will scale cleanly
- Key question: does the INT8/INT6 decode work at 7B scale?

**Phase 21O prerequisites**:
1. Extract 7B layer0 f32 weights from Qwen2.5-7B
2. Regenerate 7B INT8 sidecar from f32 layer0
3. Regenerate 7B INT6 sidecar from same f32 layer0
4. Run 4-prompt file canary on all three paths
5. Compare output_abs_sum and offline cosine

**Alternative**: Option A (multi-layer) is also valid but less critical — layer0 plumbing is the prerequisite for any multi-layer work.

---

## 8. Safety Scan

| Item | Status |
|------|--------|
| Model files staged | ❌ No |
| Sidecar files staged | ❌ No |
| Decoded weights staged | ❌ No |
| Binaries staged | ❌ No |
| Secrets detected | ❌ No |
| Huge logs staged | ❌ No |
| Tag created | ✅ Yes |

---

## 9. Checkpoint Metadata

| Field | Value |
|-------|-------|
| **Checkpoint file** | `examples/speculative/results/PRT_PHASE21N_05B_PRT_V2_THREE_PATH_CHECKPOINT.md` |
| **JSON file** | `examples/speculative/results/phase21n_05b_prt_v2_three_path_checkpoint.json` |
| **Tag** | `PRT_PHASE21N_05B_PRT_V2_THREE_PATH_BASELINE_CHECKPOINT` at `ee29e208f` |
| **Verdict** | `PASS_05B_PRT_V2_THREE_PATH_BASELINE_CHECKPOINT` |
| **Timestamp** | 2026-05-14T17:56:00-04:00 |
| **Models staged** | No |
| **Secrets** | No |
| **Tags altered** | No |

---

## 10. Phase History Summary

| Phase | Verdict | Key finding |
|-------|---------|-------------|
| 21H-S | PASS | f32 is [K,M], INT8 is [K,M] row-major |
| 21H-T | PASS | Corrected INT8 decode formula |
| 21H-U | PASS | INT8 offline cosine 0.99996084 |
| 21H-V | PASS | INT6 offline cosine 0.99936563 |
| 21J | PASS | File prompts stable; inline unstable |
| 21K | PASS | INT8 4/4 canary exits 0 |
| 21L | PASS | INT8 baseline checkpoint |
| 21M | PASS | INT6 runtime path works, 4/4 exits 0 |
| **21N** | **PASS** | **Three-path baseline frozen** |