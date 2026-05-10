# PRT Phase 18B — PRT-Resident Replacement Feasibility

## Verdict

**BLOCKED_FFN_UP_MINORITY_FRACTION** — FFN_UP is only ~10% of model weights. PRT-Resident Replacement cannot solve the 32B RAM wall alone.

## Context

- Phase 18A recommended PRT-Resident Replacement to break the 32B RAM wall
- Hypothesis: skip loading FFN_UP dense weights, use INT6 sidecars as primary representation
- Critical question: how much RAM does FFN_UP actually consume vs. the rest of the model?

---

## Loader / Tensor Path Findings

**GGUF Loading Path:**
1. `llama_model_load_from_file()` → `llama_model_load()` → `gguf_init_from_file()`
2. `llama_model_loader::init_mappings()` — mmaps tensor data regions from GGUF file
3. `llama_model_loader::load_data_for(ggml_tensor*)` — accesses tensor data via mmap or read path
4. All tensors are eagerly loaded into RAM via `load_all_data()` — no lazy loading

**Key structural facts about GGUF tensor loading:**
- `ggml_backend_tensor_alloc()` allocates tensor buffer
- mmap path: `cur->data = mapping->addr() + w.offs` (zero-copy, still in RAM)
- read path: `file->read_raw(cur->data, ggml_nbytes(cur))` (full copy into RAM)
- No path exists to skip a tensor's memory allocation during normal model loading

**PRT sidecar loading:**
- `llama_set_prt_sidecar_int6(layer, int8_data, scales, M, N)` registers sidecar pointer
- `g_prt_int8_data[layer]` and `g_prt_int8_scales[layer]` are set
- PRT custom op reads from these global pointers at compute time
- Native FFN_UP tensor is still loaded normally — PRT runs alongside, not instead of

**The critical blocker:** Even if we could skip loading FFN_UP into `g_prt_int8_data`, the graph builder (`build_ffn`) receives `cur` (the dense tensor) from the model loader. The compute path still references `cur->data` unless we replace `build_lora_mm` entirely. The condition `up && prt_layer && g_prt_sidecar_data[il]` means PRT only activates when a sidecar is registered — it does NOT prevent the dense tensor from being loaded.

---

## Memory Architecture Discovery

**Actual 14B model weight breakdown (Q4_K_M):**

| Component | Size | Fraction |
|-----------|------|----------|
| FFN_UP (40 layers) | 0.845 GB | **10.1%** |
| Attention weights (Q,K,V,O) | 0.94 GB | 11.2% |
| Other: embeddings, norms, rope, output | 6.58 GB | **78.7%** |
| **Total** | **8.37 GB** | 100% |

**FFN_UP is only 10% of total model weights!**

**Estimated 32B breakdown (Q4_K_M):**

| Component | Size | Fraction |
|-----------|------|----------|
| FFN_UP (64 layers) | 3.20 GB | 17% |
| Attention weights | 1.50 GB | 8% |
| Other: embeddings, norms, rope, output | 13.80 GB | **75%** |
| **Total** | **~18.50 GB** | 100% |

**The dominant RAM consumer is NOT FFN_UP — it's the "Other" category at 75-79%.**

---

## RAM Savings Analysis

**If we skip ALL FFN_UP dense loading (best case):**
- 14B savings: 0.85 GB
- 32B savings: 3.20 GB
- 32B RAM needed: ~19.3 GB (weights + KV + runtime)
- Available: ~11 GB
- **Shortfall: ~8.3 GB — still infeasible**

**If we also halve attention via aggressive quantization:**
- 32B attention savings: 0.75 GB
- Total 32B savings: ~4 GB
- 32B RAM needed: ~18.5 GB
- **Shortfall: ~7.5 GB — still infeasible**

**Conclusion: FFN_UP replacement alone cannot bridge the 32B RAM gap.**

---

## Interpretation

**Can PRT sidecars become resident replacement, not extra memory?**
Only for FFN_UP — and FFN_UP is the minority fraction. For PRT to materially reduce 32B RAM, it would need to replace attention weights or embedding tables, which it cannot do (those require different representation and compute).

**Is true RAM reduction feasible in llama.cpp?**
Only for FFN_UP tensors. The rest of the model (attention, embeddings, norms, rope) cannot be skipped or replaced by PRT sidecars without fundamental architectural changes.

**What code path must be changed next?**
1. The dominant memory consumer is attention/embeddings/rope — not FFN_UP
2. KV cache quantization is the only path that reduces the Other category
3. mmap/lazy-load for non-critical tensors would help but requires deep ggml changes
4. No single technique solves 32B — combination of FFN_UP skip + KV compression + mmap might approach viability

**Is backend/ggml integration required?**
Yes — to implement true lazy loading, mmap-based tensor access, and custom tensor backends for the dominant memory components.

---

## New Direction Required

The PRT-Resident Replacement hypothesis failed at the architectural level. FFN_UP is only 10-17% of model weight memory. The remaining 83% must be addressed through other means:

1. **KV Cache Compression** — targets memory used during generation, not weights
2. **mmap-based lazy loading** — only load tensor data when layer is active, unload after
3. **Attention weight compression** — INT4 or INT2 for attention projections
4. **Vocabulary/embedding compression** — embedding table quantization

**Phase 18B finding changes the roadmap:** PRT-Resident Replacement is a dead end for 32B RAM. The next experiment must focus on KV cache compression or mmap-based loading.

---

## Recommended Next Phase

**Phase 18C: KV Cache Compression Probe**
- Measure actual KV cache memory at ctx=256/512/1024 for 14B
- Estimate 32B KV memory requirements
- Determine if KV compression alone (INT8) brings 32B from infeasible to borderline

**Alternative Phase 18C: mmap-based layer streaming probe**
- Measure if mmap loading of layers can overlap with compute enough to be usable
- Test with 14B by streaming non-FFN_UP tensors

**Phase 18C should NOT:** Generate new 32B sidecars, attempt FFN_UP skip on 32B, claim RAM problem solved, download 32B model.

---

## Allowed Claims

- FFN_UP is ~10% of 14B weights, not the dominant memory consumer
- PRT-Resident Replacement alone cannot solve the 32B RAM wall
- The dominant RAM consumer is attention/embeddings/norms/rope at 75-79% of weights
- Phase 18B closed the PRT-Resident Replacement path for 32B

## Forbidden Claims

- RAM problem solved
- 32B will run on current hardware via PRT
- FFN_UP skipping meaningfully reduces 32B RAM
- Production readiness of any approach