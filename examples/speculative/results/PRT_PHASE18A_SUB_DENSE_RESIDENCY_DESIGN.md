# PRT Phase 18A — Sub-Dense Residency Design

## Verdict

**RECOMMEND_PRT_RESIDENT_REPLACEMENT** with parallel **KV_CACHE_COMPRESSION** probe

## Context

- Phase 16: Qwen2.5-14B INT6 PRT validated (40 layers, K=5120, M=13824)
- Phase 17B: 32B Q4_K_M needs ~22-24GB total, current hardware has ~11GB available — blocked by RAM, not technique
- Core problem: PRT sidecars currently run **on top of** full dense weights — RAM is additive, not substitutive
- Matt's goal: change the compute/weight representation to break the RAM wall, not brute-force GPU-shaped inference on CPU

---

## The RAM Wall

| Model | Weights (Q4_K_M) | KV @ ctx=256 | Runtime overhead | Total needed | Available | Shortfall |
|-------|-----------------|--------------|-----------------|--------------|-----------|-----------|
| 7B | ~4.4GB | ~0.5GB | ~0.5GB | ~5.4GB | ~11GB | ✅ -5.6GB |
| 14B | ~8.4GB | ~1.0GB | ~1.0GB | ~10.4GB | ~11GB | ✅ -0.6GB |
| 32B | ~18.5GB | ~2.0GB | ~2.0GB | ~22.5GB | ~11GB | ❌ -11.5GB |

**Key insight:** The 14B borderline works because we barely fit. 32B fails because dense weights alone exceed available RAM before adding any PRT overhead.

**Why sidecars alone don't solve RAM:** Current PRT loads the full dense GGUF first, then layers on INT6 sidecars. Total RAM = dense weights + KV + sidecars. For 32B: 18.5GB + 2GB + ~2GB = ~22.5GB needed vs 11GB available.

---

## Option 1 — Layer Streaming

**Concept:** Load one layer at a time from disk, keep only current + next layers in RAM.

**RAM savings:** Potentially unlimited — could handle arbitrarily large models if streaming from fast storage.

**Speed cost:** 
- Per-layer load from NVMe: ~3-5GB/layer for 32B × NVMe bandwidth (~2-3 GB/s) → ~2-3 seconds per layer
- 32 layers × 2.5s = 80 seconds just for layer loads before first token
- Even with prefetch: ~1-2 seconds per token at context depth
- For short responses (n<32): unusable. For long streams: marginal.

**Complexity:** High — requires async I/O, layer lifecycle management, KV preservation across layer loads.

**First safe experiment:** mmap probe on 14B — measure if partial model access via mmap() achieves acceptable page-fault overhead at page-cache hit rates above 80%.

**Pass/Fail criteria:** If mmap page-fault time per layer is < 500ms at 80%+ cache hit → viable; if > 2s → unusable for real-time generation.

**Why it might fail:** NVMe read latency (~100µs) is fine, but page-in for 3-5GB layer cold-start hits OOM killer or thrashes page cache. Page cache pressure during multi-layer streaming would degrade other system services.

**Rank:** 4th — too slow for 32B streaming even if technically feasible.

---

## Option 2 — Hot/Cold Residency

**Concept:** Mark certain layers as "hot" (always resident) and others as "cold" (loaded on demand, with prefetch).

**RAM savings:** Depends on hot set size. If hot = 8 layers at 3.5GB/layer = ~28GB > RAM. Need hot subset.

**Observations from Phase 16:**
- `--prt-force-native 11,15` forced native for layers 11 and 15 — these were chosen empirically
- Possibly layers with highest attention weight or FFN usage in training data
- Not all layers contribute equally to generation quality

**Complexity:** Medium — requires layer importance profiling, runtime switching, fallback path.

**First safe experiment:** Use 14B with PRT active on 20/40 layers (20 hot, 20 cold). Measure whether generation degrades below 0.995 cosine threshold for cold-layer outputs. If degradation > 5% semantic quality drop → approach invalid.

**Pass/Fail criteria:** Layer importance ranking test — generate same prompt with all layers native vs selective cold (top/bottom deciles). If cosine > 0.995 and timing improves > 20% → proceed; otherwise abandon.

**Why it might fail:** Every layer participates in every forward pass in vanilla llama.cpp — no conditional computation path exists. Adding layer-skip logic requires graph modification.

**Rank:** 3rd — needs profiling and graph modification, not immediately safe.

---

## Option 3 — PRT-Resident Replacement (Recommended)

**Concept:** Do not load FFN_UP dense weights into RAM at all for layers with INT6 sidecars. Use sidecars as the **only** representation. The GGUF loader skips or deallocates those tensor regions after sidecar installation.

**RAM savings:** 
- For 32B (64 layers): Each FFN_UP block (K=8192, M=20480 in 32B?) is the largest single tensor per layer
- Skipping FFN_UP dense load per layer saves: Q4_K_M storage per layer × 64 layers
- Estimate: 32B FFN_UP per layer ~6-8GB Q4 → skipping saves ~6-8GB total
- Combined with existing sidecar size (~2GB) → net savings: ~4-6GB per layer set
- **This could bring 32B total from 22.5GB needed down to ~14-16GB needed** → still tight but potentially fits with KV compression

**Implementation path:**
1. GGUF loader: allocate model but do not map FFN_UP tensors into CPU-accessible memory
2. Instead, create custom tensor descriptors for those regions backed by INT6 sidecar files
3. PRT callback reads from sidecar file, not from ggml_context tensor
4. Native fallback path still works for non-PRT layers

**Complexity:** Medium-High — requires modifying loader behavior, tensor descriptor lifecycle, and PRT callback paths.

**First safe experiment:** 
- Load 14B but deliberately skip loading FFN_UP tensors (don't map them into RAM)
- Use llama-prt-ffn-up-extract to extract layer0, then delete/munmap the region
- Verify tensor is still accessible via custom descriptor
- This is a toy probe with no production consequence

**Pass/Fail criteria:** 
- Can we open 14B GGUF, skip FFN_UP mapping, and still run native inference on non-FFN_UP tensors? If yes → proceed. If not → hard block.
- This tests the feasibility of **not loading** FFN_UP tensors before we test substitution.

**Why it might fail:** 
- llama.cpp's ggml_context expects all tensors to be present and mapped during init
- Skipping tensors may trigger validation errors or NULL pointer derefs
- KV cache and attention still need other tensors — FFN_UP is not the only memory consumer
- The skipped tensor regions may still be referenced in computation graph

**Key insight:** This is the only approach that can make RAM go **down** (not just tolerate being high). All other approaches add overhead. This one replaces.

**Rank:** 1st — highest leverage, targets root cause (additive RAM), replace not augment.

---

## Option 4 — Compressed Tensor Residency

**Concept:** Keep some tensors permanently in INT6/INT8 rather than decompressing to FP16/F32 at load time.

**RAM savings:** 
- FP16 FFN_UP: M×K×2 bytes
- INT6 sidecar: M×K×0.75 bytes (4:3 packing) 
- Savings per layer: ~62.5% of FFN_UP storage
- For 32B: 64 layers × ~6GB Q4 FFN_UP × 0.625 = ~240GB savings (!) but this overstates — we can't skip loading entirely

**But:** This is what we already do with sidecars. The difference is whether we decompress on load (current) or keep decompressed (overhead). Actually, INT6 sidecar IS compressed — the overhead is decompress time, not storage.

**Complexity:** Low — already done in Phase 14/15/16.

**First safe experiment:** Already done — Phase 15 validated INT6 on 7B.

**Pass/Fail criteria:** N/A — already validated.

**Why it might fail:** Decompression cost appears during generation, not load. Already accounted for in Phase 16 timing measurements.

**Rank:** 2nd (tie) — already validated, not a new path.

---

## Option 5 — KV Cache Compression

**Concept:** Quantize KV cache to INT8/FP16 after initial fill. Sliding window attention for long contexts.

**RAM savings:**
- For 14B @ ctx=256: KV cache ~1GB (FP16). Quantized to INT8: ~0.5GB. Savings: ~0.5GB.
- For 32B @ ctx=512: KV cache ~4GB (FP16). Quantized to INT8: ~2GB. Savings: ~2GB.

**Observations from Phase 16:**
- We used ctx=256 and ctx=512 successfully with 14B
- KV cache at ctx=256 is already small — further compression gives diminishing returns
- ctx=2048 for 32B would need ~16GB KV → even INT8 would need ~8GB

**Complexity:** Medium — KV cache quantization requires kernel changes in attention pathway.

**First safe experiment:** 
- Measure KV memory for 14B at ctx=256/512/1024/2048 via llama-memory-breakdown
- Estimate KV for 32B at same contexts via model size scaling
- If KV > 4GB at ctx=512 for 32B → KV compression alone won't solve RAM problem

**Pass/Fail criteria:** If KV at ctx=512 for 32B is < 4GB → KV compression helps. If > 6GB → not enough alone.

**Why it might fail:** KV cache is accessed every token — quantization introduces decode overhead that affects every generation step, not just initial load.

**Rank:** 2nd (tie) — worth measuring but not highest leverage alone.

---

## Option 6 — Predictive Prefetch / Async I/O

**Concept:** While CPU computes layer N, async-load layer N+1 into page cache. Overlap compute and I/O.

**RAM savings:** None directly — this is a latency mitigation, not a memory reduction.

**NVMe bandwidth context:** 
- OptiPlex 7010 has NVMe SSD (likely ~3-5 GB/s sequential read)
- Per-layer load time for 32B: ~3-5GB / 3.5 GB/s = ~1-1.5 seconds per layer
- Layer streaming at 32B: ~64 layers × 1.2s = 76 seconds cold start
- Even with perfect prefetch: token processing time = max(compute, I/O) → dominated by I/O

**Complexity:** High — requires async pipeline, careful scheduling, KV preservation across layer boundaries.

**First safe experiment:** Measure actual NVMe sequential read speed with `dd` and page-cache behavior for 4GB reads.

**Pass/Fail criteria:** If sequential read speed is > 1 GB/s and page-cache hit rate for repeated layer access is > 80% → prefetch helps. Otherwise → I/O dominates compute.

**Why it might fail:** The I/O time per layer likely exceeds the actual compute time for a single layer on CPU — prefetch doesn't help if compute is already faster than I/O.

**Rank:** 5th — latency improvement, not primary RAM solution.

---

## Option 7 — Hybrid Sidecar Model Format

**Concept:** Create a derivative GGUF where selected dense tensors are replaced by PRT sidecar metadata. No loading of original tensor data.

**RAM savings:** Same as Option 3 — this is the on-disk representation of the PRT-resident replacement concept.

**Complexity:** High — requires new GGUF-like format or extension. Downstream tooling needs updates.

**First safe experiment:** 
- Take Qwen2.5-0.5B (380MB, fully loadable)
- Create a variant where FFN_UP is replaced by sidecar reference only
- Verify model loads and runs with sidecar substitution

**Pass/Fail criteria:** If 0.5B hybrid runs correctly with FFN_UP sidecar substitution → proceed to 7B probe.

**Why it might fail:** Format tooling needs work. llama.cpp's GGUF loader won't natively understand "tensor is actually a sidecar reference."

**Rank:** 1st (tie with Option 3) — but needs format work first.

---

## Option 8 — Backend/ggml Residency Integration

**Concept:** Find where in ggml/llama.cpp tensors are loaded and residency decisions are made. Add lazy-load, mmap, or custom tensor backends.

**Current residency behavior:**
- `gguf_init_from_file()` maps entire GGUF into memory
- All tensors are allocated in ggml_context during init
- No on-demand loading — everything is eagerly loaded
- `ggml_view_*` operations create tensor views but don't load data

**Key functions to examine:**
- `gguf_init_from_file()` — entry point, loads all metadata + tensors
- `ggml_used_mem()` — memory accounting
- Custom tensor operations (Phase 10E work) — where sidecar backends could be registered

**Complexity:** High — requires deep ggml knowledge and API changes.

**First safe experiment:**
- Audit `gguf_init_from_file()` and `ggml_init()` — identify where tensors are allocated
- Add instrumentation to count bytes allocated per tensor type
- Build a memory map of 14B GGUF by tensor class (attention weights, FFN weights, embeddings, norms)

**Pass/Fail criteria:** A clear memory map showing which tensor classes dominate RAM. FFN_UP is typically 60-70% of model weights. If we can replace FFN_UP → reduce model RAM by ~60%.

**Why it might fail:** Even lazy loading requires changes to how tensors are accessed in the compute graph. Custom backends need to be integrated at ggml level.

**Rank:** 4th (tie) — foundational work, long lead time.

---

## Ranking Table

| Option | RAM Savings | Speed Risk | Complexity | 32B Relevance | First Exp Cost | Rank |
|--------|-------------|------------|------------|---------------|----------------|------|
| 3. PRT-Resident Replacement | ~4-6GB (high) | Medium | Med-High | Direct | Low (skip load probe) | **1st** |
| 7. Hybrid Format | ~4-6GB (high) | Medium | High | Direct | Medium (0.5B toy) | **2nd** |
| 5. KV Compression | ~0.5-2GB (low) | Low | Medium | Helpful | Low (memory probe) | **3rd** |
| 2. Hot/Cold Residency | ~1-3GB (medium) | High | Med-High | Partial | Medium (profiling) | **4th** |
| 4. Compressed Tensor | ~0GB (done) | Low | Low | Already validated | None | 5th |
| 8. Backend/ggml Integration | TBD (audit) | Med-High | High | Foundational | Low (audit) | 5th |
| 6. Predictive Prefetch | ~0GB | Low | High | Mitigates I/O | Medium (benchmark) | 6th |
| 1. Layer Streaming | Unlimited (theoretical) | Very High | High | Yes, but too slow | Medium (mmap probe) | 7th |

---

## Recommended Phase 18B

**Experiment: PRT-Resident Replacement Feasibility Probe**

Goal: Determine whether we can skip loading FFN_UP tensors from GGUF without breaking the model.

**Specific test (14B):**
1. Load 14B GGUF normally via `gguf_init_from_file()`
2. Inspect memory usage (how much is FFN_UP vs other tensors)
3. Attempt to selectively skip FFN_UP tensor initialization — test if tensor descriptor can point to sidecar instead of GGUF region
4. Run generation with partial model (non-FFN_UP tensors only) to confirm base model integrity
5. If integrity confirmed → test PRT sidecar replacement on layer0 only

**What this answers:** Can we make RAM go DOWN (replace) rather than stay flat or go up (add)? If yes, 32B becomes tractable.

**Why not 32B:** Too expensive to download. 14B tells us what we need to know — if the approach works at 14B, it applies to 32B with same ratio.

**Alternatives if this fails:** 
- Fall back to KV compression probe (Phase 18C)
- Or audit ggml init path for lazy-load opportunities (Phase 18D)

---

## What Not To Do Next

- Do **not** download 32B model on this hardware
- Do **not** generate full 32B sidecar set
- Do **not** claim PRT solves the RAM wall yet
- Do **not** make speedup promises — we haven't measured compute overhead of replacement yet
- Do **not** rewrite llama.cpp wholesale — incremental changes only
- Do **not** mix Phase 18 work with public claims or benchmark postings
- Do **not** run long-context tests on 32B without residency strategy confirmed

---

## Allowed Claims

- Phase 18A is a design/feasibility document
- RAM wall (not technique) is the primary blocker for 32B on CPU
- PRT-resident replacement is the highest-leverage hypothesis
- KV compression and layer streaming are secondary options
- 14B is the maximum reliably feasible model size on current hardware

## Forbidden Claims

- RAM problem is solved
- 32B will definitely run on this hardware
- Production readiness of any approach
- Universal speedup from any technique
- GPU comparison or parity claims
- Larger-than-14B support until proven