# Phase 28AJ: Native C++ Layer Pager Design

## Verdict: PASS_PHASE28AJ_NATIVE_PAGER_DESIGN | PASS_HOOK_POINTS_IDENTIFIED | PASS_PAGER_API_SKETCHED | PASS_V0_STRATEGY_DEFINED | RECOMMEND_CPP_PAGER_PROBE

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`eae0f1e9d`

---

## C. Native Pager Objective

### Design Target (ultimate)
A native C++ layer/tensor pager in llama.cpp that:
1. Loads only active layer/tensor groups into memory
2. Optionally loads residual sidecar data
3. Enforces resident-memory budget (≤16GB target)
4. Supports future 30B paged inference without OS swap
5. Works with existing GGUF model loader

### Prototype Target (Phase 28AK)
- C++ standalone pager probe (no llama.cpp integration yet)
- Validates C++ file IO behavior matches Python 28AH/28AI results
- Tests real `read()` vs `mmap()` performance at C++ level
- Creates fake layer files of known sizes
- No ggml graph, no model execution

### Future Runtime Target (post-28AK)
- Native llama.cpp integration via identified hook points
- Per-layer or per-tensor-group loading via modified model loader
- Budget enforcement via evict/prefetch callbacks

---

## D. Hook Point Candidates

### Primary Hook: `llama_model_loader` — `load_all_data()`

**File:** `src/llama-model-loader.cpp:7897`
**Signature:** `bool load_all_data(llama_context & ctx, ggml_backend_buffer_type_t buft, llama_mlocks * mlock_mmaps, ...)`

**What it does:** Loads ALL model tensor data into backend buffer. Currently mmap'd or read into host buffer.

**Hook opportunity:** Could be modified to load only active tensors, with lazy loading for non-active layers. However, `load_all_data` currently assumes all tensors are loaded before returning.

### Secondary Hook: `llama_mmap` — file-backed memory mapping

**File:** `src/llama-mmap.cpp`, `src/llama-model-loader.cpp:1326-1343`
**Key function:** `llama_mmap::llama_mmap(file, prefetch, numa)`

**What it does:** Memory-maps GGUF file directly. Supports `posix_fadvise(POSIX_FADV_SEQUENTIAL)` for read-ahead hint.

**Hook opportunity:** `llama_mmap` already supports partial mapping. `unmap_fragment()` at line 616 can unmap portions of the mapped file. Could be used for eviction if paired with mmap reference counting.

### Tensor Access: `llama_model_loader::get_tensor_meta()`

**File:** `src/llama-model-loader.cpp:846-862`
**Signature:** `ggml_tensor * get_tensor_meta(const char * name) const`

**What it does:** Returns tensor metadata (shape, dtype, offset) without loading data. Hook here to redirect tensor loading to pager instead of direct mmap.

### Model Loader Entry: `llama_model::load_tensors()`

**File:** `src/llama-model.cpp:2612`
**Signature:** `bool llama_model::load_tensors(llama_model_loader & ml)`

**What it does:** Orchestrates tensor loading. `use_mmap` flag controls whether data is mmap'd or copied. At line 2620: `const bool use_mmap_buffer = true;`

**Hook opportunity:** Insert pager layer between tensor creation and data loading. Could create tensors as views into pager-managed memory.

### Existing PRT Sidecar Pattern: `llama_prt_sidecar_extract.cpp`

**File:** `examples/speculative/llama_prt_sidecar_extract.cpp`
**What it shows:** Sidecar loading is already understood in the codebase. `phase10b_shadow_test.cpp` loads sidecars from `/tmp/prt_sidecars/` and uses them in actual matmul operations.

**Hook opportunity:** Build on existing sidecar loading pattern, extend to per-layer sidecar management.

### Backend Buffer: `ggml_backend_buffer`

**File:** `ggml/src/ggml-backend.cpp`
**Hook opportunity:** If tensors can be made to reference pager-managed buffers instead of direct GGUF mmaps, eviction becomes possible without changing model loader logic.

---

## E. Pager API Sketch

```cpp
// ── Configuration ────────────────────────────────────────────────────────────

struct prt_pager_config {
    // Memory budget
    size_t max_resident_bytes;     // e.g., 8ULL * 1024 * 1024 * 1024 for 8GB budget

    // Paging behavior
    int    prefetch_distance;       // layers ahead to prefetch (default 1)
    int    window_size;             // max active layers (default 4)

    // Data source
    std::string storage_root;        // directory with layer files
    std::string residual_manifest; // path to manifest.json (optional)

    // Policy
    enum class Policy {
        BASE_ONLY,          // base weights only, no residuals
        SELECTIVE_RESIDUAL, // base + selected residual families
        ALL_RESIDUAL,       // load all residuals
        BUDGET_GREEDY,      // fill budget by score/byte
    } policy;

    // IO mode
    bool use_mmap;                  // use mmap() vs read()
    bool use_direct_io;            // use O_DIRECT if available
    bool checksum_enabled;          // verify CRC16 after load
    bool prefetch_async;           // async prefetch thread (future)
};

// ── Pager Core ───────────────────────────────────────────────────────────────

class prt_layer_pager {
public:
    explicit prt_layer_pager(const prt_pager_config & config);
    ~prt_layer_pager();

    // ── Layer operations ────────────────────────────────────────────────────

    // Load layer into resident memory (blocking)
    bool load_layer(int layer_idx);

    // Prefetch layer ahead (non-blocking hint)
    void prefetch_layer(int layer_idx);

    // Evict layer from resident memory
    void evict_layer(int layer_idx);

    // Load specific tensor from layer (for residual sidecars)
    bool load_tensor(int layer_idx, const std::string & tensor_name);

    // Load residual for a layer
    bool load_residual(int layer_idx, const std::string & tensor_name);

    // ── Budget enforcement ─────────────────────────────────────────────────

    // Enforce memory budget — evict oldest layers until under budget
    void enforce_budget();

    // Check if adding this layer would exceed budget
    bool can_add_layer(int layer_idx, size_t layer_bytes);

    // ── Stats ──────────────────────────────────────────────────────────────

    size_t resident_bytes() const;
    size_t resident_residual_bytes() const;
    int    active_layer_count() const;
    int    prefetch_hit_count() const;
    int    eviction_count() const;

    struct stats {
        size_t total_bytes_read;
        size_t total_prefetch_bytes;
        size_t peak_resident_bytes;
        int    reads;
        int    prefetches;
        int    evictions;
        int    cache_hits;
        double read_time_ms;
        double wall_time_ms;
    };
    stats get_stats() const;

    // ── Error handling ─────────────────────────────────────────────────────

    enum class error {
        FILE_NOT_FOUND,
        CHECKSUM_MISMATCH,
        BUDGET_EXCEEDED,
        MAP_FAILED,
        LOAD_FAILED,
    };
    error last_error() const;
    std::string error_string(error e) const;
};
```

---

## F. Integration Strategy Comparison

### Strategy 1 — Loader-level paging (hardest)

**Approach:** Modify `llama_model::load_tensors()` and `llama_model_loader::load_all_data()` to load only active layers.

**Pros:**
- Direct control over tensor loading
- No ggml graph changes needed

**Cons:**
- Modifies core model loader
- Breaks assumptions that all tensors are loaded before model use
- High risk of breaking existing functionality

**Verdict:** ❌ Not v0

### Strategy 2 — Backend-buffer paging (medium)

**Approach:** Create tensors as views into pager-managed buffers instead of direct GGUF mmaps. Use ggml's buffer abstraction to handle paging.

**Pros:**
- Uses existing ggml buffer abstraction
- Could work without changing model loader
- Supports eviction via buffer interface

**Cons:**
- ggml tensor lifetime may be tied to model object
- Raw pointers stored in ggml_tensor may break if backing data evicted
- Complex to get right

**Verdict:** ❌ Not v0 (too risky for first prototype)

### Strategy 3 — Sidecar-only paging first (easiest)

**Approach:** Base model loaded normally via existing loader. Only residual sidecar files are managed by external pager. Sidecars are loaded on demand and overlaid on base tensors.

**Pros:**
- Minimal llama.cpp changes
- Works with existing model loader
- Sidecar loading pattern already proven in `phase10b_shadow_test.cpp`
- First working prototype is achievable

**Cons:**
- Does NOT solve 30B base weight residency problem
- Only useful if base model already fits in memory
- For 30B Q2 (26GB), base weights still need paging

**Verdict:** ⚠️ Partial — good for residual loading, but insufficient for 30B

### Strategy 4 — External preprocessor (stepping stone)

**Approach:** Generate smaller Q2 model + residual sidecars from full model. Load via normal llama.cpp loader. External pager handles only the sidecars.

**Pros:**
- Clean separation — no llama.cpp modification needed
- Works with standard model loader
- Could produce production-ready sidecar format

**Cons:**
- Preprocessing step required
- Does not solve base weight paging
- Full model conversion needed first

**Verdict:** ⚠️ Stepping stone, not final solution

### Recommended v0 Strategy: **Sidecar-only pager + external preprocessor**

**Rationale:**
- Immediate win: residual sidecar loading is the PRT core innovation
- Proven pattern in `phase10b_shadow_test.cpp`
- No llama.cpp modification required
- Clear path to production: preprocessor generates sidecars, pager loads them
- Separate effort: base weight paging can follow once sidecar path works

**Limitation acknowledged:** For 30B Q2 at 26GB aggregate, sidecar-only paging does NOT solve the base weight residency problem. Base weights still need paging. However, the **Q2+ternary approach** addresses this: using Q2 base (smaller) + residual (high quality) achieves the same goal without needing base weight paging.

**Conclusion:** If we use Q2 base for 30B (instead of Q4), the base model itself is small enough to fit. The residual sidecar approach then adds quality recovery. This makes sidecar-only paging viable as v0.

---

## G. Fake Native Prototype Design

**File:** `examples/speculative/prt_pager_probe.cpp`

### Purpose
Validate C++ file IO/paging behavior (read vs mmap, RSS, bandwidth) before touching llama.cpp runtime. Compare against Python 28AH/28AI results.

### Behavior
1. Parse command-line args (layers, layer-mb, storage-dir, mode, policy)
2. Create fake layer files in storage directory (same sizes as 28AI: ~2.18 MB/layer)
3. Simulate layer traversal: load → prefetch → evict
4. Track: wall time, read time, RSS, resident bytes, evictions, prefetches, checksums
5. Print structured stats
6. Clean up temp files (unless --keep)

### Key Differences from Python 28AI
- Real C++ file IO (`open`/`read` vs Python `open()`)
- Real C++ mmap (via `mmap()` syscalls)
- Can use `posix_fadvise()` for cache hint control
- Direct RSS measurement via `/proc/self/status`
- Compile and link against llama.cpp build system

### Expected Outputs
- Compare C++ read() vs mmap() bandwidth (should match Python ~2500 MB/s for structured files)
- Confirm RSS stays bounded
- Validate evict/prefetch counting
- Build system integration test

---

## H. Major Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| llama.cpp assumes all tensors resident before compute | **HIGH** | Hook at load-time, not compute-time; ensure active tensors are always resident |
| ggml stores raw pointers to tensor data | **HIGH** | Use buffer views, not direct pointers; ensure evicted tensors are never referenced |
| mmap page cache hides real NVMe performance | **MEDIUM** | Use `posix_fadvise(POSIX_FADV_DONTNEED)` after load; measure cold vs warm |
| Tensor lifetime tied to model object | **HIGH** | Clear separation: pager manages backing files, model loader creates tensor views |
| Eviction unsafe if graph references tensor | **HIGH** | Only evict layers not in current compute graph; track in-flight layers |
| Prefetch thread safety | **MEDIUM** | Single-threaded prefetch initially; add mutex later |
| Format mismatch with GGUF | **LOW** | Use existing GGUF metadata parsing; don't invent new format |
| Residual decode overhead | **LOW** | Dequant is fast; overhead measurable in probe |
| C++ complexity | **MEDIUM** | Start with standalone probe, not integrated llama.cpp |
| Sidecar format instability | **LOW** | Trit format already defined in Phase 28Z-28AA |

---

## I. Recommended Next Phase

**Phase 28AK: Standalone C++ Pager Probe**

Scope:
- Implement `examples/speculative/prt_pager_probe.cpp`
- Create fake GGUF-compatible layer files matching Qwen2.5-0.5B FFN_UP sizes
- Test read vs mmap, RSS, checksum, eviction counting
- Compare results to Python 28AI (should match ~2500 MB/s for read mode)
- Commit source + report only

Why standalone first:
- Validates C++ IO behavior before llama.cpp integration
- Proves the pager concept works in C++ environment
- Builds familiarity with llama.cpp build system
- No risk of breaking existing functionality

**Alternative:** Phase 28AK could be source-audit-only if hook points are unclear from static analysis alone. But static analysis of llama-model-loader.cpp is sufficient to proceed with probe design.

---

## J. Files Created
- `examples/speculative/results/PHASE28AJ_NATIVE_LAYER_PAGER_DESIGN.md` — this report
- `examples/speculative/results/phase28aj_native_layer_pager_design.json` — structured verdict

---

## K. Safety Scan

```
No .gguf/.bin/.safetensors files staged ✅
No 30B files accessed ✅
No sidecars generated ✅
No model data staged ✅
No secrets detected ✅
No tags touched ✅
```

**Safety verdict:** CLEAN

---

## Verdict Flags
- PASS_PHASE28AJ_NATIVE_PAGER_DESIGN
- PASS_HOOK_POINTS_IDENTIFIED
- PASS_PAGER_API_SKETCHED
- PASS_V0_STRATEGY_DEFINED
- RECOMMEND_CPP_PAGER_PROBE

---

## L. Tags Touched?
NO.