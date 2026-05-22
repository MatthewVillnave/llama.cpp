# Phase 28AL: Sidecar Pager Integration Design

## Verdict: PASS_PHASE28AL_SIDECAR_PAGER_DESIGN | PASS_SIDECAR_ONLY_SCOPE_DEFINED | PASS_PAGER_API_DEFINED | PASS_INTEGRATION_FLOW_DEFINED | RECOMMEND_STANDALONE_SIDECAR_PAGER | RECOMMEND_RUNTIME_FLAG_LATER

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`116db3472`

---

## C. Sidecar-Only v0 Objective

### What This Controls
- Residual sidecar files (`.trit` format from Phase 28AA)
- Manifest-driven layer activation
- Budgeted residual set (only selected families/tensors loaded)
- On-demand per-layer loading instead of pre-loading all sidecars at startup

### What Remains With Normal Llama.cpp Loader
- Base model weights (Q2/Q4 GGUF loaded normally via `llama_model_loader`)
- All base tensor metadata and data
- Graph construction and execution
- KV cache management

### What This Can Prove
- Residual paging mechanics work at C++ level ✅
- Budgeted sidecar activation (not all residuals loaded, budget enforced) ✅
- Per-layer on-demand loading vs batch loading ✅
- Selected tensor-family residuals (FFN_UP/DOWN/GATE only) ✅
- Integration hook points for future sidecar overlay in compute path ✅

### What This Cannot Prove (Yet)
- Dense 30B base-weight paging (deferred — fighting ggml assumptions)
- Full 30B feasibility (requires real model testing)
- Generation quality improvement (requires runtime tests)
- Speedup (requires actual benchmarks)
- Production readiness

---

## D. Existing Sidecar Hook Points

### Primary Hook: `prt_shadow.h` — Global Sidecar Map

**File:** `examples/speculative/prt_shadow.h`

```cpp
struct SidecarLoad {
    int layer;
    size_t size;
    float * data;  // PRT sidecar: |w| per element, float32
};

static std::unordered_map<int, SidecarLoad> g_sidecars;
static bool g_sidecars_loaded = false;

// Load all sidecars at startup
bool prt_load_sidecars(const char * map_path);

// Get sidecar for layer (returns nullptr if not loaded)
const SidecarLoad * prt_get_sidecar(int layer);

// Check if sidecar exists
bool prt_has_sidecar(int layer);

// Free all sidecars
void prt_unload_sidecars();
```

**Key insight:** The existing pattern is BATCH loading — all 28 sidecars loaded at once via `prt_load_sidecars()`. This is the integration point to add PAGER behavior instead.

### Integration point in `phase10b_shadow_test.cpp`:
```cpp
// Lines 99-125: Load all 28 sidecars
// Lines 178-188: PRT matmul uses loaded sidecars
// Lines 256-275: Memory accounting
// Lines 352: Cleanup
```

### Secondary Hook: `llama_prt_sidecar_extract.cpp`

**File:** `examples/speculative/llama_prt_sidecar_extract.cpp`
**What it does:** Generates sidecar files from full GGUF model

**Key function:** Reads GGUF tensor data and writes residual sidecar files

### Tertiary Hooks:
- `llama-model-loader.cpp` — base model loading (not modified in v0)
- `llama-model.cpp` — model construction (not modified in v0)
- `ggml-backend` buffer allocation — where tensor views are created

---

## E. Recommended File Placement

### v0: `examples/speculative/prt_sidecar_pager.h` + `prt_sidecar_pager.cpp`

**Rationale:**
- Easiest experimental location — no core pollution
- Compiles standalone, no llama.cpp linker dependency for v0 probe
- Clear separation from runtime `prt_shadow.h`
- Can be moved to `src/` later when integration is proven

**Not in `tools/`:** That's for standalone utilities, not integration prototypes

**Not in `src/` yet:** Higher risk — reserve for when pager is stable

### Future (post-v0):
- `examples/speculative/prt_sidecar_pager.h` → `src/ggml/ggml-pager.h` (if integrated)
- Or keep in `examples/speculative/` permanently for experimental code

---

## F. Pager API Design

### Configuration

```cpp
struct prt_sidecar_pager_config {
    std::string sidecar_root;        // e.g., "/tmp/prt_sidecars/"
    std::string manifest_path;       // e.g., "/tmp/prt_sidecars/manifest.json"
    size_t max_resident_bytes;       // e.g., 512 * 1024 * 1024 for 512MB budget
    int prefetch_distance;           // layers ahead to prefetch (default 1)
    int window_size;                 // max active layers (default 4)
    bool use_mmap;                   // use mmap() vs read() [default: false — read() faster]
    bool checksum_enabled;            // verify CRC16 after load [default: true]
    bool strict_budget;              // abort if budget exceeded [default: true]
    std::string policy;              // "all_validated", "mlp_all", "attention_partial", "budget_greedy"
};
```

### Residual View

```cpp
// Lightweight view into residual data — no copy, just pointer + metadata
struct prt_residual_view {
    int layer_idx;                   // which layer this residual belongs to
    std::string tensor_family;      // "ffn_up", "ffn_down", "attn_q", etc.
    size_t size_bytes;              // total size
    size_t scale_offset;            // offset to scale data in file
    size_t data_offset;             // offset to packed data in file
    const void * data_ptr;          // direct pointer (valid while resident)
    bool is_null;                   // true if no residual for this layer
};
```

### Pager Class

```cpp
class prt_sidecar_pager {
public:
    // ── Lifecycle ─────────────────────────────────────────────────────────

    explicit prt_sidecar_pager(const prt_sidecar_pager_config & config);
    ~prt_sidecar_pager();

    // Load manifest and validate sidecar index
    bool init();

    // Free all resident data and close file handles
    void shutdown();

    // ── Layer operations ──────────────────────────────────────────────────

    // Activate layer: load base layer + selected residuals into memory
    // Returns false if budget exceeded (and strict_budget is true)
    bool activate_layer(int layer_idx);

    // Prefetch future layers (non-blocking hint)
    void prefetch_layer(int layer_idx);

    // Evict layer from resident memory
    void evict_layer(int layer_idx);

    // Evict all layers
    void evict_all();

    // ── Residual access ──────────────────────────────────────────────────

    // Get residual view for layer + tensor family
    // Returns null view if not loaded (caller should fallback to base)
    prt_residual_view get_residual(int layer_idx, const std::string & tensor_family);

    // ── Budget ──────────────────────────────────────────────────────────

    // Enforce memory budget — evict oldest layers until under max_resident_bytes
    void enforce_budget();

    // Check if adding a layer would exceed budget
    bool can_add_layer(int layer_idx, size_t layer_and_residual_bytes);

    // ── Stats ───────────────────────────────────────────────────────────

    struct stats {
        size_t peak_resident_bytes;
        size_t current_resident_bytes;
        size_t total_bytes_read;
        int activations;
        int prefetches;
        int evictions;
        int cache_hits;      // prefetch hit (already loaded)
        double wall_time_ms;
        double read_time_ms;
    };

    stats get_stats() const;
    void reset_stats();

    // ── Error handling ─────────────────────────────────────────────────

    enum class error {
        NONE = 0,
        MANIFEST_NOT_FOUND,
        MANIFEST_PARSE_ERROR,
        SIDECAR_FILE_NOT_FOUND,
        CHECKSUM_MISMATCH,
        BUDGET_EXCEEDED,
        MAP_FAILED,
        READ_FAILED,
    };

    error last_error() const { return last_error_; }
    const char * error_string(error e) const;

private:
    // Internal: load one sidecar file
    bool load_sidecar_file(int layer_idx, const std::string & tensor_family);

    // Internal: load manifest from disk
    bool load_manifest();

    // Internal: find manifest entry for layer + family
    const ManifestEntry * find_entry(int layer_idx, const std::string & family) const;

    // Internal: evict oldest layer from active window
    void evict_oldest_layer();

    // Internal: track resident info
    struct LayerState {
        bool is_resident = false;
        size_t resident_bytes = 0;
        std::map<std::string, prt_residual_view> residuals;  // family → view
    };

    prt_sidecar_pager_config config_;
    std::map<int, LayerState> layer_states_;  // layer_idx → state
    int active_window_start_ = 0;
    error last_error_ = error::NONE;
    stats stats_;
};
```

### Ownership/Lifetime Rules
1. `prt_sidecar_pager` owns all loaded data — caller only holds `prt_residual_view` (non-owning pointer)
2. When a layer is evicted, all its residual views become invalid
3. Caller must not retain residual views beyond layer activation scope
4. Manifest entries are read-only — no modification during pager lifetime

### Error Handling
- All public methods return `bool` — `true` = success, `false` = error (check `last_error()`)
- `CHECKSUM_MISMATCH` → log warning, optionally abort based on `strict_budget`
- `BUDGET_EXCEEDED` → abort if strict, otherwise evict and retry
- `SIDECAR_FILE_NOT_FOUND` → return null view, caller uses base fallback

### Budget Enforcement
- Checked at `activate_layer()` — if adding would exceed `max_resident_bytes`, evict oldest layers first
- If even after eviction the budget cannot be met, return `false` with `BUDGET_EXCEEDED`
- `evict_oldest_layer()` removes the layer with smallest index (FIFO eviction)

---

## G. Integration Flow

### Future Runtime Flow (v0 — disabled flag)

```
1. CLI config: --enable-prt-sidecar-pager --sidecar-root /path/to/sidecars/ --sidecar-budget-mb 512
   ↓
2. llama.cpp model loader: base model loads normally (existing path unchanged)
   ↓
3. PRT sidecar pager: init() called with config, loads manifest only
   ↓
4. Before processing each layer L:
   a. pager.activate_layer(L)
      - Load selected residual files for layer L
      - Update active window
      - Evict old layers if budget exceeded
      - Stats updated
   b. prefetch_layer(L+N) for N in [1, prefetch_distance]
   ↓
5. During PRT matmul for layer L:
   a. auto view = pager.get_residual(L, "ffn_up")
   b. if (!view.is_null):
        - use sidecar data for PRT computation
      else:
        - fallback to base path (existing behavior)
        - log: "no sidecar for L.Layer.ffn_up"
   ↓
6. Memory guard (optional): if pager.resident_bytes() > ram_budget → abort
   ↓
7. Stats emitted at end: peak_resident, total_reads, evictions, cache_hits
```

### Where Activation Occurs
- In the per-layer compute loop of PRT active path
- Hook point: `prt_active.h` or `prt_shadow.h` — add `activate_layer()` call before each layer's compute
- For v0, this is a NEW call inserted before the existing `prt_shadow()` or `prt_active()` call

### Where Residual Tensor View Is Consumed
- In `prt_shadow_matmul()` or equivalent PRT matmul function
- `prt_residual_view.data_ptr` passed to kernel as alternative input
- Fallback: if `is_null`, use existing base tensor approach

### How Fallback Works
```cpp
auto view = pager.get_residual(layer_idx, "ffn_up");
if (view.is_null) {
    // Use existing base path — no change to existing behavior
    prt_shadow_base_path(layer_idx, ...);
} else {
    // Use sidecar overlay
    prt_shadow_with_sidecar(layer_idx, view.data_ptr, ...);
}
```

### How Stats Are Emitted
- `pager.get_stats()` called at end of inference
- Output to stderr/log as structured summary:
```
=== PRT Sidecar Pager Stats ===
Peak resident: 342 MB
Current resident: 127 MB
Total bytes read: 1.2 GB
Activations: 28, Prefetches: 56, Evictions: 24
Cache hits: 14, Misses: 42
Wall time: 1.8s, Read time: 1.2s
```

---

## H. Preprocessor Workflow (Future)

### External preprocessor generates sidecar package:

```
Input:
  - Source/reference GGUF model
  - Q2 base assumption (target quantization)
  - Policy (all_validated / mlp_all / etc.)
  - Residual budget (MB)

Step 1: Load GGUF metadata
  → Identify tensor families: FFN_UP, FFN_DOWN, FFN_GATE, attn_q, attn_output

Step 2: Compute ternary residuals per selected tensor
  → For each tensor: residual = original_Q4 - Q2_base
  → Pack to .trit format (3-bit/trit, float32 scales, CRC16 header)

Step 3: Write per-layer residual files
  → /sidecar_root/layer_000.ffn_up.trit
  → /sidecar_root/layer_000.ffn_down.trit
  → etc.

Step 4: Write manifest.json
  → Index of all sidecar files with offsets, sizes, checksums

Step 5: Write validation_summary.json
  → Per-family residual statistics
  → Budget compliance check
  → Quality metrics (if reference available)

Step 6: Run manifest validator
  → prt_validate_manifest.py --manifest /sidecar_root/manifest.json
  → Verify all entries readable, checksums valid

Step 7: Run budget calculator
  → prt_residual_budget.py --manifest /sidecar_root/manifest.json --policy mlp_all
  → Verify total residual bytes < budget

Output: /sidecar_root/ sidecar package ready for runtime
```

### Package Structure
```
/sidecar_root/
  manifest.json           # index of all sidecar files
  validation_summary.json # quality/budget summary
  layer_000.ffn_up.trit  # per-layer residual files
  layer_000.ffn_down.trit
  layer_000.ffn_gate.trit
  layer_000.attn_q.trit
  layer_000.attn_output.trit
  ...
  layer_055.ffn_up.trit
```

---

## I. Staged Implementation Plan

### Phase 28AM: Standalone Sidecar Pager
- `prt_sidecar_pager.h` + `prt_sidecar_pager.cpp`
- Fake `.trit` files + fake manifest
- Full API: activate/prefetch/evict/get_view/enforce_budget
- No llama.cpp linkage
- Compile to `/tmp/prt_sidecar_pager_test`
- **Goal:** Validates pager mechanics in isolation

### Phase 28AN: Connect to Manifest + Trit Reader
- Integrate `prt_manifest_validate.py` logic into C++ (or link against Python via tiny wrapper)
- Read real `.trit` files using the `read_trit()` logic from Phase 28AA
- Validate checksums
- **Goal:** Pager handles real `.trit` format (with synthetic data)

### Phase 28AO: Runtime-Adjacent Probe
- Link `prt_sidecar_pager.cpp` against built llama.cpp library
- Write a small probe that loads fake sidecars and simulates layer activation
- Measure: RSS, read time, budget enforcement, eviction correctness
- **Goal:** Validates C++ linkage and runtime integration path

### Phase 28AP: Integration Behind Disabled Flag
- Add `--enable-prt-sidecar-pager` CLI flag (disabled by default)
- Instantiate `prt_sidecar_pager` when flag is set
- Call `activate_layer()` in existing per-layer loop (behind flag)
- No sidecar overlay in compute path yet — just pager mechanics
- **Goal:** Validates integration without changing compute semantics

### Phase 28AQ: Offline Residual Decode Parity
- Use real `.trit` files (from small model slice) to test decode parity
- Compare: residual from `.trit` vs residual computed from original model
- **Goal:** Validates `.trit` format correctness and decode path

### No generation until:
- All pager stages pass
- Budget enforcement verified
- Offline parity confirmed
- Integration tested behind flag

---

## J. Risks and Blockers

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Residual tensor lifetime** — if compute path caches pointer to residual, eviction invalidates it | **HIGH** | Strict rule: residual views only valid during `activate_layer` scope; compute consumes immediately |
| **Sidecar data format mismatch** — `.trit` decode output doesn't match what compute expects | **HIGH** | Phase 28AQ offline parity test before integration |
| **Manifest/tensor naming mismatch** — layer names in manifest don't match what PRT code expects | **HIGH** | Use same tensor family naming as `prt_shadow.h` (ffn_up, ffn_down, etc.) |
| **Pager stats/logging overhead** — per-layer activation adds latency | **LOW** | Disable stats collection in production builds |
| **Page cache hiding real behavior** — repeated runs always fast | **LOW** | Use `posix_fadvise(POSIX_FADV_DONTNEED)` after load to drop pages; measure cold vs warm |
| **Missing residual fallback masking errors** — fallback works but underlying issue hidden | **MEDIUM** | Log warning when fallback is taken; track fallback count in stats |
| **Accidental full residual load** — policy loads all families and exceeds budget | **MEDIUM** | Budget enforcement in `activate_layer()` — abort if exceeded and strict_budget=true |
| **Checksum overhead** — CRC16 validation adds per-file latency | **LOW** | Make checksum toggleable; disable for production if overhead matters |
| **Thread safety** — prefetch thread races with main activation | **MEDIUM** | Single-threaded prefetch initially; add mutex if async prefetch needed |
| **Integration with existing PRT modes** — `prt_shadow.h` already has sidecar loading | **MEDIUM** | Sidecar pager REPLACES `prt_load_sidecars()` pattern, doesn't augment it — cleaner separation |

---

## K. Recommended Next Phase

**Phase 28AM: Standalone Sidecar Pager Implementation**

Implement `prt_sidecar_pager.h/.cpp` with:
- Full API as designed above
- Fake `.trit` files (synthetic residual data) matching real `.trit` format structure
- Fake manifest.json
- Budget enforcement
- Eviction counting
- No llama.cpp linkage

**Why standalone first:**
- Validates complete pager API before touching any runtime code
- Catches API design issues early
- No risk of breaking existing functionality

---

## L. Files Created
- `examples/speculative/results/PHASE28AL_SIDECAR_PAGER_INTEGRATION_DESIGN.md` — this report
- `examples/speculative/results/phase28al_sidecar_pager_integration_design.json` — structured verdict

---

## M. Safety Scan

```
No .gguf/.bin/.safetensors/.pt/.pth files staged ✅
No 30B files accessed ✅
No sidecars generated ✅
No model data staged ✅
No secrets detected ✅
No tags touched ✅
```

**Safety verdict:** CLEAN

---

## Verdict Flags
- PASS_PHASE28AL_SIDECAR_PAGER_DESIGN
- PASS_SIDECAR_ONLY_SCOPE_DEFINED
- PASS_PAGER_API_DEFINED
- PASS_INTEGRATION_FLOW_DEFINED
- RECOMMEND_STANDALONE_SIDECAR_PAGER
- RECOMMEND_RUNTIME_FLAG_LATER

---

## N. Tags Touched?
NO.