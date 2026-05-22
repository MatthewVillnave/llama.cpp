// Phase 28AM: Standalone Sidecar Pager
// Sidecar-only pager — no llama.cpp, no ggml, no external dependencies.
// Manages residual sidecar files with manifest-driven activation, budget enforcement, eviction.

#ifndef PRT_SIDECAR_PAGER_H
#define PRT_SIDECAR_PAGER_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <map>
#include <vector>
#include <unordered_map>

// ── Types ─────────────────────────────────────────────────────────────────────

struct prt_residual_view {
    const uint8_t * data = nullptr;
    size_t size = 0;
    bool is_null = true;
    std::string reason;  // why is_null ("not_loaded", "evicted", "file_missing", "checksum_failed")
};

struct prt_sidecar_pager_stats {
    size_t resident_bytes = 0;
    size_t peak_resident_bytes = 0;
    size_t total_bytes_read = 0;
    size_t reads = 0;
    size_t prefetches = 0;
    size_t evictions = 0;
    size_t cache_hits = 0;    // prefetch hit (already loaded)
    size_t cache_misses = 0; // prefetch miss (loaded on demand)
    size_t fallbacks = 0;    // get_residual returned null_view
    size_t budget_rejects = 0;  // activate_layer rejected due to budget
};

struct prt_sidecar_pager_config {
    std::string sidecar_root;       // e.g., "/tmp/prt_sidecars/"
    std::string manifest_path;      // e.g., "/tmp/prt_sidecars/manifest.json"
    size_t max_resident_bytes = 512 * 1024 * 1024;  // 512MB default
    int prefetch_distance = 1;
    int window_size = 4;
    bool use_mmap = false;          // read() is faster — default false
    bool checksum_enabled = true;
    bool strict_budget = true;      // abort if budget exceeded
    std::string policy = "all_validated";
};

// ── Manifest types (simplified JSON) ───────────────────────────────────────────

struct ManifestEntry {
    int layer;
    std::string tensor_family;  // "ffn_up", "ffn_down", "attn_q", etc.
    std::string file;           // filename relative to sidecar_root
    size_t size_bytes;
    size_t offset;             // offset within file (for multi-tensor files)
    uint16_t checksum;         // CRC16
    bool present = false;
};

// ── Pager ─────────────────────────────────────────────────────────────────────

class prt_sidecar_pager {
public:
    explicit prt_sidecar_pager(const prt_sidecar_pager_config & config);
    ~prt_sidecar_pager();

    // ── Lifecycle ──────────────────────────────────────────────────────────

    // Load manifest and validate sidecar index. Returns false on error.
    bool init();

    // Free all resident data and close file handles
    void shutdown();

    // ── Layer operations ───────────────────────────────────────────────────

    // Activate layer: load selected residuals into memory, enforce budget
    // Returns false if budget exceeded (and strict_budget is true)
    bool activate_layer(int layer_idx);

    // Prefetch future layers (non-blocking hint)
    void prefetch_layer(int layer_idx);

    // Evict layer from resident memory
    void evict_layer(int layer_idx);

    // ── Residual access ──────────────────────────────────────────────────

    // Get residual view for layer + tensor family.
    // Returns null view if not loaded (caller should fallback to base).
    prt_residual_view get_residual(int layer_idx, const std::string & tensor_family);

    // ── Budget ───────────────────────────────────────────────────────────

    // Enforce memory budget — evict oldest layers until under max_resident_bytes
    void enforce_budget();

    // Check if adding this layer would exceed budget
    bool can_add_layer(int layer_idx, size_t layer_and_residual_bytes);

    // ── Stats ────────────────────────────────────────────────────────────

    prt_sidecar_pager_stats get_stats() const { return stats_; }
    void reset_stats() { stats_ = prt_sidecar_pager_stats(); }

    // ── Error ─────────────────────────────────────────────────────────────

    enum class error {
        NONE = 0,
        MANIFEST_NOT_FOUND,
        MANIFEST_PARSE_ERROR,
        SIDECAR_FILE_NOT_FOUND,
        CHECKSUM_MISMATCH,
        BUDGET_EXCEEDED,
        READ_FAILED,
    };

    error last_error() const { return last_error_; }
    const char * error_string(error e) const;

private:
    bool load_sidecar_file(const ManifestEntry & entry);
    bool load_sidecar_data(const std::string & path, size_t offset, size_t size, uint16_t expected_crc);
    const ManifestEntry * find_entry(int layer_idx, const std::string & family) const;
    void evict_oldest_layer();
    uint16_t crc16(const uint8_t * data, size_t len);

    prt_sidecar_pager_config config_;
    error last_error_ = error::NONE;
    prt_sidecar_pager_stats stats_;

    // Manifest
    std::vector<ManifestEntry> manifest_entries_;

    // Layer state
    struct LayerState {
        bool is_resident = false;
        size_t resident_bytes = 0;
        std::map<std::string, prt_residual_view> residuals;  // family → view
    };
    std::map<int, LayerState> layer_states_;  // layer_idx → state
    int active_window_start_ = 0;

    // File cache (for mmap mode, keep FDs open; for read mode, keep data)
    struct FileCache {
        const uint8_t * data = nullptr;
        size_t size = 0;
        bool is_loaded = false;
    };
    std::map<std::string, FileCache> file_cache_;
};

#endif  // PRT_SIDECAR_PAGER_H