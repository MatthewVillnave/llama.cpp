// Phase 28AN: Real Manifest + .trit Reader
// Extends Phase 28AM sidecar pager with:
// - Support for real Phase 28Y manifest schema (top-level + per-tensor)
// - Minimal .trit header reader
// - Manifest enum for dual-schema support

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
    std::string reason;
};

struct prt_sidecar_pager_stats {
    size_t resident_bytes = 0;
    size_t peak_resident_bytes = 0;
    size_t total_bytes_read = 0;
    size_t reads = 0;
    size_t prefetches = 0;
    size_t evictions = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t fallbacks = 0;
    size_t budget_rejects = 0;
    size_t trit_validated = 0;
    size_t trit_checksum_ok = 0;
    size_t trit_checksum_fail = 0;
};

struct prt_sidecar_pager_config {
    std::string sidecar_root;
    std::string manifest_path;
    size_t max_resident_bytes = 512 * 1024 * 1024;
    int prefetch_distance = 1;
    int window_size = 4;
    bool use_mmap = false;
    bool checksum_enabled = true;
    bool strict_budget = true;
    bool validate_trit_header = true;
    std::string policy = "all_validated";
};

// ── Manifest schemas ───────────────────────────────────────────────────────────

enum class manifest_schema {
    UNKNOWN = 0,
    SIDECAR_LEGACY,   // prt_sidecar_manifest.example.json — files[{layer,filename,bytes}]
    PHASE28Y,        // format_name/format_version/source_model + entries[{layer_index,family,...}]
};

// ── .trit header (32 bytes, little-endian) ─────────────────────────────────────

struct trit_header {
    uint32_t magic;        // 0x54524954 = "TRIT"
    uint16_t ver_major;
    uint16_t ver_minor;
    uint32_t rows;
    uint32_t cols;
    uint16_t block_rows;
    uint16_t block_cols;
    uint16_t n_scales;     // u16 max 65535
    uint32_t payload_offset;  // = HEADER_SIZE + payload_bytes
    uint32_t scale_offset;    // aligned to 4 bytes
    uint16_t checksum;
    static constexpr size_t SIZE = 32;
    static constexpr uint32_t MAGIC_VALUE = 0x54524954;
    static constexpr uint16_t VER_MAJOR = 0;
    static constexpr uint16_t VER_MINOR = 1;
};

// ── Pager ─────────────────────────────────────────────────────────────────────

class prt_sidecar_pager {
public:
    explicit prt_sidecar_pager(const prt_sidecar_pager_config & config);
    ~prt_sidecar_pager();

    bool init();
    void shutdown();

    // Layer operations
    bool activate_layer(int layer_idx);
    void prefetch_layer(int layer_idx);
    void evict_layer(int layer_idx);

    // Residual access
    prt_residual_view get_residual(int layer_idx, const std::string & tensor_family);

    // Budget
    void enforce_budget();
    bool can_add_layer(int layer_idx, size_t layer_and_residual_bytes);

    // Stats
    prt_sidecar_pager_stats get_stats() const { return stats_; }
    void reset_stats() { stats_ = prt_sidecar_pager_stats(); }

    // Manifest schema
    manifest_schema detected_schema() const { return schema_; }

    // Error
    enum class error {
        NONE = 0,
        MANIFEST_NOT_FOUND,
        MANIFEST_PARSE_ERROR,
        SIDECAR_FILE_NOT_FOUND,
        TRIT_BAD_MAGIC,
        TRIT_BAD_VERSION,
        TRIT_CHECKSUM_FAIL,
        CHECKSUM_MISMATCH,
        BUDGET_EXCEEDED,
        READ_FAILED,
    };

    error last_error() const { return last_error_; }
    const char * error_string(error e) const;

private:
    bool load_manifest_phase28y();
    bool load_manifest_legacy();
    bool load_trit_header(const std::string & path, trit_header & out);
    bool load_sidecar_data(const std::string & path, size_t offset, size_t size, uint16_t expected_crc);
    const void * find_entry(int layer_idx, const std::string & family) const;
    void evict_oldest_layer();
    uint16_t crc16(const uint8_t * data, size_t len);
    uint16_t compute_trit_crc(const uint8_t * header_30bytes);

    prt_sidecar_pager_config config_;
    error last_error_ = error::NONE;
    manifest_schema schema_ = manifest_schema::UNKNOWN;
    prt_sidecar_pager_stats stats_;

    // Per-tensor entry (unified across schemas)
    struct TensorEntry {
        int layer = -1;
        std::string family;
        std::string file;
        size_t size = 0;
        uint16_t checksum = 0;
        bool required = true;
        int rows = 0;
        int cols = 0;
    };
    std::vector<TensorEntry> entries_;

    // Schema-specific metadata
    struct {
        std::string format_name;
        std::string source_model;
        std::string base_quant;
        int layer_count = 0;
        std::vector<std::string> tensor_families;
    } manifest_meta_;

    // Layer state
    struct LayerState {
        bool is_resident = false;
        size_t resident_bytes = 0;
        std::map<std::string, prt_residual_view> residuals;
    };
    std::map<int, LayerState> layer_states_;
    int active_window_start_ = 0;

    // File cache
    struct FileCache {
        const uint8_t * data = nullptr;
        size_t size = 0;
        bool is_loaded = false;
    };
    std::map<std::string, FileCache> file_cache_;
};

#endif  // PRT_SIDECAR_PAGER_H