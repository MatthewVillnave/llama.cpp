// Phase 28AN: Sidecar Pager Implementation
// Supports Phase 28Y real manifest + legacy sidecar manifest schema
// Minimal .trit header reader (no full decode)

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>
#include <map>

#include "prt_sidecar_pager.h"

// ── CRC16 ─────────────────────────────────────────────────────────────────────

static uint16_t crc16_table[256] = {0};
static bool crc16_initialized = false;

static void init_crc16_once() {
    if (crc16_initialized) return;
    for (int i = 0; i < 256; i++) {
        uint16_t crc = (uint16_t)i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) crc = (crc >> 1) ^ 0xA001;
            else crc >>= 1;
        }
        crc16_table[i] = crc;
    }
    crc16_initialized = true;
}

static uint16_t crc16_update(uint16_t crc, const uint8_t * data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 1) ^ crc16_table[data[i]];
    }
    return crc;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

prt_sidecar_pager::prt_sidecar_pager(const prt_sidecar_pager_config & config)
    : config_(config), last_error_(error::NONE) {
    init_crc16_once();
}

prt_sidecar_pager::~prt_sidecar_pager() {
    shutdown();
}

// ── Manifest detection ────────────────────────────────────────────────────────

bool prt_sidecar_pager::init() {
    FILE * mf = fopen(config_.manifest_path.c_str(), "r");
    if (!mf) {
        last_error_ = error::MANIFEST_NOT_FOUND;
        return false;
    }
    fseek(mf, 0, SEEK_END);
    long fsize = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    std::vector<char> buf(fsize + 1);
    fread(buf.data(), 1, fsize, mf);
    buf[fsize] = '\0';
    fclose(mf);

    bool has_format_version = (strstr(buf.data(), "\"format_version\"") != nullptr);
    bool has_files_array = (strstr(buf.data(), "\"files\"") != nullptr);

    if (has_format_version) {
        schema_ = manifest_schema::PHASE28Y;
        return load_manifest_phase28y();
    } else if (has_files_array) {
        schema_ = manifest_schema::SIDECAR_LEGACY;
        return load_manifest_legacy();
    } else {
        last_error_ = error::MANIFEST_PARSE_ERROR;
        return false;
    }
}

// ── Manifest helpers ───────────────────────────────────────────────────────────

static int json_find_int(const char * buf, const char * key) {
    const char * p = strstr(buf, key);
    if (!p) return -1;
    p += strlen(key);
    while (*p == ' ' || *p == ':' || *p == '"' || *p == '\n' || *p == '\r') p++;
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    return val;
}

static std::string json_find_str(const char * buf, const char * key) {
    const char * p = strstr(buf, key);
    if (!p) return "";
    p += strlen(key);
    while (*p == ' ' || *p == ':' || *p == '"' || *p == '\n' || *p == '\r') p++;
    if (*p == '"') p++;
    const char * end = p;
    while (*end && *end != '"') end++;
    return std::string(p, end - p);
}

// ── Phase 28Y manifest parser ──────────────────────────────────────────────────

bool prt_sidecar_pager::load_manifest_phase28y() {
    FILE * mf = fopen(config_.manifest_path.c_str(), "r");
    if (!mf) return false;
    fseek(mf, 0, SEEK_END);
    long fsize = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    std::vector<char> buf(fsize + 1);
    fread(buf.data(), 1, fsize, mf);
    buf[fsize] = '\0';
    fclose(mf);

    manifest_meta_.format_name = json_find_str(buf.data(), "\"format_name\"");
    manifest_meta_.source_model = json_find_str(buf.data(), "\"source_model\"");
    manifest_meta_.base_quant = json_find_str(buf.data(), "\"base_quant\"");
    manifest_meta_.layer_count = json_find_int(buf.data(), "\"layer_count\"");

    const char * entries_start = strstr(buf.data(), "\"entries\"");
    if (!entries_start) {
        // Try "files" array in legacy-like format with top-level fields
        return load_manifest_legacy();
    }

    const char * p = entries_start;
    while (true) {
        const char * pi = strstr(p, "\"layer_index\"");
        const char * pl = strstr(p, "\"layer_id\"");
        if (!pi && !pl) break;
        if (pi && (!pl || pi <= pl)) { p = pi; p += 14; }
        else { p = pl; p += 11; }
        while (*p == ' ' || *p == ':' || *p == '\n' || *p == '\r') p++;
        int layer = atoi(p);

        std::string tensor_name = json_find_str(p, "\"tensor_name\"");
        std::string tensor_family = json_find_str(p, "\"tensor_family\"");
        if (tensor_family.empty()) tensor_family = tensor_name;

        std::string file_path = json_find_str(p, "\"file_path\"");

        size_t byte_size = 0;
        const char * bs = strstr(p, "\"byte_size\"");
        if (bs) {
            bs += 11;
            while (*bs == ' ' || *bs == ':' || *bs == '\n' || *bs == '\r') bs++;
            byte_size = (size_t)atoll(bs);
        }

        int rows = json_find_int(p, "\"rows\"");
        int cols = json_find_int(p, "\"cols\"");

        bool required = true;
        const char * status_p = strstr(p, "\"status\"");
        if (status_p) {
            std::string status_val = json_find_str(status_p, "\"status\"");
            required = (status_val != "missing" && status_val != "skipped");
        }

        if (!file_path.empty()) {
            TensorEntry e;
            e.layer = layer;
            e.family = tensor_family;
            e.file = file_path;
            e.size = byte_size;
            e.rows = rows;
            e.cols = cols;
            e.required = required;
            e.checksum = 0xFFFF;
            entries_.push_back(e);
        }

        p++;  // advance to avoid infinite loop
    }

    return entries_.size() > 0;
}

// ── Legacy manifest parser ────────────────────────────────────────────────────

bool prt_sidecar_pager::load_manifest_legacy() {
    FILE * mf = fopen(config_.manifest_path.c_str(), "r");
    if (!mf) return false;
    fseek(mf, 0, SEEK_END);
    long fsize = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    std::vector<char> buf(fsize + 1);
    fread(buf.data(), 1, fsize, mf);
    buf[fsize] = '\0';
    fclose(mf);

    const char * p = buf.data();

    while ((p = strstr(p, "\"layer\":")) != nullptr) {
        p += 8;
        while (*p == ' ' || *p == ':' || *p == '\n' || *p == '\r') p++;
        int layer = atoi(p);

        const char * fn_p = strstr(p, "\"filename\"");
        if (!fn_p) break;
        fn_p += 11;
        while (*fn_p == ' ' || *fn_p == ':' || *fn_p == '"' || *fn_p == '\n' || *fn_p == '\r') fn_p++;
        if (*fn_p == '"') fn_p++;
        const char * fn_end = fn_p;
        while (*fn_end && *fn_end != '"') fn_end++;
        std::string filename(fn_p, fn_end - fn_p);

        const char * bytes_p = strstr(fn_end, "\"bytes\"");
        size_t bytes = 0;
        if (bytes_p) {
            bytes_p += 7;
            while (*bytes_p == ' ' || *bytes_p == ':' || *bytes_p == '\n' || *bytes_p == '\r') bytes_p++;
            bytes = (size_t)atoll(bytes_p);
        }

        TensorEntry e;
        e.layer = layer;
        e.family = "ffn_up";
        e.file = filename;
        e.size = bytes;
        e.required = true;
        e.checksum = 0xFFFF;
        entries_.push_back(e);

        p = fn_end;
    }

    return entries_.size() > 0;
}

// ── .trit header reader ──────────────────────────────────────────────────────

bool prt_sidecar_pager::load_trit_header(const std::string & path, trit_header & out) {
    uint8_t header[trit_header::SIZE];
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;

    size_t n = fread(header, 1, trit_header::SIZE, f);
    fclose(f);

    if (n < trit_header::SIZE) return false;

    out.magic = *(uint32_t *)(header + 0);
    out.ver_major = *(uint16_t *)(header + 4);
    out.ver_minor = *(uint16_t *)(header + 6);
    out.rows = *(uint32_t *)(header + 8);
    out.cols = *(uint32_t *)(header + 12);
    out.block_rows = *(uint16_t *)(header + 16);
    out.block_cols = *(uint16_t *)(header + 18);
    out.n_scales = *(uint16_t *)(header + 20);
    out.payload_offset = *(uint32_t *)(header + 22);
    out.scale_offset = *(uint32_t *)(header + 26);
    out.checksum = *(uint16_t *)(header + 30);

    if (out.magic != trit_header::MAGIC_VALUE) {
        last_error_ = error::TRIT_BAD_MAGIC;
        return false;
    }

    if (out.ver_major != trit_header::VER_MAJOR || out.ver_minor != trit_header::VER_MINOR) {
        last_error_ = error::TRIT_BAD_VERSION;
        return false;
    }

    uint16_t computed = compute_trit_crc(header);
    if (computed != out.checksum) {
        last_error_ = error::TRIT_CHECKSUM_FAIL;
        stats_.trit_checksum_fail++;
        return false;
    }

    stats_.trit_checksum_ok++;
    return true;
}

uint16_t prt_sidecar_pager::compute_trit_crc(const uint8_t * header_30bytes) {
    uint16_t crc = 0;
    for (size_t i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= header_30bytes[i];
    }
    return crc & 0xFFFF;
}

// ── Layer operations ──────────────────────────────────────────────────────────

void prt_sidecar_pager::shutdown() {
    layer_states_.clear();
    for (auto & kv : file_cache_) {
        delete[] kv.second.data;
        kv.second.data = nullptr;
        kv.second.is_loaded = false;
    }
    file_cache_.clear();
}

bool prt_sidecar_pager::activate_layer(int layer_idx) {
    stats_.activation_attempts++;

    std::vector<const TensorEntry *> entries;
    for (const auto & e : entries_) {
        if (e.layer == layer_idx) entries.push_back(&e);
    }

    if (entries.empty()) {
        return false;
    }

    size_t layer_total = 0;
    for (const auto * e : entries) layer_total += e->size;

    if (stats_.resident_bytes + layer_total > config_.max_resident_bytes) {
        if (config_.eviction_lru) {
            // LRU mode: evict oldest layers until we fit
            while (stats_.resident_bytes + layer_total > config_.max_resident_bytes) {
                int oldest = -1;
                for (const auto & kv : layer_states_) {
                    if (kv.second.is_resident && kv.first != layer_idx &&
                        (oldest == -1 || kv.first < oldest)) {
                        oldest = kv.first;
                    }
                }
                if (oldest == -1) break;
                evict_layer(oldest);
                stats_.lru_evictions++;
            }
            if (stats_.resident_bytes + layer_total > config_.max_resident_bytes) {
                // Still can't fit — this layer is too big
                stats_.budget_rejects++;
                last_error_ = error::BUDGET_EXCEEDED;
                return false;
            }
        } else {
            // Strict reject mode
            enforce_budget();
            if (stats_.resident_bytes + layer_total > config_.max_resident_bytes) {
                stats_.budget_rejects++;
                last_error_ = error::BUDGET_EXCEEDED;
                return false;
            }
        }
    }
    LayerState & ls = layer_states_[layer_idx];
    if (ls.is_resident) {
        stats_.cache_hits++;
        stats_.activation_successes++;
        return true;
    }

    for (const auto * e : entries) {
        std::string full_path = config_.sidecar_root;
        if (!full_path.empty() && full_path.back() != '/' && e->file[0] != '/') full_path += "/";
        full_path += e->file;

        if (config_.validate_trit_header) {
            trit_header th;
            if (load_trit_header(full_path, th)) {
                stats_.trit_validated++;
                if (load_sidecar_data(full_path, 0, e->size, e->checksum)) {
                    auto it = file_cache_.find(full_path);
                    if (it != file_cache_.end() && it->second.is_loaded) {
                        prt_residual_view view;
                        view.data = it->second.data;
                        view.size = e->size;
                        view.is_null = false;
                        view.reason = "";
                        ls.residuals[e->family] = view;
                        ls.resident_bytes += e->size;
                    }
                } else {
                    prt_residual_view view;
                    view.is_null = true;
                    view.reason = "file_read_failed";
                    ls.residuals[e->family] = view;
                }
            } else {
                prt_residual_view view;
                view.is_null = true;
                view.reason = (last_error_ == error::TRIT_CHECKSUM_FAIL) ? "trit_checksum_fail" : "trit_header_invalid";
                ls.residuals[e->family] = view;
            }
        } else {
            if (load_sidecar_data(full_path, 0, e->size, e->checksum)) {
                auto it = file_cache_.find(full_path);
                if (it != file_cache_.end() && it->second.is_loaded) {
                    prt_residual_view view;
                    view.data = it->second.data;
                    view.size = e->size;
                    view.is_null = false;
                    view.reason = "";
                    ls.residuals[e->family] = view;
                    ls.resident_bytes += e->size;
                }
            } else {
                prt_residual_view view;
                view.is_null = true;
                view.reason = "file_missing";
                ls.residuals[e->family] = view;
            }
        }
    }

    ls.is_resident = true;
    stats_.resident_bytes += ls.resident_bytes;
    stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
    stats_.reads++;
    stats_.cache_misses++;
    stats_.activation_successes++;

    int window_start = std::max(0, layer_idx - config_.window_size + 1);
    for (auto & kv : layer_states_) {
        if (kv.first < window_start && kv.second.is_resident) {
            evict_layer(kv.first);
        }
    }

    return true;
}

void prt_sidecar_pager::prefetch_layer(int layer_idx) {
    auto it = layer_states_.find(layer_idx);
    if (it != layer_states_.end() && it->second.is_resident) {
        stats_.cache_hits++;
        return;
    }

    std::vector<const TensorEntry *> entries;
    for (const auto & e : entries_) {
        if (e.layer == layer_idx) entries.push_back(&e);
    }

    if (entries.empty()) return;

    size_t layer_total = 0;
    for (const auto * e : entries) layer_total += e->size;

    if (stats_.resident_bytes + layer_total > config_.max_resident_bytes) return;

    LayerState & ls = layer_states_[layer_idx];
    for (const auto * e : entries) {
        std::string full_path = config_.sidecar_root;
        if (!full_path.empty() && full_path.back() != '/' && e->file[0] != '/') full_path += "/";
        full_path += e->file;

        if (load_sidecar_data(full_path, 0, e->size, e->checksum)) {
            auto fit = file_cache_.find(full_path);
            if (fit != file_cache_.end() && fit->second.is_loaded) {
                prt_residual_view view;
                view.data = fit->second.data;
                view.size = e->size;
                view.is_null = false;
                view.reason = "";
                ls.residuals[e->family] = view;
                ls.resident_bytes += e->size;
            }
        }
    }

    ls.is_resident = true;
    stats_.resident_bytes += ls.resident_bytes;
    stats_.prefetches++;
}

void prt_sidecar_pager::evict_layer(int layer_idx) {
    auto it = layer_states_.find(layer_idx);
    if (it == layer_states_.end() || !it->second.is_resident) return;

    stats_.resident_bytes -= it->second.resident_bytes;
    it->second.is_resident = false;
    it->second.residuals.clear();
    stats_.evictions++;
}

void prt_sidecar_pager::enforce_budget() {
    while (stats_.resident_bytes > config_.max_resident_bytes && !layer_states_.empty()) {
        int oldest = -1;
        for (const auto & kv : layer_states_) {
            if (kv.second.is_resident && (oldest == -1 || kv.first < oldest)) {
                oldest = kv.first;
            }
        }
        if (oldest == -1) break;
        evict_layer(oldest);
    }
}

bool prt_sidecar_pager::can_add_layer(int layer_idx, size_t bytes) {
    return stats_.resident_bytes + bytes <= config_.max_resident_bytes;
}

// ── Residual access ────────────────────────────────────────────────────────────

prt_residual_view prt_sidecar_pager::get_residual(int layer_idx, const std::string & tensor_family) {
    // Phase 28BP-A: manifest coverage guard — short-circuit before activation
    if (!covers(layer_idx, tensor_family)) {
        stats_.fallbacks++;
        stats_.null_views++;
        stats_.not_in_manifest++;
        prt_residual_view v; v.is_null = true; v.reason = "not_in_manifest"; return v;
    }

    auto lit = layer_states_.find(layer_idx);
    if (lit == layer_states_.end() || !lit->second.is_resident) {
        stats_.fallbacks++;
        stats_.null_views++;
        stats_.layer_not_activated++;
        prt_residual_view v; v.is_null = true; v.reason = "layer_not_activated"; return v;
    }

    auto fit = lit->second.residuals.find(tensor_family);
    if (fit == lit->second.residuals.end()) {
        stats_.fallbacks++;
        stats_.null_views++;
        stats_.tensor_not_found++;
        prt_residual_view v; v.is_null = true; v.reason = "tensor_not_found"; return v;
    }

    if (fit->second.is_null) stats_.null_views++;
    else stats_.non_null_views++;
    return fit->second;
}

// ── Phase 28BP-A: coverage guard helpers ─────────────────────────────────────

bool prt_sidecar_pager::covers(int layer_idx, const std::string & tensor_family) const {
    for (const auto & e : entries_) {
        if (e.layer == layer_idx && e.family == tensor_family) return true;
    }
    return false;
}

bool prt_sidecar_pager::has_layer(int layer_idx) const {
    for (const auto & e : entries_) {
        if (e.layer == layer_idx) return true;
    }
    return false;
}

// ── Internal ──────────────────────────────────────────────────────────────────

bool prt_sidecar_pager::load_sidecar_data(const std::string & path, size_t offset, size_t size, uint16_t expected_crc) {
    auto it = file_cache_.find(path);
    if (it != file_cache_.end() && it->second.is_loaded) {
        return true;
    }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        last_error_ = error::SIDECAR_FILE_NOT_FOUND;
        return false;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t * data = new uint8_t[fsize];
    size_t n = fread(data, 1, fsize, f);
    fclose(f);

    if (n != (size_t)fsize) {
        delete[] data;
        last_error_ = error::READ_FAILED;
        return false;
    }

    if (config_.checksum_enabled && expected_crc != 0xFFFF) {
        uint16_t crc = crc16_update(0, data, n);
        if (crc != expected_crc) {
            last_error_ = error::CHECKSUM_MISMATCH;
            delete[] data;
            return false;
        }
    }

    FileCache fc;
    fc.data = data;
    fc.size = n;
    fc.is_loaded = true;
    file_cache_[path] = fc;
    stats_.total_bytes_read += n;

    return true;
}

const void * prt_sidecar_pager::find_entry(int layer_idx, const std::string & family) const {
    for (const auto & e : entries_) {
        if (e.layer == layer_idx && e.family == family) return &e;
    }
    return nullptr;
}

const char * prt_sidecar_pager::error_string(error e) const {
    switch (e) {
        case error::NONE: return "none";
        case error::MANIFEST_NOT_FOUND: return "manifest_not_found";
        case error::MANIFEST_PARSE_ERROR: return "manifest_parse_error";
        case error::SIDECAR_FILE_NOT_FOUND: return "sidecar_file_not_found";
        case error::TRIT_BAD_MAGIC: return "trit_bad_magic";
        case error::TRIT_BAD_VERSION: return "trit_bad_version";
        case error::TRIT_CHECKSUM_FAIL: return "trit_checksum_fail";
        case error::CHECKSUM_MISMATCH: return "checksum_mismatch";
        case error::BUDGET_EXCEEDED: return "budget_exceeded";
        case error::READ_FAILED: return "read_failed";
        default: return "unknown";
    }
}
