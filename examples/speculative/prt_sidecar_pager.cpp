// Phase 28AM: Standalone Sidecar Pager Implementation
// Sidecar-only pager — no llama.cpp, no ggml, no external dependencies.

#include "prt_sidecar_pager.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>

// ── CRC16 ─────────────────────────────────────────────────────────────────────

uint16_t prt_sidecar_pager::crc16(const uint8_t * data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 1) { crc = (crc >> 1) ^ 0xA001; }
            else { crc >>= 1; }
        }
    }
    return crc;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

prt_sidecar_pager::prt_sidecar_pager(const prt_sidecar_pager_config & config)
    : config_(config), last_error_(error::NONE) {}

prt_sidecar_pager::~prt_sidecar_pager() {
    shutdown();
}

bool prt_sidecar_pager::init() {
    // Simple manifest format:
    // {
    //   "entries": [
    //     {"layer": 0, "family": "ffn_up", "file": "layer_000.ffn_up.trit", "size": 1234, "offset": 0, "crc": 0xABCD},
    //     ...
    //   ]
    // }

    FILE * mf = fopen(config_.manifest_path.c_str(), "r");
    if (!mf) {
        last_error_ = error::MANIFEST_NOT_FOUND;
        return false;
    }

    // Read entire manifest into buffer
    fseek(mf, 0, SEEK_END);
    long fsize = ftell(mf);
    fseek(mf, 0, SEEK_SET);
    std::vector<char> buf(fsize + 1);
    fread(buf.data(), 1, fsize, mf);
    buf[fsize] = '\0';
    fclose(mf);

    // Simple JSON parsing — look for "layer": N and "family": "X"
    // Not a full JSON parser — just enough for test manifests
    const char * p = buf.data();
    while ((p = strstr(p, "\"layer\"")) != nullptr) {
        // Find layer number
        p += 7;
        while (*p == ' ' || *p == ':') p++;
        int layer = atoi(p);

        // Find family
        const char * fam_start = strstr(p, "\"family\"");
        if (!fam_start) break;
        fam_start += 9;
        while (*fam_start == ' ' || *fam_start == ':' || *fam_start == '"') fam_start++;
        const char * fam_end = fam_start;
        while (*fam_end && *fam_end != '"') fam_end++;
        std::string family(fam_start, fam_end - fam_start);

        // Find file
        const char * file_start = strstr(fam_end, "\"file\"");
        if (!file_start) break;
        file_start += 7;
        while (*file_start == ' ' || *file_start == ':' || *file_start == '"') file_start++;
        const char * file_end = file_start;
        while (*file_end && *file_end != '"') file_end++;
        std::string file(file_start, file_end - file_start);

        // Find size
        const char * size_start = strstr(file_end, "\"size\"");
        if (!size_start) break;
        size_start += 6;
        while (*size_start == ' ' || *size_start == ':') size_start++;
        size_t size = (size_t)atoll(size_start);

        ManifestEntry entry;
        entry.layer = layer;
        entry.tensor_family = family;
        entry.file = file;
        entry.size_bytes = size;
        entry.offset = 0;
        entry.checksum = 0xFFFF;  // placeholder
        entry.present = true;
        manifest_entries_.push_back(entry);

        p = file_end;
    }

    return manifest_entries_.size() > 0;
}

void prt_sidecar_pager::shutdown() {
    layer_states_.clear();
    for (auto & kv : file_cache_) {
        if (kv.second.data) {
            // data is owned by caller in read() mode; just clear
            kv.second.data = nullptr;
            kv.second.is_loaded = false;
        }
    }
    file_cache_.clear();
}

// ── Layer operations ──────────────────────────────────────────────────────────

bool prt_sidecar_pager::activate_layer(int layer_idx) {
    // Find entries for this layer
    std::vector<const ManifestEntry *> entries;
    for (const auto & e : manifest_entries_) {
        if (e.layer == layer_idx) entries.push_back(&e);
    }

    if (entries.empty()) {
        // No entries for this layer — normal, return true
        return true;
    }

    // Calculate total size for this layer
    size_t layer_total = 0;
    for (const auto * e : entries) layer_total += e->size_bytes;

    // Check budget
    if (stats_.resident_bytes + layer_total > config_.max_resident_bytes) {
        // Try to enforce budget first
        enforce_budget();
        if (stats_.resident_bytes + layer_total > config_.max_resident_bytes) {
            if (config_.strict_budget) {
                last_error_ = error::BUDGET_EXCEEDED;
                stats_.budget_rejects++;
                return false;
            }
        }
    }

    // Ensure layer state exists
    LayerState & ls = layer_states_[layer_idx];
    if (ls.is_resident) {
        stats_.cache_hits++;
        return true;
    }

    // Load each entry
    for (const auto * e : entries) {
        // Build full path
        std::string full_path = config_.sidecar_root;
        if (!full_path.empty() && full_path.back() != '/' && e->file[0] != '/') full_path += "/";
        full_path += e->file;

        if (load_sidecar_data(full_path, e->offset, e->size_bytes, e->checksum)) {
            // Create residual view
            auto it = file_cache_.find(full_path);
            if (it != file_cache_.end() && it->second.is_loaded) {
                prt_residual_view view;
                view.data = it->second.data + e->offset;
                view.size = e->size_bytes;
                view.is_null = false;
                view.reason = "";
                ls.residuals[e->tensor_family] = view;
                ls.resident_bytes += e->size_bytes;
            }
        } else {
            // File missing or read error — mark as null
            prt_residual_view view;
            view.is_null = true;
            view.reason = last_error_ == error::CHECKSUM_MISMATCH ? "checksum_failed" : "file_missing";
            ls.residuals[e->tensor_family] = view;
        }
    }

    ls.is_resident = true;
    stats_.resident_bytes += ls.resident_bytes;
    stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
    stats_.reads++;
    stats_.cache_misses++;

    // Update active window
    active_window_start_ = std::max(0, layer_idx - config_.window_size + 1);

    // Evict layers outside window
    for (auto & kv : layer_states_) {
        if (kv.first < active_window_start_) {
            evict_layer(kv.first);
        }
    }

    return true;
}

void prt_sidecar_pager::prefetch_layer(int layer_idx) {
    // Check if already loaded
    auto it = layer_states_.find(layer_idx);
    if (it != layer_states_.end() && it->second.is_resident) {
        stats_.cache_hits++;
        return;
    }

    // Try to load (but don't enforce budget — just prefetch)
    std::vector<const ManifestEntry *> entries;
    for (const auto & e : manifest_entries_) {
        if (e.layer == layer_idx) entries.push_back(&e);
    }

    if (entries.empty()) return;

    size_t layer_total = 0;
    for (const auto * e : entries) layer_total += e->size_bytes;

    if (stats_.resident_bytes + layer_total > config_.max_resident_bytes) {
        // Can't prefetch due to budget — not an error
        return;
    }

    LayerState & ls = layer_states_[layer_idx];
    for (const auto * e : entries) {
        std::string full_path = config_.sidecar_root;
        if (!full_path.empty() && full_path.back() != '/' && e->file[0] != '/') full_path += "/";
        full_path += e->file;

        if (load_sidecar_data(full_path, e->offset, e->size_bytes, e->checksum)) {
            auto fit = file_cache_.find(full_path);
            if (fit != file_cache_.end() && fit->second.is_loaded) {
                prt_residual_view view;
                view.data = fit->second.data + e->offset;
                view.size = e->size_bytes;
                view.is_null = false;
                view.reason = "";
                ls.residuals[e->tensor_family] = view;
                ls.resident_bytes += e->size_bytes;
            }
        } else {
            prt_residual_view view;
            view.is_null = true;
            view.reason = "prefetch_failed";
            ls.residuals[e->tensor_family] = view;
        }
    }

    ls.is_resident = true;
    stats_.resident_bytes += ls.resident_bytes;
    stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
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
        // Find oldest resident layer
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

// ── Residual access ───────────────────────────────────────────────────────────

prt_residual_view prt_sidecar_pager::get_residual(int layer_idx, const std::string & tensor_family) {
    auto lit = layer_states_.find(layer_idx);
    if (lit == layer_states_.end()) {
        stats_.fallbacks++;
        prt_residual_view v;
        v.is_null = true;
        v.reason = "layer_not_activated";
        return v;
    }

    if (!lit->second.is_resident) {
        stats_.fallbacks++;
        prt_residual_view v;
        v.is_null = true;
        v.reason = "layer_evicted";
        return v;
    }

    auto fit = lit->second.residuals.find(tensor_family);
    if (fit == lit->second.residuals.end()) {
        stats_.fallbacks++;
        prt_residual_view v;
        v.is_null = true;
        v.reason = "tensor_not_found";
        return v;
    }

    return fit->second;
}

// ── Internal ──────────────────────────────────────────────────────────────────

bool prt_sidecar_pager::load_sidecar_file(const ManifestEntry & entry) {
    std::string full_path = config_.sidecar_root;
    if (!full_path.empty() && full_path.back() != '/' && entry.file[0] != '/') full_path += "/";
    full_path += entry.file;
    return load_sidecar_data(full_path, entry.offset, entry.size_bytes, entry.checksum);
}

bool prt_sidecar_pager::load_sidecar_data(const std::string & path, size_t offset, size_t size, uint16_t expected_crc) {
    // Check cache first
    auto it = file_cache_.find(path);
    if (it != file_cache_.end() && it->second.is_loaded) {
        return true;
    }

    // Load file
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

    // Checksum if enabled
    if (config_.checksum_enabled) {
        uint16_t crc = crc16(data, n);
        if (crc != expected_crc && expected_crc != 0xFFFF) {
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

const ManifestEntry * prt_sidecar_pager::find_entry(int layer_idx, const std::string & family) const {
    for (const auto & e : manifest_entries_) {
        if (e.layer == layer_idx && e.tensor_family == family) return &e;
    }
    return nullptr;
}

const char * prt_sidecar_pager::error_string(error e) const {
    switch (e) {
        case error::NONE: return "none";
        case error::MANIFEST_NOT_FOUND: return "manifest_not_found";
        case error::MANIFEST_PARSE_ERROR: return "manifest_parse_error";
        case error::SIDECAR_FILE_NOT_FOUND: return "sidecar_file_not_found";
        case error::CHECKSUM_MISMATCH: return "checksum_mismatch";
        case error::BUDGET_EXCEEDED: return "budget_exceeded";
        case error::READ_FAILED: return "read_failed";
        default: return "unknown";
    }
}