// Phase 28BM: PRT sidecar runtime link implementation
// Contains the implementation of prt_get_residual_view(), prt_init_pager(),
// prt_shutdown_pager(), and prt_get_pager_stats().
// This file is compiled into libllama.so and linked to all consumers.
// Single definition ensures no ODR violations.

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

#include "prt_sidecar_runtime_link.h"
#include "prt_sidecar_pager.h"
#include <string>

// Runtime globals are defined once in src/prt_sidecar_pager_globals.cpp.
// This file is extern-only to avoid splitting pager state across translation units.
extern std::unordered_map<int, SidecarLoad> g_sidecars;
extern bool g_sidecars_loaded;
extern prt_sidecar_pager* g_prt_pager;
extern bool g_prt_pager_enabled;

// ── Residual view wrapper ────────────────────────────────────────────────────

// Get residual view for a layer + tensor family.
// Routes to pager if enabled and layer is available in pager manifest.
// Falls back to legacy g_sidecars path otherwise.
// Returns a null view with reason set if neither pager nor legacy has the tensor.
prt_residual_view prt_get_residual_view(int layer_idx, const std::string& tensor_family) {
    prt_residual_view view;
    view.is_null = true;
    view.reason = "not_found";

    if (g_prt_pager != nullptr && g_prt_pager_enabled) {
        // Pager path
        view = g_prt_pager->get_residual(layer_idx, tensor_family);
        if (!view.is_null) {
            return view;
        }
        // Pager returned null — fall through to legacy
    }

    // Legacy path: check g_sidecars map
    // Legacy stores sidecars by layer only, not by tensor_family.
    // For compatibility, if legacy has a sidecar for this layer, return a
    // compatible view (the legacy pointer as a uint8_t* with the sidecar size).
    // Note: legacy sidecars are float* while pager sidecars are uint8_t*
    // — caller is responsible for type awareness.
    auto it = g_sidecars.find(layer_idx);
    if (it != g_sidecars.end() && it->second.data != nullptr) {
        view.data = reinterpret_cast<const uint8_t*>(it->second.data);
        view.size = it->second.size;
        view.is_null = false;
        view.reason = "legacy";
        return view;
    }

    // Neither pager nor legacy has this tensor
    view.is_null = true;
    view.reason = "not_found";
    return view;
}

// ── Pager initialization helper ──────────────────────────────────────────────

// Init pager from config. Call only when --enable-prt-sidecar-pager is set.
// Returns true on success, false on failure.
// Sets g_prt_pager and g_prt_pager_enabled on success.
bool prt_init_pager(const prt_sidecar_pager_config& config) {
    if (g_prt_pager != nullptr) {
        // Already initialized
        return true;
    }
    g_prt_pager = new prt_sidecar_pager(config);
    if (!g_prt_pager->init()) {
        delete g_prt_pager;
        g_prt_pager = nullptr;
        g_prt_pager_enabled = false;
        return false;
    }
    g_prt_pager_enabled = true;
    return true;
}

// Shutdown and free pager. Safe to call even if not initialized.
void prt_shutdown_pager() {
    if (g_prt_pager != nullptr) {
        g_prt_pager->shutdown();
        delete g_prt_pager;
        g_prt_pager = nullptr;
        g_prt_pager_enabled = false;
    }
}

// ── Stats helper ─────────────────────────────────────────────────────────────

prt_sidecar_pager_stats prt_get_pager_stats() {
    if (g_prt_pager != nullptr) {
        return g_prt_pager->get_stats();
    }
    return prt_sidecar_pager_stats();
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL
