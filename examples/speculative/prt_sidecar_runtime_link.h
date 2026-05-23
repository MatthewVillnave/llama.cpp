// Phase 28AU: Runtime Link Stub — prt_get_residual_view() behind disabled flag
// Provides lookup routing between legacy g_sidecars and pager, no generation change.

#ifndef PRT_SIDECAR_RUNTIME_LINK_H
#define PRT_SIDECAR_RUNTIME_LINK_H

// Compile guard: only active when experimental sidecar pager is enabled
#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

#include "prt_sidecar_pager.h"
#include <unordered_map>
#include <string>

// g_sidecars is declared extern in prt_shadow.h (weak static definition there)
// Access via extern here
struct SidecarLoad;
extern std::unordered_map<int, SidecarLoad> g_sidecars;

// ── Global pager object (nullptr by default) ────────────────────────────────

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
    // Legacy path: check g_sidecars map (declared in prt_shadow.h)
    // g_sidecars is std::unordered_map<int, SidecarLoad> from prt_shadow.h
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

#endif  // PRT_SIDECAR_RUNTIME_LINK_H