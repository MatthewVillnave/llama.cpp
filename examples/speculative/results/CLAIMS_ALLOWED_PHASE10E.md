# CLAIMS_ALLOWED_PHASE10E.md

## Allowed Claims After Phase 10E

The following claims are **ALLOWED** based on Phase 10E results:

### PRT Research Claims

1. ✅ **PRT_3P standalone sidecars passed offline/full-layer construction**
   - Sidecar files are valid: correct size (2048×11008×4 = 90,177,536 bytes)
   - All values finite: no NaN, no Inf
   - No path string contamination in binary data
   - Standalone PRT compute produces correct output with synthetic activation

2. ✅ **Sidecar loader and standalone PRT compute appear clean**
   - Loader pointer arithmetic verified: all pointers within allocated range
   - Standalone PRT math validated: correct output outside llama.cpp
   - File format validated: no corruption, no contamination

3. ✅ **llama.cpp graph interception and custom-op substitution were proven structurally**
   - `ggml_map_custom2` successfully intercepts ffn_up computation graph nodes
   - Custom op callback fires for all 36 layers (verified with logging)
   - Graph substitution mechanism is functional at the infrastructure level

4. ✅ **All-layer sidecar lookup/substitution infrastructure reached the graph**
   - `llama_set_prt_sidecar()` successfully stores per-layer sidecar pointers
   - Per-layer lookup via `g_prt_sidecar_data[layer]` works correctly
   - All 36 layers receive custom op substitution

5. ✅ **Active generation branch is blocked by ggml custom-op memory corruption**
   - Corruption confirmed: output contains path string fragments
   - Root cause: ggml_map_custom2 integration issue, not custom op code
   - Sidecar loading + custom op + ggml graph = memory corruption

6. ✅ **No end-to-end PRT speedup or quality claim is allowed**
   - Cannot measure speedup with corrupted output
   - Cannot measure quality with corrupted output
   - Production integration is blocked

### Technical Claims

7. ✅ **PRT custom op fires for all layers** — verified via logging (prt_op_entry count = 36+ per run)

8. ✅ **Per-layer sidecar lookup works** — `g_prt_sidecar_data[layer]` correctly returns non-null pointers when sidecar is loaded

9. ✅ **Custom op identity fallback works** — when sidecar=nil, identity copy produces correct output (confirmed with llama-completion baseline)

10. ✅ **ggml_map_custom2 substitution reaches the graph** — the substitution mechanism is proven functional at the llama-graph level