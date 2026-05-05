# PRT Phase 13D-S: PRT Shim Disabled-Mode Parity Fix

**Date:** 2026-05-04 21:40 EDT  
**Branch:** `experimental/prt-phase13-model-generalization`  
**Previous HEAD:** `fb014ed84`

---

## Root Cause

The llama-prt-posix binary had UNCONDITIONAL sidecar loading and callback installation, even when no `--prt-mode` flag was provided:

```cpp
// BEFORE (unconditional - bug)
load_all_sidecars(model);            // line 430 - runs EVERY time
if (g_sidecars.empty()) return 1;  // exits if sidecars missing
if (!validate_prt_sidecars()) return 1;
cparams.cb_eval = prt_eval_callback;  // line 438 - installed ALWAYS
cparams.cb_eval_user_data = &g_prt_state;
```

Even with `g_prt_debug_mode == 0` (default when no --prt-mode), the binary:
1. Always loaded sidecars from /tmp/prt_sidecars/
2. Always installed PRT callback
3. Always ran PRT replacements in callback mode

The callback was corrupting the FFN output - causing path fragment strings like "ffn_up_layer0_prt.bin" to leak into generation.

---

## Fix Applied

Added guards in `phase10e0_layer0_replacement.cpp`:

```cpp
// AFTER (guarded - fix)
if (g_prt_debug_mode > 0) {
    load_all_sidecars(model);
    if (g_sidecars.empty()) return 1;
    if (!validate_prt_sidecars()) return 1;
}

// Callback only installed when PRT mode is active
if (g_prt_debug_mode > 0) {
    cparams.cb_eval = prt_eval_callback;
    cparams.cb_eval_user_data = &g_prt_state;
}
```

---

## Test Results

| Test | Path Fragments Found | Verdict |
|------|-------------------|----------|
| Native llama-cli | 0 | ✅ CLEAN |
| PRT disabled (before fix) | 40+ | ❌ CORRUPT |
| PRT disabled (after fix) | 0 | ✅ CLEAN |

Key output after fix:
- `[DEBUG] token 17: n=8 buf_hex: 20 63 61 70 69 74 61 6C str=' capital'`
- NO `ffn_up_layer0_prt.bin` fragments

---

## Binary Rebuild

| Metric | Before | After |
|--------|--------|-------|
| SHA256 | ce959f62...d7 | 16643f031...9 |
| Build date | May 1 23:34 | May 4 21:34 |
| Object size | 34832 bytes | (recompiled) |

**Compilation command:**
```bash
cd build/examples/speculative && /usr/bin/c++ -O3 -DNDEBUG ... -c ../examples/speculative/phase10e0_layer0_replacement.cpp
```

**Link command:**
```bash
/usr/bin/c++ -O3 -DNDEBUG examples/speculative/CMakeFiles/llama-prt-posix.dir/phase10e0_layer0_replacement.cpp.o -o bin/llama-prt-posix ...
```

---

## Verdict

**PASS** — PRT disabled mode now matches native clean behavior.

In disabled mode (no --prt-mode flag):
- ✅ Sidecar loading prevented
- ✅ Callbacks not installed  
- ✅ No PRT logs printed
- ✅ No path fragment corruption
- ✅ Output token "capital" shows correctly

---

## Next Steps

1. **Active PRT dry run** — Test with `--prt-mode 5700` after disabled mode is verified
2. **Debug active mode numerical output** — If active produces garbage, that's a separate sidecar/math issue, not the shim bug
3. **Consider removing debug prints from callback** — The callback's [DEBUG] lines should only appear in verbose mode

---

## Files Changed

- `examples/speculative/phase10e0_layer0_replacement.cpp` — Added `g_prt_debug_mode > 0` guards around sidecar loading and callback installation