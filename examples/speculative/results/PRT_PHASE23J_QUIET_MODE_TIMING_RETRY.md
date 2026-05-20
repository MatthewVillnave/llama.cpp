# PRT Phase 23J — Quiet Mode Timing Retry

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`  
**Previous HEAD:** `77066892c` (Phase 23I checkpoint)  
**New HEAD:** — (BLOCKED: compile errors with static init order)  
**Verdict:** `FAIL_QUIET_GUARD_INCOMPLETE`  
**Date:** Sun 2026-05-17

---

## Verdict: FAIL_QUIET_GUARD_INCOMPLETE

Attempted to add `PR T_V2_LOG_LEVEL` env guard to suppress `[PRT-NATIVE]` debug printfs via `g_prt_log_level` control. Multiple compile failures due to C++ static initialization order.

Specifically:
1. `g_prt_log_level` was declared at line ~103 in `src/llama-graph.cpp`, but the struct that needed to set it was `PRTEnvAutoInit` (around line 78)
2. Static initialization order in C++ means the struct runs before the global `g_prt_log_level` is initialized

Multiple failed fixes:
- Add `extern` declaration before struct → compile errors
- Move `g_prt_log_level` declaration earlier → same error  
- Add static init struct for log level → circular dep with other struct
- Use inline lambda for in-place init → not supported in C++11/14

---

## What Was Tried

1. Added `extern int g_prt_log_level;` before `PRTEnvAutoInit` → failed with forward declaration errors
2. Moved `g_prt_log_level = 2;` before the structs → still failed because the struct's constructor still ran before initialization
3. Created separate `PRTLogLevelInit` static struct → circular dependency with `PRTEnvAutoInit` which also reads the same env var
4. Tried using sed to add inline lambda → sed corrupted the file

Also tried removing the duplicate lines but each attempt made things worse.

---

## Why This Failed

The core problem is C++ static initialization order (not code correctness):

- Line ~58: `int g_prt_ggml_op_layer = -1;`
- Line ~78-105: `PRTEnvAutoInit` struct (constructor runs at static init)
- Line ~104: `int g_prt_log_level = 2;` (this is AFTER PRTEnvAutoInit)

When the static init runs, `g_prt_env_auto_init` executes its constructor (which reads env vars), but by the time it finds and uses `g_prt_log_level`, that variable might not be in the right state. This is complex static init order.

---

## What The Code Shows

The [PRT-NATIVE] spam is controlled by:
- `g_prt_log_level = 2` (default = debug)
- Guard: `if (g_prt_log_level >= 2)` in various places prints the spam

The `--log-disable` flag is supposed to suppress logging, but it only affects llama.cpp's internal logging, not the PRT debug prints that happen at inference time.

The correct fix would require adding static initialization infrastructure similar to how `LLAMA_LOG` is handled - but this is beyond minimal change scope.

---

## Current State

- Branch remains at Phase 23I: `77066892c`
- No code changes committed
- No timing data captured
- No speedup claim exists

---

## Recommendation

Options:
1. **Skip timing for now** — Phase 23J is too complex for minimal change. Accept that timing requires manual terminal execution.
2. **Just document the root cause** — Phase 23H timing was blocked by `[PRT-NATIVE]` spam, not by a fixable code issue. The blocker is the debug spam itself.
3. **Manual timing only** — Run timing in a real terminal with `PRT_V2_LOG_LEVEL=0` env set manually, not via the wrapper.

The timing is blocked, not by code correctness, but by debug output volume. The real fix requires proper infrastructure for log level control at static init time, which is beyond minimal change scope.