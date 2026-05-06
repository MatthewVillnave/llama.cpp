# PRT Phase 13Q: Clean Output Extraction / Debug Log Separation

**Verdict: NEEDS_PRT_LOG_ROUTING**

## Summary

Runner modification to add `--separate-stderr` flag was implemented and tested, but the approach cannot work because llama-cli uses a PTY for both stdout AND stderr. When the child calls `os.setsid()`, the PTY slave becomes the controlling terminal for the session, and both stdout (fd 1) and stderr (fd 2) point to it. This means PRT debug logs written with `fprintf(stderr, ...)` go to the same PTY as stdout.

Result: `stdout_contains_prt_debug: True` for PRT active runs. The logs are not separable externally.

## What Was Tried

1. Added `--separate-stderr` flag to Python PTY runner
2. Captured PTY output on master fd
3. Tested with native (no PRT) and PRT active modes
4. Native: `stdout_contains_prt_debug: False` ✓
5. PRT active: `stdout_contains_prt_debug: True` ✗

## Root Cause Analysis

**llama-cli PTY setup (from phase13o_pty_argv_runner.py):**
```python
os.setsid()  # new session
os.dup2(slave_fd, 0)  # stdin = PTY slave
os.dup2(slave_fd, 1)  # stdout = PTY slave  
os.dup2(slave_fd, 2)  # stderr = PTY slave
os.execvp(argv[0], argv)
```

**PRT debug log calls (src/llama-graph.cpp):**
```c
fprintf(stderr, "[PRT-11BB-AUTH] IL=%d cur=[%lld,%lld] name=%s\n", ...);
```

Since stderr IS the PTY slave, both stdout (generation) and stderr (PRT logs) arrive at the parent via the same PTY master. They cannot be separated without code-level changes in llama.cpp.

## C/C++ Evidence

All PRT debug logs use `fprintf(stderr, ...)`:
- `[PRT] Sidecar set: ...` — llama.cpp:1271
- `[PRT-11BB-AUTH] IL=%d...` — llama-graph.cpp:1110, 1121, 1128, 1136

No `--prt-quiet`, `--prt-log-file`, or similar flags exist in the current codebase.

## Options for Fix

**Option A (Preferred): Add `--prt-log-file` flag**
- Add a global `g_prt_log_file = NULL` in llama-graph.cpp
- Add CLI flag `--prt-log-file /path/to/prt_debug.log` in llama.cpp
- When set, open FILE* and write PRT logs there instead of stderr
- If stderr is a TTY (PTY), writes still go to TTY, but a separate file captures them

**Option B: Add `--prt-quiet` flag**
- Add `g_prt_quiet = false` global
- When `--prt-quiet` is set, skip all `fprintf(stderr, "[PRT-...)` calls
- Provides clean output but loses debug capability

**Option C: Redirect stderr to pipe in Python runner**
- Would require complex IPC to pass pipe fd to child
- Very invasive, likely breaks llama-cli's TTY detection

## Recommended Next

Implement Option A (`--prt-log-file`) since it:
1. Doesn't change the debug output format
2. Allows clean output capture without losing logs
3. Is a minimal, targeted change

Add to llama-graph.cpp:
```c
static FILE * g_prt_log_file = NULL;

void llama_set_prt_log_file(const char * path) {
    g_prt_log_file = fopen(path, "w");
}

// Then in PRT log calls:
fprintf(g_prt_log_file ? g_prt_log_file : stderr, "[PRT-11BB-AUTH] ...\n", ...);
```

Then test the runner with stdout-only capture (no PRT logs in PTY output).

## Safety Check

- No models staged: ✓
- No secrets leaked: ✓
- Tags untouched: ✓

## Test Results

**Native (no PRT):**
- exit_code: 0, timed_out: False, elapsed: 0.604s
- saw_generation_timing: True ✓
- stdout_contains_prt_debug: False ✓

**PRT active:**
- exit_code: 0, timed_out: False, elapsed: 2.348s
- saw_prt_shape: True ✓
- saw_generation_timing: True ✓
- stdout_contains_prt_debug: True ✗ (debug logs interleaved)
- saw_prt_true_replacement: True ✓

**Conclusion:** External separation impossible. Code-level log routing required.