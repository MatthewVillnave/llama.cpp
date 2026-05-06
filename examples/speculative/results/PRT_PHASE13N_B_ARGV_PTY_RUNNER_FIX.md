# PRT Phase 13N-B: Fix argv-safe PTY runner

**Verdict: PARTIAL** (runner works for capture, exit detection needs fix)

## Summary

Python PTY argv runner (phase13o) using `pty.openpty + os.execvp` avoids the shell quoting issue. It successfully captures output and PRT logs:
- Test 3 showed PRT active: "[PRT-11BB] IL=23... Generation: 20.2 t/s"
- All 3 tests captured stdout/stderr successfully

However, all 3 tests timed out because `llama-cli` doesn't exit after generation - it stays in interactive mode waiting for stdin input. Under PTY this causes the runner to hang even though generation completed.

## Tests Run

| Test | Command | Result | Output |
|------|---------|-------|-------|
| 1 | native short | timed_out=True | captured, shows "Exiting..." |
| 2 | native full | timed_out=True | captured |
| 3 | active PRT | timed_out=True | captured, PRT logs visible |

## Findings

**Python PTY runner fixes shell quoting** - No more "invalid argument: capital" error.

**PRT works through Python PTY runner** - Test 3 shows PRT inference happening.

**Exit detection issue** - llama-cli doesn't exit after generation, stays interactive.

## Fixes Applied

1. Rewrote runner to use `pty.openpty()` + `os.execvp()` directly
2. Fixed waitpid reaping bug (ChildProcessError after first reap)
3. Added non-blocking mode on PTY master
4. Added proper child death detection

## Root Cause: llama-cli interactive mode

llama-cli with `-p prompt -n N` still runs in interactive mode after generating N tokens. It waits for stdin to produce more output. Under PTY, stdin is open/connected, so the process never exits unless we close stdin or send EOF.

**Solutions:**
1. Close stdin in child after exec (os.close(0))
2. Use file-based prompt (-f file) instead of -p
3. Send EOF to child after short delay
4. Accept timeout as expected behavior for interactive CLI tools

## Output Captured (example from Test 3)

```
[PRT-11BB] IL=23 hidden=896 ffn=4864 tokens=1 in_sum(4)=-0.5472 out_sum(4)=-1.4082
[ Prompt: 24.3 t/s | Generation: 20.2 t/s ]
```

This proves:
- PTY runner passes args correctly with spaces
- PRT is active and running inference
- Output is captured

## System State

- Binary: `./build/bin/llama-cli` (May 6 10:42)
- Sidecars: `/tmp/prt_sidecars/` (24 files, 17MB each)
- Disk: ~151GB free

## Safety Check

- No models staged: ✓
- No secrets leaked: ✓  
- Tags untouched: ✓

## Recommended Next

1. Fix exit detection - close stdin in child after exec, or accept timeout behavior
2. Once exit detection fixed → re-run Phase 13N full suite
3. Test with `-f file` instead of `-p prompt` alternative

**Blocked pending exit detection fix or acceptance of timeout behavior.**