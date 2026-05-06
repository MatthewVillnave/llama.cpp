# PRT Phase 13N-A: PTY Runner Full-Prompt Isolation

**Verdict: SHELL_QUOTING_BUG**

## Summary

Proven root cause: The PTY runner uses `script -q -c "$*"` which joins arguments with spaces and passes to shell for tokenization. This breaks when prompts contain spaces, because shell tokenizes the single string instead of preserving argument boundaries.

## Test Results

| Test | Command | Result | Notes |
|------|---------|--------|-------|
| A | Direct `script -q -c "./build/bin/... -p test"` | PASS | Output "Hello" |
| B | Direct `script -q -c "./build/bin/... -p The capital of France is"` | PASS | Output "The" |
| C | Runner with `-p test` | TIMEOUT | 60s timeout reached |
| D | Runner with `-p The capital of France is` | error: invalid argument: capital | exit 0 but wrong |
| E | Runner with `-p The_capital_of_France_is` | TIMEOUT | 60s timeout |

## Root Cause Analysis

**Runner (phase13m_pty_runner.sh) line 33:**
```bash
OUTPUT=$(timeout "$TIMEOUT" script -q -c "$*" /dev/null 2>&1) || true
```

When `$*` joins all arguments with spaces:
```
script -q -c "./build/bin/llama-cli -m /path -p test -n 1 ..."
```

This gets passed to `sh -c` which re-tokenizes the string. With `-p test`, the token `-p` gets value `test`. But with `-p The capital of France is`, the shell parses:
- `-p` = "The"
- `capital` = unrecognized flag → error!
- `of`, `France`, `is` = additional tokens

**Direct script works because** the entire command is already a quoted string passed to script -c, preserving the embedded quotes around the prompt.

## Evidence

1. Test B (direct script full prompt) works → script -q -c works with full prompts
2. Test D (runner full prompt) fails with error "invalid argument: capital" → prompt got split
3. Test C (runner short prompt) also timed out → something else wrong with runner output capture

The runner has two issues:
1. **Shell quoting bug** for prompts with spaces
2. **Output capture issue** - both C and E time out with 0-byte output despite command running

## Fix: Python PTY argv Runner

Created `phase13o_pty_argv_runner.py` that:
- Uses `pty.fork()` + `os.execvp()` 
- Takes argv directly (no shell string)
- Handles arguments properly with any content

However, initial test timed out - needs debugging.

## System State

- Binary: `./build/bin/llama-cli` (May 6 10:42)
- Sidecars: `/tmp/prt_sidecars/` (24 files, 17MB each)
- Disk: 151GB free (32% used)
- RAM: 410MB free

## Safety Check

- No models staged: ✓
- No secrets leaked: ✓
- Tags untouched: ✓

## Recommended Next

1. Debug Python argv runner (initial test timed out)
2. Or fix runner to use argv array instead of `$*` string
3. Once PTY runner handles full prompts → re-run Phase 13N validation

**Blocked until PTY runner can handle full prompts properly.**