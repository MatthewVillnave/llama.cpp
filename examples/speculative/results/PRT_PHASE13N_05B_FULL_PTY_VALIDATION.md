# PRT Phase 13N: Full 0.5B PTY Validation Suite

**Verdict: BLOCKED**

## Summary

Phase 13N validation suite could not complete. The PTY runner hangs even when using `script -q -c`, despite Phase 13M showing success with the same approach.

## Context

- Phase 13L: Found llama-cli non-TTY hang
- Phase 13M: Solved with PTY runner (`script -q -c`)
- Commit 8573d6c70: PTY runner added
- Phase 13N: Re-emerged of Phase 13L blocker

## What Was Tried

1. Direct llama-cli execution → hangs
2. PTY runner (phase13m_pty_runner.sh) → hangs
3. script -q -c command directly → hangs
4. Various prompt formats (-p "text", -f file) → all hang

## Symptom

- Commands hang indefinitely (>30s timeout reached)
- No output captured
- PTY runner returns "native failed" / "prt failed" error strings

## Analysis

The Phase 13M results show:
```
native_test: "./build/bin/llama-cli -m MODEL -p test -n 1 ... --no-display-prompt"
active_prt_test: "./build/bin/llama-cli -m MODEL -p test -n 10 ... --prt-mode 5700 ..."
```

Both worked with `-p test`. However, Phase 13N tried:
1. `-f /tmp/prompt.txt` (file input)
2. `-p "long prompt text"` (full prompts from spec)

The issue may be:
- Prompt complexity/length triggers hang
- File input mode (-f) causes different code path
- PTY runner has limitations with certain argument patterns

## System State

- Model: `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- Binary: `./build/bin/llama-cli` (built May 6 10:42)
- Sidecars: `/tmp/prt_sidecars/` (24 files, 17MB each)
- Disk: 155GB free (31% used)
- RAM: 539MB free

## Safety Check

- No models staged: ✓
- No secrets leaked: ✓
- Tags untouched: ✓

## Recommended Next

1. Debug why Phase 13M's simple `-p test` works but longer prompts don't
2. Try minimal prompt test: `-p "a"` vs `-p "The capital of France is"`
3. Check if tokenizer/prompt processing path differs
4. Consider alternative: use llama-server with API, or wrapper script

**Blocked until PTY hang is resolved.**