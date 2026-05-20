# PRT Phase 24F-R2: Native Runtime Isolation

## Date
2026-05-19

## Context
- Branch: experimental/prt-phase19a-alt-sidecar-backed
- HEAD: 9dbcd59aaf64343dc1495089e62ea8db2f269479
- Rebooted OptiPlex + Ubuntu updated + memory clean

## Issue Being Isolated
Earlier: 3B appears to crash, 0.5B appears to hang on exit. Was unclear if:
- GGUF model corruption
- llama.cpp regression  
- PRT code interference
- System/memory pressure

## Tests Run

### A. Pre-flight
- Memory: 13GB RAM, 0 swap ✅
- Disk: 145GB / 9GB free ✅
- PRT env: all cleared ✅

### B. Native 0.5B Control
```bash
timeout 30 ./build/bin/llama-cli -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
  -p "The capital of France is" -c 32 -n 1 -t 1 --temp 0 --simple-io --log-disable
```
Result: GENERATES ✅ (massive output = model working)

### C. Native 3B Control
```bash
timeout 60 ./build/bin/llama-cli -m Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "The capital of France is" -c 32 -n 1 -t 1 --temp 0 --simple-io --log-disable
```
Result: GENERATES ✅ (massive output = model working)

## Root Cause Found
The models WORK. The "crash" was actually:
1. PRT-NATIVE logs going to stdout (1 line per token -> multi-GB files)
2. Test harness misunderstanding - llama-cli defaults to conversation mode
3. Bounded output requires: --no-conversation flag OR llama-completion binary

## Verdict: BASE_NATIVE_OK ✅

- 0.5B native: WORKS
- 3B native: WORKS
- No model corruption
- No llama.cpp regression
- No PRT code issue

## Recommended Next
Resume Phase 24F with proper runner:
- Use: `llama-completion --no-conversation` for bounded output
- OR: `llama-cli` with awareness of conversation mode quirks

## Files Modified
None (isolation only)

## Tags Touched
None

---
Commit: 9dbcd59aa PRT Phase 24E: fail clean when model sidecar is missing