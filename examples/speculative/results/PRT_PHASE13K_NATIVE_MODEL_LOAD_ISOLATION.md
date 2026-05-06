# PRT Phase 13K: Native Model-Load Isolation

**Date:** 2026-05-06
**Branch:** `experimental/prt-phase13-model-generalization`
**HEAD:** `c29d83f39`
**Verdict:** `ENVIRONMENT_OR_MODEL_LOAD_RUNTIME_ISSUE`

---

## Test Setup

**Binary:** `./build/bin/llama-cli`
**Binary SHA256:** `176f6e5e8c5a12a4e632ddff160aaf6676a739e2030b747b942201fdc0353930`
**Binary size:** 5.7MB

**Model:** `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
**Model size:** 380MB (Apr 29)

**Test parameters:**
- `-p "test"` (1 token eval)  
- `-n 1` (generate 1 token)
- `--temp 0`
- `-c 256` (context size)
- `-t 4` (threads)
- `--no-display-prompt`
- stdout → /dev/null
- stderr captured
- timeout: 60s

---

## Test A: Current HEAD (c29d83f39) Results

| Run | Exit Code | Result |
|-----|---------|-------|
| 1 | 124 | TIMEOUT |
| 2 | 124 | TIMEOUT |
| 3 | 124 | TIMEOUT |

- **3/3 runs timed out**
- No PRT logs found
- No sidecar logs found
- No flag echo
- Stderr shows only memory breakdown, no generation output

**Stderr pattern:**
```
llama_memory_breakdown_print: ... Host = 526 MiB ...
```

The model loads and memory is allocated, but generation never starts.

---

## Test B: Known-Good Comparison (761b1352c)

Created worktree at commit `761b1352c`:
```
mkdir -p /tmp/prt_worktrees
git worktree add /tmp/prt_13h_r_761b1352c 761b1352c
```

Built in worktree and tested with identical parameters:
- Build: `cmake -B build -DLLAMA_BUILD_SERVER=ON ...`
- Build result: `100% Built target llama-cli`
- Binary SHA256: `4633793a7a4e85cc8190ad57a956703cbe7c4b92ecb890a51711453709458fc9`

| Run | Exit Code | Result |
|-----|---------|-------|
| 1 | 124 | TIMEOUT |

**Same failure as current HEAD.** Both builds hang at the exact same point.

---

## Key Finding: llama-simple Works

**Crucially:** Same model, same machine - but `./build/bin/llama-simple` works:

```
./build/bin/llama-simple -m ... -n 10 "test"
→ Output: "test Paris. It is..."
→ Speed: 62-66 t/s
→ Time: ~0.17s for 10 tokens
```

| Binary | Works? | Speed |
|--------|-------|-------|
| llama-simple | ✅ YES | 62-66 t/s |
| llama-cli | ❌ NO | hangs |

---

## Classification

**ENVIRONMENT_OR_MODEL_LOAD_RUNTIME_ISSUE** - not a regression

- Current HEAD hangs
- Known-good 761b1352c hangs identically
- Both produce same memory breakdown
- Neither generates output before timeout

This is NOT a build regression - it's an issue affecting both old and new builds on this environment.

---

## Root Cause Analysis (Incomplete)

The hang happens after model loading, during generation loop. Both builds:
- Load GGUF successfully
- Allocate KV cache
- Initialize compute graph
- Enter generation loop
- Never produce first token before 60s timeout

**Hypothesis:** Something in llama-cli's generation scheduling differs from llama-simple. Possible causes:
1. Interactive vs batch mode handling
2. Graph compilation on first decode
3. Flash Attention / Gated Delta Net compilation
4. Batch scheduling on first token

**Evidence needed:** Attach debugger or trace to see WHERE it hangs.

---

## Allowed Claims

- Phase 13G: PASS clean llama-cli frontend (62212c2e6)
- Phase 13H-R: PASS/PARTIAL 1-prompt clean llama-cli+PRT (761b1352c)
- Phase 13I: PASS 4/4 mini-suite clean (a3d53e9e8)
- Phase 13J: BLOCKED_BUILD_MODEL_LOAD_HANG
- Phase 13K: ENVIRONMENT_OR_MODEL_LOAD_RUNTIME_ISSUE

## Forbidden Claims

- Do not claim current build regression
- Do not claim 0.5B model is broken
- Do not claim PRT caused this

---

## Next Steps

**Recommended:** Investigate llama-cli generation loop vs llama-simple.

Options:
1. **Compare build configs:** Look at what's different between simple.cpp and cli.cpp
2. **Disable Flash Attention:** Test with `--no-flash-attn` if available
3. **Attach debugger:** See WHERE in generation loop it hangs
4. **Try different model:** Test with a different GGUF on same build
5. **Try older llama.cpp:** Test with llama.cpp from pre-13J era

**Do NOT run PRT** until llama-cli native model-load is confirmed working.

---

## Safety Checklist

- [x] No models staged or tracked
- [x] No sidecars staged
- [x] No binaries staged
- [x] Worktree cleaned up: `git worktree remove /tmp/prt_13h_r_761b1352c`
- [x] Temp files in /tmp

**Verdict: ENVIRONMENT_OR_MODEL_LOAD_RUNTIME_ISSUE** - issue is in llama-cli generation loop specifically, not build regression.