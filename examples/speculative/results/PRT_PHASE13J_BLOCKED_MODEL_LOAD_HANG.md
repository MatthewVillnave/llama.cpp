# PRT Phase 13J: BLOCKED — Model-Load Hang

**Date:** 2026-05-06
**Branch:** `experimental/prt-phase13-model-generalization`
**HEAD:** `5fffcc721` (PRT: document low-disk test IO protocol)
**Verdict:** `BLOCKED_BUILD_MODEL_LOAD_HANG`

---

## What Happened

Phase 13J was attempting a full 8-prompt n=40 quality/timing suite for the Qwen2.5-0.5B model, following the Phase 13I timeout-safe mini-suite success.

After disk cleanup and a fresh `cmake --build` rebuild of llama-cli, the binary failed to generate tokens. The model loaded successfully (verified via verbose-prompt output and llama-bench), but llama-cli's generation loop hung indefinitely, producing only repeated `> ` spinner characters on stderr — never completing even a single token.

## Environment at Time of Block

| Item | Value |
|------|-------|
| Binary | `./build/bin/llama-cli` (rebuilt May 6 10:21 with `-DLLAMA_BUILD_SERVER=ON`) |
| Binary size | 5.7MB |
| Binary timestamp | May 5 16:29 (old) / May 6 10:21 (rebuilt) |
| Model | `/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` |
| Model file size | 398MB |
| Model verified loadable | YES (llama-simple, llama-bench both work) |
| Number of hung processes | 1 (llama-cli, 100% CPU, 0 tokens generated) |
| Native llama-cli model-load | HUNG — generation loop does not progress |
| Active PRT attempted | NO — blocked by model-load hang |
| Any valid quality/timing result from 13J | NO |

## Machine State at Block

- **Disk:** 170GB free (24% used)
- **RAM:** 11GB available, 4.2GB used
- **Swap:** 2.3GB free, 1.7GB used
- **Temp files cleaned:** ~600MB of phase13 output logs removed

## Key Observations

1. **Model is healthy** — `llama-simple` (simple.cpp) generates correctly at ~62-66 t/s on the same model
2. **llama-cli model load works** — the GGUF metadata, tensor loading, KV cache, and compute graph allocation all succeed
3. **llama-cli generation hangs** — the autoregressive decode loop stalls after the prompt eval phase
4. **llama-bench works** — confirms the model+backend work for batched pp/tg benchmarks
5. **Rebuild didn't fix it** — rebuilt with fresh cmake, still hangs
6. **llama-simple vs llama-cli difference** — likely in how batch scheduling, interactive stdin handling, or the REPL loop interacts with the decode graph

## What Phase 13J Is NOT

- **NOT a PRT failure** — no PRT code was active
- **NOT a sidecar failure** — no sidecars were loaded
- **NOT a quality result** — no valid token output was produced
- **NOT a model corruption** — the model file is intact and works in other tools

## Known-Good Ladder (Reference)

| Phase | Commit | Verdict | Notes |
|-------|--------|---------|-------|
| 13G | 62212c2e6 | PASS | Clean llama-cli frontend, PRT flags built |
| 13H | 0bede7146 | INVALID | Used llama-simple, not llama-cli |
| 13H-R | 761b1352c | PASS/PARTIAL | llama-cli active canary, 1/8 prompts clean |
| 13I | a3d53e9e8 | PASS | Timeout-safe 4/4 mini-suite, native+PRT both clean |
| 13J | — | BLOCKED | Model-load hang, no valid results |

## Next Safe Resume Phase: 13K

Before retrying Phase 13J, a model-load isolation test (Phase 13K) must pass:

**Phase 13K Goal:** Prove `./build/bin/llama-cli` can load Qwen2.5-0.5B reliably with zero PRT flags.

Minimal test:
- `n_predict=1`
- prompt: "test"
- stdout → /dev/null
- stderr captured
- timeout wrapper
- no PRT flags, no sidecars

Only after native model-load is confirmed stable should Phase 13J be retried.

## See Also

- `examples/speculative/results/PRT_PHASE13_STATUS_REORIENTATION.md` — full context and safety framing
- `examples/speculative/results/PRT_PHASE13I_TIMEOUT_SAFE_MINISUITE.md` — last clean PRT result
- `examples/speculative/results/PRT_PHASE13H_R_LLAMA_CLI_QUALITY_TIMING.md` — last clean llama-cli result