# Claims Allowed: Phase 11BM

**Version:** 1.0
**Date:** 2026-05-02
**Scope:** Route A + L12+L15 Fallback Policy

---

## Verified Claims (Allowed)

### Core
- Route A + L12+L15 is the current release-candidate branch
- Uses PRT mode 5700 with `--prt-force-native 12,15`
- 34/36 layers use PRT (94.4% coverage)
- 2/36 layers use native fallback (5.6% coverage)

### Speed
- Tested prompts show ~1.8–1.85x speedup vs native
- Speedup confirmed at n=100 on "Once upon a time in a"
- Speedup confirmed at n=50 on factual prompts
- Speedup persists across narrative, code, and factual prompt types

### Quality
- n=100 known-failure prompt "Once upon a time in a" completed successfully
- Output matches native byte-for-byte on tested known-failure prompt
- First 8 tokens identical to native on known failure
- Token-level comparison confirms correctness

### Architecture
- Callbacks disabled (--prt-mode 5700, not 5605)
- callback_overwrites = 0
- No overwrite/correction cheat path
- Pure Route A path — PRT replaces native FFN_UP

### Counters
- native_fallback_calls correctly counts L12+L15 native usage
- prt_true_replacement_calls shows ~3700 calls at n=100
- Sidecar checksums match across runs (L0: -354.098145, L35: 39.010246)

### Robustness
- No collapse/repetition observed in tested prompts
- No output degradation vs native on tested prompts
- Fallback policy stable across multiple runs

---

## Allowed Phrasing

Use these terms:
- "release-candidate experimental branch"
- "production-candidate fallback policy"
- "Route A with L12+L15 native anchors"
- "approximately 1.85x speedup"
- "validated on known failure prompts"
- "tested on narrative, code, and factual prompts"

---

*End of Allowed Claims*
