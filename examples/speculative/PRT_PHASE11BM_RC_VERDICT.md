# PRT Phase 11BM Release-Candidate Verdict

**Branch:** `phase11bm-archived-20260502`
**Date:** 2026-05-02
**Status:** RELEASE-CANDIDATE (NOT PRODUCTION)

---

## Mechanical Status

| Component | Status |
|-----------|--------|
| Route A true graph replacement | ✓ WORKING |
| Custom op integration | ✓ WORKING |
| Native FFN_UP skip for PRT layers | ✓ WORKING |
| Callbacks disabled | ✓ CONFIRMED |
| callback_overwrites counter | ✓ = 0 |
| native_fallback_calls counter | ✓ CORRECT |

**Verdict:** Mechanically sound.

---

## Speed Status

| Mode | Configuration | Observed Speedup |
|------|--------------|-----------------|
| Pure Route A all36 | 36 layers PRT | ~2.0x (FAILS quality) |
| Route A + L12+L15 fallback | 34 layers PRT, 2 native | **~1.85x** |

**Verdict:** Speedup preserved at ~1.85x with L12+L15 fallback.

---

## Quality Status

| Prompt | n | Native | L12+L15 | Match |
|--------|---|--------|---------|-------|
| "Once upon a time in a" | 100 | ✓ | ✓ | **IDENTICAL** |
| "What is the capital of France?" | 50 | ✓ | ✓ | MATCHED |

**Known Failure Recovery:** "Once upon a time in a" — Pure Route A produces "far away land", L12+L15 produces "small village" matching native. **FIXED.**

**Verdict:** Quality preserved on tested prompts.

---

## Resource Limitations

- Test machine: 16GB RAM, 4GB swap (nearly full)
- Ollama runners consume ~4.7GB when running
- n=100 stable after killing Ollama
- JSON/structured prompts SIGKILL at n>30 with memory pressure
- Not a mode failure — resource constraint on test machine

**Verdict:** Hardware limitation, not PRT mode issue.

---

## Production Status

**NOT PRODUCTION READY.**

Allowed designation:
- "release-candidate experimental branch"
- "production-candidate fallback policy"

Forbidden designation:
- "production-ready"
- "universally safe"

---

## Next Steps

1. Run broader suite on machine with more RAM (>32GB)
2. Validate JSON/structured prompts at n=50+
3. Test additional narrative prompts
4. Confirm L13+L14 as alternative fallback pair

---

*End of Phase 11BM RC Verdict*
