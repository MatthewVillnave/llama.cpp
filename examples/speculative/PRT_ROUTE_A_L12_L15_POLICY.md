# PRT Route A + L12+L15 Native Fallback Policy

**Version:** 1.0
**Date:** 2026-05-02
**Status:** Release-Candidate Experimental Policy

---

## Why L12+L15 Exists

Pure Route A all36 produces incorrect output on specific prompts — most notably `"Once upon a time in a"`.

**Observed behavior:**
- Pure Route A all36 → token 0 = " far" (WRONG)
- Native mode → token 0 = " small village" (CORRECT)

The failure is caused by a nonlinear layer-interaction issue in the L12-L17 range. Single-layer native fallback is insufficient. Pair-wise fallback is required.

**Fix discovered:** Forcing L12 AND L15 to use native build_lora_mm (instead of PRT custom op) restores native output quality.

---

## Why Pure all36 Route A Is Brittle

**Root cause:** Layer interaction dependency

Testing revealed:
- L12 alone → WRONG (" far")
- L15 alone → WRONG (" far")
- L12+L15 → CORRECT (" small")
- L12+L16 → WRONG (" far")
- L13+L14 → CORRECT (" small")
- L14+L17 → CORRECT (" small")
- L12-L17 (all 6) → WRONG (" far")

**Conclusion:** The failure is NOT about "which layers are PRT" in isolation. It is about specific layer-pair combinations producing the correct hidden state trajectory.

This is a state-dependent bug in the PRT approximation — not a simple missing-layer problem.

---

## Native Fallback Layer Policy

**Primary policy:** L12 + L15 force-native

```
--prt-force-native 12,15
```

**Alternative policy:** L13 + L14 (also fixes known failure)

```
--prt-force-native 13,14
```

**Conservative policy:** L0-L5 or L6-L11 (fixes but uses 6 native layers)

```
--prt-force-native 0,1,2,3,4,5
# or
--prt-force-native 6,7,8,9,10,11
```

**Why NOT L0-L5 by default:** Costs 6 native layers (~83% PRT) instead of 2 (~94% PRT). Only use if L12+L15 fails on a prompt.

---

## Expected Counters

For 36-layer model, n=100 tokens:

| Counter | Expected Value |
|---------|--------------|
| callback_overwrites | **0** (disabled) |
| native_fallback_calls | 16 (2 layers × 8 generation steps) |
| prt_true_replacement_calls | ~3700 (34 layers × ~109 calls) |
| native_ffn_up_calls | 0 |
| identity_fallback_calls | 0 |

---

## Forbidden: Callback Overwrite Path

**The callback correction path is DISABLED.**

```
Forbidden: --prt-mode 5605 (or any callback mode)
Allowed:  --prt-mode 5700 (pure Route A + force-native)
```

**Why forbidden:** Callback modes achieve correctness by running native AFTER PRT, then overwriting. This destroys the speed path and is equivalent to "native + some PRT" — not a true PRT speedup.

**Required property:** `callback_overwrites == 0`

---

## Policy Summary

| Property | Value |
|----------|-------|
| PRT mode | 5700 |
| Force-native layers | 12,15 |
| PRT layer coverage | 34/36 (94.4%) |
| Native layer coverage | 2/36 (5.6%) |
| Expected speedup | ~1.85x vs native |
| Quality | Matches native on tested prompts |
| Callbacks | Disabled |
| Overwrite path | Forbidden |

---

*End of Route A L12+L15 Policy*
