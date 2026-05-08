# PRT Phase 14P — Full 8-Prompt 7B INT8 Validation

## Verdict

**PASS_7B_INT8_FULL_VALIDATION** ✅

## Context

- Phase 14K: 7B INT8 single prompt canary — PASS
- Phase 14L: 7B INT8 4-prompt validation — PASS
- Phase 14M: 7B INT8 repeatability (10 runs) — PASS
- Phase 14N: 7B INT8 repeatability checkpoint tagged
- Phase 14O: Decision to run full 8-prompt 7B validation before optimization or publication
- **This phase**: Full 8-prompt suite against Qwen2.5-7B native and INT8 PRT

---

## Quality Table

| # | Prompt | Native Output | INT8 Output | Match | Quality |
|---|--------|--------------|-------------|-------|---------|
| 1 | "The capital of France is" | "The capital of France is Paris." | "The capital of France is Paris." | ✅ exact | ✅ |
| 2 | "Write a Python function that reverses a list." | "Certainly! Below is a Python function that reverses a list. This function uses slicing..." | "Certainly! Below is a Python function that reverses a list. This function uses slicing..." | ✅ semantic | ✅ code plausible |
| 3 | "Once upon a time in a" | "Once upon a time in a far-off land, there was a kingdom known for its lush forests, sparkling rivers, and kind-hearted people. The kingdom was ruled by a wise king and his queen," | "Once upon a time in a far-off land, there was a kingdom known for its lush forests, sparkling rivers, and kind-hearted people. The kingdom was ruled by a wise king and his queen," | ✅ exact | ✅ |
| 4 | "Explain CPU inference in one sentence." | "CPU inference refers to the process of using a CPU to perform the prediction or decision-making tasks of a machine learning model." | "CPU inference refers to the process of using a CPU to perform the prediction or decision-making tasks of a machine learning model." | ✅ exact | ✅ |
| 5 | "Return JSON with keys name and status." | Valid JSON `{"name": "Example", "status": "Active"}` with markdown wrapping | Valid JSON `{"name": "Example", "status": "Active"}` with markdown wrapping | ✅ exact | ✅ JSON valid |
| 6 | "The fastest way to sort a list in Python is" | "The fastest way to sort a list in Python, in terms of average-case time complexity, is to use the `sorted()` function or the `.sort()` method, which both utilize the Timsort" | "The fastest way to sort a list in Python, in terms of average-case time complexity, is to use the `sorted()` function or the `.sort()` method, which both utilize the Timsort" | ✅ exact | ✅ code plausible |
| 7 | "In two sentences, explain what RAM does." | "RAM (Random Access Memory) temporarily stores data that is being used by the computer's processor so that information can be accessed quickly. It allows the computer to run applications and access files swiftly by providing a" | "RAM (Random Access Memory) temporarily stores data that is being used by the computer's processor so that information can be accessed quickly. It allows the computer to run applications and access files swiftly by providing a" | ✅ exact | ✅ |
| 8 | "Complete this phrase: artificial intelligence is" | "artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction." | "artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction." | ✅ exact | ✅ |

**Quality summary**: 8/8 exact or semantic matches, 0 quality degradations, JSON valid on prompt 5, code plausible on prompts 2 and 6.

---

## Evidence

| Check | Result |
|-------|--------|
| PRT_SHAPE_DETAIL | n_layer=28, hidden=3584, ffn=18944, format=int8, 28/28 layers |
| Sidecars loaded | 28/28 (all 8 INT8 runs) |
| INT8 format evidence | `[PRT] Loaded 28/28 sidecars` in all 8 INT8 runs |
| Fallback behavior | Layers 11 and 15 (force-native); all other 26 layers use INT8 sidecars |
| Path/debug contamination | Stderr contains `[PRT]` messages only — not in output text |
| No collapse/repetition | 0 instances across all 8 prompts |

---

## Timing Table

| # | Native t/s | INT8 t/s | Ratio (INT8/Nat) |
|---|------------|----------|-----------------|
| 1 | 9.6 | 9.5 | 0.990 |
| 2 | 8.6 | 8.7 | 1.012 |
| 3 | 8.7 | 8.7 | 1.000 |
| 4 | 8.9 | 8.8 | 0.989 |
| 5 | 8.6 | 8.6 | 1.000 |
| 6 | 8.6 | 8.5 | 0.988 |
| 7 | 8.5 | 8.3 | 0.976 |
| 8 | 8.5 | 8.4 | 0.988 |

**Native avg**: 8.75 t/s | **Native median**: 8.60 t/s | **Native min/max**: 8.5 / 9.6 t/s  
**INT8 avg**: 8.69 t/s | **INT8 median**: 8.55 t/s | **INT8 min/max**: 8.3 / 9.5 t/s  
**Avg ratio**: 0.993 | **Median ratio**: 0.989 | **Min ratio**: 0.976 | **Max ratio**: 1.012

All 8 ratios **≥ 0.90** ✅ — pass criterion met on every prompt individually.

---

## Interpretation

**Q: Does 7B INT8 preserve quality across the full 8 prompts?**  
**A**: Yes. 8/8 exact or semantic matches, 0 quality degradations. All special cases (JSON, code) produced valid/plausible output.

**Q: Does 7B INT8 retain at least 90% native throughput?**  
**A**: Yes. Every individual ratio is above 0.97×. The average is 0.993× and the median is 0.989×. This is stronger than the Phase 14M repeatability pass, confirming the result holds across the full prompt suite.

**Q: Is the 7B claim now symmetric with 3B?**  
**A**: Yes. 3B was validated with 8 prompts. 7B is now also validated with 8 prompts, both at near-native or better throughput and with 0 quality degradations. The 7B result now has the same credibility structure as the 3B result.

**Q: What claim is now allowed?**  
**A**: See "Allowed claims" below. The 7B INT8 full 8-prompt validation result completes the quality story for this model size, matching the 3B structure.

---

## Allowed Claims

- Qwen2.5-7B INT8 PRT passed full 8-prompt validation with 8/8 semantic matches and 0 quality degradations on this measured CPU setup.
- Qwen2.5-7B INT8 PRT retained ~99.3% average and ~98.9% median native throughput across the full 8-prompt suite on this setup.
- Sidecars loaded 28/28 and fallback was limited to force-native layers 11 and 15.
- The 7B result is now symmetric with the 3B INT8 validation in structure and strength.

## Forbidden Claims

- ❌ No universal speedup claim.
- ❌ No production readiness claim.
- ❌ No larger-than-7B extrapolation.
- ❌ No GPU comparison.
- ❌ No exact equivalence beyond tested prompts/tasks.
- ❌ No claim outside this measured CPU setup.

---

## Recommended Next Phase

**Phase 14Q**: Tag/freeze full 7B INT8 validation checkpoint (`PRT_PHASE14Q_7B_INT8_FULL_VALIDATION_CHECKPOINT`).

The full 8-prompt validation completes the 7B quality story. The next step is to tag this result and proceed to either:
- **Path A**: Lab writeup package (Phase 14Q version)
- **Path B**: INT4 sidecar prototype

The tag and writeup are the most valuable immediate next steps given the completeness of this validation.

## Safety

| Check | Status |
|-------|--------|
| Models staged? | NO |
| Sidecars staged? | NO |
| Binaries staged? | NO |
| Temp logs staged? | NO |
| Secrets detected? | NO |
| Existing tags touched? | NO |