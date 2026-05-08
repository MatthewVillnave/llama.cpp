# PRT Public X Thread — Version A (Credibility / Measured Tone)

---

**1/ Lab update: PRT Phase 14 is complete.**

---

**2/ The problem:**
Phase 13 proved the active replacement path worked in llama.cpp — quality preserved, clean outputs. But float32 sidecars were too memory-heavy and throughput was materially below native.

---

**3/ The pivot:**
Phase 14 switched to packed INT8 sidecars — ~4× smaller per layer, less memory traffic, same replacement path.

---

**4/ The result:**
0.5B, 3B, and 7B Qwen2.5 Q4_K_M validated on my CPU setup. All three sizes passed full quality suites. No production claim — this is measured research.

---

**5/ 3B numbers:**
- Native avg: 21.0 t/s
- INT8 PRT avg: 21.0 t/s
- Ratio: **1.000×** (10 independent runs)
- Quality: 8/8 semantic matches, 0 degradations

Native parity, repeatably.

---

**6/ 7B numbers:**
- Native avg: 8.75 t/s | INT8 PRT avg: 8.69 t/s
- Ratio: **0.993× average**, 0.989× median (full 8-prompt suite)
- Quality: 8/8 exact or semantic matches, 0 degradations
- Sidecars: 28/28 loaded cleanly

Near-native, stable, validated.

---

**7/ Claim boundary:**
❌ Not production
❌ Not universal speedup
❌ Not GPU comparison
❌ Not larger than 7B
✅ Measured CPU research checkpoint on Qwen2.5 Q4_K_M.

---

**8/ Core insight:**
The idea was not dead. The representation was too heavy.

Float32 sidecars failed as a speed solution. Packed INT8 sidecars made the replacement path viable.

---

**9/ Next work:**
- Longer-context / larger-n stability
- INT4 sidecar prototype
- Native ggml/backend integration
- Optimization toward native-beating throughput
- Broader benchmark suites

Active research path. More to come.

---

---

# PRT Public X Thread — Version B (Punchier)

---

**1/ PRT Phase 14 done.**

---

**2/ Float32 sidecars killed the idea.**
Too heavy. Memory traffic dominated. Active replacement worked — but it was too slow.

---

**3/ INT8 changed the story.**
Same replacement logic. Smaller sidecars. Less traffic. Near-native throughput.

---

**4/ The bridge worked. It was too heavy.**
We didn't abandon the idea. We changed the material.

---

**5/ 3B:** 21.0 t/s native → 21.0 t/s INT8 PRT. 1.000×. Repeatable. 8/8 semantic. 0 degradations.
**7B:** 8.75 → 8.69 t/s. 0.993× avg. Full 8-prompt validation. 8/8 matches. 0 degradations.

---

**6/ Not production. Not universal. Not GPU. Measured setup only — Qwen2.5 Q4_K_M on consumer CPU.**

---

**7/ The idea survived measurement. The float32 representation didn't.**

---

**8/ Next: longer contexts, INT4 sidecars, native backend integration.**
Research continues.

---

# Approved short claim (for copy-paste)

> "Packed INT8 sidecars let PRT preserve tested output quality while recovering near-native llama.cpp throughput on Qwen2.5-3B and 7B in my measured CPU setup."
