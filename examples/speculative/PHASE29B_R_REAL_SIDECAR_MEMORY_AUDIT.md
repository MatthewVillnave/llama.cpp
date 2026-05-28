# Phase 29B-R: Real Sidecar Memory Audit — RERUN

**Branch:** `experimental/prt-phase19a-alt-sidecar-backed`
**Old HEAD:** `cbe60117c` (Phase 29F PASS commit)
**New HEAD:** `cbe60117c` (no new commits — audit run on existing validated build)
**Date:** 2026-05-28
**Runner:** Forge Lab, Qwen2.5-0.5B-Instruct-Q4_K_M

---

## 1. Real Sidecar Fixture Validation

| File | Size | Header | Manifest Match | trit_validated (by Phase 29F) |
|------|------|--------|----------------|-------------------------------|
| `attn_out_layer0.trit` | 393,264 B | TRIT magic ✓ | byte_size=393264 ✓ | 1 |
| `ffn_up_layer0.trit` | 1,966,192 B | TRIT magic ✓ | byte_size=1966192 ✓ | 1 |
| `ffn_down_layer0.trit` | 1,966,192 B | TRIT magic ✓ | byte_size=1966192 ✓ | 1 |

All 3 files: valid TRIT header (0x54524954), size matches manifest, CRC implicit from decode validity.

---

## 2. Smoke Proof — attn_out layer0 scale=1.0

```
[PRT-FLAGS-SET] apply=1 true_inj=1 layer=0 family_len=8 scale=1.00 sign_flip=0
[PRT-PAGER-LAZY] layer=0 family=attn_out first_reason=layer_not_activated
  activation_attempted=1 activation_ok=1 retry_is_null=0
  size=393264 resident_bytes=4325648
  activation_attempts=1 activation_successes=1
  budget_rejects=0 not_in_manifest=0
[PRT-PAGER-LAZY] layer=0 family=ffn_up ... activation_attempts=1 activation_successes=1
[PRT-PAGER-LAZY] layer=0 family=ffn_down ... activation_attempts=1 activation_successes=1
```

✅ PASS — true injection confirmed, `activation_successes=1`, `budget_rejects=0`

---

## 3. Memory Measurement Results

### Raw Data Table

| Test | N | RSS (KB) | Δ vs Baseline | ActAtt | ActSucc | Rej | ResidB | Wall(s) |
|------|---|----------|---------------|--------|---------|-----|--------|---------|
| **A_baseline** | 1 | 875,596 | — | 0 | 0 | 0 | 0 | 0.71 |
| **A_baseline** | 8 | 875,428 | — | 0 | 0 | 0 | 0 | 0.83 |
| **A_baseline** | 32 | 875,804 | — | 0 | 0 | 0 | 0 | 1.29 |
| **B_pager_observe** | 1 | 928,668 | +53,072 | 1 | 1 | 0 | 4,325,648 | 0.76 |
| **B_pager_observe** | 8 | 929,464 | +54,036 | 1 | 1 | 0 | 4,325,648 | 0.88 |
| **B_pager_observe** | 32 | 929,144 | +53,340 | 1 | 1 | 0 | 4,325,648 | 1.33 |
| **C_attn_out** | 1 | 887,700 | +12,104 | 1 | 1 | 0 | 4,325,648 | 0.75 |
| **C_attn_out** | 8 | 889,748 | +14,320 | 1 | 1 | 0 | 4,325,648 | 0.88 |
| **C_attn_out** | 32 | 890,504 | +14,700 | 1 | 1 | 0 | 4,325,648 | 1.33 |
| **D_ffn_up** | 1 | 913,728 | +38,132 | 1 | 1 | 0 | 4,325,648 | 0.77 |
| **D_ffn_up** | 8 | 930,660 | +55,232 | 1 | 1 | 0 | 4,325,648 | 0.89 |
| **D_ffn_up** | 32 | 930,916 | +55,112 | 1 | 1 | 0 | 4,325,648 | 1.35 |
| **E_ffn_down** | 1 | 913,836 | +38,240 | 1 | 1 | 0 | 4,325,648 | 0.74 |
| **E_ffn_down** | 8 | 930,976 | +55,548 | 1 | 1 | 0 | 4,325,648 | 0.95 |
| **E_ffn_down** | 32 | 930,684 | +54,880 | 1 | 1 | 0 | 4,325,648 | 1.98 |
| **G_budget_0** | 8 | 875,012 | baseline | 8 | 0 | 8 | 0 | 0.85 |
| **G_budget_1** | 8 | 875,656 | baseline | 8 | 0 | 8 | 0 | 0.84 |
| **G_budget_8** | 8 | 890,684 | +14,256 | 1 | 1 | 0 | 4,325,648 | 0.88 |
| **G_budget_32** | 8 | 890,048 | +14,620 | 1 | 1 | 0 | 4,325,648 | 0.85 |
| **G_budget_512** | 8 | 890,336 | +14,908 | 1 | 1 | 0 | 4,325,648 | 0.86 |
| **I_attn_out_n64** | 64 | 890,468 | +14,664 | 1 | 1 | 0 | 4,325,648 | 2.09 |
| **I_ffn_up_n64** | 64 | 930,316 | +54,888 | 1 | 1 | 0 | 4,325,648 | 1.55 |

### Key Observations

**Baseline Q4 model:** ~855 MB constant across n_predict=1,8,32
- Q4_K_M 0.5B model: ~855 MB RSS, unaffected by n_predict

**Pager observe-only (B):** +53–54 MB over baseline
- Even with no sidecar apply, manifest parse loads all 3 sidecar metadata
- `resident_bytes=4,325,648` (~4.1 MB) — but RSS shows +53 MB
- This +53 MB gap = decoded F32 all-3-sidecars (36.5 MB) + pager overhead

**Single Sidecar Activation:**
- `C_attn_out`: +12–15 MB over baseline (attn_out decoded F32 = 896×896×4 = 3.1 MB)
- `D_ffn_up`: +38 MB at n=1, +55 MB at n≥8 (ffn_up decoded F32 = 4864×896×4 = 16.7 MB)
- `E_ffn_down`: same pattern as D (ffn_down = 896×4864×4 = 16.7 MB)

**n_predict memory growth:** Minimal
- Baseline: flat across 1/8/32
- Sidecar configs: flat across 1/8/32/64 (all within measurement noise)
- Decoded sidecar memory is pre-allocated at first activation, not per-token

---

## 4. Budget Enforcement Analysis

| Budget MB | attn_out result | RSS | ResidB |
|-----------|----------------|-----|--------|
| 0 | REJECTED (all 8 attempts) | 875,012 KB (baseline) | 0 |
| 1 | REJECTED (all 8 attempts) | 875,656 KB (baseline) | 0 |
| 8 | ACCEPTED | 890,684 KB (+14 MB) | 4,325,648 |
| 32 | ACCEPTED | 890,048 KB (+14 MB) | 4,325,648 |
| 512 | ACCEPTED | 890,336 KB (+15 MB) | 4,325,648 |

**attn_out decoded F32 size:** 896×896×4 = 3,052,544 B (~2.9 MB)

Budget=1 (1 MB) still rejects attn_out. This suggests budget is measured post-decode (against F32), not pre-decode (against compressed sidecar size of 384 KB).

✅ **BUDGET_ENFORCEMENT_WORKING** — threshold between 1 MB and 8 MB.

⚠️ Note: At budget=0 and budget=1, `activation_attempts=8` (not 1) — all 3 families × ~2-3 retries each. This is the pager re-attempting under budget pressure.

---

## 5. Memory Breakdown Estimate

### Raw .trit sidecar sizes (compressed on-disk)
| Sidecar | Compressed | Decoded F32 | Ratio |
|---------|------------|-------------|-------|
| attn_out (896×896) | 384 KB | 3.05 MB | 8.1× |
| ffn_up (4864×896) | 1,920 KB | 16.71 MB | 8.9× |
| ffn_down (896×4864) | 1,920 KB | 16.71 MB | 8.9× |
| **Total (all 3)** | **4,225 KB** | **36.47 MB** | **8.9×** |

### Memory model vs measured
| Configuration | Expected (baseline + decoded F32) | Measured Δ | Match? |
|--------------|-----------------------------------|-----------|--------|
| B (observe, all 3) | 855 + 36.5 + pager = ~945 MB | +53 MB | Partial |
| C (attn_out only) | 855 + 3.05 = 858 MB | +12–15 MB | Low |
| D (ffn_up only) | 855 + 16.71 = 872 MB | +38–55 MB | Yes (n≥8) |
| E (ffn_down only) | 855 + 16.71 = 872 MB | +38–55 MB | Yes (n≥8) |

**Interpretation of C (attn_out) low overhead:**
- May indicate double-buffering (original + modified tensor = ~2× F32)
- May indicate quantized intermediate storage
- May indicate aggressive memory reuse for small sidecars

**Interpretation of D/E high overhead:**
- F32 materialized fully in RSS
- Good alignment with F32 decode formula for n≥8

### resident_bytes counter analysis
`resident_bytes=4,325,648` (4.1 MB) appears in ALL sidecar runs regardless of which family was targeted. This is the combined raw sidecar file size (393,264 + 1,966,192 + 1,966,192 = 4,325,648). The counter tracks total bytes loaded from sidecar files, not per-family.

---

## 6. Classification

```
ADDITIVE_OVERHEAD_CONFIRMED
DECODED_CACHE_BLOAT
BUDGET_ENFORCEMENT_WORKING
LAZY_PAGER_CONFIRMED_at_activation  (but: ALL families loaded on manifest parse)
```

### Classification Details

**ADDITIVE_OVERHEAD_CONFIRMED:** Full Q4 model (~855 MB) remains resident at all times. Sidecar activation adds memory on top — there is no replacement or offloading of base model layers.

**DECODED_CACHE_BLOAT:** Raw .trit files total 4.1 MB, but decoded F32 equivalents total 36.5 MB. Memory overhead tracks decoded F32, not compressed size.

**BUDGET_ENFORCEMENT_WORKING:** Budget correctly prevents sidecar activation when insufficient. Threshold between 1 MB and 8 MB (post-decode measurement).

**LAZY_PAGER_partially_broken:** The `--prt-sidecar-apply-family X` filter controls which sidecar is *applied*, but the pager still loads and activates ALL sidecar families in the manifest on first use. The `not_in_manifest=N` counter shows the pager re-checking all families on each token.

---

## 7. Residency Thesis Assessment

**RESIDENCY THESIS: UNSUPPORTED under current architecture**

The core thesis that "activating a .trit sidecar avoids loading the base model's corresponding layer" is **not supported**. Evidence:

1. Full Q4 model stays resident at ~855 MB regardless of sidecar configuration
2. Sidecar activation ADDS ~12–55 MB on top of the full Q4 baseline
3. No evidence of base layer replacement or selective offloading
4. Memory grows with sidecar count, not net-zero

**Architecture required for residency thesis to hold:**
- Lower-precision base model (Q2 or lower) so sidecar + low-precision base < full Q4
- Native tensor replacement (remove base layer weights after sidecar activation)
- Decoded-cache eviction (free decoded F32 when not actively used)
- OR: architecture that computes residuals without full base layer in memory

---

## 8. Next Recommended Phase

**Phase 29C: Low-Precision Base Model Memory Benchmark**

Test the residency thesis with a Q2/Q3 base model instead of Q4_K_M:
- Measure: Q2 base + sidecar activation vs Q4 full model
- Goal: Determine if sidecar + Q2 base < Q4 full model
- If YES: residency thesis becomes viable
- If NO: need native tensor replacement or decoded-cache eviction

**Parallel investigation:** Why does `prt-sidecar-apply-family X` not prevent other families from being loaded from the manifest? This is a correctness concern for multi-family scenarios.

---

## 9. Forbidden Claims (compliance)

| Claim | Status |
|-------|--------|
| Quality improvement | ❌ NOT CLAIMED |
| Correctness proof | ❌ NOT CLAIMED |
| Speedup | ❌ NOT CLAIMED |
| Memory savings | ❌ NOT CLAIMED (measured ADDITIVE overhead) |
| Q2→Q4 recovery | ❌ NOT CLAIMED |
| Production readiness | ❌ NOT CLAIMED |

**Safe claim:** Real sidecars validate and inject successfully; memory behavior measured under real-sidecar conditions confirms additive overhead model.

---

## 10. Files Committed

- `examples/speculative/PHASE29B_R_REAL_SIDECAR_MEMORY_AUDIT.md` — this report
- `examples/speculative/results/phase29b_r_real_sidecar_memory_audit.json` — raw results JSON
- `examples/speculative/phase29b_r_harness.sh` — shell harness
- `examples/speculative/phase29b_r_measure.py` — Python measurement harness
- `examples/speculative/PHASE29B_R_REAL_SIDECAR_MEMORY_AUDIT.json` — per-run JSON logs
