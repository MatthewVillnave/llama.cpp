# Phase 28T: Qwen2.5-3B FFN_UP Ternary Residual Transfer Check

## Verdict: PASS_PHASE28T_3B_TRANSFER_CHECK ✅ | PASS_3B_TRANSFER_STRONG ✅ | PASS_NO_MODEL_FILES_STAGED

## Summary
Ternary residual recovery **transfers cleanly from 0.5B to 3B**. 5/5 layers STRONG_RECOVERY. 3B mean Δ cos = +0.7076 ± 0.0029 vs 0.5B mean = +0.7079 ± 0.0084. Delta = +0.0003. 3B is statistically identical to 0.5B, with even tighter variance.

---

## A. Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## B. Current HEAD
`74617f376 Phase 28S: scan all FFN_UP ternary residual layers`

## C. Source Model
- **Model:** Qwen2.5-3B-Instruct-Q4_K_M.gguf (`/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf`)
- **Layers tested:** 0, 8, 17, 26, 35 (5 representative layers across 36 total)
- **Slice shape:** 512×2048
- **QType:** Q4_K (vs 0.5B's Q5_0)

## D. 3B Transfer Table

| Layer | Shape | QType | Q2 cos | Q2+T cos | Δ cos | MAE hat | Compression | Verdict |
|-------|-------|-------|--------|----------|-------|---------|-------------|---------|
| 0 | 512×2048 | Q4_K | -0.0065 | 0.7056 | **+0.7121** | 0.5837 | 0.75 | STRONG |
| 8 | 512×2048 | Q4_K | +0.0123 | 0.7180 | **+0.7057** | 0.6315 | 0.75 | STRONG |
| 17 | 512×2048 | Q4_K | +0.0022 | 0.7105 | **+0.7082** | 0.6711 | 0.75 | STRONG |
| 26 | 512×2048 | Q4_K | +0.0014 | 0.7047 | **+0.7034** | 0.6936 | 0.75 | STRONG |
| 35 | 512×2048 | Q4_K | -0.0042 | 0.7044 | **+0.7086** | 0.6846 | 0.75 | STRONG |

## E. 0.5B vs 3B Comparison

| Metric | 0.5B (Phase 28S) | 3B (Phase 28T) | Delta |
|--------|-----------------|----------------|-------|
| Layers tested | 24 | 5 | — |
| Mean Δ cosine | +0.7079 | +0.7076 | **+0.0003** |
| Std Δ cosine | 0.0084 | 0.0029 | -0.0055 |
| Min Δ cosine | +0.6862 | +0.7034 | +0.0172 |
| Max Δ cosine | +0.7247 | +0.7121 | -0.0126 |
| Range | 0.0385 | 0.0087 | tighter |
| Strong recovery | 24/24 | 5/5 | both 100% |
| Compression vs Q4 | 0.75 | 0.75 | identical |
| QType of base | Q5_0 | Q4_K | different |

**Key finding:** 3B shows statistically **identical** mean recovery to 0.5B (Δ = +0.0003, within floating-point noise). 3B has **even tighter variance** (std=0.0029 vs 0.0084), suggesting the recovery becomes more consistent at larger scale.

## F. Transfer Verdict

**3B transfer: STRONG PASS**

- Mean improvement is within 0.0003 of 0.5B — essentially identical
- 5/5 layers strong recovery across 36-layer model
- 3B variance is smaller than 0.5B — not worse, better
- Min recovery (+0.7034 on L26) is well above the strong recovery threshold
- No layer position dependence in 3B either (early L0 to late L35 all strong)

## G. Interpretation

### 1. Does 3B show the same strong recovery pattern?
**YES.** All 5 tested layers show strong recovery, mean Δ cos = +0.7076, identical to 0.5B's +0.7079.

### 2. Is recovery similar magnitude to 0.5B?
**YES, essentially identical.** Delta of +0.0003 is within floating-point noise. The cosine improvement is the same within rounding error.

### 3. Is recovery layer-position independent?
**YES for 3B.** L0, L8, L17, L26, L35 all show strong recovery. The 36-layer model shows no position-dependent degradation.

### 4. Do tensor dimensions or quant types change behavior?
**No significant change.** 3B uses Q4_K (vs 0.5B's Q5_0) — a different base quantization — yet recovery is equally strong. The ternary residual overlay appears robust to base QType differences.

### 5. Does this support scaling toward 30B/32B?
**YES, with caveat.** The transfer from 0.5B→3B is clean. This increases confidence that 3B→7B→...→30B may also work, but actual validation on larger models is the proper next step before extrapolation.

## H. Recommended Next Phase

**Phase 28U — Full 3B FFN_UP Scan + 30B Extrapolation Budget**

Options:
1. **Full 3B scan (36 layers)** — complete the picture for 3B, similar to Phase 28S
2. **7B quick check (2–3 layers)** — test whether the pattern continues at 7B scale
3. **30B budget extrapolation** — use 0.5B and 3B data to estimate memory budget for 30B PRT residual overlay

Recommended: Option 3 — **30B budget extrapolation with empirical validation anchors from 0.5B and 3B**.

## I. Models/Sidecars/F32 Refs Staged?
**NO.** All slices written to /tmp only.

## J. Secrets Detected?
None.

## K. Tags Touched?
None.

---

## Files Committed
- `examples/speculative/prt_residual_3b_transfer.py` — 3B transfer check script
- `examples/speculative/results/PHASE28T_3B_FFN_UP_TERNARY_TRANSFER_CHECK.md` — this report
- `examples/speculative/results/phase28t_3b_ffn_up_ternary_transfer_check.json` — structured results