# Phase 22P: AVX2 Shape Policy Checkpoint

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## HEAD
`ace78db8e`

---

## 1. Executive Summary

PRT-v2 now has a shape-aware backend policy:
- **N≤4** → PRT AVX2 kernel (decode/small-N runtime shapes)
- **N>4** → real native `build_lora_mm` (GGML Q4 matmul, prefill/context shapes)
- Slow scalar PRT fallback **eliminated** in tested 0.5B layer0 scenarios
- This is a **backend policy baseline**, not an end-to-end speedup claim

---

## 2. Frozen Technical Baseline

| Item | Value |
|------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | `ace78db8e` |
| Policy env | `PRT_V2_DECODE_ONLY=1` (default ON in PRT-v2 experimental mode) |
| Op | `GGML_OP_PRT_FFN_UP` |
| Backend | `ops.cpp` AVX2 for N≤4; native `build_lora_mm` for N>4 |
| Native fallback | Real llama.cpp Q4 matmul via `build_lora_mm` (not scalar PRT fallback) |
| Layer | layer0 only |
| Sidecar | INT8 decoded-f32 (from `ffn_up_layer0_prt.int8`) |
| Prompt method | File prompts only |
| Scratch | `/media/matthew-villnave/VL_usb/prt_scratch` |
| Routing check | `cur->ne[1]` (n_tokens) in both `llama-graph.cpp` and `prt_graph_replace.h` |

---

## 3. Prior Evidence (Phases 22K–22M)

### N=2 AVX2 Correctness (Phase 22K / 22L)
- Scalar abs4: `8.409693`
- AVX2 abs4: `8.409694` ✓
- ~5x kernel speedup vs scalar
- Validated with file prompts: Hi, capital

### N=4 AVX2 Correctness (Phase 22M)
- Scalar abs4: `16.082027`
- AVX2 abs4: `16.082024` ✓
- 100% AVX2 coverage for c=4 (3 N=2 + 8 N=4 calls)
- Stable across decode steps

### Shape Policy (Phase 22N)
- N=2: single-token decode steps (every generation call)
- N=4: small multi-token decode windows
- N>4: prefill / context fill shapes
- Clear separation enables clean routing

---

## 4. Phase 22O Policy Evidence

### Matrix

| Test | N=2 PRT | N=4 PRT | N≤4 AVX2 | N>4 native_prefill | Scalar fallback | REJECT |
|------|---------|---------|----------|---------------------|-----------------|--------|
| cap c=4 n=1 | 3 | 8 | **11** | 2 (N=16) | **0** | 0 |
| cap c=16 n=1 | 3 | 0 | **3** | 2 (N=16) | **0** | 0 |
| cap c=64 n=1 | 2 | 0 | **2** | 7 (N=34) | **0** | 0 |

### Key Points
- All N≤4 calls route to PRT AVX2 ✓
- All N>4 calls route to native GGML Q4 matmul ✓
- Scalar PRT fallback count = **0** across all tests ✓
- REJECT count = **0** across all tests ✓
- AVX2 N=2 output stable: abs4=8.409694 (matches Phase 22K validated runs) ✓

---

## 5. Correct Interpretation

**This checkpoint means:**
- PRT-v2 has a tested shape-aware routing policy
- N≤4 decode paths use AVX2 kernel (no scalar fallback on those shapes)
- N>4 prefill paths use native GGML Q4 matmul (no scalar PRT fallback)
- Routing decision is made at graph construction time, before tensor allocation

**This checkpoint does NOT mean:**
- End-to-end speedup proven (no measurement run)
- Production readiness
- Multi-layer quality
- All-layer support
- 7B or 14B support
- INT6 routed through this policy (not yet implemented)
- Exact token-match output equivalence

---

## 6. Allowed Claims

✓ PRT-v2 has a tested shape policy
✓ N≤4 routes to PRT AVX2 in tested 0.5B layer0 scenarios
✓ N>4 routes to native GGML Q4 prefill in tested 0.5B layer0 scenarios
✓ Scalar PRT fallback was eliminated in tested 0.5B layer0 scenarios
✓ INT8 decoded-f32 path works with this policy

---

## 7. Forbidden Claims

✗ End-to-end speedup
✗ Production readiness
✗ Multi-layer support
✗ All-layer support
✗ 7B INT6 support
✗ 14B support
✗ Broad semantic equivalence
✗ Exact token match

---

## 8. Recommended Next

**Phase 23A** — Route INT6 decoded-f32 through the same policy/backend:
- INT6 sidecar → decode to f32 W
- N≤4 → PRT AVX2
- N>4 → native prefill
- 0.5B layer0 first, no 7B until 0.5B passes

**Alternative:** Phase 22Q — End-to-end timing comparison for 0.5B INT8 policy path

---

## 9. Safety Scan

| Item | Status |
|------|--------|
| Model files staged | NO |
| Sidecars staged | NO |
| f32 refs staged | NO |
| Captures staged | NO |
| Prompts staged | NO |
| Binaries staged | NO |
| Huge logs staged | NO |
| Secrets | NO |
| Scratch drive used | YES |
| System disk | 44G free |
| Scratch disk | 51G free |

---

## 10. Verdict

**PASS_AVX2_SHAPE_POLICY_BASELINE_CHECKPOINT**

Frozen at `ace78db8e`. Policy wired. Evidence captured. Ready for Phase 23A INT6 routing.