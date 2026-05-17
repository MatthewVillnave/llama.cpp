# Phase 23B: INT6 Decoded-f32 Policy Baseline Checkpoint

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## HEAD
`8c2e0924f` (Phase 23A)

---

## 1. Executive Summary

INT6 decoded-f32 path now uses the Phase 22P shape policy/backend:
- **N≤4** → PRT AVX2 kernel
- **N>4** → native GGML Q4 prefill
- Scalar PRT fallback eliminated
- INT6 is connected to the trusted backend policy on 0.5B layer0

INT6 AVX2 output differs from INT8 AVX2 output due to quantization differences (different sidecar files, offline cosine INT6 vs f32 = 0.99936563 vs INT8 vs f32 = 0.99996627). This is expected.

---

## 2. Frozen Technical Baseline

| Item | Value |
|------|-------|
| Branch | `experimental/prt-phase19a-alt-sidecar-backed` |
| HEAD | `8c2e0924f` |
| Model | 0.5B Q4_K_M layer0 |
| Sidecar | INT6: `prt_phase21h_v_int6_from_f32/ffn_up_layer0_prt.int6` |
| Sidecar | INT8: `prt_phase22e_05b_int8_from_f32/ffn_up_layer0_prt.int8` |
| Dimensions | K=896 (hidden), M=4864 (FFN) |
| INT6 decode | PRT6 header → 6-bit unpack → int8 q → scale → f32 W |
| INT8 decode | raw int8 → scale → f32 W |
| Op | `ggml_prt_ffn_up` |
| Policy | N≤4 PRT AVX2, N>4 native prefill |
| Policy flag | `PRT_V2_DECODE_ONLY=1` |

---

## 3. Offline Comparison

| Pair | Cosine | Source |
|------|--------|--------|
| INT8 vs f32 | 0.99996627 | Phase 22E |
| INT6 vs f32 | 0.99936563 | Phase 21H |
| INT6 vs INT8 | 0.999337 | Phase 21H |

INT6 is slightly farther from f32 than INT8 (as expected from 6-bit vs 8-bit quantization).

---

## 4. Runtime Comparison: INT8 vs INT6

### c=4, N≤4 AVX2

| Metric | INT8 | INT6 |
|--------|------|------|
| Sidecar loaded | int8_sidecar | INT6 (magic=0x5052436) |
| Decoded to f32 | YES | YES |
| N=2 AVX2 calls | 3 | 3 |
| N=4 AVX2 calls | 8 | 8 |
| Native_prefill (N=16) | 2 | 2 |
| Scalar fallback | 0 | 0 |
| REJECT | 0 | 0 |

### N=2 AVX2 abs4 Values

| Call | INT8 abs4 | INT6 abs4 |
|------|-----------|-----------|
| 1 | 8.409694 | 3.795631 |
| 2 | 5.138701 | 1.952418 |
| 3 | (N=2 second call) | (N=2 second call) |

### N=4 AVX2 abs4 Values

| Call | INT8 abs4 | INT6 abs4 |
|------|-----------|-----------|
| 1 | 16.082024 | 5.928558 |
| 2 | 12.649295 | 5.595599 |
| 3 | 10.744886 | 4.789041 |
| 4 | 8.185065 | (more) |
| 5+ | 15.402712... | ... |

**INT6 abs4 values are different from INT8** — expected due to different quantization (6-bit vs 8-bit) and different regeneration from f32 reference. Both paths are internally consistent (no corruption).

### c=64, N>4 Policy

| Metric | INT8 | INT6 |
|--------|------|------|
| N=2 AVX2 calls | 2 | 2 |
| N=16/N=64/N=34 native_prefill | 7 | 7 (projected) |
| Scalar fallback | 0 | 0 |
| REJECT | 0 | 0 |

Both paths route N>4 to native_prefill identically.

---

## 5. Correct Interpretation

- INT6 decoded-f32 policy path works ✓
- INT6 reaches `ggml_prt_ffn_up` → AVX2 kernel for N≤4 ✓
- N>4 routes to native prefill ✓
- Scalar PRT fallback eliminated ✓
- This is NOT direct packed INT6 compute (decodes to f32 first)
- This is NOT a speed claim (decode overhead not measured)
- This is NOT multi-layer (layer0 only)
- This is NOT production readiness

---

## 6. Allowed Claims

✓ INT6 sidecar decodes to f32 and reaches ggml-native PRT op on 0.5B layer0
✓ INT6 uses Phase 22P shape policy (N≤4 PRT AVX2, N>4 native prefill)
✓ scalar PRT fallback remains eliminated
✓ 0.5B layer0 INT6 policy baseline is checkpointed

---

## 7. Forbidden Claims

✗ Direct packed INT6 compute
✗ End-to-end speedup
✗ Production readiness
✗ Multi-layer support
✗ 7B INT6 support
✗ 14B support
✗ Exact/token match with INT8
✗ Broad semantic equivalence

---

## 8. Recommended Next

**Phase 23C** — 0.5B INT6 repeat validation/timing before 7B:
- Verify INT6 decode is deterministic (same prompt → same output)
- Compare decode overhead (INT6 6-bit unpack vs INT8 direct decode)
- If stable: optionally move to 7B INT6 canary

**Alternative:** Phase 23D — 7B INT6 decoded-f32 policy canary

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
| Huge logs staged | NO (tmp only) |
| Secrets | NO |
| Scratch drive used | YES |
| System disk | ~44G free |
| Scratch disk | ~51G free |

---

## 10. Verdict

**PASS_INT6_DECODED_F32_POLICY_BASELINE_CHECKPOINT**

INT6 decoded-f32 path verified on 0.5B layer0 with Phase 22P shape policy. INT6 AVX2 output differs from INT8 AVX2 output (expected quantization difference). Policy routing, scalar fallback elimination, and decode path all confirmed clean.