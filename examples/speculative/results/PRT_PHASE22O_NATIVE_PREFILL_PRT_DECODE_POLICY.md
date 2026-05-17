# Phase 22O: Native Prefill / PRT Decode Policy Implementation

## Branch
`experimental/prt-phase19a-alt-sidecar-backed`

## Previous HEAD
`d6f24ab0d` (Phase 22N: shape policy decision)

## New HEAD
TBD after commit

---

## D. Policy Env/Flag

```
PRT_V2_DECODE_ONLY=1
```

- `PRT_V2_DECODE_ONLY=1`: N<=4 → PRT AVX2, N>4 → native `build_lora_mm`
- `PRT_V2_DECODE_ONLY` not set: original behavior (all N go through PRT)

Implemented via:
- `g_prt_decode_only` global in `llama-graph.cpp`
- Auto-initialized from `PRT_V2_DECODE_ONLY=1` env var
- Checked in two places: ggml_prt_ffn_up path (llama-graph.cpp) and build_prt_ffn_up path (prt_graph_replace.h)

---

## E. Routing Method

**In llama-graph.cpp** (ggml_prt_ffn_up path):
```c
int nt = (int)cur->ne[1];
if (g_prt_decode_only && nt > 4) {
    prt_logf("[PRT_V2_POLICY] decode_only=1 N=%d action=native_prefill layer=%d\n", nt, il);
    tmp = this->build_lora_mm(up, cur);  // native path
} else {
    if (g_prt_decode_only) {
        prt_logf("[PRT_V2_POLICY] decode_only=1 N=%d action=prt layer=%d\n", nt, il);
    }
    // Call ggml_prt_ffn_up (AVX2 kernel)
}
```

**In prt_graph_replace.h** (build_prt_ffn_up path):
```c
int n_tokens = (int)cur->ne[1];
if (g_prt_decode_only && n_tokens > 4) {
    fprintf(stderr, "[PRT_V2_POLICY] decode_only=1 N=%d action=native_prefill layer=%d\n", n_tokens, layer_id);
    return nullptr;  // force native in caller (build_ffn in llama-graph.cpp)
}
```

---

## F. N<=4 Behavior

All N=1, N=2, N=4 → PRT AVX2 kernel via `ggml_prt_ffn_up`

```
[PRT_V2_POLICY] decode_only=1 N=1 action=prt layer=0
[PRT_V2_POLICY] decode_only=1 N=2 action=prt layer=0
[PRT_V2_POLICY] decode_only=1 N=4 action=prt layer=0
[PRT_V2_AVX2] path=N2_TEMP_STORE ...
[PRT_V2_AVX2] path=N4_TEMP_STORE ...
```

---

## G. N>4 Behavior

All N>4 → native `build_lora_mm` (GGML Q4 matmul, no PRT involvement)

```
[PRT_V2_POLICY] decode_only=1 N=16 action=native_prefill layer=0
[PRT_V2_POLICY] decode_only=1 N=34 action=native_prefill layer=0
```

No scalar PRT fallback. No REJECT log. No PRT NUMERIC output for N>4.

---

## H. c=4 Result (Capital, c=4, n=1)

| Metric | Value |
|--------|-------|
| N=1 PRT calls | 4 |
| N=2 PRT calls | 3 |
| N=4 PRT calls | 8 |
| N>4 native_prefill | 2 (N=16) |
| REJECT | 0 |
| AVX2 N=2 abs4 | 8.409694 (stable) |
| AVX2 N=4 abs4 | matches scalar |
| **Result** | **PASS** |

---

## I. c=16 Result (Capital, c=16, n=1)

| Metric | Value |
|--------|-------|
| N=1 PRT calls | 4 |
| N=2 PRT calls | 3 |
| N=4 PRT calls | 0 |
| N>4 native_prefill | 2 (N=16) |
| REJECT | 0 |
| AVX2 N=2 abs4 | 8.409694 (stable) |
| **Result** | **PASS** |

No N=4 in c=16 (expected — short prompt). N=16 prefill → native_prefill ✓.

---

## J. c=64 Result (Capital, c=64, n=1)

| Metric | Value |
|--------|-------|
| N=2 PRT calls | 2 |
| N>4 native_prefill | 7 (N=34, other prefill shapes) |
| REJECT | 0 |
| AVX2 N=2 abs4 | 8.409694, 5.138701 (stable, matches previous runs) |
| **Result** | **PASS** |

N=30/N=34 now routes to native_prefill instead of scalar PRT fallback. N=2 AVX2 output unchanged.

---

## K. Scalar PRT Fallback Count

With `decode_only=1`:
- c=4: 0 scalar PRT fallback
- c=16: 0 scalar PRT fallback
- c=64: 0 scalar PRT fallback

**Scalar PRT fallback eliminated for all tested configurations.**

---

## L. Native Prefill Count

| Test | N>4 native_prefill calls |
|------|--------------------------|
| c=4 | 2 (N=16) |
| c=16 | 2 (N=16) |
| c=64 | 7 (N=34 + others) |

---

## M. AVX2 Count

| Test | AVX2 N=2 | AVX2 N=4 | Total PRT |
|------|----------|----------|----------|
| c=4 | 3 | 8 | 11 |
| c=16 | 3 | 0 | 3 |
| c=64 | 2 | 0 | 2 |

---

## N. Output Validity

AVX2 N=2 output identical with/without decode_only policy:
```
abs4=8.409694 y0=1.955871 y1=-0.076392 y2=-0.011000 y3=1.304417  (decode_only=1)
abs4=8.409694 y0=1.955871 y1=-0.076392 y2=-0.011000 y3=1.304417  (decode_only=0)
```

**Output valid.** No regression in decode path.

---

## O. Optional 7B Result
Not run. 0.5B policy passes cleanly.

---

## P. Verdict

**PASS_NATIVE_PREFILL_PRT_DECODE_POLICY + PASS_N_GT_4_NATIVE_PREFILL + PASS_N_LE_4_PRT_AVX2**

- Policy routing works correctly in both graph paths
- N<=4 uses PRT AVX2 ✓
- N>4 uses native `build_lora_mm` ✓
- Scalar PRT fallback eliminated ✓
- No REJECT for N>4 ✓
- Output valid (no regression) ✓

---

## Q. Recommended Next

**Phase 22P — Checkpoint AVX2 shape policy baseline**

- Document all shapes, backends, and routing decisions
- Verify 7B works with decode_only=1 (optional)
- Then decide: INT6 or further optimization

---

## Safety Checklist

| Item | Status |
|------|--------|
| Models staged | No |
| Sidecars staged | No |
| Credentials | No |
| Tags touched | No |
| System disk free | 44G |
| Scratch disk free | 51G |