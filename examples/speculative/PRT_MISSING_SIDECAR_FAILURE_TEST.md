# PRT Missing Sidecar Failure Test

**Date:** 2026-05-02
**Purpose:** Document the intentional failure test for missing sidecars

---

## Test Scenario

**Premise:** A required PRT sidecar is accidentally missing from `/tmp/prt_sidecars/`

**Expected behavior (before patch):** Silent fallback to native, `native_fallback_calls` incremented, generation continues with wrong quality assumptions

**Expected behavior (after patch):** Fatal startup error, clear message, exit code 1

---

## Test Commands

### Test: Missing L5 sidecar

```bash
# Backup L5
mv /tmp/prt_sidecars/ffn_up_layer5_prt.bin /tmp/prt_sidecars/ffn_up_layer5_prt.bin.bak

# Run (should fail)
./build/bin/llama-prt-posix \
  -m models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  -p "Once upon a time in a" -n 10 \
  --prt-mode 5700 --prt-force-native "12,15"

echo "Exit code: $?"  # Should be 1

# Restore
mv /tmp/prt_sidecars/ffn_up_layer5_prt.bin.bak /tmp/prt_sidecars/ffn_up_layer5_prt.bin
```

### Expected Output

```
[PRT] Loaded 35/36 sidecars
[PRT-11BP] === Sidecar Validation ===
[PRT-11BP] PRT mode=5700, Route A active
[PRT-ERROR] Layer 5: required PRT sidecar MISSING
[PRT-11BP] Required PRT layers: 34
[PRT-11BP] Force-native layers: 2 (skipped — allowed via --prt-force-native)
[PRT-ERROR] FATAL: 1 required sidecar(s) missing. Exiting.
[PRT-ERROR] Use --prt-force-native to mark layers as native if sidecar is unavailable.
Exit code: 1
```

---

## What the Patch Validates

### 1. Layer-Specific Error Messages

```
[PRT-ERROR] Layer 5: required PRT sidecar MISSING
```

The error tells you exactly which layer is missing — not just "sidecar missing."

### 2. Force-Native Exclusion

L12 and L15 sidecars are **not required** when using `--prt-force-native 12,15`:

```
[PRT-11BP] Force-native layers: 2 (skipped — allowed via --prt-force-native)
```

This means you can safely run with L12+L15 native fallback even if those sidecars are missing — they won't be used anyway.

### 3. Help Message

```
[PRT-ERROR] Use --prt-force-native to mark layers as native if sidecar is unavailable.
```

Tells the user how to recover if a sidecar is genuinely unavailable.

---

## Sidecar Count Validation

The validation also confirms the count of required vs loaded sidecars:

```
[PRT-11BP] Required PRT layers: 34
[PRT-11BP] Force-native layers: 2 (skipped — allowed via --prt-force-native)
```

- 36 total layers
- 2 force-native (L12, L15) → not required
- 34 required PRT layers → must all be present

If any of the 34 required sidecars is missing → FATAL.

---

## Test Coverage

| Missing Layer | Force-Native? | Expected Result |
|-------------|--------------|----------------|
| L5 | No | FATAL — required layer missing |
| L12 | Yes | NO ERROR — force-native, sidecar unused |
| L15 | Yes | NO ERROR — force-native, sidecar unused |
| L0 | No | FATAL — required layer missing |
| L35 | No | FATAL — required layer missing |

---

## Reproduction Log

```
=== TEST 2: Missing L5 sidecar (not force-native) → should FATAL ===
[PRT-11BP] === Sidecar Validation ===
[PRT-11BP] PRT mode=5700, Route A active
[PRT-ERROR] Layer 5: required PRT sidecar MISSING
[PRT-11BP] Required PRT layers: 34
[PRT-11BP] Force-native layers: 2 (skipped — allowed via --prt-force-native)
[PRT-ERROR] FATAL: 1 required sidecar(s) missing. Exiting.
[PRT-ERROR] Use --prt-force-native to mark layers as native if sidecar is unavailable.

Exit code: 1
```

---

## Post-Restore Verification

After restoring the sidecar:

```
[PRT-11BP] === Sidecar Validation ===
[PRT-11BP] PRT mode=5700, Route A active
[PRT-11BP] Required PRT layers: 34
[PRT-11BP] Force-native layers: 2 (skipped — allowed via --prt-force-native)
[PRT-11BP] === Sidecar Checksums ===
[PRT-11BP] L0:  -354.098145
[PRT-11BP] L12: -292.085388
[PRT-11BP] L15: -105.420593
[PRT-11BP] L35: 39.010246
[PRT-11BP] VALIDATION PASSED — all required sidecars present
[11BD] callback_overwrites: 0
[11BD] prt_true_replacement_calls: 476
[11BD] native_fallback_calls: 16
```

Generation resumes normally.

---

*End of Missing Sidecar Failure Test*
