# PRT Phase 13D-R: Output Corruption Isolation

**Date:** 2026-05-04 21:30 EDT
**Branch:** `experimental/prt-phase13-model-generalization`
**HEAD:** `f9ace2944`

---

## Environment

| Metric | Value |
|--------|-------|
| Disk | 80% (45GB free) |
| RAM | 10GB available |
| Swap | 2.3GB free |

---

## Key Finding: PRT Binary Produces Garbage Even With PRT Disabled

**Critical discovery:** The `llama-prt-posix` binary produces path-fragment garbage output EVEN WHEN `--prt-mode` is NOT set (PRT nominally "disabled"). This means the corruption is NOT caused by PRT's active path — it exists in the base PRT binary regardless of PRT mode.

This strongly suggests the PRT binary was built with debug callbacks or logging that interferes with generation output in ALL modes.

---

## Test Results

### Test A — Native llama-cli (no PRT hooks)
```
Command: ./build/bin/llama-cli -m ... -p "The capital of France is" -n 20
Output: "The capital of France is Paris."
Result: ✅ CLEAN
```

### Test B — PRT binary, no --prt-mode flag
```
Command: ./build/bin/llama-prt-posix -m ... -p "The capital of France is" -n 20 (NO --prt-mode)
Output: token 19: '-deceond_sidecars/ffn_up_layer0_prt.bin'
Last cosine: 0.999973
Result: ❌ GARBAGE (path fragments even with PRT disabled)
```

### Test C — PRT binary with --prt-mode 5700 (active PRT)
```
Command: ./build/bin/llama-prt-posix -m ... -p "The capital of France is" -n 20 --prt-mode 5700 --prt-force-native 11,15
Output: token 15: ' optionallydecars/ffn_up_layer0_prt.bin'
Result: ❌ GARBAGE (path fragments with PRT active)
```

---

## Interpretation

| Test | Binary | PRT Mode | Output | Verdict |
|------|--------|----------|--------|---------|
| A | llama-cli (upstream build) | N/A | CLEAN | ✅ |
| B | llama-prt-posix (PRT build) | disabled | GARBAGE | ❌ |
| C | llama-prt-posix (PRT build) | active | GARBAGE | ❌ |

**Conclusion: NATIVE_ISSUE in llama-prt-posix binary.** The llama-prt-posix binary has a bug that corrupts output in ALL modes, not just PRT-active modes.

The garbage contains path fragments like `ffn_up_layer0_prt.bin` and `sidecars/ffn_up_layer0_prt.bin` — these are sidecar paths that appear in the binary's debug logging or callback code. This suggests:
1. The PRT binary's callback debug logging is writing file paths into the generation stream OR
2. The binary was built with a debug/stripped binary that has path strings embedded in a way that leaks into output OR
3. Something in the PRT callback infrastructure prints file paths to stdout when it shouldn't

---

## Next Steps

1. **Rebuild llama-prt-posix from clean build** — the current binary may have been built with debug instrumentation that writes to stdout
2. **Check if `llama-cli` binary is the upstream version** (no PRT hooks) vs llama-prt-posix (PRT hooks compiled in)
3. **Audit callback stdout/stderr** — ensure no debug prints go to stdout in non-debug builds
4. **Compare binary builds** — sha256 of clean upstream llama-cli vs PRT binary

---

## Binary Identity

| Binary | SHA256 | Size |
|--------|--------|------|
| llama-cli | `7a87a7ae721361fe44004aae0d4bd5b4a91dcea8dde9c618bbbdf5145828dc88` | 5.7MB |
| llama-prt-posix | `ce959f62e39fdf75f81c7691ff40d12fca9f008eca94b1e3936205233091e6d7` | 41KB |

Note: llama-prt-posix is tiny (41KB) — it's a shim. The actual llama.cpp library (libllama.so) is shared. The shim adds PRT callbacks that are active even without --prt-mode.

---

## Verdict

**NATIVE_ISSUE** — The PRT binary's base state (without --prt-mode) produces garbage. This is a binary/build issue, not a PRT active-path issue.

**Recommended next step:** Rebuild llama-prt-posix after auditing callback debug prints in phase10e0_layer0_replacement.cpp and ensuring no stdout pollution from debug paths in non-verbose builds.