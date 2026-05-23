# Phase 28AZ: Python/C++ Trit Parity

## Verdict
PASS_PHASE28AZ_PYTHON_CPP_TRIT_PARITY

## What Is Proven
- Python `.trit` writer output is accepted by the real C++ `prt_trit_decoder`.
- C++ fixture output is accepted by the Python `.trit` reader.
- Packed 3-bit ternary arrays round-trip exactly across Python and C++ for the covered cases.
- Header offsets are now deterministic across languages: `payload_offset=32`, `scale_offset=32+packed_payload_bytes`.
- Corrupt magic and truncated payload cases reject deterministically in both language paths.

## What Is Not Proven
- This is not residual matmul correctness.
- This does not prove residual overlay quality recovery.
- This does not prove runtime generation correctness.
- Generation must not be attempted yet.

## Claim Boundary
Only canonical .trit packed ternary encoding/decoding and header compatibility are proven. No residual matmul correctness, runtime math correctness, model generation, or quality claims are made.

## Results
- Branch: `experimental/prt-phase19a-alt-sidecar-backed`
- HEAD before: `7fdcfdcb09609011e2ad66c177f6d21f28ef17be`
- HEAD after: `7fdcfdcb09609011e2ad66c177f6d21f28ef17be`
- Tests: 18 total, 18 pass, 0 fail
- Python -> C++ parity: PASS
- C++ -> Python parity: PASS
- C++ 28AY harness: PASS
- Corrupt rejection: PASS

## Edge Coverage
- bit offsets 0 through 7
- spanning bit offsets 6 and 7
- row widths not divisible by 8
- multiple block_rows/block_cols shapes
- all-zero trits
- alternating -1/0/+1 pattern
- deterministic random seeded pattern

## Files Changed
- `examples/speculative/prt_trit_io.py`
- `examples/speculative/prt_trit_cpp_parity_probe.cpp`
- `examples/speculative/phase28az_python_cpp_trit_parity.py`
- `examples/speculative/results/phase28az_python_cpp_trit_parity.json`
- `examples/speculative/PHASE28AZ_PYTHON_CPP_TRIT_PARITY.md`
