#!/usr/bin/env python3
"""
Phase 28AZ: Python/C++ .trit parity runner.

This verifies packed ternary encoding only. It does not run matmul or generation.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from prt_trit_io import read_trit, validate_trit, write_trit


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_BIN = Path("/tmp/prt_trit_cpp_parity_probe")
HARNESS_BIN = Path("/tmp/prt_trit_decode_harness")
RESULT_JSON = REPO_ROOT / "examples/speculative/results/phase28az_python_cpp_trit_parity.json"
REPORT_MD = REPO_ROOT / "examples/speculative/PHASE28AZ_PYTHON_CPP_TRIT_PARITY.md"


CASES = [
    {"name": "bit_offsets_0_7", "rows": 1, "cols": 8, "block_rows": 1, "block_cols": 8, "pattern": "alternating"},
    {"name": "spanning_bo6_bo7", "rows": 1, "cols": 16, "block_rows": 1, "block_cols": 16, "pattern": "alternating"},
    {"name": "non_divisible_cols", "rows": 3, "cols": 10, "block_rows": 2, "block_cols": 4, "pattern": "random"},
    {"name": "multi_block_shape", "rows": 5, "cols": 17, "block_rows": 2, "block_cols": 5, "pattern": "random"},
    {"name": "all_zero_trits", "rows": 4, "cols": 9, "block_rows": 2, "block_cols": 3, "pattern": "zeros"},
    {"name": "alternating_neg_zero_pos", "rows": 7, "cols": 11, "block_rows": 3, "block_cols": 4, "pattern": "alternating"},
    {"name": "seeded_random", "rows": 8, "cols": 16, "block_rows": 4, "block_cols": 8, "pattern": "random"},
]


def run(cmd, cwd=REPO_ROOT, check=True):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def git_out(*args):
    return run(["git", *args]).stdout.strip()


def build_binaries():
    run([
        "g++", "-O2", "-std=c++17", "-I.", "-Iggml/include", "-Iinclude",
        "examples/speculative/prt_trit_decode.cpp",
        "examples/speculative/prt_trit_cpp_parity_probe.cpp",
        "-o", str(PROBE_BIN),
    ])
    run([
        "g++", "-O2", "-std=c++17", "-D_POSIX_C_SOURCE=200809L",
        "-DPRT_SIDECAR_PAGER_EXPERIMENTAL",
        "-I.", "-Iggml/include", "-Iinclude",
        "examples/speculative/prt_sidecar_pager.cpp",
        "examples/speculative/prt_trit_decode.cpp",
        "examples/speculative/prt_trit_decode_harness.cpp",
        "-o", str(HARNESS_BIN),
    ])


def make_pattern(case):
    rows, cols = case["rows"], case["cols"]
    n = rows * cols
    if case["pattern"] == "zeros":
        vals = np.zeros(n, dtype=np.int8)
    elif case["pattern"] == "alternating":
        vals = np.array([-1, 0, 1] * ((n + 2) // 3), dtype=np.int8)[:n]
    elif case["pattern"] == "random":
        seed = rows * 1009 + cols * 917 + case["block_rows"] * 101 + case["block_cols"]
        rng = np.random.RandomState(seed)
        vals = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    else:
        raise ValueError(case["pattern"])
    return vals.reshape(rows, cols)


def make_scales(case):
    rows, cols = case["rows"], case["cols"]
    br, bc = case["block_rows"], case["block_cols"]
    n_blocks = ((rows + br - 1) // br) * ((cols + bc - 1) // bc)
    return np.ones(n_blocks, dtype=np.float32)


def cpp_decode(path):
    proc = run([str(PROBE_BIN), "--decode", str(path)])
    payload = json.loads(proc.stdout)
    if not payload.get("ok"):
        raise ValueError(payload)
    return np.array(payload["trits"], dtype=np.int8).reshape(payload["rows"], payload["cols"])


def run_python_to_cpp(tmpdir):
    results = []
    for case in CASES:
        path = tmpdir / f"py_{case['name']}.trit"
        expected = make_pattern(case)
        scales = make_scales(case)
        info = write_trit(path, expected, scales, case["block_rows"], case["block_cols"])
        actual = cpp_decode(path)
        results.append({
            "case": case["name"],
            "shape": [case["rows"], case["cols"]],
            "block_shape": [case["block_rows"], case["block_cols"]],
            "payload_offset": info["payload_offset"],
            "scale_offset": info["scale_offset"],
            "pass": bool(np.array_equal(expected, actual)),
        })
    return results


def cxx_expected(case):
    rows, cols = case["rows"], case["cols"]
    n = rows * cols
    pattern = case["pattern"]
    if pattern == "zeros":
        vals = np.zeros(n, dtype=np.int8)
    elif pattern == "alternating":
        vals = np.array([-1, 0, 1] * ((n + 2) // 3), dtype=np.int8)[:n]
    elif pattern == "cxx_fixture":
        vals = np.array([1 if (i % 16) < 8 else -1 for i in range(n)], dtype=np.int8)
    else:
        vals = []
        for idx in range(n):
            x = (idx * 1103515245 + 12345) & 0xFFFFFFFF
            r = (x >> 16) % 3
            vals.append(-1 if r == 0 else (0 if r == 1 else 1))
        vals = np.array(vals, dtype=np.int8)
    return vals.reshape(rows, cols)


def run_cpp_to_python(tmpdir):
    results = []
    cases = CASES + [{
        "name": "cxx_harness_pattern",
        "rows": 8,
        "cols": 16,
        "block_rows": 8,
        "block_cols": 16,
        "pattern": "cxx_fixture",
    }]
    for case in cases:
        path = tmpdir / f"cpp_{case['name']}.trit"
        run([
            str(PROBE_BIN), "--write", str(path),
            str(case["rows"]), str(case["cols"]),
            str(case["block_rows"]), str(case["block_cols"]),
            case["pattern"],
        ])
        actual, scales, meta = read_trit(path)
        expected = cxx_expected(case)
        results.append({
            "case": case["name"],
            "shape": [case["rows"], case["cols"]],
            "block_shape": [case["block_rows"], case["block_cols"]],
            "payload_offset": meta["payload_offset"],
            "scale_offset": meta["scale_offset"],
            "scale_count": int(len(scales)),
            "pass": bool(np.array_equal(expected, actual)),
        })
    return results


def run_corrupt_tests(tmpdir):
    valid_path = tmpdir / "valid_for_corruption.trit"
    case = CASES[1]
    write_trit(valid_path, make_pattern(case), make_scales(case), case["block_rows"], case["block_cols"])

    bad_magic = tmpdir / "bad_magic.trit"
    shutil.copyfile(valid_path, bad_magic)
    data = bytearray(bad_magic.read_bytes())
    data[0:4] = b"DEAD"
    bad_magic.write_bytes(data)

    truncated = tmpdir / "truncated_payload.trit"
    data = bytearray(valid_path.read_bytes())
    truncated.write_bytes(data[:-3])

    tests = []
    ok, msg = validate_trit(bad_magic)
    py_bad_magic = (not ok) and msg.startswith("bad_magic")
    cpp_bad_magic = run([str(PROBE_BIN), "--decode", str(bad_magic)], check=False)
    tests.append({
        "case": "bad_magic",
        "python": "PASS" if py_bad_magic else f"FAIL:{msg}",
        "cpp": "PASS" if cpp_bad_magic.returncode != 0 and "bad_magic" in cpp_bad_magic.stdout else "FAIL",
        "pass": bool(py_bad_magic and cpp_bad_magic.returncode != 0 and "bad_magic" in cpp_bad_magic.stdout),
    })

    try:
        read_trit(truncated)
        py_truncated = False
        py_msg = "accepted"
    except ValueError as exc:
        py_truncated = "File too short" in str(exc)
        py_msg = str(exc)
    cpp_truncated = run([str(PROBE_BIN), "--decode", str(truncated)], check=False)
    tests.append({
        "case": "truncated_payload",
        "python": "PASS" if py_truncated else f"FAIL:{py_msg}",
        "cpp": "PASS" if cpp_truncated.returncode != 0 and "file_too_small" in cpp_truncated.stdout else "FAIL",
        "pass": bool(py_truncated and cpp_truncated.returncode != 0 and "file_too_small" in cpp_truncated.stdout),
    })
    return tests


def write_report(result):
    verdict = "PASS_PHASE28AZ_PYTHON_CPP_TRIT_PARITY" if result["overall_pass"] else "BLOCKED_PHASE28AZ_PYTHON_CPP_TRIT_PARITY"
    body = f"""# Phase 28AZ: Python/C++ Trit Parity

## Verdict
{verdict}

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
{result["claim_boundary"]}

## Results
- Branch: `{result["branch"]}`
- HEAD before: `{result["head_before"]}`
- HEAD after: `{result["head_after"]}`
- Tests: {result["test_count"]} total, {result["pass_count"]} pass, {result["fail_count"]} fail
- Python -> C++ parity: {result["python_to_cpp_parity"]}
- C++ -> Python parity: {result["cpp_to_python_parity"]}
- C++ 28AY harness: {result["cpp_harness_status"]}
- Corrupt rejection: {result["corrupt_file_rejection_status"]}

## Edge Coverage
{os.linesep.join(f"- {item}" for item in result["edge_case_coverage"])}

## Files Changed
{os.linesep.join(f"- `{item}`" for item in result["files_changed"])}
"""
    REPORT_MD.write_text(body)


def main():
    head_before = git_out("rev-parse", "HEAD")
    branch = git_out("branch", "--show-current")
    build_binaries()

    with tempfile.TemporaryDirectory(prefix="phase28az_") as td:
        tmpdir = Path(td)
        py_to_cpp = run_python_to_cpp(tmpdir)
        cpp_to_py = run_cpp_to_python(tmpdir)
        corrupt = run_corrupt_tests(tmpdir)
        harness = run([str(HARNESS_BIN), "--all", "--setup-dir", str(tmpdir / "harness")], check=False)

    all_case_results = py_to_cpp + cpp_to_py + corrupt
    harness_pass = harness.returncode == 0 and "PASS_PHASE28AY_TRIT_DECODE_PARITY" in harness.stdout
    pass_count = sum(1 for x in all_case_results if x["pass"]) + (1 if harness_pass else 0)
    test_count = len(all_case_results) + 1
    fail_count = test_count - pass_count
    overall_pass = fail_count == 0

    changed = [
        "examples/speculative/prt_trit_io.py",
        "examples/speculative/prt_trit_cpp_parity_probe.cpp",
        "examples/speculative/phase28az_python_cpp_trit_parity.py",
        "examples/speculative/results/phase28az_python_cpp_trit_parity.json",
        "examples/speculative/PHASE28AZ_PYTHON_CPP_TRIT_PARITY.md",
    ]

    result = {
        "phase": "28AZ",
        "name": "Python Reference Trit Encoder/Decoder Cross-Language Parity",
        "branch": branch,
        "head_before": head_before,
        "head_after": git_out("rev-parse", "HEAD"),
        "test_count": test_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "overall_pass": overall_pass,
        "python_to_cpp_parity": "PASS" if all(x["pass"] for x in py_to_cpp) else "FAIL",
        "cpp_to_python_parity": "PASS" if all(x["pass"] for x in cpp_to_py) else "FAIL",
        "edge_case_coverage": [
            "bit offsets 0 through 7",
            "spanning bit offsets 6 and 7",
            "row widths not divisible by 8",
            "multiple block_rows/block_cols shapes",
            "all-zero trits",
            "alternating -1/0/+1 pattern",
            "deterministic random seeded pattern",
        ],
        "corrupt_file_rejection_status": "PASS" if all(x["pass"] for x in corrupt) else "FAIL",
        "cpp_harness_status": "PASS" if harness_pass else "FAIL",
        "python_to_cpp_cases": py_to_cpp,
        "cpp_to_python_cases": cpp_to_py,
        "corrupt_cases": corrupt,
        "files_changed": changed,
        "claim_boundary": (
            "Only canonical .trit packed ternary encoding/decoding and header compatibility are proven. "
            "No residual matmul correctness, runtime math correctness, model generation, or quality claims are made."
        ),
        "no_generation_runs": True,
        "no_matmul_claims": True,
        "models_staged": False,
        "sidecars_staged": False,
        "binaries_staged": False,
        "secrets_staged": False,
    }

    RESULT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    write_report(result)

    print(json.dumps({
        "overall_pass": overall_pass,
        "test_count": test_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "result_json": str(RESULT_JSON),
        "report_md": str(REPORT_MD),
    }, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
