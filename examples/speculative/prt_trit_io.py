#!/usr/bin/env python3
"""
PRT Phase 28Z: .trit binary format writer/reader
Offline tool for writing and reading ternary residual overlay files.
Supports synthetic and real tensor slices.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

# --- .trit file format v0 ---
MAGIC = 0x54524954  # b"TRIT" little-endian u32
VERSION_MAJOR = 0
VERSION_MINOR = 1
HEADER_SIZE = 32
TRIT_ENCODING_BITS = 3  # 3 bits per trit, values 0/1/7 mapping to {0,+1,-1}


def pack_trits(trits, rows, cols):
    """Pack ternary values {-1,0,1} into bytes. Returns bytes.
    Uses 3 bits per trit, packed LSB-first, splitting across bytes when needed.
    """
    total = rows * cols
    n_bytes = (total * TRIT_ENCODING_BITS + 7) // 8
    packed = bytearray(n_bytes)
    for i in range(total):
        v = int(trits[i])
        if v == 0:
            bits = 0
        elif v == 1:
            bits = 1
        elif v == -1:
            bits = 7  # 0b111
        else:
            raise ValueError(f"Invalid trit value: {v}")
        bit_pos = i * TRIT_ENCODING_BITS  # absolute bit position in stream
        byte_idx = bit_pos // 8
        bit_off = bit_pos % 8
        # We need to write 3 bits. If bit_off <= 5, all 3 fit in one byte.
        # If bit_off >= 6, they split across two bytes.
        if bit_off <= 5:
            packed[byte_idx] |= bits << bit_off
        else:
            # Split: lower bits of 'bits' go to current byte,
            # upper (3 - (8 - bit_off)) bits spill to next byte
            bits_in_first = 8 - bit_off          # how many bits fit in current byte
            bits_in_second = 3 - bits_in_first    # how many spill to next byte
            packed[byte_idx] |= (bits & ((1 << bits_in_first) - 1)) << bit_off
            packed[byte_idx + 1] |= (bits >> bits_in_first) & ((1 << bits_in_second) - 1)
    return bytes(packed)


def unpack_trits(packed_bytes, rows, cols):
    """Unpack bytes back to ternary array {-1,0,1}."""
    total = rows * cols
    trits = np.zeros(total, dtype=np.int8)
    for i in range(total):
        bit_pos = i * TRIT_ENCODING_BITS
        byte_idx = bit_pos // 8
        bit_off = bit_pos % 8
        if bit_off <= 5:
            bits = (packed_bytes[byte_idx] >> bit_off) & 0x7
        else:
            bits_in_first = 8 - bit_off
            bits_in_second = 3 - bits_in_first
            bits = (packed_bytes[byte_idx] >> bit_off) & ((1 << bits_in_first) - 1)
            bits |= (packed_bytes[byte_idx + 1] & ((1 << bits_in_second) - 1)) << bits_in_first
        if bits == 0:
            trits[i] = 0
        elif bits == 1:
            trits[i] = 1
        elif bits == 7:
            trits[i] = -1
        else:
            trits[i] = 0  # invalid bits map to 0
    return trits.reshape(rows, cols)


def compute_header_crc(header_bytes):
    """Simple CRC16 on header bytes."""
    crc = 0
    for b in header_bytes[:30]:  # skip last 2 bytes (checksum field)
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF
        crc ^= b
    return crc & 0xFFFF


def write_trit(path, ternary, scales, block_rows=512, block_cols=256):
    """
    Write ternary residual + scales to .trit file.

    Args:
        path: output file path
        ternary: 2D np.ndarray of int8 {-1, 0, 1}, shape (rows, cols)
        scales: 1D np.ndarray of float32, shape (n_scales,)
        block_rows: block size for scale grouping (default 512)
        block_cols: block size for scale grouping (default 256)
    """
    rows, cols = ternary.shape
    n_scales = len(scales)

    # Compute packed payload size
    payload_bits = rows * cols * TRIT_ENCODING_BITS
    payload_bytes = (payload_bits + 7) // 8

    # Compute offsets
    scale_offset = HEADER_SIZE + payload_bytes
    # Align to 4 bytes
    if scale_offset % 4:
        scale_offset = (scale_offset // 4 + 1) * 4

    # Pack trits
    packed_payload = pack_trits(ternary.flatten(), rows, cols)

    # Header format: <4sHHIIHHHIIH — 32 bytes total
    # magic(4) + vmaj(2) + vmin(2) + rows(4) + cols(4) + block_rows(2) + block_cols(2)
    #       + n_scales(2,u16) + payload_offset(4) + scale_offset(4) + checksum(2) = 32 bytes
    # n_scales is u16 (max 65535) since typical tensor scales << 65535
    header = struct.pack(
        "<4sHHIIHHHIIH",
        b"TRIT",
        VERSION_MAJOR, VERSION_MINOR,
        rows, cols,
        block_rows, block_cols,
        n_scales,
        HEADER_SIZE + payload_bytes,
        scale_offset,
        0  # checksum placeholder
    )
    # checksum over first 30 bytes
    crc = compute_header_crc(header[:30])
    header = struct.pack(
        "<4sHHIIHHHIIH",
        b"TRIT",
        VERSION_MAJOR, VERSION_MINOR,
        rows, cols,
        block_rows, block_cols,
        n_scales,
        HEADER_SIZE + payload_bytes,
        scale_offset,
        crc
    )

    # Write
    with open(path, "wb") as f:
        f.write(header)
        f.write(packed_payload)
        # Padding to scale_offset
        padding_needed = scale_offset - (HEADER_SIZE + payload_bytes)
        if padding_needed:
            f.write(b"\x00" * padding_needed)
        # Write scales
        f.write(scales.tobytes())

    return {
        "path": path,
        "rows": rows, "cols": cols,
        "block_rows": block_rows, "block_cols": block_cols,
        "payload_bytes": payload_bytes,
        "scale_count": n_scales,
        "scale_offset": scale_offset,
        "total_bytes": scale_offset + n_scales * 4,
        "checksum": crc
    }


def read_trit(path):
    """
    Read .trit file, return (ternary, scales, metadata).

    Returns:
        ternary: 2D np.ndarray int8 {-1,0,1}
        scales: 1D np.ndarray float32
        metadata: dict
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too short: {len(data)} < {HEADER_SIZE}")

    (magic_check, vmaj, vmin, rows, cols, block_rows, block_cols,
     n_scales, payload_offset, scale_offset,
     checksum) = struct.unpack_from("<4sHHIIHHHIIH", data, 0)

    if magic_check != b"TRIT":
        raise ValueError(f"Bad magic: {magic_check!r}")
    if vmaj != VERSION_MAJOR or vmin != VERSION_MINOR:
        raise ValueError(f"Unsupported version: {vmaj}.{vmin}")

    # Verify checksum
    header_for_crc = data[:30]
    expected_crc = compute_header_crc(header_for_crc)
    if checksum != expected_crc:
        raise ValueError(f"Checksum mismatch: {checksum} vs {expected_crc}")

    # Read payload
    payload_bytes = payload_offset - HEADER_SIZE
    packed_payload = data[HEADER_SIZE:payload_offset]

    # Read scales
    scales = np.frombuffer(data[scale_offset:scale_offset + n_scales * 4],
                           dtype=np.float32).copy()

    # Unpack trits
    ternary = unpack_trits(packed_payload, rows, cols)

    metadata = {
        "version": f"{vmaj}.{vmin}",
        "rows": rows, "cols": cols,
        "block_rows": block_rows, "block_cols": block_cols,
        "n_scales": n_scales,
        "payload_bytes": payload_bytes,
        "scale_offset": scale_offset,
        "checksum": checksum,
        "total_bytes": len(data),
        "file_path": path
    }

    return ternary, scales, metadata


def validate_trit(path):
    """Validate .trit file header + checksum without reading full payload."""
    with open(path, "rb") as f:
        data = f.read(HEADER_SIZE)

    if len(data) < HEADER_SIZE:
        return False, "file_too_short"

    magic_check, vmaj, vmin, rows, cols, block_rows, block_cols, \
        n_scales, payload_offset, scale_offset, checksum = struct.unpack_from(
            "<4sHHIIHHHIIH", data, 0)

    if magic_check != b"TRIT":
        return False, f"bad_magic:{magic_check!r}"
    if vmaj != VERSION_MAJOR:
        return False, f"unsupported_version:{vmaj}.{vmin}"

    header_crc = compute_header_crc(data[:30])
    if checksum != header_crc:
        return False, f"checksum_fail:{checksum}vs{header_crc}"

    return True, "valid"


def estimate_trit_bytes(rows, cols, n_scales):
    """Estimate .trit file size without writing."""
    payload_bits = rows * cols * TRIT_ENCODING_BITS
    payload_bytes = (payload_bits + 7) // 8
    scale_offset = HEADER_SIZE + payload_bytes
    if scale_offset % 4:
        scale_offset = (scale_offset // 4 + 1) * 4
    return scale_offset + n_scales * 4


# --- Synthetic helpers ---
def make_synthetic_ternary(rows, cols, seed=123):
    """Create deterministic synthetic ternary array."""
    rng = np.random.RandomState(seed)
    vals = rng.choice([-1, 0, 1], size=rows * cols, p=[0.25, 0.5, 0.25])
    return vals.reshape(rows, cols).astype(np.int8)


def make_synthetic_scales(ternary, block_rows=512, block_cols=256):
    """Compute mean_nonzero_abs scale per block row."""
    rows, cols = ternary.shape
    n_block_rows = (rows + block_rows - 1) // block_rows
    n_block_cols = (cols + block_cols - 1) // block_cols
    n_scales = n_block_rows * n_block_cols
    scales = np.zeros(n_scales, dtype=np.float32)
    idx = 0
    for r in range(0, rows, block_rows):
        br = min(block_rows, rows - r)
        for c in range(0, cols, block_cols):
            bc = min(block_cols, cols - c)
            block = ternary[r:r+br, c:c+bc]
            nonzero = block[block != 0]
            if nonzero.size > 0:
                scales[idx] = float(np.mean(np.abs(nonzero)))
            else:
                scales[idx] = 0.1
            idx += 1
    return scales


# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="PRT .trit writer/reader")
    p.add_argument("--mode", required=True,
                   choices=["write_synthetic", "read", "validate", "roundtrip", "roundtrip_repeat"])
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--cols", type=int, default=2048)
    p.add_argument("--block-rows", type=int, default=512)
    p.add_argument("--block-cols", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--trit-path", default="/tmp/prt_phase28z.trit",
                   help="Path for .trit file")
    p.add_argument("--out-json", default=None,
                   help="Path for JSON report")
    return p.parse_args()


def run_roundtrip(rows, cols, block_rows, block_cols, seed, trit_path, out_json, repeat=False):
    """Write synthetic .trit, read back, verify."""
    if repeat:
        seed = seed  # same seed = same data

    ternary = make_synthetic_ternary(rows, cols, seed)
    scales = make_synthetic_scales(ternary, block_rows, block_cols)

    # Estimate
    est_bytes = estimate_trit_bytes(rows, cols, len(scales))

    # Write
    write_info = write_trit(trit_path, ternary, scales, block_rows, block_cols)

    # Read back
    ternary_out, scales_out, meta = read_trit(trit_path)

    # Validate
    valid, msg = validate_trit(trit_path)
    ternary_match = np.array_equal(ternary, ternary_out)
    scales_match = np.allclose(scales, scales_out, rtol=1e-5, atol=1e-6)
    actual_bytes = write_info["total_bytes"]

    result = {
        "mode": "roundtrip_repeat" if repeat else "roundtrip",
        "seed": seed,
        "shape": [rows, cols],
        "block_size": [block_rows, block_cols],
        "estimated_bytes": est_bytes,
        "actual_bytes": actual_bytes,
        "bytes_match": est_bytes == actual_bytes,
        "ternary_match": bool(ternary_match),
        "scales_match": bool(scales_match),
        "checksum_valid": valid,
        "checksum_msg": msg,
        "validation_pass": valid and ternary_match and scales_match,
        "write_info": write_info,
        "read_meta": meta
    }

    return result


def main():
    args = parse_args()

    if args.mode == "validate":
        valid, msg = validate_trit(args.trit_path)
        status = "PASS" if valid else "FAIL"
        print(f"\nTRIT VALIDATE: {status}")
        print(f"  File: {args.trit_path}")
        print(f"  Result: {msg}")
        result = {"status": status, "file": args.trit_path, "result": msg}

    elif args.mode in ("roundtrip", "roundtrip_repeat"):
        is_repeat = args.mode == "roundtrip_repeat"
        result = run_roundtrip(
            args.rows, args.cols,
            args.block_rows, args.block_cols,
            args.seed, args.trit_path,
            args.out_json,
            repeat=is_repeat
        )

        print(f"\n{'='*60}")
        print(f"TRIT ROUNDTRIP: {'PASS' if result['validation_pass'] else 'FAIL'}")
        print(f"{'='*60}")
        print(f"Shape: {result['shape']}")
        print(f"Block size: {result['block_size']}")
        print(f"Seed: {result['seed']}")
        print(f"Bytes — est: {result['estimated_bytes']}, actual: {result['actual_bytes']} {'✅' if result['bytes_match'] else '❌'}")
        print(f"Ternary match: {'✅' if result['ternary_match'] else '❌'}")
        print(f"Scales match: {'✅' if result['scales_match'] else '❌'}")
        print(f"Checksum: {'✅' if result['checksum_valid'] else '❌'} ({result['checksum_msg']})")
        print(f"File: {args.trit_path}")

    elif args.mode == "write_synthetic":
        ternary = make_synthetic_ternary(args.rows, args.cols, args.seed)
        scales = make_synthetic_scales(ternary, args.block_rows, args.block_cols)
        info = write_trit(args.trit_path, ternary, scales, args.block_rows, args.block_cols)
        print(f"\nWritten: {args.trit_path}")
        print(f"  Shape: {args.rows} x {args.cols}")
        print(f"  Total bytes: {info['total_bytes']}")
        result = info

    elif args.mode == "read":
        ternary, scales, meta = read_trit(args.trit_path)
        print(f"\nRead: {args.trit_path}")
        print(f"  Shape: {ternary.shape}")
        print(f"  Ternary unique: {np.unique(ternary)}")
        print(f"  Scales: {scales[:5]}...")
        print(f"  Meta: {meta}")
        result = {"ternary_shape": list(ternary.shape), "meta": meta}

    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nJSON written to {args.out_json}")


if __name__ == "__main__":
    main()