#!/usr/bin/env python3
"""Dequantize ffn_up layer 0 using Q4_K."""
import struct
import statistics
import zlib
import sys

QK_K = 256

def dequantize_q4_k(data_bytes, n_elements):
    """Port of ggml dequantize_row_q4_K."""
    n_blocks = n_elements // QK_K
    result = []
    BLOCK_SIZE = 144
    
    for i in range(n_blocks):
        off = i * BLOCK_SIZE
        
        d_val = struct.unpack_from('<e', data_bytes, off)[0]
        dmin_val = struct.unpack_from('<e', data_bytes, off + 2)[0]
        
        scales = data_bytes[off + 4 : off + 20]
        qs = bytearray(data_bytes[off + 20 : off + 148])
        
        qp = 0
        for j in range(QK_K // 64):
            s0 = scales[j * 2]
            s1 = scales[j * 2 + 1]
            
            sc0 = s0 & 0xF
            m0 = (s0 >> 4) & 0xF
            sc1 = s1 & 0xF
            m1 = (s1 >> 4) & 0xF
            
            d1 = d_val * sc0
            m1v = dmin_val * m0
            d2 = d_val * sc1
            m2v = dmin_val * m1
            
            # 32 low nibbles
            for l in range(32):
                result.append(d1 * (qs[qp] & 0xF) - m1v)
                qp += 1
            # 32 high nibbles
            for l in range(32):
                result.append(d2 * (qs[qp] >> 4) - m2v)
                qp += 1
    
    return result

def main():
    model_path = "/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    output_path = "/tmp/ffn_up_layer0_float.bin"
    
    abs_offset = 292392096
    n_elements = 2048 * 11008
    n_bytes = 12681216
    
    print("=== PRT Phase 10A: Dequant Layer 0 ffn_up ===", file=sys.stderr)
    
    with open(model_path, 'rb') as f:
        f.seek(abs_offset)
        raw_bytes = f.read(n_bytes)
    
    print(f"Read {len(raw_bytes)} bytes", file=sys.stderr)
    
    float_data = dequantize_q4_k(raw_bytes, n_elements)
    print(f"Elements: {len(float_data)}", file=sys.stderr)
    
    abs_vals = [abs(x) for x in float_data]
    print(f"min={min(float_data):.4f} max={max(float_data):.4f} mean={statistics.mean(float_data):.4f}", file=sys.stderr)
    print(f"First 20: {[round(x,4) for x in float_data[:20]]}", file=sys.stderr)
    
    float_bytes = struct.pack(f'{len(float_data)}f', *float_data)
    with open(output_path, 'wb') as f:
        f.write(float_bytes)
    
    print(f"Wrote {len(float_bytes)} bytes CRC32={zlib.crc32(float_bytes)}", file=sys.stderr)
    print("SUCCESS", file=sys.stderr)
    return 0

if __name__ == '__main__':
    exit(main())