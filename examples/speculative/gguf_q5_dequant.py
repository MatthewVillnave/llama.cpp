#!/usr/bin/env python3
"""
GGUF Q5_0 Dequantization Helper for Phase 28P
Dequantizes Q5_0 blocks from GGUF file into float32 rows.
For Qwen2.5 ffn_up tensors: [896, 4864], dtype=Q5_0.
"""
import struct
import sys
import json

def dequantize_q5_0_block(quants_bytes, scale, dmin, start_elem):
    """
    Dequantize a Q5_0 block (32 elements) into float32 values.
    Q5_0 stores 5 bits per element: 4 bits in main quant + 1 high bit from extra.
    Layout per 32 elements:
    - scale: float32 (4 bytes)
    - dmin: float32 (4 bytes)
    - main_quants: 20 bytes (each nibble = 2 elements, packed)
    - extra_bits: 4 bytes (1 bit per element, high bits)
    
    Total: 28 bytes per 32 elements.
    """
    values = []
    
    for elem in range(32):
        # Get the 4-bit main quant from nibbles
        byte_idx = elem // 8
        low_nibble = quants_bytes[byte_idx] & 0x0F
        high_nibble = (quants_bytes[byte_idx] >> 4) & 0x0F
        
        # Lower nibble for even element, upper for odd
        if elem % 2 == 0:
            q = low_nibble
        else:
            q = high_nibble
        
        # Map quant to value
        # Q5_0: 16 levels from -max_q to +max_q
        # scale * (q - 16) / 16 + dmin? Need to check exact formula
        val = scale * (q / 16.0) + dmin
        values.append(val)
    
    return values


def extract_q5_0_slice(data, tensor_offset, slice_rows, slice_cols, ne0, ne1):
    """
    Extract and dequantize a Q5_0 tensor slice.
    ne0 = columns, ne1 = rows (but in GGML tensors ne[0]=cols, ne[1]=rows for 2D)
    For ffn_up: ne0=4864, ne1=896
    """
    rows_per_block = 32
    
    # Bytes per block for Q5_0 = 28
    bytes_per_block = 28
    
    # For rows: group into blocks of 32
    # Each row has ne0 elements = 4864 for ffn_up
    
    result = []
    
    # For the first slice_rows, read rows_per_block groups
    for r in range(min(slice_rows, ne1)):
        block_idx = r // rows_per_block
        elem_in_block = r % rows_per_block
        
        block_start = tensor_offset + block_idx * (bytes_per_block * (ne0 // 32))
        
        # Read scale and dmin
        scale = struct.unpack('<f', data[block_start:block_start+4])[0]
        dmin = struct.unpack('<f', data[block_start+4:block_start+8])[0]
        
        # Q5_0 main quants start at byte 8 (after scale + dmin)
        main_quants_offset = block_start + 8
        n_main_bytes = 20  # 32 elements * 4 bits / 8 bits/byte
        
        row_values = []
        for c in range(min(slice_cols, ne0)):
            elem_in_row_block = c // 32
            elem_in_block = c % 32
            
            block_byte_offset = elem_in_row_block * bytes_per_block
            block_start_in_row = block_start + block_byte_offset
            
            # Scale and dmin for this sub-block
            sub_scale = struct.unpack('<f', data[block_start_in_row:block_start_in_row+4])[0]
            sub_dmin = struct.unpack('<f', data[block_start_in_row+4:block_start_in_row+8])[0]
            
            # Main quant index within sub-block
            main_offset = main_quants_offset + block_byte_offset
            
            if elem_in_block % 2 == 0:
                q = data[main_offset + elem_in_block // 2] & 0x0F
            else:
                q = (data[main_offset + elem_in_block // 2] >> 4) & 0x0F
            
            val = sub_scale * (q / 16.0) + sub_dmin
            row_values.append(val)
        
        result.append(row_values)
    
    return result


def extract_q5_0_slice_v2(data, tensor_offset, slice_rows, slice_cols, ne0, ne1):
    """
    Simpler approach: read row by row using per-row Q5_0 encoding.
    Q5_0 can be per-row or per-block depending on the quantization scheme.
    For now, just do a quick approximation that works.
    """
    rows_per_block = 32
    
    result = []
    for r in range(min(slice_rows, ne1)):
        block_idx = r // rows_per_block
        elem_in_block = r % rows_per_block
        
        # Each row spans ne0 elements = 4864
        # For Q5_0, each row is quantized separately
        # Row size in bytes: ne0 * 5 / 8 + overhead = ceil(4864 * 5 / 8) + 8 = 3040 + 8 = 3048?
        # Actually 4864 * 5 = 24320 bits = 3040 bytes + 8 bytes (scale + min) = 3048
        row_size_bytes = (ne0 * 5 + 7) // 8 + 8
        row_offset = tensor_offset + block_idx * (rows_per_block * row_size_bytes)
        
        scale = struct.unpack('<f', data[row_offset:row_offset+4])[0]
        dmin = struct.unpack('<f', data[row_offset+4:row_offset+8])[0]
        
        quants_start = row_offset + 8
        row_values = []
        for c in range(min(slice_cols, ne0)):
            # Each element takes 5 bits
            bit_pos = c * 5
            byte_pos = quants_start + bit_pos // 8
            bit_offset = bit_pos % 8
            
            if byte_pos + 1 < len(data):
                # Extract 5 bits spanning 2 bytes
                bits_low = data[byte_pos]
                bits_high = data[byte_pos + 1] if byte_pos + 1 < len(data) else 0
                
                # Combine bits
                if bit_offset <= 3:
                    # First part in byte_pos, second crosses to byte_pos+1
                    q = ((bits_low >> bit_offset) & 0x1F)
                else:
                    # Cross-byte
                    shift = bit_offset - 4
                    q = ((bits_low >> shift) & 0x0F) | ((bits_high << (8 - shift)) & 0x10)
                    q = q & 0x1F
            else:
                q = 16  # neutral mid-point
            
            val = scale * (q / 16.0) + dmin
            row_values.append(val)
        
        result.append(row_values)
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: gguf_q5_dequant.py <gguf_path> <layer> <out_json>", file=sys.stderr)
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    layer = int(sys.argv[2])
    out_json = sys.argv[3]
    
    with open(gguf_path, 'rb') as f:
        data = f.read()
    
    # Find tensor info using existing C++ tool output as reference
    # From our earlier run:
    # blk.0.ffn_up.weight type=6(Q5_0) offset=151217920 size=2996224
    # ne0=4864, ne1=896
    
    ne0 = 4864  # cols
    ne1 = 896   # rows
    tensor_type = 6  # Q5_0
    
    # Calculate tensor offset for specified layer
    # Layer offsets: 0=151217920, 1=162101760, ...
    layer_offsets = {
        0: 151217920, 1: 162101760, 2: 172985600, 3: 182745856,
        4: 192463104, 5: 203303936, 6: 213064192, 7: 222781440,
        8: 233622272, 9: 243382528, 10: 253099776, 11: 263940608,
        12: 273700864, 13: 283418112, 14: 294258944, 15: 304019200,
        16: 313736448, 17: 324577280, 18: 334337536, 19: 344054784,
        20: 354895616, 21: 365779456, 22: 376663296, 23: 387547136
    }
    
    tensor_offset = layer_offsets.get(layer, 151217920 + layer * (162101760 - 151217920))
    
    slice_rows = min(512, ne1)
    slice_cols = min(1024, ne0)
    
    # Try extraction
    print(f"Extracting layer {layer}: {slice_rows}x{slice_cols}", file=sys.stderr)
    print(f"Tensor offset: {tensor_offset}", file=sys.stderr)
    
    # For Q5_0, row size = (ne0 * 5 + 7) // 8 + 8 (scale + min)
    row_size = (ne0 * 5 + 7) // 8 + 8
    
    result = []
    rows_per_block = 32
    bytes_per_block = row_size * rows_per_block
    
    for r in range(slice_rows):
        block_idx = r // rows_per_block
        elem_in_block = r % rows_per_block
        
        row_base = tensor_offset + block_idx * bytes_per_block + elem_in_block * row_size
        
        if row_base + 8 > len(data):
            break
        
        scale = struct.unpack('<f', data[row_base:row_base+4])[0]
        dmin = struct.unpack('<f', data[row_base+4:row_base+8])[0]
        
        quants_start = row_base + 8
        row_vals = []
        
        for c in range(slice_cols):
            bit_pos = c * 5
            byte_idx = bit_pos // 8
            bit_off = bit_pos % 8
            
            qp = quants_start + byte_idx
            if qp + 1 < len(data):
                if bit_off <= 3:
                    q = (data[qp] >> bit_off) & 0x1F
                else:
                    q_lo = data[qp] >> (bit_off - 4)
                    q_hi = data[qp + 1] << (12 - bit_off)
                    q = (q_lo | q_hi) & 0x1F
            else:
                q = 16
            
            val = scale * (q / 16.0) + dmin
            row_vals.append(val)
        
        result.append(row_vals)
    
    output = {
        "layer": layer,
        "tensor_type": "Q5_0",
        "shape": [len(result), len(result[0]) if result else 0],
        "slice": result,
        "note": "Approximate Q5_0 dequantization. Verify with phase 28P real-slice baseline."
    }
    
    with open(out_json, 'w') as f:
        json.dump(output, f)
    
    print(f"Written to {out_json}", file=sys.stderr)
    print(f"Shape: {len(result)}x{len(result[0]) if result else 0}", file=sys.stderr)


if __name__ == "__main__":
    main()