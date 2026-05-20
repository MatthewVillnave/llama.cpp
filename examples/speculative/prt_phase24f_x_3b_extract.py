#!/usr/bin/env python3
# PRT Phase 24F-X: Extract 3B layer0 FFN_UP to f32
# Adapted from Phase15B-D logic using Python gguf

import gguf
import numpy as np
import os
import sys

MODEL = '/home/matthew-villnave/models/gguf/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf'
OUTPUT_F32 = os.environ.get('OUTPUT_F32', 
    '/media/matthew-villnave/VL_usb/prt_scratch/f32_refs/prt_phase24f_3b_layer0_W_f32.bin')
OUTPUT_INT8 = os.environ.get('OUTPUT_INT8',
    '/media/matthew-villnave/VL_usb/prt_scratch/sidecars/prt_sidecars_3b_int8_phase24f/ffn_up_layer0_prt.int8')

K = 2048
M = 11008

def extract_ffn_up():
    print(f"=== PRT Phase 24F-X: Extract 3B FFN_UP ===")
    print(f"Model: {MODEL}")
    
    reader = gguf.GGUFReader(MODEL)
    
    # Tensor at index 4
    t = reader.get_tensor(4)
    print(f"Tensor: {t.name}, shape: {t.shape}, type: {t.tensor_type}")
    
    # Dequantize Q4_K
    raw = t.data
    f32 = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q4_K)
    
    # Transpose from (M, K) to (K, M)
    f32 = f32.T
    
    print(f"F32 shape: {f32.shape}")
    print(f"Min: {f32.min():.4f}, Max: {f32.max():.4f}, Norm: {np.linalg.norm(f32):.4f}")
    
    # Save f32
    f32.tofile(OUTPUT_F32)
    print(f"Saved f32: {os.path.getsize(OUTPUT_F32)} bytes")
    
    # Generate INT8
    scales = np.max(np.abs(f32), axis=0)
    scales_safe = np.where(scales > 1e-8, scales, 1.0)
    int8_data = np.round(f32 / scales_safe * 127).astype(np.int8)
    
    # Reconstruct for validation
    f32_recon = int8_data.astype(np.float32) * scales_safe / 127.0
    cosine = np.sum(f32 * f32_recon) / (np.linalg.norm(f32) * np.linalg.norm(f32_recon) + 1e-8)
    
    print(f"Cosine: {cosine:.6f}")
    
    # Save INT8
    os.makedirs(os.path.dirname(OUTPUT_INT8), exist_ok=True)
    with open(OUTPUT_INT8, 'wb') as f:
        scales.astype(np.float32).tofile(f)
        int8_data.tofile(f)
    
    print(f"Saved INT8: {os.path.getsize(OUTPUT_INT8)} bytes")
    print("=== DONE ===")

if __name__ == '__main__':
    extract_ffn_up()
