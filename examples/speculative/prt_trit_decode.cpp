// Phase 28AY: .trit decode parity bridge
// Ternary decoder: 3 bits/trit, per-block f32 scales, row-major decode

#include "prt_trit_decode.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

// ── CRC16 ────────────────────────────────────────────────────────────────

static uint16_t crc16_30(const uint8_t * data) {
    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= data[i];
    }
    return crc & 0xFFFF;
}

// ── Triton header (must match prt_sidecar_pager.h) ───────────────────────

struct trit_hdr {
    uint32_t magic;        // 0x54524954
    uint16_t ver_major;
    uint16_t ver_minor;
    uint32_t rows;
    uint32_t cols;
    uint16_t block_rows;
    uint16_t block_cols;
    uint16_t n_scales;
    uint32_t payload_offset;
    uint32_t scale_offset;
    uint16_t checksum;
    static constexpr size_t SIZE = 32;
    static constexpr uint32_t MAGIC = 0x54495254;  // "TRIT" as-stored LE u32
    static constexpr uint16_t VMAJ = 0;
    static constexpr uint16_t VMIN = 1;
};

// ── Decoder ───────────────────────────────────────────────────────────────

prt_trit_decoder::prt_trit_decoder() {}
prt_trit_decoder::~prt_trit_decoder() {}

prt_decoded_view prt_trit_decoder::decode_bytes(
        const uint8_t * file_bytes,
        size_t file_size,
        uint32_t rows,
        uint32_t cols,
        uint16_t block_rows,
        uint16_t block_cols,
        uint16_t n_scales,
        const float * scales) {

    prt_decoded_view view;
    view.rows = rows;
    view.cols = cols;
    view.format = prt_data_format::DECODED_F32;

    if (rows == 0 || cols == 0) {
        view.reason = "invalid_dims";
        stats_.decode_errors++;
        return view;
    }

    if (block_rows == 0) {
        view.reason = "invalid_block_rows";
        stats_.decode_errors++;
        return view;
    }

    // Compute expected payload size: 3 bits per trit, packed
    // n_blocks_row = ceil(cols / block_cols) (block_cols used for scale index)
    uint32_t n_block_rows = (rows + block_rows - 1) / block_rows;
    uint32_t n_block_cols = (cols + block_cols - 1) / block_cols;
    uint32_t n_blocks = n_block_rows * n_block_cols;
    size_t n_trits = (size_t)rows * cols;
    size_t packed_bytes = (n_trits * 3 + 7) / 8;

    if (block_cols == 0) {
        view.reason = "invalid_block_cols";
        stats_.decode_errors++;
        return view;
    }

    size_t expected_file = 32 + packed_bytes + (size_t)n_scales * 4;
    if (file_size < expected_file) {
        view.reason = "file_too_small";
        stats_.decode_errors++;
        return view;
    }

    // Allocate decoded buffer
    view.data = new float[n_trits];
    memset(view.data, 0, n_trits * sizeof(float));
    view.is_null = false;

    const uint8_t * payload = file_bytes + 32;

    uint32_t scale_idx = 0;
    for (uint32_t rb_start = 0; rb_start < rows; rb_start += block_rows) {
        for (uint32_t cb_start = 0; cb_start < cols; cb_start += block_cols) {
            uint32_t br = (rb_start + block_rows <= rows) ? block_rows : rows - rb_start;
            uint32_t bc = (cb_start + block_cols <= cols) ? block_cols : cols - cb_start;

            float scale = 1.0f;
            if (scales && scale_idx < n_scales) {
                scale = scales[scale_idx];
            }
            scale_idx++;

            for (uint32_t r = 0; r < br; r++) {
                for (uint32_t c = 0; c < bc; c++) {
                    uint32_t global_r = rb_start + r;
                    uint32_t global_c = cb_start + c;
                    uint32_t trit_idx = global_r * cols + global_c;
                    uint32_t byte_idx = (trit_idx * 3) / 8;
                    uint32_t bit_off   = (trit_idx * 3) % 8;
                    uint8_t bits;
                    if (bit_off > 5) {
                        // Spanning: bit_off is 6 or 7 (trit crosses byte boundary)
                        // HIGH bits: (8 - bit_off) bits from first byte starting at bit_off
                        // LOW  bits: remaining (3 - high_count) bits from second byte starting at bit 0
                        // Canonical spanning order: LOW bits occupy bit positions [0, high_count-1],
                        // HIGH bits occupy bit positions [high_count, 3-1]
                        // Reconstruction: bits = (LOW << high_count) | HIGH
                        uint32_t high_count = 8 - bit_off;    // bits from first byte (HIGH)
                        uint32_t low_count  = 3 - high_count; // bits from second byte (LOW)
                        uint8_t high_part = (payload[byte_idx] >> bit_off) & ((1 << high_count) - 1);
                        uint8_t low_part  = payload[byte_idx + 1] & ((1 << low_count) - 1);
                        bits = (low_part << high_count) | high_part;
                    } else {
                        bits = (payload[byte_idx] >> bit_off) & 0x7;
                    }
                    float val = prt_unpack_trit_3bit(bits);
                    view.data[trit_idx] = val * scale;
                }
            }
        }
    }

    stats_.files_decoded++;
    return view;
}

prt_decoded_view prt_trit_decoder::decode_file(const std::string& path) {
    prt_decoded_view view;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        view.reason = "file_open_failed";
        stats_.decode_errors++;
        return view;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize < (long)trit_hdr::SIZE) {
        fclose(f);
        view.reason = "file_too_small";
        stats_.decode_errors++;
        return view;
    }

    std::vector<uint8_t> buf(fsize);
    if (fread(buf.data(), 1, fsize, f) != (size_t)fsize) {
        fclose(f);
        view.reason = "file_read_failed";
        stats_.decode_errors++;
        return view;
    }
    fclose(f);

    const uint8_t * hdr = buf.data();
    trit_hdr th;
    th.magic = *(uint32_t*)(hdr + 0);
    th.ver_major = *(uint16_t*)(hdr + 4);
    th.ver_minor = *(uint16_t*)(hdr + 6);
    th.rows = *(uint32_t*)(hdr + 8);
    th.cols = *(uint32_t*)(hdr + 12);
    th.block_rows = *(uint16_t*)(hdr + 16);
    th.block_cols = *(uint16_t*)(hdr + 18);
    th.n_scales = *(uint16_t*)(hdr + 20);
    th.payload_offset = *(uint32_t*)(hdr + 22);
    th.scale_offset = *(uint32_t*)(hdr + 26);
    th.checksum = *(uint16_t*)(hdr + 30);

    // Validate magic
    if (th.magic != trit_hdr::MAGIC) {
        view.reason = "bad_magic";
        stats_.decode_errors++;
        return view;
    }

    // Validate version
    if (th.ver_major != trit_hdr::VMAJ || th.ver_minor != trit_hdr::VMIN) {
        view.reason = "bad_version";
        stats_.decode_errors++;
        return view;
    }

    // Validate checksum
    uint16_t computed = crc16_30(hdr);
    if (computed != th.checksum) {
        view.reason = "checksum_fail";
        stats_.checksum_fail++;
        return view;
    }
    stats_.checksum_ok++;

    // Extract scales
    std::vector<float> scales(th.n_scales);
    const float * scales_ptr = nullptr;
    if (th.n_scales > 0 && th.scale_offset > 0 && th.scale_offset < buf.size()) {
        memcpy(scales.data(), hdr + th.scale_offset, th.n_scales * sizeof(float));
        scales_ptr = scales.data();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    view = decode_bytes(buf.data(), buf.size(),
                        th.rows, th.cols,
                        th.block_rows, th.block_cols,
                        th.n_scales, scales_ptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    stats_.last_decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return view;
}