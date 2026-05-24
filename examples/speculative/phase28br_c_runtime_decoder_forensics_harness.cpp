// Phase 28BR-C: Runtime Decoder Integration Forensics
// Classifies where NaN/Inf enters the runtime path in decoded R buffer

#include "prt_trit_decode.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>

static uint16_t crc16_30(const uint8_t * data) {
    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= data[i];
    }
    return crc & 0xFFFF;
}

// Compute CRC16 of entire file
static uint16_t file_crc16(const uint8_t* data, size_t size) {
    uint16_t crc = 0;
    for (size_t i = 0; i < size; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= data[i];
    }
    return crc & 0xFFFF;
}

std::string hex_dump(const uint8_t* data, size_t len) {
    std::ostringstream oss;
    for (size_t i = 0; i < len; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
        if ((i + 1) % 16 == 0) oss << "\n";
        else if ((i + 1) % 4 == 0) oss << " ";
    }
    return oss.str();
}

void scan_buffer(const float* data, size_t n,
                 size_t& nan_count, size_t& inf_count,
                 float& min_val, float& max_val,
                 double& abs_sum) {
    nan_count = inf_count = 0;
    min_val = INFINITY;
    max_val = -INFINITY;
    abs_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        if (std::isnan(v)) nan_count++;
        else if (std::isinf(v)) inf_count++;
        else {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            abs_sum += std::fabs(v);
        }
    }
}

struct trit_hdr {
    uint32_t magic;
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
    static constexpr uint32_t MAGIC = 0x54495254;
};

void print_sample(std::ostream& out, const float* data, size_t n, const char* label) {
    out << label << " samples: ";
    size_t indices[] = {0, 1, 8, 16, 100, 896, 1000};
    for (auto idx : indices) {
        if (idx < n) out << "[" << idx << "]=" << std::showpos << data[idx] << std::noshowpos << " ";
    }
    out << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path-to.trit>\n", argv[0]);
        return 1;
    }

    const char* path = argv[1];
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> buf(fsize);
    if (fread(buf.data(), 1, fsize, f) != (size_t)fsize) {
        fprintf(stderr, "Read failed\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    printf("=== RAW BYTE IDENTITY ===\n");
    printf("File: %s\n", path);
    printf("Size: %ld bytes\n", fsize);

    uint16_t whole_crc = file_crc16(buf.data(), buf.size());
    printf("CRC16(whole): 0x%04X\n", whole_crc);
    printf("First 32 bytes hex: "); fflush(stdout);
    for (int i = 0; i < 32; i++) printf("%02X", buf[i]);
    printf("\n");
    printf("Last 32 bytes hex: "); fflush(stdout);
    for (int i = (int)fsize - 32; i < (int)fsize; i++) printf("%02X", buf[i]);
    printf("\n");

    // Compare with direct CLI xxd
    printf("File size matches expected: %s\n",
           (fsize == 303216) ? "YES" : "NO (expected 303216)");

    printf("\n=== HEADER PARSE ===\n");
    trit_hdr th;
    th.magic = *(uint32_t*)(buf.data() + 0);
    th.ver_major = *(uint16_t*)(buf.data() + 4);
    th.ver_minor = *(uint16_t*)(buf.data() + 6);
    th.rows = *(uint32_t*)(buf.data() + 8);
    th.cols = *(uint32_t*)(buf.data() + 12);
    th.block_rows = *(uint16_t*)(buf.data() + 16);
    th.block_cols = *(uint16_t*)(buf.data() + 18);
    th.n_scales = *(uint16_t*)(buf.data() + 20);
    th.payload_offset = *(uint32_t*)(buf.data() + 22);
    th.scale_offset = *(uint32_t*)(buf.data() + 26);
    th.checksum = *(uint16_t*)(buf.data() + 30);

    printf("magic: 0x%08X (expected 0x%08X) %s\n",
           th.magic, trit_hdr::MAGIC,
           th.magic == trit_hdr::MAGIC ? "OK" : "MISMATCH");
    printf("ver_major: %u (expected 0) %s\n",
           th.ver_major, th.ver_major == 0 ? "OK" : "MISMATCH");
    printf("ver_minor: %u (expected 1) %s\n",
           th.ver_minor, th.ver_minor == 1 ? "OK" : "MISMATCH");
    printf("rows: %u\n", th.rows);
    printf("cols: %u\n", th.cols);
    printf("block_rows: %u\n", th.block_rows);
    printf("block_cols: %u\n", th.block_cols);
    printf("n_scales: %u\n", th.n_scales);
    printf("payload_offset: %u (expected 32) %s\n",
           th.payload_offset, th.payload_offset == 32 ? "OK" : "MISMATCH");
    printf("scale_offset: %u\n", th.scale_offset);

    uint32_t n_blocks = ((th.rows + th.block_rows - 1) / th.block_rows) *
                        ((th.cols + th.block_cols - 1) / th.block_cols);
    printf("n_blocks (computed): %u\n", n_blocks);
    printf("n_scales matches n_blocks: %s\n",
           th.n_scales == n_blocks ? "YES" : "NO");

    uint32_t n_trits = th.rows * th.cols;
    uint32_t packed_bytes = (n_trits * 3 + 7) / 8;
    uint32_t expected_file = 32 + packed_bytes + th.n_scales * 4;
    printf("expected_decoded_bytes: %u (rows*cols*4)\n", n_trits * 4);
    printf("expected_file_size: %u (actual: %ld) %s\n",
           expected_file, fsize,
           expected_file == (uint32_t)fsize ? "OK" : "MISMATCH");

    uint16_t hdr_crc = crc16_30(buf.data());
    printf("header_crc: 0x%04X (stored: 0x%04X) %s\n",
           hdr_crc, th.checksum,
           hdr_crc == th.checksum ? "OK" : "MISMATCH");

    // Reference values from Python prt_trit_io.py
    bool header_matches_ref = true;
    if (th.rows != 896) header_matches_ref = false;
    if (th.cols != 896) header_matches_ref = false;
    if (th.block_rows != 32) header_matches_ref = false;
    if (th.block_cols != 48) header_matches_ref = false;
    if (th.n_scales != 532) header_matches_ref = false;
    if (th.payload_offset != 32) header_matches_ref = false;

    printf("header_matches_python_ref: %s\n", header_matches_ref ? "YES" : "NO");

    printf("\n=== DECODED R SCAN (before any contribution) ===\n");
    prt_trit_decoder decoder;
    prt_decoded_view view = decoder.decode_file(path);

    if (view.is_null) {
        printf("decode_file FAILED: %s\n", view.reason.c_str());
        return 1;
    }

    printf("decode success: rows=%zu cols=%zu\n", view.rows, view.cols);

    size_t nan_count = 0, inf_count = 0;
    float min_val = INFINITY, max_val = -INFINITY;
    double abs_sum = 0.0;
    scan_buffer(view.data, view.rows * view.cols, nan_count, inf_count, min_val, max_val, abs_sum);

    double mean_abs = abs_sum / (double)(view.rows * view.cols);
    printf("nan_count: %zu\n", nan_count);
    printf("inf_count: %zu\n", inf_count);
    printf("finite: %s\n", (nan_count == 0 && inf_count == 0) ? "YES" : "NO");
    printf("min_val: %+f\n", min_val);
    printf("max_val: %+f\n", max_val);
    printf("abs_sum: %f\n", abs_sum);
    printf("mean_abs: %f\n", mean_abs);

    printf("\nFirst 16 values: ");
    for (int i = 0; i < 16; i++) {
        printf("%+f ", view.data[i]);
    }
    printf("\n");

    printf("Last 16 values: ");
    size_t total = view.rows * view.cols;
    for (int i = -16; i < 0; i++) {
        printf("%+f ", view.data[total + i]);
    }
    printf("\n");

    print_sample(std::cout, view.data, total, "key");

    // Check which indices are NaN/Inf
    std::vector<size_t> nan_indices, inf_indices;
    for (size_t i = 0; i < total; i++) {
        if (std::isnan(view.data[i])) nan_indices.push_back(i);
        else if (std::isinf(view.data[i])) inf_indices.push_back(i);
    }
    if (!nan_indices.empty()) {
        printf("NaN indices (first 20): ");
        for (size_t i = 0; i < std::min(size_t(20), nan_indices.size()); i++) {
            printf("%zu ", nan_indices[i]);
        }
        printf("\n");
    }
    if (!inf_indices.empty()) {
        printf("Inf indices (first 20): ");
        for (size_t i = 0; i < std::min(size_t(20), inf_indices.size()); i++) {
            printf("%zu ", inf_indices[i]);
        }
        printf("\n");
    }

    // Now simulate contribution loop
    printf("\n=== CONTRIBUTION LOOP SIMULATION ===\n");
    printf("Simulating synthetic X=I @ R contribution loop...\n");

    // For identity X, Y = R. Check Y for non-finite.
    size_t y_nan_count = 0, y_inf_count = 0;
    double y_abs_sum = 0.0;
    float y_max_abs = 0.0f;

    for (size_t i = 0; i < total; i++) {
        float v = view.data[i];
        if (std::isnan(v)) y_nan_count++;
        else if (std::isinf(v)) y_inf_count++;
        else {
            y_abs_sum += std::fabs(v);
            if (std::fabs(v) > y_max_abs) y_max_abs = std::fabs(v);
        }
    }

    printf("Y nan_count: %zu\n", y_nan_count);
    printf("Y inf_count: %zu\n", y_inf_count);
    printf("Y finite: %s\n", (y_nan_count == 0 && y_inf_count == 0) ? "YES" : "NO");
    printf("Y abs_sum: %f\n", y_abs_sum);
    printf("Y max_abs: %f\n", y_max_abs);

    // Spanning trit analysis
    printf("\n=== SPANNING TRIT ANALYSIS ===\n");
    printf("block_rows=%u block_cols=%u -> n_blocks=%u\n", th.block_rows, th.block_cols, n_blocks);

    // Find which block contains index 896 (first NaN position)
    uint32_t br = 896 / th.cols;
    uint32_t bc = 896 % th.cols;
    uint32_t block_r = 896 / (th.block_rows * th.cols);  // rough
    printf("First NaN at index 896 -> row=%u col=%u\n", br, bc);

    // Check spanning case in the decoder
    printf("bit_off>5 count in loop for block_rows=%u block_cols=%u:\n", th.block_rows, th.block_cols);
    uint64_t spanning_count = 0;
    uint64_t normal_count = 0;
    for (uint32_t rb_start = 0; rb_start < th.rows; rb_start += th.block_rows) {
        for (uint32_t cb_start = 0; cb_start < th.cols; cb_start += th.block_cols) {
            uint32_t br_cur = (rb_start + th.block_rows <= th.rows) ? th.block_rows : th.rows - rb_start;
            uint32_t bc_cur = (cb_start + th.block_cols <= th.cols) ? th.block_cols : th.cols - cb_start;
            for (uint32_t r = 0; r < br_cur; r++) {
                for (uint32_t c = 0; c < bc_cur; c++) {
                    uint32_t global_r = rb_start + r;
                    uint32_t global_c = cb_start + c;
                    uint32_t trit_idx = global_r * th.cols + global_c;
                    uint32_t bit_off = (trit_idx * 3) % 8;
                    if (bit_off > 5) spanning_count++;
                    else normal_count++;
                }
            }
        }
    }
    printf("spanning trits: %lu\n", spanning_count);
    printf("normal trits: %lu\n", normal_count);
    printf("total: %lu (expected %u)\n", spanning_count + normal_count, th.rows * th.cols);

    delete[] view.data;

    printf("\n=== DONE ===\n");
    return 0;
}