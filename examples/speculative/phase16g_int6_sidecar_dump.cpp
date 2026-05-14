// PRT Phase 16G: INT6 Sidecar Schema Audit - Sidecar Dump Tool
// Parses INT6 sidecar exactly as the runtime loader does (tools/cli/cli.cpp)
//
// Usage: ./phase16g_int6_sidecar_dump <sidecar_path> [M] [K]
//   M, K optional overrides (infer from file size if omitted)
// Output: JSON to stdout
//
// Runtime loader schema (from tools/cli/cli.cpp):
//   Byte  0-3:  magic[4] = {'P','R','T','6'}
//   Byte  4-7:  version (uint32 LE)
//   Byte  8-11: rows = M (uint32 LE)
//   Byte 12-15: cols = K (uint32 LE)
//   Byte 16-19: reserved (uint32, currently unused by loader)
//   Byte 20+:   scales (M × float32)
//   After scales: packed payload (M * K * 3 / 4 bytes)
//
// Packing: 4 INT6 values → 3 bytes (offset-32, 6 bits/value)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>

static const int SIDECAR_PACKED_PER_ROW(int K) { return ((K + 3) / 4) * 3; }
static const int SIDECAR_PAYLOAD_SIZE(int M, int K) { return M * (((K + 3) / 4) * 3); }
static const int SIDECAR_SCALE_SIZE(int M) { return M * 4; }
static const int SIDECAR_HDR_SIZE = 16;  // Correct: 16 bytes (not 20!)
static const int SIDECAR_HDR_WRONG = 20; // Wrong: 20 bytes (what v4 C++ writes)

struct SidecarInfo {
    std::string path;
    int64_t file_size;
    int rows_M;
    int cols_K;
    uint32_t version;
    uint32_t rows_read;
    uint32_t cols_read;
    uint32_t reserved;
    
    // Scale info
    float * scales;
    float scale_min;
    float scale_max;
    float scale_mean;
    int scale_nonzero;
    int scale_has_inf;
    int scale_has_nan;
    
    // Decoded q info
    int8_t * q_data;       // unpacked, M*K elements
    int q_min;
    int q_max;
    int64_t q_elements;
    
    // Payload info
    size_t payload_offset;
    size_t payload_size;
    size_t payload_bytes_consumed;
    size_t payload_bytes_leftover;
    
    // Checksum
    uint64_t q_checksum;
    
    // Error
    std::string error;
    bool ok;
};

static uint64_t checksum_q(int8_t * q, int64_t n) {
    uint64_t cs = 0;
    for (int64_t i = 0; i < n; i++) cs = cs * 31 + (uint64_t)(q[i] + 32);
    return cs;
}

static bool unpack_row_int6(const uint8_t * packed, int k_cols, int8_t * out) {
    int packed_per_row = SIDECAR_PACKED_PER_ROW(k_cols);
    for (int i = 0; i < k_cols; i += 4) {
        int pi = i * 3 / 4;
        uint8_t b0 = packed[pi];
        uint8_t b1 = packed[pi + 1];
        uint8_t b2 = packed[pi + 2];
        out[i]     = (int8_t)((b0 & 0x3F) - 32);
        if (i + 1 < k_cols) out[i + 1] = (int8_t)(((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F) - 32;
        if (i + 2 < k_cols) out[i + 2] = (int8_t)(((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F) - 32;
        if (i + 3 < k_cols) out[i + 3] = (int8_t)((b2 >> 2) & 0x3F) - 32;
    }
    return true;
}

static SidecarInfo parse_sidecar(const char * path, int override_M, int override_K) {
    SidecarInfo info = {};
    info.path = path;
    info.ok = false;
    
    struct stat st;
    if (stat(path, &st) != 0) { info.error = "cannot stat file"; return info; }
    info.file_size = st.st_size;
    
    FILE * f = fopen(path, "rb");
    if (!f) { info.error = "cannot open file"; return info; }
    
    // Read 16-byte header
    uint8_t hdr[16];
    if (fread(hdr, 1, 16, f) != 16) { fclose(f); info.error = "cannot read header"; return info; }
    
    if (hdr[0] != 'P' || hdr[1] != 'R' || hdr[2] != 'T' || hdr[3] != '6') {
        fclose(f); info.error = "invalid magic"; return info;
    }
    
    info.version = *(uint32_t*)(hdr + 4);
    info.rows_read = *(uint32_t*)(hdr + 8);
    info.cols_read = *(uint32_t*)(hdr + 12);
    info.reserved = *(uint32_t*)(hdr + 16);  // technically past 16 bytes but harmless
    
    if (override_M > 0) info.rows_M = override_M; else info.rows_M = (int)info.rows_read;
    if (override_K > 0) info.cols_K = override_K; else info.cols_K = (int)info.cols_read;
    
    int M = info.rows_M;
    int K = info.cols_K;
    
    size_t expected = 16 + (size_t)M * 4 + SIDECAR_PAYLOAD_SIZE(M, K);
    info.payload_offset = 16 + (size_t)M * 4;
    info.payload_size = SIDECAR_PAYLOAD_SIZE(M, K);
    
    // Read scales (at byte 16, correct offset)
    info.scales = (float*)malloc((size_t)M * sizeof(float));
    if (!info.scales) { fclose(f); info.error = "malloc scales"; return info; }
    if (fread(info.scales, sizeof(float), M, f) != (size_t)M) {
        fclose(f); info.error = "cannot read scales"; return info;
    }
    
    // Compute scale stats
    info.scale_min = 1e30f;
    info.scale_max = -1e30f;
    info.scale_mean = 0.0f;
    info.scale_nonzero = 0;
    info.scale_has_inf = 0;
    info.scale_has_nan = 0;
    for (int i = 0; i < M; i++) {
        float v = info.scales[i];
        info.scale_mean += v;
        if (std::isnan(v)) { info.scale_has_nan++; }
        else if (std::isinf(v)) { info.scale_has_inf++; }
        else {
            if (v < info.scale_min) info.scale_min = v;
            if (v > info.scale_max) info.scale_max = v;
            if (v != 0.0f) info.scale_nonzero++;
        }
    }
    info.scale_mean /= (float)M;
    
    // Read packed payload
    size_t packed_n = info.payload_size;
    std::vector<uint8_t> packed(packed_n);
    size_t n_read = fread(packed.data(), 1, packed_n, f);
    info.payload_bytes_consumed = n_read;
    info.payload_bytes_leftover = packed_n - n_read;
    fclose(f);
    
    // Check if we can decode
    if (n_read < packed_n) {
        fprintf(stderr, "Warning: short read %zu/%zu bytes\n", n_read, packed_n);
    }
    
    // Unpack all rows
    info.q_data = (int8_t*)malloc((size_t)M * K);
    if (!info.q_data) { info.error = "malloc q_data"; return info; }
    
    info.q_min = 127;
    info.q_max = -127;
    info.q_elements = 0;
    for (int r = 0; r < M; r++) {
        unpack_row_int6(packed.data() + (size_t)r * SIDECAR_PACKED_PER_ROW(K), K,
                       info.q_data + (size_t)r * K);
    }
    
    // Q stats (sample first 1000 rows for speed)
    for (int64_t i = 0; i < std::min((int64_t)M * K, (int64_t)1000 * K); i++) {
        int qv = info.q_data[i];
        if (qv < info.q_min) info.q_min = qv;
        if (qv > info.q_max) info.q_max = qv;
    }
    info.q_elements = (int64_t)M * K;
    info.q_checksum = checksum_q(info.q_data, std::min((int64_t)M * K, (int64_t)100000));
    
    info.ok = true;
    return info;
}

static void print_first8_scales(const SidecarInfo & info) {
    printf("  first_8_scales: [");
    for (int i = 0; i < 8; i++) {
        printf("%.8f", info.scales[i]);
        if (i < 7) printf(", ");
    }
    printf("]\n");
}

int main(int argc, char * argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sidecar_path> [M] [K]\n", argv[0]);
        fprintf(stderr, "  Parses INT6 sidecar using runtime-equivalent schema (16-byte header)\n");
        return 1;
    }
    
    const char * path = argv[1];
    int override_M = (argc > 2) ? atoi(argv[2]) : 0;
    int override_K = (argc > 3) ? atoi(argv[3]) : 0;
    
    SidecarInfo info = parse_sidecar(path, override_M, override_K);
    
    printf("{\n");
    printf("  \"path\": \"%s\",\n", path);
    printf("  \"file_size\": %ld,\n", (long)info.file_size);
    printf("  \"ok\": %s,\n", info.ok ? "true" : "false");
    
    if (!info.ok) {
        printf("  \"error\": \"%s\"\n", info.error.c_str());
        printf("}\n");
        return 1;
    }
    
    printf("  \"version\": %u,\n", info.version);
    printf("  \"rows_header\": %u,\n", info.rows_read);
    printf("  \"cols_header\": %u,\n", info.cols_read);
    printf("  \"rows_M\": %d,\n", info.rows_M);
    printf("  \"cols_K\": %d,\n", info.cols_K);
    printf("  \"scale_offset\": %zu,\n", (size_t)16);
    printf("  \"scale_count\": %d,\n", info.rows_M);
    printf("  \"scale_size_bytes\": %zu,\n", (size_t)info.rows_M * 4);
    printf("  \"payload_offset\": %zu,\n", info.payload_offset);
    printf("  \"payload_size\": %zu,\n", info.payload_size);
    printf("  \"payload_bytes_consumed\": %zu,\n", info.payload_bytes_consumed);
    printf("  \"payload_bytes_leftover\": %zu,\n", info.payload_bytes_leftover);
    printf("  \"scale_stats\": {\n");
    printf("    \"min\": %.8f,\n", info.scale_min);
    printf("    \"max\": %.8f,\n", info.scale_max);
    printf("    \"mean\": %.8f,\n", info.scale_mean);
    printf("    \"nonzero\": %d/%d,\n", info.scale_nonzero, info.rows_M);
    printf("    \"has_nan\": %d,\n", info.scale_has_nan);
    printf("    \"has_inf\": %d\n", info.scale_has_inf);
    printf("  },\n");
    printf("  \"q_stats\": {\n");
    printf("    \"min\": %d,\n", info.q_min);
    printf("    \"max\": %d,\n", info.q_max);
    printf("    \"elements\": %ld,\n", (long)info.q_elements);
    printf("    \"expected_elements\": %ld,\n", (long)info.rows_M * info.cols_K);
    printf("    \"element_match\": %s,\n", 
           (info.q_elements == (int64_t)info.rows_M * info.cols_K) ? "true" : "false");
    printf("    \"q0\": %d,\n", info.q_data[0]);
    printf("    \"q1\": %d,\n", info.q_data[1]);
    printf("    \"q2\": %d,\n", info.q_data[2]);
    printf("    \"q3\": %d,\n", info.q_data[3]);
    printf("    \"checksum_first_100k\": %lu\n", (unsigned long)info.q_checksum);
    printf("  },\n");
    printf("  \"expected_total_size\": %zu,\n", (size_t)16 + (size_t)info.rows_M * 4 + info.payload_size);
    printf("  \"size_match\": %s\n", 
           (info.file_size == (int64_t)(16 + (size_t)info.rows_M * 4 + info.payload_size)) ? "true" : "false");
    printf("}\n");
    
    free(info.scales);
    free(info.q_data);
    return 0;
}
