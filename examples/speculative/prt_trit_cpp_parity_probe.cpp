// Phase 28AZ: C++ probe for Python/C++ .trit parity
// Build:
//   g++ -O2 -std=c++17 -I. -Iggml/include -Iinclude \
//       examples/speculative/prt_trit_decode.cpp \
//       examples/speculative/prt_trit_cpp_parity_probe.cpp \
//       -o /tmp/prt_trit_cpp_parity_probe

#include "prt_trit_decode.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint16_t crc16_30(const uint8_t * data) {
    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= data[i];
    }
    return crc & 0xFFFF;
}

static uint8_t bits_for_trit(int v) {
    if (v > 0) return 0x01;
    if (v < 0) return 0x07;
    return 0x00;
}

static int trit_for_index(size_t idx, const std::string& pattern) {
    if (pattern == "zeros") return 0;
    if (pattern == "alternating") {
        static const int vals[3] = {-1, 0, 1};
        return vals[idx % 3];
    }
    if (pattern == "cxx_fixture") return (idx % 16) < 8 ? 1 : -1;

    // Deterministic non-cryptographic pattern, intentionally dependency-free.
    uint32_t x = (uint32_t)idx * 1103515245u + 12345u;
    int r = (int)((x >> 16) % 3);
    return r == 0 ? -1 : (r == 1 ? 0 : 1);
}

static bool write_fixture(const std::string& path,
                          uint32_t rows,
                          uint32_t cols,
                          uint16_t block_rows,
                          uint16_t block_cols,
                          const std::string& pattern) {
    const size_t n_trits = (size_t) rows * cols;
    const size_t packed_bytes = (n_trits * 3 + 7) / 8;
    const size_t scale_offset = 32 + packed_bytes;
    const size_t total = scale_offset + 4;
    std::vector<uint8_t> buf(total, 0);

    buf[0] = 0x54; buf[1] = 0x52; buf[2] = 0x49; buf[3] = 0x54;
    uint16_t vmaj = 0, vmin = 1, ns = 1;
    uint32_t po = 32, so = (uint32_t) scale_offset;
    memcpy(&buf[4], &vmaj, 2);
    memcpy(&buf[6], &vmin, 2);
    memcpy(&buf[8], &rows, 4);
    memcpy(&buf[12], &cols, 4);
    memcpy(&buf[16], &block_rows, 2);
    memcpy(&buf[18], &block_cols, 2);
    memcpy(&buf[20], &ns, 2);
    memcpy(&buf[22], &po, 4);
    memcpy(&buf[26], &so, 4);

    uint16_t crc = crc16_30(buf.data());
    memcpy(&buf[30], &crc, 2);

    for (size_t idx = 0; idx < n_trits; idx++) {
        uint8_t bits = bits_for_trit(trit_for_index(idx, pattern));
        uint32_t byte_idx = (idx * 3) / 8;
        uint32_t bit_off = (idx * 3) % 8;
        if (bit_off > 5) {
            uint32_t high_count = 8 - bit_off;
            uint32_t low_count = 3 - high_count;
            uint8_t high_part = bits & ((1 << high_count) - 1);
            uint8_t low_part = (bits >> high_count) & ((1 << low_count) - 1);
            buf[32 + byte_idx] |= high_part << bit_off;
            buf[32 + byte_idx + 1] |= low_part;
        } else {
            buf[32 + byte_idx] |= bits << bit_off;
        }
    }

    float scale = 1.0f;
    memcpy(&buf[scale_offset], &scale, 4);

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t written = fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
    return written == buf.size();
}

static int decode_to_csv(const std::string& path) {
    prt_trit_decoder dec;
    prt_decoded_view view = dec.decode_file(path);
    if (view.is_null) {
        printf("{\"ok\":false,\"reason\":\"%s\"}\n", view.reason.c_str());
        return 2;
    }

    printf("{\"ok\":true,\"rows\":%zu,\"cols\":%zu,\"trits\":[", view.rows, view.cols);
    for (size_t i = 0; i < view.rows * view.cols; i++) {
        int t = 0;
        if (view.data[i] > 0.5f) t = 1;
        else if (view.data[i] < -0.5f) t = -1;
        if (i) printf(",");
        printf("%d", t);
    }
    printf("]}\n");
    delete[] view.data;
    return 0;
}

static void usage(const char * prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --decode PATH\n", prog);
    fprintf(stderr, "  %s --write PATH ROWS COLS BLOCK_ROWS BLOCK_COLS PATTERN\n", prog);
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--decode") == 0) {
        return decode_to_csv(argv[2]);
    }
    if (argc == 8 && strcmp(argv[1], "--write") == 0) {
        bool ok = write_fixture(argv[2],
                                (uint32_t) strtoul(argv[3], nullptr, 10),
                                (uint32_t) strtoul(argv[4], nullptr, 10),
                                (uint16_t) strtoul(argv[5], nullptr, 10),
                                (uint16_t) strtoul(argv[6], nullptr, 10),
                                argv[7]);
        printf("{\"ok\":%s,\"path\":\"%s\"}\n", ok ? "true" : "false", argv[2]);
        return ok ? 0 : 2;
    }
    usage(argv[0]);
    return 1;
}
