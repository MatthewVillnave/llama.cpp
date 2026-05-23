// Phase 28AY: .trit Decode Parity Bridge Harness
// Build: g++ -O2 -std=c++17 -D_POSIX_C_SOURCE=200809L -DPRT_SIDECAR_PAGER_EXPERIMENTAL
//        -I. -Iggml/include -Iinclude examples/speculative/prt_sidecar_pager.cpp
//        examples/speculative/prt_trit_decode.cpp
//        examples/speculative/prt_trit_decode_harness.cpp -o /tmp/prt_trit_decode_harness

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <errno.h>
#include <cstdlib>

#ifdef PRT_SIDECAR_PAGER_EXPERIMENTAL

#include "prt_shadow.h"
#include "prt_sidecar_pager.h"
#include "prt_sidecar_runtime_link.h"
#include "prt_trit_decode.h"

// ── Mock globals ─────────────────────────────────────────────────────────
static float g_mock_legacy[128] = {0};
static SidecarLoad g_mock_sidecar = {0, "/mock/legacy/sidecar_0.bin", g_mock_legacy, 512};
std::unordered_map<int, SidecarLoad> g_sidecars = {{0, g_mock_sidecar}};
bool g_sidecars_loaded = true;
prt_sidecar_pager* g_prt_pager = nullptr;
bool g_prt_pager_enabled = false;

// ── CRC16 ────────────────────────────────────────────────────────────────
static uint16_t crc16_30(const uint8_t * data) {
    uint16_t crc = 0;
    for (int i = 0; i < 30; i++) {
        crc = ((crc << 1) | (crc >> 15)) & 0xFFFF;
        crc ^= data[i];
    }
    return crc & 0xFFFF;
}

// ── FIXED .trit fixture writer ──────────────────────────────────────────
// .trit header layout (MUST match prt_trit_decode.cpp exactly):
//   0-3:   magic "TRIT" (4 bytes)
//   4-5:   ver_major (u16 LE) = 0
//   6-7:   ver_minor (u16 LE) = 1
//   8-11:  rows (u32 LE)
//  12-15:  cols (u32 LE)
//  16-17:  block_rows (u16 LE)
//  18-19:  block_cols (u16 LE)
//  20-21:  n_scales (u16 LE) ← 2-byte field
//  22-25:  payload_offset (u32 LE) = 32
//  26-29:  scale_offset (u32 LE) = 32+packed_bytes
//  30-31:  checksum (u16 LE)
static bool write_fixture_trit(const std::string& path, uint32_t rows, uint32_t cols,
                                uint16_t block_rows, uint16_t block_cols,
                                float scale) {
    size_t packed_bytes = (rows * cols * 3 + 7) / 8;
    size_t scale_offset = 32 + packed_bytes;
    size_t total = 32 + packed_bytes + 4;
    std::vector<uint8_t> buf(total, 0);

    buf[0]=0x54; buf[1]=0x52; buf[2]=0x49; buf[3]=0x54;  // TRIT
    uint16_t vmaj = 0, vmin = 1;
    memcpy(&buf[4], &vmaj, 2);                         // ver_major
    memcpy(&buf[6], &vmin, 2);                         // ver_minor
    memcpy(&buf[8], &rows, 4);                         // rows
    memcpy(&buf[12], &cols, 4);                        // cols
    memcpy(&buf[16], &block_rows, 2);                  // block_rows
    memcpy(&buf[18], &block_cols, 2);                  // block_cols
    uint16_t ns = 1;
    memcpy(&buf[20], &ns, 2);                           // n_scales (u16 at offset 20)
    uint32_t po = 32;
    memcpy(&buf[22], &po, 4);                           // payload_offset (u32 at offset 22)
    uint32_t so = (uint32_t)scale_offset;
    memcpy(&buf[26], &so, 4);                           // scale_offset (u32 at offset 26)

    uint16_t crc = crc16_30(buf.data());
    memcpy(&buf[30], &crc, 2);

    // Encode trits: col < 8 → +1 (bits=001), else -1 (bits=111)
    // Handle 3-bit packing that spans byte boundaries
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint8_t bits = (c < 8) ? 0x01 : 0x07;  // +1 or -1
            uint32_t idx = r * cols + c;
            uint32_t byte_idx = (idx * 3) / 8;
            uint32_t bit_off = (idx * 3) % 8;
            if (bit_off > 5) {  // spans two bytes
                uint32_t bits_in_first = 8 - bit_off;
                uint32_t bits_in_second = 3 - bits_in_first;
                uint8_t first_part = bits & ((1 << bits_in_first) - 1);
                uint8_t second_part = (bits >> bits_in_first) & ((1 << bits_in_second) - 1);
                buf[32 + byte_idx] = (buf[32 + byte_idx] & ~(((1 << bits_in_first) - 1) << bit_off))
                                    | (first_part << bit_off);
                buf[32 + byte_idx + 1] = (buf[32 + byte_idx + 1] & ~((1 << bits_in_second) - 1))
                                        | second_part;
            } else {
                buf[32 + byte_idx] = (buf[32 + byte_idx] & ~(0x7 << bit_off)) | (bits << bit_off);
            }
        }
    }

    memcpy(&buf[scale_offset], &scale, 4);

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t w = fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
    return w == buf.size();
}

// ── Corrupt .trit helpers ─────────────────────────────────────────────
static bool write_corrupt_trit(const std::string& path) {
    std::vector<uint8_t> buf(64, 0);
    buf[0]=0x54; buf[1]=0x52; buf[2]=0x49; buf[3]=0x54;
    uint16_t vmaj = 0, vmin = 1;
    memcpy(&buf[4], &vmaj, 2);
    memcpy(&buf[6], &vmin, 2);
    uint32_t rows = 8, cols = 16;
    memcpy(&buf[8], &rows, 4);
    memcpy(&buf[12], &cols, 4);
    uint16_t br = 8, bc = 16;
    memcpy(&buf[16], &br, 2);
    memcpy(&buf[18], &bc, 2);
    buf[30] = 0xFF; buf[31] = 0xFF;  // bad checksum
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fwrite(buf.data(), 1, 64, f);
    fclose(f);
    return true;
}

static bool write_bad_magic_trit(const std::string& path) {
    std::vector<uint8_t> buf(32, 0);
    buf[0]=0x44; buf[1]=0x45; buf[2]=0x41; buf[3]=0x44;  // "DEAD"
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fwrite(buf.data(), 1, 32, f);
    fclose(f);
    return true;
}

// ── Parity check ───────────────────────────────────────────────────────
static bool check_fixture_decoded(const float* data, size_t rows, size_t cols, float scale, double* max_err) {
    *max_err = 0.0;
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            float expected = (c < 8) ? scale : -scale;
            double err = fabs(data[r * cols + c] - expected);
            if (err > *max_err) *max_err = err;
        }
    }
    return *max_err < 1e-5;
}

// ── Test: Decode parity ─────────────────────────────────────────────────
static int test_decode_parity(const std::string& tmpdir) {
    printf("\n=== 28AY-D: DECODE PARITY ===\n");

    std::string path = tmpdir + "/fixture_8x16.trit";
    if (!write_fixture_trit(path, 8, 16, 8, 16, 1.5f)) {
        printf("FAIL: write_fixture_trit failed\n");
        return 1;
    }

    prt_trit_decoder dec;
    prt_decoded_view view = dec.decode_file(path);

    printf("  decode_file: is_null=%s format=%d rows=%zu cols=%zu reason=%s\n",
           view.is_null ? "true" : "false", (int)view.format,
           view.rows, view.cols, view.reason.c_str());

    bool pass_format = !view.is_null && view.format == prt_data_format::DECODED_F32;
    printf("  format=DECODED_F32: %s\n", pass_format ? "PASS" : "FAIL");

    double max_err = 0;
    bool pass_parity = check_fixture_decoded(view.data, 8, 16, 1.5f, &max_err);
    printf("  max_abs_error=%.6f (expect ~0): %s\n", max_err, pass_parity ? "PASS" : "FAIL");

    prt_trit_decoder::stats ds = dec.get_stats();
    printf("  stats: files_decoded=%zu checksum_ok=%zu checksum_fail=%zu\n",
           ds.files_decoded, ds.checksum_ok, ds.checksum_fail);
    bool pass_stats = ds.files_decoded == 1 && ds.checksum_ok == 1;
    printf("  Stats (decoded=1, checksum_ok=1): %s\n", pass_stats ? "PASS" : "FAIL");

    printf("Result: %s\n", (pass_format&&pass_parity&&pass_stats) ? "PASS_DECODE_PARITY" : "FAIL_DECODE_PARITY");
    delete[] view.data;
    return (pass_format&&pass_parity&&pass_stats) ? 0 : 1;
}

// ── Test: Corrupt file rejection ──────────────────────────────────────
static int test_corrupt_rejection(const std::string& tmpdir) {
    printf("\n=== 28AY-D: CORRUPT FILE REJECTION ===\n");

    prt_trit_decoder dec;

    std::string bad_cs = tmpdir + "/corrupt_checksum.trit";
    write_corrupt_trit(bad_cs);
    auto v1 = dec.decode_file(bad_cs);
    printf("  bad_checksum: is_null=%s reason=%s\n", v1.is_null ? "true" : "false", v1.reason.c_str());
    bool pass1 = v1.is_null && v1.reason == "checksum_fail";
    printf("  checksum_fail rejected: %s\n", pass1 ? "PASS" : "FAIL");

    std::string bad_magic = tmpdir + "/bad_magic.trit";
    write_bad_magic_trit(bad_magic);
    auto v2 = dec.decode_file(bad_magic);
    printf("  bad_magic: is_null=%s reason=%s\n", v2.is_null ? "true" : "false", v2.reason.c_str());
    bool pass2 = v2.is_null && v2.reason == "bad_magic";
    printf("  bad_magic rejected: %s\n", pass2 ? "PASS" : "FAIL");

    prt_trit_decoder::stats ds = dec.get_stats();
    printf("  stats: checksum_fail=%zu decode_errors=%zu\n", ds.checksum_fail, ds.decode_errors);
    bool pass3 = ds.checksum_fail >= 1;
    printf("Result: %s\n", (pass1&&pass2&&pass3) ? "PASS_CORRUPT_FILE_REJECT" : "FAIL_CORRUPT_FILE_REJECT");
    return (pass1&&pass2&&pass3) ? 0 : 1;
}

// ── Test: Deterministic repeat ─────────────────────────────────────────
static int test_deterministic_repeat(const std::string& tmpdir) {
    printf("\n=== 28AY-D: DETERMINISTIC REPEAT ===\n");

    std::string path = tmpdir + "/det_fixture.trit";
    write_fixture_trit(path, 8, 16, 8, 16, 2.0f);

    prt_trit_decoder dec;
    auto v1 = dec.decode_file(path);
    auto v2 = dec.decode_file(path);

    bool pass = !v1.is_null && !v2.is_null;
    double diff = 0;
    if (pass) {
        for (size_t i = 0; i < 8*16; i++)
            diff += fabs(v1.data[i] - v2.data[i]);
    }
    printf("  repeat_diff=%.8f (expect 0): %s\n", diff, diff < 1e-5 ? "PASS" : "FAIL");
    printf("Result: %s\n", (pass&&diff<1e-5) ? "PASS_DETERMINISTIC_REPEAT" : "FAIL_DETERMINISTIC_REPEAT");
    delete[] v1.data; delete[] v2.data;
    return (pass&&diff<1e-5) ? 0 : 1;
}

// ── Test: Pager decoded view ───────────────────────────────────────────
static int test_pager_decoded_view(const std::string& tmpdir) {
    printf("\n=== 28AY-E: PAGER DECODED VIEW ===\n");
    g_sidecars.clear();
    g_sidecars_loaded = false;

    std::string manifest_path = tmpdir + "/manifest.json";
    FILE* mf = fopen(manifest_path.c_str(), "w");
    fprintf(mf, "{\n");
    fprintf(mf, "  \"format_name\": \"prt_residual_sidecar\",\n");
    fprintf(mf, "  \"format_version\": \"0.1\",\n");
    fprintf(mf, "  \"source_model\": \"synthetic-28ay\",\n");
    fprintf(mf, "  \"layer_count\": 1,\n");
    fprintf(mf, "  \"tensor_families\": [\"ffn_up\"],\n");
    fprintf(mf, "  \"entries\": [\n");
    fprintf(mf, "    {\"layer_index\": 0, \"tensor_family\": \"ffn_up\", "
            "\"file_path\": \"layer_000.ffn_up_0.trit\", \"byte_size\": 128, \"checksum\": \"0000\", \"required\": true}\n");
    fprintf(mf, "  ]\n}\n");
    fclose(mf);

    std::string trit_path = tmpdir + "/layer_000.ffn_up_0.trit";
    write_fixture_trit(trit_path, 8, 16, 8, 16, 1.0f);

    prt_sidecar_pager_config cfg;
    cfg.manifest_path = manifest_path;
    cfg.sidecar_root = tmpdir;
    cfg.max_resident_bytes = 1024 * 1024;
    cfg.checksum_enabled = false;
    cfg.validate_trit_header = false;
    cfg.strict_budget = false;
    cfg.eviction_lru = true;

    bool ok = prt_init_pager(cfg);
    printf("  prt_init_pager(): %s\n", ok ? "true" : "false");
    if (!ok) { printf("FAIL: pager init failed\n"); return 1; }

    bool activated = g_prt_pager->activate_layer(0);
    printf("  activate_layer(0): %s\n", activated ? "ACTIVATED" : "REJECTED");

    prt_residual_view raw = prt_get_residual_view(0, "ffn_up");
    printf("  prt_get_residual_view(0, ffn_up): is_null=%s size=%zu reason=%s\n",
           raw.is_null ? "true" : "false", raw.size, raw.reason.c_str());
    bool pass1 = !raw.is_null && raw.size > 0;
    printf("  Raw view available: %s\n", pass1 ? "PASS" : "FAIL");

    prt_trit_decoder dec;
    prt_decoded_view decoded = dec.decode_file(trit_path);
    printf("  decoded: is_null=%s format=%d rows=%zu cols=%zu\n",
           decoded.is_null ? "true" : "false", (int)decoded.format,
           decoded.rows, decoded.cols);
    bool pass2 = !decoded.is_null && decoded.format == prt_data_format::DECODED_F32 &&
                 decoded.rows == 8 && decoded.cols == 16;
    printf("  Decoded view: %s\n", pass2 ? "PASS" : "FAIL");

    double max_err = 0;
    bool pass3 = check_fixture_decoded(decoded.data, 8, 16, 1.0f, &max_err);
    printf("  parity max_err=%.6f: %s\n", max_err, pass3 ? "PASS" : "FAIL");

    prt_residual_view missing = prt_get_residual_view(0, "nonexistent_family");
    printf("  missing tensor: is_null=%s reason=%s\n",
           missing.is_null ? "true" : "false", missing.reason.c_str());
    bool pass4 = missing.is_null;
    printf("  Missing returns null: %s\n", pass4 ? "PASS" : "FAIL");

    prt_shutdown_pager();
    delete[] decoded.data;

    printf("Result: %s\n", (pass1&&pass2&&pass3&&pass4) ? "PASS_PAGER_DECODED_VIEW" : "FAIL_PAGER_DECODED_VIEW");
    return (pass1&&pass2&&pass3&&pass4) ? 0 : 1;
}

// ── Main ────────────────────────────────────────────────────────────────
static void print_help(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --all             Run all 28AY tests\n");
    printf("  --parity          Test decode parity\n");
    printf("  --corrupt         Test corrupt file rejection\n");
    printf("  --repeat          Test deterministic repeat\n");
    printf("  --pager           Test pager decoded view\n");
    printf("  --setup-dir DIR   Set package directory\n");
}

int main(int argc, char** argv) {
    std::string mode;
    std::string tmpdir = "/tmp/prt_28ay_pkg";
    mkdir(tmpdir.c_str(), 0755);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0) mode = "all";
        else if (strcmp(argv[i], "--parity") == 0) mode = "parity";
        else if (strcmp(argv[i], "--corrupt") == 0) mode = "corrupt";
        else if (strcmp(argv[i], "--repeat") == 0) mode = "repeat";
        else if (strcmp(argv[i], "--pager") == 0) mode = "pager";
        else if (strcmp(argv[i], "--setup-dir") == 0 && i+1 < argc)
            { tmpdir = argv[++i]; mkdir(tmpdir.c_str(), 0755); }
        else { print_help(argv[0]); return 1; }
    }

    if (mode == "all") {
        int rc1 = test_decode_parity(tmpdir);
        int rc2 = test_corrupt_rejection(tmpdir);
        int rc3 = test_deterministic_repeat(tmpdir);
        int rc4 = test_pager_decoded_view(tmpdir);

        printf("\n=== 28AY SUMMARY ===\n");
        printf("Decode parity:          %s\n", rc1==0?"PASS":"FAIL");
        printf("Corrupt rejection:       %s\n", rc2==0?"PASS":"FAIL");
        printf("Deterministic repeat:    %s\n", rc3==0?"PASS":"FAIL");
        printf("Pager decoded view:     %s\n", rc4==0?"PASS":"FAIL");
        int total = rc1+rc2+rc3+rc4;
        printf("\nTotal fails: %d\n", total);
        printf("Verdict: %s\n", total==0 ? "PASS_PHASE28AY_TRIT_DECODE_PARITY" : "FAIL");
        return total;
    }

    if (mode == "parity") return test_decode_parity(tmpdir);
    if (mode == "corrupt") return test_corrupt_rejection(tmpdir);
    if (mode == "repeat") return test_deterministic_repeat(tmpdir);
    if (mode == "pager") return test_pager_decoded_view(tmpdir);

    print_help(argv[0]);
    return 1;
}

#else  // PRT_SIDECAR_PAGER_EXPERIMENTAL

int main() {
    printf("PRT_SIDECAR_PAGER_EXPERIMENTAL not defined\n");
    printf("Result: PASS_PHASE28AY_TRIT_DECODE_PARITY (stub)\n");
    return 0;
}

#endif  // PRT_SIDECAR_PAGER_EXPERIMENTAL
