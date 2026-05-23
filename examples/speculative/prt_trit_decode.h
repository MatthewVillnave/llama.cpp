// Phase 28AY: .trit decode parity bridge
// Defines decoded view contract + ternary decoder

#ifndef PRT_TRIT_DECODE_H
#define PRT_TRIT_DECODE_H

#include <cstdint>
#include <cstddef>
#include <string>

// ── Decoded view contract ─────────────────────────────────────────────────

enum class prt_data_format {
    RAW_TRIT       = 0,  // Raw .trit file bytes, no decode
    DECODED_Q8     = 1,  // Decoded Q8_0 block (quantized)
    DECODED_F32    = 2,  // Decoded floating-point f32 buffer
    UNKNOWN        = 3,
};

struct prt_decoded_view {
    float * data = nullptr;        // owned buffer, caller must delete[]
    size_t rows = 0;
    size_t cols = 0;
    prt_data_format format = prt_data_format::UNKNOWN;
    bool is_null = true;
    std::string reason;
};

// ── Decoder ────────────────────────────────────────────────────────────────

class prt_trit_decoder {
public:
    prt_trit_decoder();
    ~prt_trit_decoder();

    // Decode a .trit file to DECODED_F32
    // Returns null view on error with reason set.
    prt_decoded_view decode_file(const std::string& path);

    // Decode from raw bytes (header already consumed)
    prt_decoded_view decode_bytes(const uint8_t * file_bytes,
                                  size_t file_size,
                                  uint32_t rows,
                                  uint32_t cols,
                                  uint16_t block_rows,
                                  uint16_t block_cols,
                                  uint16_t n_scales,
                                  const float * scales);

    // Stats
    struct stats {
        size_t files_decoded = 0;
        size_t checksum_ok = 0;
        size_t checksum_fail = 0;
        size_t decode_errors = 0;
        double last_decode_ms = 0;
    };
    stats get_stats() const { return stats_; }
    void reset_stats() { stats_ = stats(); }

private:
    stats stats_;
};

// ── Inline helpers ─────────────────────────────────────────────────────────

// 3-bits-per-trit unpack:
// pattern 0b000 → 0
// pattern 0b001 → 1
// pattern 0b111 → -1
// all other patterns → 0 (reserved/unused)
inline float prt_unpack_trit_3bit(uint8_t bits) {
    if (bits == 0x01) return  1.0f;
    if (bits == 0x07) return -1.0f;
    return 0.0f;
}

#endif  // PRT_TRIT_DECODE_H
