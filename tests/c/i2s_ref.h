// Canonical I2_S reference (layout "128/32"), independent of the engine sources.
//
// A tensor of n ternary weights is stored in n/4 bytes followed by a 32-byte
// tail whose first 4 bytes are the f32 scale. Weights are grouped in blocks of
// 128; block b occupies bytes [32*b, 32*b+32). Inside a block, byte gp (0..31)
// holds the 2-bit codes of weights {gp, 32+gp, 64+gp, 96+gp} in bits
// [7:6], [5:4], [3:2], [1:0]. Codes: 0 -> -1, 1 -> 0, 2 -> +1.
//
// This is the layout of the official microsoft/bitnet-b1.58-2B-4T-gguf file
// (verified by tests/py/test_gguf_layout.py against the BF16 master weights).
#ifndef BITNETX_I2S_REF_H
#define BITNETX_I2S_REF_H

#include <stdint.h>
#include <string.h>

// t[i] in {-1,0,+1}; n must be a multiple of 128; out has n/4 bytes.
static inline void i2s_ref_pack(const int8_t *t, int64_t n, uint8_t *out) {
    memset(out, 0, (size_t)(n / 4));
    for (int64_t i = 0; i < n; i++) {
        const int64_t blk = i / 128, j = i % 128;
        const int grp = (int)(j / 32), gp = (int)(j % 32);
        out[blk * 32 + gp] |= (uint8_t)((t[i] + 1) << (6 - 2 * grp));
    }
}

static inline int i2s_ref_code(const uint8_t *packed, int64_t i) {
    const int64_t blk = i / 128, j = i % 128;
    const int grp = (int)(j / 32), gp = (int)(j % 32);
    return (packed[blk * 32 + gp] >> (6 - 2 * grp)) & 3;
}

// The engine's kernels return the dot product of the *codes* (w+1) with the
// int8 activations; ggml-cpu.c then subtracts sum(y) ("act_sums"):
//   sum_i (w_i+1) y_i - sum_i y_i = sum_i w_i y_i.
static inline int32_t i2s_ref_dot_codes(const uint8_t *packed, const int8_t *y, int64_t n) {
    int32_t s = 0;
    for (int64_t i = 0; i < n; i++) s += i2s_ref_code(packed, i) * (int32_t)y[i];
    return s;
}

// xorshift32: deterministic data without libc rand() differences.
static inline uint32_t i2s_ref_rng(uint32_t *st) {
    uint32_t x = *st; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return *st = x;
}

#endif
