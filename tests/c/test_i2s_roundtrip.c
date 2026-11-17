// quantize_i2_s -> canonical 128/32 bytes -> dequantize_row_i2_s must be the
// identity on ternary input {-s, 0, +s}.
// Sources under test: 3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "i2s_ref.h"

extern void   dequantize_row_i2_s(const uint8_t *, float *, int64_t, float);
extern size_t quantize_i2_s(const float *, void *, int64_t, int64_t, const float *);

int main(void) {
    enum { ROWS = 3, N = 256, TOT = ROWS * N };
    const float S = 1.25f;
    static int8_t  t[TOT];
    static float   src[TOT], out[TOT];
    static uint8_t buf[TOT / 4 + 32], ref[TOT / 4];
    uint32_t st = 12345u;
    for (int i = 0; i < TOT; i++) { t[i] = (int8_t)(i2s_ref_rng(&st) % 3) - 1; }
    t[0] = 1;
    for (int i = 0; i < TOT; i++) src[i] = t[i] * S;

    memset(buf, 0xAA, sizeof buf);
    size_t sz = quantize_i2_s(src, buf, ROWS, N, NULL);
    float sc; memcpy(&sc, buf + TOT / 4, 4);
    i2s_ref_pack(t, TOT, ref);

    int bad_bytes = 0;
    for (int i = 0; i < TOT / 4; i++) bad_bytes += buf[i] != ref[i];

    dequantize_row_i2_s(buf, out, TOT, sc);
    int bad_rt = 0;
    for (int i = 0; i < TOT; i++) bad_rt += out[i] != src[i];

    // dequantize of the *reference* bytes must be exact (reader is canonical)
    dequantize_row_i2_s(ref, out, TOT, S);
    int bad_deq = 0;
    for (int i = 0; i < TOT; i++) bad_deq += out[i] != src[i];

    printf("size=%zu (expect %d) scale=%g (expect %g)\n", sz, TOT / 4 + 32, sc, S);
    printf("dequantize(reference bytes) mismatches: %d / %d\n", bad_deq, TOT);
    printf("quantize bytes != canonical 128/32:     %d / %d\n", bad_bytes, TOT / 4);
    printf("quantize->dequantize mismatches:        %d / %d\n", bad_rt, TOT);
    return (bad_bytes || bad_rt || bad_deq || sz != TOT / 4 + 32 || sc != S) ? 1 : 0;
}
