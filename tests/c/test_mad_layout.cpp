// src/ggml-bitnet-mad.cpp (the pre-refactor kernel file; NOT compiled by the
// current build, see tests/README.md): its quantize_i2_s must emit the
// canonical 128/32 bytes on every architecture. On ARM it uses QK_I2_S=64
// (layout 64/16), cf. microsoft/BitNet#585.
#include <cstdio>
#include <cstdlib>
extern "C" {
#include "i2s_ref.h"
size_t quantize_i2_s(const float *, void *, int64_t, int64_t, const float *);
}
int main() {
    enum { N = 512 };
    static int8_t t[N]; static float src[N]; static uint8_t buf[N / 4 + 32], ref[N / 4];
    uint32_t st = 99u;
    for (int i = 0; i < N; i++) { t[i] = (int8_t)(i2s_ref_rng(&st) % 3) - 1; src[i] = t[i] * 0.5f; }
    quantize_i2_s(src, buf, 1, N, NULL);
    i2s_ref_pack(t, N, ref);
    int bad = 0;
    for (int i = 0; i < N / 4; i++) bad += buf[i] != ref[i];
    printf("mad.cpp quantize_i2_s bytes != canonical 128/32: %d / %d\n", bad, N / 4);
    return bad ? 1 : 0;
}
