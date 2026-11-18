// ARM/scalar CPU path for I2_S mul_mat, exactly as ggml-cpu.c calls it
// (ggml-cpu.c "I2_S GEMV/GEMM fast path"):
//   ggml_gemm_i2_i8_s(n, tmp, n_rows, W, Y, 4, n_rows)  -> tmp[col*n_rows + row]
//   ggml_gemv_i2_i8_s(n, tmp, ne01,   W, y, 1, n_rows)  -> tmp[row]
// followed by  dst = (tmp - act_sum) * weight_scale / act_scale.
// So each kernel must return  sum_i code(w_i) * y_i  with code = w+1, reading the
// canonical 128/32 layout. Reference: integer arithmetic in i2s_ref.h.
// Sources under test: ggml-cpu/quants.c, ggml-cpu/ggml-cpu-i2s.c
#include <stdio.h>
#include <stdlib.h>
#include "i2s_ref.h"

extern void ggml_vec_dot_i2_i8_s(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
extern void ggml_gemv_i2_i8_s(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
extern void ggml_gemm_i2_i8_s(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);

// mode 0: ggml_vec_dot_i2_i8_s + ggml_gemv_i2_i8_s;  mode 1: ggml_gemm_i2_i8_s
static int run(int N, int ROWS, int mode) {
    enum { COLS = 4 };
    int8_t  *t  = malloc((size_t)ROWS * N);
    uint8_t *W  = malloc((size_t)ROWS * N / 4 + 32);
    int8_t  *Y  = malloc((size_t)COLS * N);
    float   *s  = malloc(sizeof(float) * ROWS * COLS);
    int32_t *rf = malloc(sizeof(int32_t) * ROWS * COLS);
    uint32_t st = 777u + (uint32_t)N;
    for (int i = 0; i < ROWS * N; i++) t[i] = (int8_t)(i2s_ref_rng(&st) % 3) - 1;
    for (int i = 0; i < COLS * N; i++) Y[i] = (int8_t)((int)(i2s_ref_rng(&st) % 255) - 127);
    i2s_ref_pack(t, (int64_t)ROWS * N, W);
    memset(W + ROWS * N / 4, 0, 32);
    for (int c = 0; c < COLS; c++)
        for (int r = 0; r < ROWS; r++)
            rf[c * ROWS + r] = i2s_ref_dot_codes(W + (size_t)r * N / 4, Y + (size_t)c * N, N);

    int fail = 0, bad;

    // 1) vec_dot, one weight row x one activation vector
    if (mode == 0) {
    bad = 0;
    for (int r = 0; r < ROWS; r++) {
        float v = -1e30f;
        ggml_vec_dot_i2_i8_s(N, &v, 0, W + (size_t)r * N / 4, 0, Y, 0, 1);
        bad += (v != (float)rf[r]);
    }
    printf("  n=%-5d rows=%-3d vec_dot 1x1 : %d / %d mismatches\n", N, ROWS, bad, ROWS);
    fail |= bad;

    // 2) GEMV as called from ggml-cpu.c
    bad = 0;
    for (int i = 0; i < ROWS; i++) s[i] = -1e30f;
    ggml_gemv_i2_i8_s(N, s, ROWS, W, Y, 1, ROWS);
    for (int r = 0; r < ROWS; r++) bad += (s[r] != (float)rf[r]);
    printf("  n=%-5d rows=%-3d gemv        : %d / %d mismatches\n", N, ROWS, bad, ROWS);
    fail |= bad;
    }

    // 3) GEMM as called from ggml-cpu.c: 4 activation columns, out[col*bs+row]
    if (mode == 1) {
    bad = 0;
    for (int i = 0; i < ROWS * COLS; i++) s[i] = -1e30f;
    ggml_gemm_i2_i8_s(N, s, ROWS, W, Y, COLS, ROWS);
    for (int i = 0; i < ROWS * COLS; i++) bad += (s[i] != (float)rf[i]);
    printf("  n=%-5d rows=%-3d gemm (4 col): %d / %d mismatches\n", N, ROWS, bad, ROWS * COLS);
    fail |= bad;
    }

    free(t); free(W); free(Y); free(s); free(rf);
    return fail;
}

int main(int argc, char **argv) {
    const int mode = (argc > 1 && strcmp(argv[1], "gemm") == 0) ? 1 : 0;  // default / "vecdot"
    int fail = 0;
    fail |= run(128, 16, mode);
    fail |= run(2560, 32, mode);   // hidden size of bitnet-b1.58-2B-4T
    fail |= run(6912, 16, mode);   // FFN size
    fail |= run(4224, 20, mode);   // 33 blocks of 128: exercises the non-multiple-of-32-blocks tail
    return fail ? 1 : 0;
}
