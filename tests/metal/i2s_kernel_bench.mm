// Standalone test + micro benchmark of the BitNet I2_S Metal kernels (quantize_i8_s + mul_mv_i2_s).
// Compares the GPU result with a CPU reference that replicates quantize_row_i8_s() + the integer dot product
// on weights in the canonical 128/32 layout, then times the mat-vec kernel.
//   i2s_kernel_bench <kernels.metal> <K> <N> <ncols> <NSG> <R> [iters]
// exit status: 0 = results identical, 1 = mismatch, 77 = no Metal device
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../c/i2s_ref.h"

struct kq { uint64_t nb11, nb12, nb13; int32_t ne10, ne11, ne12, ncols; };
struct km { uint64_t nb1, nb2, nb3, ws_off; int32_t ne00, ne01, ne11, ne12, ncols; };

int main(int argc, char ** argv) {
    @autoreleasepool {
        if (argc < 7) { fprintf(stderr, "usage: %s kernels.metal K N ncols NSG R [iters]\n", argv[0]); return 2; }
        const int K = atoi(argv[2]), N = atoi(argv[3]), NC = atoi(argv[4]), NSG = atoi(argv[5]), R = atoi(argv[6]);
        const int iters = argc > 7 ? atoi(argv[7]) : 200;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no Metal device\n"); return 77; }

        NSError * err = nil;
        NSString * src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&err];
        if (getenv("BX_DEBUG")) {
            src = [src stringByAppendingString:@"\nkernel void dbg_div(device const float * sc, constant float & ws, device float * o1, device float * o2, uint i [[thread_position_in_grid]]) { o1[i] = precise::divide(ws, sc[i]); o2[i] = ws / sc[i]; }\nkernel void dbg_mul(device const int * sm, constant float & q, device float * o1, device float * o2, uint i [[thread_position_in_grid]]) { o1[i] = (float) sm[i] * q; o2[i] = fma((float) sm[i], q, 0.0f); }\n"];
        }
        MTLCompileOptions * opt = [MTLCompileOptions new];
        opt.preprocessorMacros = @{ @"I2S_NSG": @(NSG), @"I2S_R": @(R) };
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { printf("metal compile error: %s\n", err.localizedDescription.UTF8String); return 1; }

        int C = NC >= 4 ? 4 : (NC >= 2 ? 2 : 1);
        NSString * mvname = [NSString stringWithFormat:@"kernel_mul_mv_i2_s_i8_c%d", C];
        id<MTLComputePipelineState> pq = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kernel_quantize_i8_s_f32"] error:&err];
        id<MTLComputePipelineState> pm = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:mvname] error:&err];
        if (!pq || !pm) { printf("pipeline error: %s\n", err.localizedDescription.UTF8String); return 1; }

        // data: ternary weights, canonical packing, f32 scale tail; activations with a few outliers
        const float ws = 1.25f;
        std::vector<int8_t> t((size_t) N*K);
        uint32_t st = 12345u + K + N;
        for (auto & v : t) v = (int8_t)(i2s_ref_rng(&st) % 3) - 1;
        std::vector<uint8_t> w((size_t) N*K/4 + 32, 0);
        i2s_ref_pack(t.data(), (int64_t) N*K, w.data());
        memcpy(w.data() + (size_t) N*K/4, &ws, 4);
        std::vector<float> x((size_t) NC*K);
        for (auto & v : x) v = ((int)(i2s_ref_rng(&st) % 2001) - 1000) / 250.0f;
        for (int c = 0; c < NC; c++) x[(size_t) c*K + (i2s_ref_rng(&st) % K)] *= 6.0f;
        if (NC > 1) memset(&x[(size_t)(NC-1)*K], 0, sizeof(float)*K); // an all-zero column: result must be exactly 0

        // CPU reference (same operations as quantize_row_i8_s + vec_dot)
        std::vector<float> ref((size_t) NC*N); std::vector<int64_t> sums; std::vector<float> sc_of;
        for (int c = 0; c < NC; c++) {
            const float * xc = &x[(size_t) c*K];
            float amax = 0.0f; for (int i = 0; i < K; i++) amax = fmaxf(amax, fabsf(xc[i]));
            const float s = amax > 0.0f ? 127.0f/amax : 0.0f;
            std::vector<int> v(K);
            for (int i = 0; i < K; i++) { int q = (int) roundf(xc[i]*s); v[i] = q > 127 ? 127 : (q < -128 ? -128 : q); }
            for (int r = 0; r < N; r++) {
                int64_t sum = 0;
                for (int i = 0; i < K; i++) sum += (int64_t) t[(size_t) r*K + i] * v[i];
                ref[(size_t) c*N + r] = s > 0.0f ? (float) sum * (ws/s) : 0.0f;
                if (getenv("BX_DEBUG") && s > 0.0f) { sums.push_back(sum); sc_of.push_back(s); }
            }
        }

        const size_t q8_bytes = (size_t) NC*K, sc_bytes = ((size_t) NC*4 + 15)/16*16, dst_bytes = (size_t) NC*N*4;
        id<MTLBuffer> bw  = [dev newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bx  = [dev newBufferWithBytes:x.data() length:x.size()*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bq  = [dev newBufferWithLength:q8_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bs  = [dev newBufferWithLength:sc_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bd  = [dev newBufferWithLength:dst_bytes options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        kq aq = { (uint64_t) K*4, (uint64_t) K*4*NC, (uint64_t) K*4*NC, K, NC, 1, NC };
        km am = { (uint64_t) N*4, (uint64_t) N*4*NC, (uint64_t) N*4*NC, (uint64_t) N*K/4, K, N, NC, 1, NC };

        auto encode = [&](id<MTLComputeCommandEncoder> e, bool quant) {
            if (quant) {
                [e setComputePipelineState:pq];
                [e setBytes:&aq length:sizeof(aq) atIndex:0];
                [e setBuffer:bx offset:0 atIndex:1]; [e setBuffer:bq offset:0 atIndex:2]; [e setBuffer:bs offset:0 atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(NC, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [e setComputePipelineState:pm];
            [e setBytes:&am length:sizeof(am) atIndex:0];
            [e setBuffer:bw offset:0 atIndex:1]; [e setBuffer:bq offset:0 atIndex:2]; [e setBuffer:bs offset:0 atIndex:3]; [e setBuffer:bd offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake((N + NSG*R - 1)/(NSG*R), (NC + C - 1)/C, 1) threadsPerThreadgroup:MTLSizeMake(32, NSG, 1)];
        };

        { // correctness
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            encode(e, true); [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.error) { printf("GPU error: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        }
        const float * got = (const float *) bd.contents;
        long bad = 0; double maxd = 0.0;
        for (size_t i = 0; i < (size_t) NC*N; i++) {
            // the GPU compiler may evaluate sum*(ws/s) as (sum*ws)*(1/s): allow 2 ulp (an error of 1 in the integer sum is >= 5e-7 relative)
            const double tol = 2.0 * 1.1920929e-7 * fabs((double) ref[i]);
            if (fabs((double) got[i] - ref[i]) > tol) { bad++; maxd = fmax(maxd, fabs((double) got[i] - ref[i])); }
        }

        if (getenv("BX_DEBUG")) {
            // scales: GPU vs CPU
            const float * gs = (const float *) bs.contents; long sbad = 0;
            for (int c = 0; c < NC; c++) {
                const float * xc = &x[(size_t) c*K]; float amax = 0.0f;
                for (int i = 0; i < K; i++) amax = fmaxf(amax, fabsf(xc[i]));
                const float s = amax > 0.0f ? 127.0f/amax : 0.0f;
                if (gs[c] != s) { sbad++; printf("scale col %d: gpu %.9g cpu %.9g\n", c, gs[c], s); }
            }
            // int8 activations
            const int8_t * gq = (const int8_t *) bq.contents; long qbad = 0;
            for (int c = 0; c < NC; c++) {
                const float * xc = &x[(size_t) c*K]; float amax = 0.0f;
                for (int i = 0; i < K; i++) amax = fmaxf(amax, fabsf(xc[i]));
                const float s = amax > 0.0f ? 127.0f/amax : 0.0f;
                for (int i = 0; i < K; i++) { int v = (int) roundf(xc[i]*s); v = v > 127 ? 127 : (v < -128 ? -128 : v); if (gq[(size_t) c*K + i] != v) qbad++; }
            }
            printf("debug: scale mismatches %ld, int8 mismatches %ld\n", sbad, qbad);
        }

        if (getenv("BX_DEBUG")) {
            const int n = 4096; std::vector<float> sc(n); uint32_t st2 = 99;
            for (auto & v : sc) v = 127.0f / (0.5f + (i2s_ref_rng(&st2) % 100000) / 1000.0f);
            id<MTLBuffer> b1 = [dev newBufferWithBytes:sc.data() length:n*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b2 = [dev newBufferWithLength:n*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> b3 = [dev newBufferWithLength:n*4 options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> pd = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dbg_div"] error:&err];
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pd]; [e setBuffer:b1 offset:0 atIndex:0]; [e setBytes:&ws length:4 atIndex:1]; [e setBuffer:b2 offset:0 atIndex:2]; [e setBuffer:b3 offset:0 atIndex:3];
            [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            const float * g1 = (const float *) b2.contents; const float * g2 = (const float *) b3.contents; long d1 = 0, d2 = 0;
            for (int i = 0; i < n; i++) { float c = ws / sc[i]; d1 += (g1[i] != c); d2 += (g2[i] != c); }
            { // isolated GPU multiply (float)int * q vs CPU
                const int m = 4096; std::vector<int> iv(m); uint32_t st3 = 7; for (auto & v : iv) v = (int)(i2s_ref_rng(&st3) % 4000001) - 2000000;
                const float qq = ws / (127.0f / 8.24247074f);
                id<MTLBuffer> c1 = [dev newBufferWithBytes:iv.data() length:m*4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> c2 = [dev newBufferWithLength:m*4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> c3 = [dev newBufferWithLength:m*4 options:MTLResourceStorageModeShared];
                id<MTLComputePipelineState> pm2 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dbg_mul"] error:&err];
                id<MTLCommandBuffer> cb2 = [q commandBuffer]; id<MTLComputeCommandEncoder> e2 = [cb2 computeCommandEncoder];
                [e2 setComputePipelineState:pm2]; [e2 setBuffer:c1 offset:0 atIndex:0]; [e2 setBytes:&qq length:4 atIndex:1]; [e2 setBuffer:c2 offset:0 atIndex:2]; [e2 setBuffer:c3 offset:0 atIndex:3];
                [e2 dispatchThreads:MTLSizeMake(m,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [e2 endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];
                const float * h1 = (const float *) c2.contents; const float * h2 = (const float *) c3.contents; long e1 = 0, e2c = 0;
                for (int i = 0; i < m; i++) { float c = (float) iv[i] * qq; e1 += (h1[i] != c); e2c += (h2[i] != c); }
                printf("debug: isolated GPU (float)int*q differs from CPU: %ld/%d (fma form: %ld/%d)\n", e1, m, e2c, m);
            }
            printf("debug: ws/s vs CPU IEEE: precise::divide differs %ld/%d, plain '/' differs %ld/%d\n", d1, n, d2, n);
        }

        if (getenv("BX_DEBUG") && sums.size() == (size_t) N*(NC > 1 ? NC-1 : NC)) {
            long a=0,b=0,c2=0,d=0; size_t n = sums.size();
            for (size_t i = 0; i < n; i++) {
                float g = got[i]; float fs = (float) sums[i]; float s_ = sc_of[i];
                a  += (g == fs * (ws/s_)); b += (g == (fs*ws)/s_); c2 += (g == fs/s_*ws); d += (g == (float)((double) sums[i] * ws / s_));
            }
            int shown = 0;
            for (size_t i = 0; i < n && shown < 6; i++) {
                float fs = (float) sums[i]; float s_ = sc_of[i];
                if (got[i] != fs * (ws/s_)) { shown++; printf("  i=%zu sum=%lld s=%.9g q=ws/s=%.9g cpu=%.9g gpu=%.9g  gpu/q=%.6f\n", i, (long long) sums[i], s_, ws/s_, fs*(ws/s_), got[i], got[i]/(ws/s_)); }
            }
            { long e=0,f=0,g=0,h=0;
              for (size_t i = 0; i < n; i++) { float fs=(float)sums[i], s_=sc_of[i], inv=1.0f/s_;
                e += got[i]==(fs*ws)*inv; f += got[i]==fs*(ws*inv); g += got[i]==(fs*inv)*ws; h += got[i]==fmaf(fs, ws*inv, 0.0f); }
              printf("debug: GPU == (sum*ws)*(1/s): %ld, sum*(ws*(1/s)): %ld, (sum*(1/s))*ws: %ld of %zu\n", e, f, g, n); }
            printf("debug: GPU == sum*(ws/s): %ld, (sum*ws)/s: %ld, sum/s*ws: %ld, exact-double: %ld of %zu\n", a, b, c2, d, n);
        }

        // timing of the mat-vec kernel alone
        double ms = 0;
        {
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            for (int i = 0; i < iters; i++) { encode(e, false); }
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3 / iters;
        }
        const double gbs = ((double) N*K/4) / (ms*1e-3) / 1e9;
        printf("K=%-5d N=%-5d ncols=%-2d NSG=%d R=%d C=%d : %s (%ld/%zu differ, max |d|=%g) | mv %.1f us, %.0f GB/s of weights\n",
               K, N, NC, NSG, R, C, bad ? "MISMATCH" : "ok", bad, (size_t) NC*N, maxd, ms*1e3, gbs);
        return bad ? 1 : 0;
    }
}
