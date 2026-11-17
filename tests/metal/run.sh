#!/usr/bin/env bash
# Build and run the standalone I2_S Metal kernel test/benchmark (macOS only).
#   tests/metal/run.sh                 correctness on the shapes of bitnet-b1.58-2B-4T + timing
#   tests/metal/run.sh K N ncols NSG R [iters]   single configuration
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
M="$ROOT/3rdparty/llama.cpp/ggml/src/ggml-metal"
OUT="${BITNETX_TEST_OUT:-/private/tmp/bitnetx-tests}"; mkdir -p "$OUT"
[ "$(uname -s)" = "Darwin" ] || { echo "SKIP: not macOS"; exit 77; }
# one Metal source: ggml-metal-impl.h (plain C structs/defines) + the marked kernel block of ggml-metal.metal
{
    echo '#include <metal_stdlib>'; echo 'using namespace metal;'
    echo '#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)'
    grep -v '^#include' "$M/ggml-metal-impl.h"
    sed -n '/BEGIN_BITNETX_I2S/,/END_BITNETX_I2S/p' "$M/ggml-metal.metal"
} > "$OUT/i2s_kernels.metal"
grep -q kernel_mul_mv_i2_s_i8 "$OUT/i2s_kernels.metal" || { echo "kernel block not found (patch 0010 not applied?)"; exit 1; }
clang++ -O2 -std=c++17 -fobjc-arc -framework Metal -framework Foundation "$ROOT/tests/metal/i2s_kernel_bench.mm" -o "$OUT/i2s_kernel_bench"
if [ $# -ge 5 ]; then exec "$OUT/i2s_kernel_bench" "$OUT/i2s_kernels.metal" "$@"; fi
rc=0
# every shape with the int8 kernels and with the bit-serial (popcount) kernels; rows per simdgroup of the latter depend on the
# number of columns processed together (1 -> 8, 2..3 -> 4, >= 4 -> 2), see ggml_metal_op_mul_mat_i2_s
for cfg in "2560 2560 1" "2560 6912 1" "6912 2560 1" "2560 640 1" "2560 6912 2" "2560 6912 4" "2560 6912 8" "2560 2560 3" "6912 2560 33"; do
    set -- $cfg
    "$OUT/i2s_kernel_bench" "$OUT/i2s_kernels.metal" "$1" "$2" "$3" "${BX_NSG:-4}" "${BX_R:-4}" || rc=1
    if [ "$3" -ge 4 ]; then rb=2; elif [ "$3" -ge 2 ]; then rb=4; else rb=8; fi
    BX_BITS=1 "$OUT/i2s_kernel_bench" "$OUT/i2s_kernels.metal" "$1" "$2" "$3" "${BX_NSG:-4}" "$rb" || rc=1
done
exit $rc
