#!/usr/bin/env bash
# A prompt of 32+ tokens on the CPU backend must not crash.
# With GGML_BLAS (Accelerate, default on macOS) the BLAS backend claims I2_S
# MUL_MAT for batches >= 32 and converts the weights with the generic
# to_float(x + i01*nb01, ..., ne00) call, which is wrong for the 2-bit packed
# rows and for dequantize_row_i2_s's 4-argument signature -> SIGSEGV.
# Only the exit status is checked here (text quality is covered elsewhere).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
it_require
prompt="$(head -c 700 "$IT_ROOT/tests/data/ppl_small.txt")"
it_complete 0 2 "$prompt" >/dev/null; rc=$?
echo "llama-completion (-ngl 0, ~150 token prompt) exit status: $rc"
[ $rc -eq 0 ] || { echo "FAIL: crashed (139 = SIGSEGV)"; exit 1; }
exit 0
