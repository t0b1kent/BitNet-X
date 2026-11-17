#!/usr/bin/env bash
# CPU path (-ngl 0) on the official I2_S GGUF must not produce degenerate text.
# On arm64 the scalar fallback of ggml_vec_dot_i2_i8_s read the weights with the
# wrong layout -> "@@@@@@@@..." (microsoft/BitNet#682, #348).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
it_require
out="$(it_complete 0 16 "$IT_PROMPT")"; rc=$?
cont="${out#"$IT_PROMPT"}"
echo "exit=$rc continuation: [$cont]"
[ $rc -eq 0 ] || { echo "FAIL: llama-completion exit $rc"; exit 1; }
if it_is_degenerate "$cont"; then echo "FAIL: degenerate continuation"; exit 1; fi
exit 0
