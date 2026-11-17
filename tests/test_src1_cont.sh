#!/usr/bin/env bash
# ggml-cpu.c must compile both with and without GGML_USE_LLAMAFILE
# (microsoft/BitNet#682 step 2: `src1_cont` is declared only under
# #if GGML_USE_LLAMAFILE but used by the unconditional I2_S fast path).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L="$ROOT/3rdparty/llama.cpp"
CC="${CC:-clang}"
INC=(-I"$L/ggml/include" -I"$L/ggml/src" -I"$L/ggml/src/ggml-cpu" -I"$ROOT/include")
rc=0
for def in "-DGGML_USE_LLAMAFILE=1" ""; do
    label="${def:-<GGML_USE_LLAMAFILE undefined>}"
    if out="$("$CC" -std=gnu11 -fsyntax-only -w "${INC[@]}" $def "$L/ggml/src/ggml-cpu/ggml-cpu.c" 2>&1)"; then
        echo "compiles  $label"
    else
        echo "ERROR     $label"
        echo "$out" | grep -E "error" | head -5
        rc=1
    fi
done
exit $rc
