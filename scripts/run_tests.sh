#!/usr/bin/env bash
# BitNet-X test runner. Prints one PASS / FAIL / SKIP / XFAIL / XPASS line per test.
#
#   scripts/run_tests.sh              unit tests + build smoke + integration tests
#   scripts/run_tests.sh --no-build   skip the build smoke test (integration tests
#                                     then need BITNETX_BIN pointing to a build)
#   scripts/run_tests.sh -v           also print each test's output
#   scripts/run_tests.sh NAME...      run only the named tests
#
# Environment (all optional):
#   BITNET_GGUF   official ggml-model-i2_s.gguf     (real-weights + integration tests)
#   BITNET_BF16   dir with the BF16 model.safetensors (real-weights layout test)
#   BITNETX_BIN   dir with llama-* binaries; default: bin/ of the smoke build if it succeeded
#   BITNETX_BUILD_DIR, BITNETX_JOBS, BITNETX_SMOKE_TARGETS, BITNETX_CMAKE_EXTRA: see tests/build_smoke.sh
#   BITNETX_TEST_OUT  scratch dir for test binaries/logs (default /private/tmp/bitnetx-tests or $TMPDIR)
#
# tests/KNOWN_FAILURES lists tests that are expected to fail on this branch
# (each open defect). A listed test that fails is reported as XFAIL and does
# not make the run fail; a listed test that passes is XPASS and DOES make the
# run fail (remove it from the list in the PR that fixes it).
# Exit status: 0 if there is no FAIL and no XPASS.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L="$ROOT/3rdparty/llama.cpp"
T="$ROOT/tests"
if [ "$(uname -s)" = "Darwin" ]; then DEF_TMP=/private/tmp; else DEF_TMP="${TMPDIR:-/tmp}"; fi
OUT="${BITNETX_TEST_OUT:-$DEF_TMP/bitnetx-tests}"
mkdir -p "$OUT"
export BITNETX_BUILD_DIR="${BITNETX_BUILD_DIR:-$DEF_TMP/bitnetx-build-smoke}"

CC="${CC:-clang}"; CXX="${CXX:-clang++}"
INC=(-I"$L/ggml/include" -I"$L/ggml/src" -I"$L/ggml/src/ggml-cpu" -I"$L/include" -I"$ROOT/include" -I"$T/c")
case "$(uname -s)" in
    Darwin) UNDEF=(-Wl,-undefined,dynamic_lookup) ;;
    *)      UNDEF=(-Wl,--unresolved-symbols=ignore-all -lm) ;;
esac
QUANTS="$L/ggml/src/ggml-cpu/quants.c"
QBASE="$L/ggml/src/ggml-quants.c"   # ggml-base: quantize_i2_s / dequantize_row_i2_s live here once patch 0001 is applied
I2S="$L/ggml/src/ggml-cpu/ggml-cpu-i2s.c"

VERBOSE=0; DO_BUILD=1; ONLY=()
for a in "$@"; do
    case "$a" in
        -v|--verbose) VERBOSE=1 ;;
        --no-build)   DO_BUILD=0 ;;
        -h|--help)    sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            ONLY+=("$a") ;;
    esac
done

# ---- test bodies (exit 0 = pass, 77 = skip, anything else = fail) -------------
cc_c()   { "$CC"  -O1 -std=gnu11 -w "${INC[@]}" "$@" "${UNDEF[@]}"; }
cc_cxx() { "$CXX" -O1 -std=c++17 -w "${INC[@]}" "$@" "${UNDEF[@]}"; }

t_py_i2s_reference()   { python3 "$T/py/test_i2s_ref.py"; }
t_gguf_layout_real()   { python3 "$T/py/test_gguf_layout.py"; }
t_i2s_roundtrip()      { cc_c "$T/c/test_i2s_roundtrip.c" "$QUANTS" "$QBASE" -o "$OUT/t_i2s_roundtrip" && "$OUT/t_i2s_roundtrip"; }
t_i2s_dot()            { cc_c "$T/c/test_i2s_dot.c" "$QUANTS" "$I2S" -o "$OUT/t_i2s_dot" && "$OUT/t_i2s_dot" vecdot; }
t_i2s_gemm()           { cc_c "$T/c/test_i2s_dot.c" "$QUANTS" "$I2S" -o "$OUT/t_i2s_dot" && "$OUT/t_i2s_dot" gemm; }
t_i2s_dot_nodotprod()  {
    case "$(uname -m)" in arm64|aarch64) ;; *) echo "SKIP: arm64 only"; return 77 ;; esac
    cc_c -march=armv8-a+nodotprod "$T/c/test_i2s_dot.c" "$QUANTS" "$I2S" -o "$OUT/t_i2s_dot_nodotprod" && "$OUT/t_i2s_dot_nodotprod" vecdot
}
t_i2s_gemm_nodotprod() {
    case "$(uname -m)" in arm64|aarch64) ;; *) echo "SKIP: arm64 only"; return 77 ;; esac
    cc_c -march=armv8-a+nodotprod "$T/c/test_i2s_dot.c" "$QUANTS" "$I2S" -o "$OUT/t_i2s_dot_nodotprod" && "$OUT/t_i2s_dot_nodotprod" gemm
}
t_mad_layout()         { "$CC" -c -w "$T/c/stubs_mad.c" -o "$OUT/stubs_mad.o" && cc_cxx "$T/c/test_mad_layout.cpp" "$ROOT/src/ggml-bitnet-mad.cpp" "$OUT/stubs_mad.o" -o "$OUT/t_mad_layout" && "$OUT/t_mad_layout"; }
t_metal_i2s_kernel()   { bash "$T/metal/run.sh"; }
t_src1_cont()          { bash "$T/test_src1_cont.sh"; }
t_setup_env()          { python3 "$T/py/test_setup_env.py"; }
t_patches_idempotent() {
    # applying twice must be a no-op the second time, and --check must agree
    "$ROOT/scripts/apply_patches.sh" && "$ROOT/scripts/apply_patches.sh" && "$ROOT/scripts/apply_patches.sh" --check
}
t_build_smoke()        {
    if [ "$DO_BUILD" -eq 0 ]; then echo "SKIP: --no-build"; return 77; fi
    bash "$T/build_smoke.sh"
}
t_it_cpu_generation()  { bash "$T/integration/it_cpu_generation.sh"; }
t_it_relu2()           { bash "$T/integration/it_relu2.sh"; }
t_it_tokenizer()       { bash "$T/integration/it_tokenizer.sh"; }
t_it_blas_batch()      { bash "$T/integration/it_blas_batch.sh"; }
t_it_metal_default()   { bash "$T/integration/it_metal_default.sh"; }
t_it_metal_int8()      { bash "$T/integration/it_metal_int8.sh"; }

ALL_TESTS=(
    patches_idempotent
    py_i2s_reference
    gguf_layout_real
    i2s_roundtrip
    i2s_dot
    i2s_dot_nodotprod
    i2s_gemm
    i2s_gemm_nodotprod
    metal_i2s_kernel
    mad_layout
    src1_cont
    setup_env
    build_smoke
    it_cpu_generation
    it_relu2
    it_tokenizer
    it_blas_batch
    it_metal_default
    it_metal_int8
)

# ---- runner ------------------------------------------------------------------
is_known_failure() { grep -Eq "^$1([[:space:]]|\$)" "$T/KNOWN_FAILURES" 2>/dev/null; }

n_pass=0; n_fail=0; n_skip=0; n_xfail=0; n_xpass=0
run_one() {
    local name="$1" log="$OUT/$1.log" rc status
    ( "t_$name" ) >"$log" 2>&1; rc=$?
    # integration tests use the smoke build unless BITNETX_BIN was given
    if [ "$name" = "build_smoke" ] && [ $rc -eq 0 ] && [ -z "${BITNETX_BIN:-}" ]; then
        export BITNETX_BIN="$BITNETX_BUILD_DIR/bin"
    fi
    if   [ $rc -eq 0 ];  then if is_known_failure "$name"; then status=XPASS; n_xpass=$((n_xpass+1)); else status=PASS; n_pass=$((n_pass+1)); fi
    elif [ $rc -eq 77 ]; then status=SKIP; n_skip=$((n_skip+1))
    else                      if is_known_failure "$name"; then status=XFAIL; n_xfail=$((n_xfail+1)); else status=FAIL; n_fail=$((n_fail+1)); fi
    fi
    printf '%-6s %s\n' "$status" "$name"
    if [ "$VERBOSE" -eq 1 ] || [ "$status" = FAIL ] || [ "$status" = XPASS ]; then sed 's/^/       | /' "$log" | tail -40; fi
}

if [ "${#ONLY[@]}" -gt 0 ]; then TESTS=("${ONLY[@]}"); else TESTS=("${ALL_TESTS[@]}"); fi
for t in "${TESTS[@]}"; do
    if ! declare -F "t_$t" >/dev/null; then echo "unknown test: $t"; exit 2; fi
    run_one "$t"
done
echo "----"
echo "PASS=$n_pass FAIL=$n_fail SKIP=$n_skip XFAIL=$n_xfail XPASS=$n_xpass   (logs: $OUT)"
[ "$n_fail" -eq 0 ] && [ "$n_xpass" -eq 0 ]
