#!/usr/bin/env bash
# Build smoke test: configure with exactly the flags setup_env.py uses on this
# architecture and build the main tools. Expects success.
#
# Environment:
#   BITNETX_BUILD_DIR      build directory (default: /private/tmp/bitnetx-build-smoke
#                          on macOS, ${TMPDIR:-/tmp}/bitnetx-build-smoke elsewhere);
#                          never inside the git tree
#   BITNETX_JOBS           parallel jobs (default 6)
#   BITNETX_SMOKE_TARGETS  space separated targets, or "all" for the full build
#                          setup_env.py performs (default: the tools the tests use)
#   BITNETX_CMAKE_EXTRA    extra cmake configure arguments (space separated)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$(uname -s)" = "Darwin" ]; then DEF_DIR=/private/tmp/bitnetx-build-smoke; else DEF_DIR="${TMPDIR:-/tmp}/bitnetx-build-smoke"; fi
BUILD="${BITNETX_BUILD_DIR:-$DEF_DIR}"
JOBS="${BITNETX_JOBS:-6}"
TARGETS="${BITNETX_SMOKE_TARGETS:-llama-cli llama-completion llama-tokenize llama-perplexity llama-quantize llama-server}"

case "$(uname -m)" in
    arm64|aarch64) ARCH_ARGS=(-DBITNET_ARM_TL1=OFF) ;;
    x86_64|AMD64)  ARCH_ARGS=(-DBITNET_X86_TL2=OFF) ;;
    *) echo "unsupported arch $(uname -m)"; exit 77 ;;
esac

mkdir -p "$BUILD"
echo "build dir: $BUILD"
"$ROOT/scripts/apply_patches.sh" || { echo "apply_patches failed"; exit 1; }

# same flags as setup_env.py compile()
# shellcheck disable=SC2086
if ! cmake -S "$ROOT" -B "$BUILD" "${ARCH_ARGS[@]}" \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON \
        ${BITNETX_CMAKE_EXTRA:-} > "$BUILD/smoke-configure.log" 2>&1; then
    echo "CONFIGURE FAILED (log: $BUILD/smoke-configure.log)"
    tail -20 "$BUILD/smoke-configure.log"
    exit 1
fi

if [ "$TARGETS" = "all" ]; then TGT=(); else TGT=(--target $TARGETS); fi
if ! cmake --build "$BUILD" --config Release -j "$JOBS" "${TGT[@]}" > "$BUILD/smoke-build.log" 2>&1; then
    echo "BUILD FAILED (log: $BUILD/smoke-build.log)"
    grep -E -B2 -A12 "Undefined symbols|error:" "$BUILD/smoke-build.log" | head -40
    exit 1
fi
echo "build ok: $(ls "$BUILD/bin" | grep -c '^llama-') llama-* binaries in $BUILD/bin"
