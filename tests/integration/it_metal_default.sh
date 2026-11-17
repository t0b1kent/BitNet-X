#!/usr/bin/env bash
# macOS only: running with GPU offload requested (-ngl 99, also the default)
# must not abort. The Metal backend has no kernels for BitNet packed types but
# claimed MUL_MAT for them -> "Asserting on type 36 ... not implemented" (SIGABRT).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ "$(uname -s)" = "Darwin" ] || { echo "SKIP: not macOS"; exit 77; }
it_require
out="$(it_complete 99 4 "$IT_PROMPT")"; rc=$?
echo "llama-completion (-ngl 99) exit status: $rc, continuation: [${out#"$IT_PROMPT"}]"
[ $rc -eq 0 ] || { echo "FAIL: aborted (134 = SIGABRT)"; exit 1; }
# the I2_S mat-vec runs on the GPU now (patch 0010): the answer must be the correct one
case "${out#"$IT_PROMPT"}" in
    " Paris"*) echo "ok: Metal continuation starts with ' Paris'" ;;
    *) echo "FAIL: Metal continuation does not start with ' Paris'"; exit 1 ;;
esac
exit 0
