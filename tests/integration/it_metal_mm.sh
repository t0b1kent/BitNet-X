#!/usr/bin/env bash
# macOS only: prefill with more than 8 columns runs the simdgroup-matrix kernel (patch 0012). Its result must be the same
# as with the matrix path switched off (GGML_METAL_I2S_NO_MM=1, bit-serial kernels): same greedy continuation, same perplexity.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ "$(uname -s)" = "Darwin" ] || { echo "SKIP: not macOS"; exit 77; }
it_require

# well over 8 tokens, so the whole prompt is one batch of > 8 columns
PROMPT="The capital of France is the city of Paris, the capital of Italy is the city of Rome, and the capital of Spain is"

mm="$(it_complete 99 8 "$PROMPT")"; rc_mm=$?
ref="$(GGML_METAL_I2S_NO_MM=1 it_complete 99 8 "$PROMPT")"; rc_ref=$?
echo "matrix path:     [${mm#"$PROMPT"}] (exit $rc_mm)"
echo "bit-serial path: [${ref#"$PROMPT"}] (exit $rc_ref)"
[ $rc_mm -eq 0 ] && [ $rc_ref -eq 0 ] || { echo "FAIL: aborted"; exit 1; }
it_is_degenerate "${mm#"$PROMPT"}" && { echo "FAIL: degenerate continuation on the matrix path"; exit 1; }
[ "$mm" = "$ref" ] || { echo "FAIL: matrix path and bit-serial path continue differently"; exit 1; }
echo "ok: same continuation"

# perplexity with -b 16: every batch has 16 columns
if [ -x "${BITNETX_BIN}/llama-perplexity" ] && [ -f "$IT_ROOT/tests/data/ppl_small.txt" ]; then
    ppl() { "${BITNETX_BIN}/llama-perplexity" -m "${BITNET_GGUF}" -f "$IT_ROOT/tests/data/ppl_small.txt" -c 256 -b 16 \
                -ngl 99 -t "${IT_THREADS}" 2>&1 | sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p'; }
    p_mm="$(ppl)"; p_ref="$(GGML_METAL_I2S_NO_MM=1 ppl)"
    echo "perplexity: matrix path $p_mm, bit-serial path $p_ref"
    [ -n "$p_mm" ] && [ "$p_mm" = "$p_ref" ] || { echo "FAIL: perplexity differs (or was not printed)"; exit 1; }
    echo "ok: same perplexity"
else
    echo "note: llama-perplexity or tests/data/ppl_small.txt missing, perplexity check skipped"
fi
exit 0
