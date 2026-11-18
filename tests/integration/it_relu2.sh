#!/usr/bin/env bash
# bitnet-b1.58 uses squared ReLU in the FFN (config.json "hidden_act": "relu2").
# With SiLU (microsoft/BitNet#588) the greedy continuation of
# "The capital of France is" is " a small city, and the capital of France is..."
# and perplexity is several times higher.
#
# Prerequisite on arm64: a correct I2_S CPU kernel. If the CPU output is
# degenerate this test is SKIPPED (it cannot tell SiLU from ReLU^2 then).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
it_require
out="$(it_complete 0 16 "$IT_PROMPT")"; rc=$?
cont="${out#"$IT_PROMPT"}"
echo "exit=$rc continuation: [$cont]"
[ $rc -eq 0 ] || { echo "FAIL: llama-completion exit $rc"; exit 1; }
if it_is_degenerate "$cont"; then
    echo "SKIP: CPU I2_S kernel output is degenerate (needs the I2_S layout fix first)"; exit 77
fi
fail=0
case "$cont" in
    " Paris"*) echo "ok: continuation starts with ' Paris'" ;;
    *) echo "FAIL: continuation does not start with ' Paris'"; fail=1 ;;
esac

# Perplexity on a small original text (tests/data/ppl_small.txt), 4 chunks of 256.
# -b 16 keeps the batch below the BLAS backend threshold (32).
# Measured on M1 Pro: ReLU^2 6.4-7.3, SiLU 59-72. Threshold 20.
if [ -x "${BITNETX_BIN}/llama-perplexity" ]; then
    ppl_line="$("${BITNETX_BIN}/llama-perplexity" -m "${BITNET_GGUF}" -f "$IT_ROOT/tests/data/ppl_small.txt" \
        -c 256 -b 16 -ngl 0 -t "${IT_THREADS}" 2>&1 | grep -E "Final estimate" | tail -1)"
    ppl="$(printf '%s' "$ppl_line" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')"
    echo "perplexity: ${ppl_line#*Final estimate: }"
    if [ -z "$ppl" ]; then echo "FAIL: no perplexity estimate"; fail=1
    elif awk -v p="$ppl" 'BEGIN{exit !(p < 20.0)}'; then echo "ok: PPL $ppl < 20"
    else echo "FAIL: PPL $ppl >= 20"; fail=1; fi
fi
exit $fail
