# shellcheck shell=bash
# Shared helpers for integration tests. Exit code 77 = SKIP.
# Needs: BITNETX_BIN (directory with llama-* binaries), BITNET_GGUF (official i2_s gguf).
IT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IT_THREADS="${BITNETX_THREADS:-4}"
IT_PROMPT="The capital of France is"

it_require() {
    if [ -z "${BITNETX_BIN:-}" ] || [ ! -x "${BITNETX_BIN}/llama-completion" ]; then
        echo "SKIP: BITNETX_BIN is not set or has no llama-completion (build first, see tests/README.md)"; exit 77
    fi
    if [ -z "${BITNET_GGUF:-}" ] || [ ! -f "${BITNET_GGUF}" ]; then
        echo "SKIP: BITNET_GGUF (official ggml-model-i2_s.gguf) is not set"; exit 77
    fi
}

# it_complete <ngl> <n_predict> <prompt> ; prints the full text (prompt+continuation),
# returns the binary's exit status. Greedy, fixed seed, no warmup, no chat template.
it_complete() {
    "${BITNETX_BIN}/llama-completion" -m "${BITNET_GGUF}" -p "$3" -n "$2" --temp 0 --seed 1 \
        -ngl "$1" -no-cnv --no-warmup -t "${IT_THREADS}" 2>"${IT_STDERR:-/dev/null}"
}

# degenerate = the same character 8+ times in a row (e.g. "@@@@@@@@") or
# fewer than 3 distinct alphabetic words in the continuation
it_is_degenerate() {
    local text="$1"
    if printf '%s' "$text" | grep -Eq '(.)\1{7,}'; then return 0; fi
    local words
    words="$(printf '%s' "$text" | tr -cs 'A-Za-z' '\n' | grep -c . || true)"
    local distinct
    distinct="$(printf '%s' "$text" | tr -cs 'A-Za-z' '\n' | grep . | sort -u | wc -l | tr -d ' ')"
    [ "${distinct:-0}" -lt 3 ] && return 0
    return 1
}
