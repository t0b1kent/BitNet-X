#!/usr/bin/env bash
# E0b-4 counting methods on each model; one server per model with 8 slots, stopped when done or on exit.
#   run_count.sh bitnet 2B 4B
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../results/cmp/count"
PY="$HOME/Developer/MicrosoftAI/.venv/bin/python"
PRISM_BIN="$HOME/Developer/MicrosoftAI/llama-prism-b12128/build/bin"
BITNET_BIN="${BITNET_BIN:-/private/tmp/bitnetx-build-int/bin}"
MDIR="$HOME/Developer/MicrosoftAI/Bonsai"
BITNET_MODEL="$HOME/Developer/MicrosoftAI/BitNet-GGUF/ggml-model-i2_s.gguf"
Q="$HERE/../results/e0b_qa/questions.json"
PORT=8101
SPID=""
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null' EXIT INT TERM
cd "$HERE"
for m in "$@"; do
    if [ "$m" = bitnet ]; then
        OUT="$RES/bitnet"; API=bitnet
        SRV=("$BITNET_BIN/llama-server" -m "$BITNET_MODEL" -ngl 99 -c 32768 --host 127.0.0.1 --port $PORT -np 8 -t 4)
    else
        OUT="$RES/bonsai2-$m"; API=chat
        SRV=("$PRISM_BIN/llama-server" -m "$MDIR/Ternary-Bonsai-2-$m-PQ2_0.gguf" -ngl 99 -c 32768 --host 127.0.0.1 --port $PORT -np 8 -t 4 --jinja)
    fi
    mkdir -p "$OUT"
    "${SRV[@]}" > "$OUT/server.log" 2>&1 &
    SPID=$!
    for i in $(seq 1 120); do curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 1; done
    echo "== $m $(date +%H:%M)"
    BX_API=$API "$PY" qa_count.py --questions "$Q" --out "$OUT" --levels 0,30,all --port $PORT --jobs 8 > "$OUT/run.log" 2>&1
    tail -8 "$OUT/run.log"
    kill $SPID; wait $SPID 2>/dev/null; SPID=""
done
echo "ALL DONE $(date +%H:%M)"
