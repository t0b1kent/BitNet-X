#!/usr/bin/env bash
# E0b and E0b-3 under the lenient protocol: 384-token budget, the last non-empty reply line is scored.
# Same questions for every model. Separates "can reason over the facts" from "obeys 'reply with only the answer'".
#   run_lenient.sh 2B 4B bitnet     (any subset; each model gets its own server, stopped when it is done or on exit)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../results/cmp/lenient"
PY="$HOME/Developer/MicrosoftAI/.venv/bin/python"
PRISM_BIN="$HOME/Developer/MicrosoftAI/llama-prism-b12128/build/bin"   # PrismML fork prism-b12128-d7b2896
BITNET_BIN="${BITNET_BIN:-/private/tmp/bitnetx-build-int/bin}"        # BitNet-X build (patches 0001-0012)
MDIR="$HOME/Developer/MicrosoftAI/Bonsai"
BITNET_MODEL="$HOME/Developer/MicrosoftAI/BitNet-GGUF/ggml-model-i2_s.gguf"
Q="$HERE/../results/e0b_qa/questions.json"
OPTS=(--max-tokens 384 --line last)
PORT=8101
SPID=""
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null' EXIT INT TERM
mkdir -p "$RES"
cd "$HERE"

for m in "$@"; do
    if [ "$m" = bitnet ]; then
        OUT="$RES/bitnet"; API=bitnet
        SRV=("$BITNET_BIN/llama-server" -m "$BITNET_MODEL" -ngl 99 -c 4096 --host 127.0.0.1 --port $PORT -np 1 -t 4)
    else
        OUT="$RES/bonsai2-$m"; API=chat
        SRV=("$PRISM_BIN/llama-server" -m "$MDIR/Ternary-Bonsai-2-$m-PQ2_0.gguf" -ngl 99 -c 8192 --host 127.0.0.1 --port $PORT -np 1 -t 4 --jinja)
    fi
    mkdir -p "$OUT"
    "${SRV[@]}" > "$OUT/server.log" 2>&1 &
    SPID=$!
    for i in $(seq 1 120); do curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 1; done
    echo "== $m E0b $(date +%H:%M)"
    BX_API=$API "$PY" qa_probe.py --out "$OUT/e0b" --worlds 5 --per-type 6 --port $PORT "${OPTS[@]}" > "$OUT/e0b.log" 2>&1
    cmp -s "$Q" "$OUT/e0b/questions.json" && echo "questions identical" || echo "WARNING: questions differ"
    echo "== $m noise $(date +%H:%M)"
    BX_API=$API "$PY" qa_noise.py --questions "$Q" --out "$OUT/noise" --levels 0,3,10,30,all --port $PORT "${OPTS[@]}" > "$OUT/noise.log" 2>&1
    kill $SPID; wait $SPID 2>/dev/null; SPID=""
done
echo "ALL DONE $(date +%H:%M)"
