#!/usr/bin/env bash
# Round 2 towards exact counting and answering: qa_count.py (judge2, program_v, scan) and qa_table.py (read once into a
# table) on each model; one server per model with 8 slots, stopped when done or on exit.
#   run_round2.sh bitnet 2B 4B
#   PILOT=1 run_round2.sh bitnet      (3 questions, level 30, output under .../pilot)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/Developer/MicrosoftAI/.venv/bin/python"
PRISM_BIN="$HOME/Developer/MicrosoftAI/llama-prism-b12128/build/bin"
BITNET_BIN="${BITNET_BIN:-/private/tmp/bitnetx-build-int/bin}"
MDIR="$HOME/Developer/MicrosoftAI/Bonsai"
BITNET_MODEL="$HOME/Developer/MicrosoftAI/BitNet-GGUF/ggml-model-i2_s.gguf"
Q="$HERE/../results/e0b_qa/questions.json"
PORT=8101
PILOT="${PILOT:-0}"
SPID=""
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null' EXIT INT TERM
cd "$HERE"
for m in "$@"; do
    if [ "$m" = bitnet ]; then
        NAME=bitnet; API=bitnet
        SRV=("$BITNET_BIN/llama-server" -m "$BITNET_MODEL" -ngl 99 -c 32768 --host 127.0.0.1 --port $PORT -np 8 -t 4)
    else
        NAME="bonsai2-$m"; API=chat
        SRV=("$PRISM_BIN/llama-server" -m "$MDIR/Ternary-Bonsai-2-$m-PQ2_0.gguf" -ngl 99 -c 32768 --host 127.0.0.1 --port $PORT -np 8 -t 4 --jinja)
    fi
    if [ "$PILOT" = 1 ]; then CO="$HERE/../results/cmp/pilot/count/$NAME"; TO="$HERE/../results/cmp/pilot/table/$NAME"; LV=30; LIM=(--limit 3)
    else CO="$HERE/../results/cmp/count/$NAME"; TO="$HERE/../results/cmp/table/$NAME"; LV=0,30,all; LIM=(); fi
    mkdir -p "$CO" "$TO"
    "${SRV[@]}" > "$TO/server.log" 2>&1 &
    SPID=$!
    for i in $(seq 1 120); do curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 1; done
    echo "== $m count round 2 $(date +%H:%M)"
    BX_API=$API "$PY" qa_count.py --questions "$Q" --out "$CO" --levels $LV --methods judge2,program_v,scan --log count2.jsonl \
        --port $PORT --jobs 8 ${LIM[@]+"${LIM[@]}"} > "$CO/run2.log" 2>&1
    tail -6 "$CO/run2.log"
    if [ "$PILOT" != 1 ]; then
        echo "== $m table $(date +%H:%M)"
        BX_API=$API "$PY" qa_table.py --questions "$Q" --out "$TO" --port $PORT --jobs 8 > "$TO/run.log" 2>&1
        tail -8 "$TO/run.log"
    fi
    kill $SPID; wait $SPID 2>/dev/null; SPID=""
done
echo "ALL DONE $(date +%H:%M)"
