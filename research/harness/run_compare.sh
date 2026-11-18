#!/usr/bin/env bash
# Comparison BitNet 2B4T vs Ternary Bonsai 2 (2B, 4B): same tasks, same harness, each model in its own chat format.
# BitNet results already exist (results/e0_dev, results/e0b_qa, results/e0b3_noise/bitnet).
#   run_compare.sh 2B 4B
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../results/cmp"
PY="$HOME/Developer/MicrosoftAI/.venv/bin/python"
BIN="$HOME/Developer/MicrosoftAI/llama-prism-b12128/build/bin"  # PrismML fork release prism-b12128-d7b2896 (hadamard v2)
MDIR="$HOME/Developer/MicrosoftAI/Bonsai"
PORT=8101
export BX_API=chat
mkdir -p "$RES"

for size in "$@"; do
    M="$MDIR/Ternary-Bonsai-2-$size-PQ2_0.gguf"
    OUT="$RES/bonsai2-$size"; mkdir -p "$OUT"
    echo "== $size: speed (server not running)"
    "$BIN/llama-bench" -m "$M" -p 512 -n 64 -t 4 -ngl 99 -r 3 2>&1 | grep -E "^\| " > "$OUT/bench.md"
    cat "$OUT/bench.md"
    echo "== $size: server"
    "$BIN/llama-server" -m "$M" -ngl 99 -c 8192 --host 127.0.0.1 --port $PORT -np 1 -t 4 --jinja > "$OUT/server.log" 2>&1 &
    SPID=$!
    for i in $(seq 1 120); do curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 1; done
    cd "$HERE"
    echo "== $size: E0 (A, B)"
    "$PY" run_modes.py --tasks ../tasks/dev_100.jsonl --modes A,B --out "$OUT/e0" --port $PORT > "$OUT/e0.log" 2>&1
    "$PY" summarize.py "$OUT/e0" > /dev/null
    echo "== $size: E0b"
    "$PY" qa_probe.py --out "$OUT/e0b" --worlds 5 --per-type 6 --port $PORT > "$OUT/e0b.log" 2>&1
    echo "== $size: E0b-3 noise curve"
    "$PY" qa_noise.py --questions ../results/e0b_qa/questions.json --out "$OUT/noise" --levels 0,3,10,30,all --port $PORT > "$OUT/noise.log" 2>&1
    kill $SPID; wait $SPID 2>/dev/null
    echo "== $size done"
done
echo "ALL DONE"
