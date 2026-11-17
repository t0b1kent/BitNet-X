#!/usr/bin/env bash
# Tokenisation of the official GGUF must equal the HF tokenizer.json result.
# Expected ids are stored in tests/data/tokenizer_expected.json (generated with
# tests/py/gen_tokenizer_expected.py from the BF16 repo's tokenizer.json).
# The GGUF has no tokenizer.ggml.pre -> llama.cpp uses the 'default'
# pre-tokenizer and splits "$3", "'s" etc. differently (microsoft/BitNet#684).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
it_require
[ -x "${BITNETX_BIN}/llama-tokenize" ] || { echo "SKIP: no llama-tokenize"; exit 77; }
python3 - "$IT_ROOT/tests/data/tokenizer_expected.json" "${BITNETX_BIN}/llama-tokenize" "${BITNET_GGUF}" <<'PY'
import json, subprocess, sys
cases = json.load(open(sys.argv[1]))["cases"]
fail = 0
for c in cases:
    r = subprocess.run([sys.argv[2], "-m", sys.argv[3], "-p", c["text"], "--ids", "--log-disable", "--no-escape"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: llama-tokenize exit", r.returncode); fail = 1; continue
    got = json.loads(r.stdout.strip().splitlines()[-1])
    ok = got == c["ids"]
    print(("ok   " if ok else "FAIL ") + f"{len(got):3d} tokens (HF: {len(c['ids'])}) for {c['text']!r}")
    if not ok:
        print("  gguf:", got); print("  hf:  ", c["ids"]); fail = 1
sys.exit(fail)
PY
