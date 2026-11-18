#!/usr/bin/env python3
"""Regenerate tests/data/tokenizer_expected.json from the HF tokenizer.json.

Needs the `tokenizers` package (install it in a virtualenv outside the git
tree) and BITNET_BF16 = directory of microsoft/bitnet-b1.58-2B-4T-bf16.
"""
import json, os, sys
from tokenizers import Tokenizer

TEXTS = [
    "In 1999, 1234567 people paid $3.50 each; it's 2,560 x 6912.",
    "The capital of France is",
    "def f(x):\n    return x**2 + 3*x - 10  # don't panic",
    "Hello,   world!  We're 100% sure: 3.14159 != 22/7.",
]
tok = Tokenizer.from_file(os.path.join(os.environ["BITNET_BF16"], "tokenizer.json"))
cases = [{"text": t, "ids": tok.encode(t).ids} for t in TEXTS]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tokenizer_expected.json")
with open(out, "w") as f:
    json.dump({"source": "microsoft/bitnet-b1.58-2B-4T-bf16 tokenizer.json (HF tokenizers)", "cases": cases}, f, indent=1)
    f.write("\n")
for c in cases:
    print(len(c["ids"]), repr(c["text"]))
