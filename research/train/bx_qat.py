#!/usr/bin/env python3
"""BitNet b1.58 2B-4T in MLX with trainable ternary layers (quantization-aware training).

mlx-lm's BitLinear is inference-only (packed uint8 weights, custom kernel, no gradients). Here every BitLinear is a
QATLinear: latent BF16 weights; the forward pass uses the same online quantization as the released model
(weights: absmean ternary {-1, 0, 1} * scale; activations: per-token absmax int8) and the straight-through estimator,
so gradients reach the latent weights and the trained model stays exactly ternary (exportable to I2_S).

    ~/Developer/MicrosoftAI/.venv-train/bin/python bx_qat.py --check     # PPL like llama-perplexity -c 256, generation, one backward step
"""
import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.models.bitnet as B
from tokenizers import Tokenizer

MASTER = os.path.expanduser("~/Developer/MicrosoftAI/BitNet-Original")
BOS, EOT = 128000, 128009


def ste(x, q):
    return x + mx.stop_gradient(q - x)


class QATLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, **_):
        super().__init__()
        self.weight = mx.zeros((out_features, in_features), dtype=mx.bfloat16)
        if bias:
            self.bias = mx.zeros((out_features,))

    def __call__(self, x):
        w = self.weight.astype(mx.float32)
        s = 1.0 / mx.maximum(mx.mean(mx.abs(w)), 1e-5)
        wq = ste(w, mx.clip(mx.round(w * s), -1, 1) / s)
        xf = x.astype(mx.float32)
        sa = 127.0 / mx.maximum(mx.max(mx.abs(xf), axis=-1, keepdims=True), 1e-5)
        xq = ste(xf, mx.clip(mx.round(xf * sa), -128, 127) / sa)
        y = xq @ wq.T
        if "bias" in self:
            y = y + self.bias
        return y.astype(x.dtype)


def load(path=MASTER, dtype=mx.bfloat16):
    cfg = json.load(open(os.path.join(path, "config.json")))
    B.BitLinear = QATLinear  # every projection becomes trainable-ternary
    args = B.ModelArgs.from_dict(cfg)
    model = B.Model(args)
    w = mx.load(os.path.join(path, "model.safetensors"))
    w = model.sanitize({k: v.astype(dtype) for k, v in w.items()})
    model.load_weights(list(w.items()), strict=True)
    mx.eval(model.parameters())
    tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    return model, tok


def encode(tok, text, bos=True):
    ids = tok.encode(text, add_special_tokens=False).ids
    return ([BOS] if bos else []) + ids


def ppl_like_llama(model, ids, n_ctx=256):
    """llama-perplexity: chunks of n_ctx, first token of each chunk set to BOS, score the second half."""
    nll, cnt = 0.0, 0
    first = n_ctx // 2
    for c in range(len(ids) // n_ctx):
        chunk = list(ids[c * n_ctx:(c + 1) * n_ctx])
        chunk[0] = BOS
        x = mx.array(chunk)[None]
        logits = model(x).astype(mx.float32)[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array(chunk[first + 1:])
        sel = mx.take_along_axis(lp[first:n_ctx - 1], tgt[:, None], axis=-1)[:, 0]
        mx.eval(sel)
        nll -= float(mx.sum(sel)); cnt += len(chunk) - 1 - first
    return math.exp(nll / cnt), cnt


def chat_ids(tok, user, system="You are a helpful AI assistant."):
    text = f"System: {system}<|eot_id|>User: {user}<|eot_id|>Assistant: "
    return encode(tok, text)


def greedy(model, tok, ids, n=40):
    out = []
    for _ in range(n):
        logits = model(mx.array(ids + out)[None])[0, -1]
        t = int(mx.argmax(logits))
        if t in (EOT, 128001):
            break
        out.append(t)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--doc", default=os.path.join(os.path.dirname(__file__), "..", "e1", "ppl_doc.txt"))
    a = ap.parse_args()
    t0 = time.time()
    model, tok = load()
    print(f"loaded in {time.time() - t0:.1f}s; peak memory {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    if not a.check:
        return
    ids = encode(tok, open(a.doc).read())
    t0 = time.time()
    p, n = ppl_like_llama(model, ids)
    print(f"PPL doc (-c 256, {n} scored tokens): {p:.4f}  [llama.cpp I2_S Metal baseline: 12.0192 +- 0.62]  {time.time() - t0:.0f}s", flush=True)
    print("generation:", repr(greedy(model, tok, chat_ids(tok, "What is the capital of France? Answer in one word."))), flush=True)
    # one backward step through the ternary layers (straight-through estimator)
    x = mx.array(ids[:129])[None]

    def loss_fn(m):
        logits = m(x[:, :-1]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, x[:, 1:]).mean()
    lv, g = nn.value_and_grad(model, loss_fn)(model)
    gq = g["model"]["layers"][0]["self_attn"]["q_proj"]["weight"]
    mx.eval(lv, gq)
    print(f"loss {float(lv):.4f}; grad |q_proj.0| mean {float(mx.mean(mx.abs(gq.astype(mx.float32)))):.3e}; "
          f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
