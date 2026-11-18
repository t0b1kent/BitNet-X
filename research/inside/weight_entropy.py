#!/usr/bin/env python3
"""How much information the ternary weights of BitNet b1.58 2B-4T actually carry.

For every projection: quantize the BF16 master exactly as the model does (absmean -> {-1, 0, 1}), count the three
values, and compute the per-weight entropy H (bits). log2(3) = 1.585 is the hard ceiling for one ternary weight.
Compares with storage: I2_S spends 2 bits per weight; base-3 packing (5 weights per byte, as llama.cpp TQ1_0) 1.6;
an ideal entropy coder H. Also the per-layer spread, to see whether some layers use the ternary alphabet less.

    ~/Developer/MicrosoftAI/.venv-train/bin/python weight_entropy.py
"""
import json
import math
import os

import mlx.core as mx

MASTER = os.path.expanduser("~/Developer/MicrosoftAI/BitNet-Original")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "inside", "bitnet", "weight_entropy.md")


def H(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def main():
    w = mx.load(os.path.join(MASTER, "model.safetensors"))
    tot = [0, 0, 0]
    per_kind, per_layer = {}, {}
    for k, v in w.items():
        if not any(p in k for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
            continue
        f = v.astype(mx.float32)
        s = 1.0 / mx.maximum(mx.mean(mx.abs(f)), 1e-5)
        q = mx.clip(mx.round(f * s), -1, 1)
        n = q.size
        neg, zero = int(mx.sum(q == -1)), int(mx.sum(q == 0))
        c = [neg, zero, n - neg - zero]
        tot = [a + b for a, b in zip(tot, c)]
        kind = k.split(".")[-2]
        layer = int(k.split(".")[2])
        for d, key in ((per_kind, kind), (per_layer, layer)):
            d.setdefault(key, [0, 0, 0])
            d[key] = [a + b for a, b in zip(d[key], c)]
    n = sum(tot)
    p = [x / n for x in tot]
    h = H(p)
    lines = [f"Ternary weights: {n / 1e9:.3f} billion. Share -1 / 0 / +1: {p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f}.",
             f"Entropy per weight H = {h:.3f} bits (ceiling log2 3 = 1.585). Upper bound of information in the ternary "
             f"layers: {n * h / 8e6:.0f} MB (at the ceiling {n * math.log2(3) / 8e6:.0f} MB).",
             f"Storage: I2_S 2 bits/weight = {n * 2 / 8e6:.0f} MB; base-3 packing 1.6 bits = {n * 1.6 / 8e6:.0f} MB; "
             f"ideal entropy coding = {n * h / 8e6:.0f} MB.", "",
             "| projection | -1 | 0 | +1 | H, bits |", "|---|---|---|---|---|"]
    for kind, c in sorted(per_kind.items()):
        m = sum(c); pp = [x / m for x in c]
        lines.append(f"| {kind} | {pp[0]:.3f} | {pp[1]:.3f} | {pp[2]:.3f} | {H(pp):.3f} |")
    hs = {l: H([x / sum(c) for x in c]) for l, c in per_layer.items()}
    lines += ["", "Per layer H: " + " ".join(f"{l}:{hs[l]:.3f}" for l in sorted(hs))]
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(text + "\n")


if __name__ == "__main__":
    main()
