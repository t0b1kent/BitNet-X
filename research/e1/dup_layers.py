#!/usr/bin/env python3
"""E1 probe: build a GGUF whose layer sequence repeats a block of layers (weights duplicated, KV cache per virtual layer).

Virtual layers = [0..A-1] + [A..B] * K + [B+1..N-1]; K=1 is the identity. Tensor bytes are copied verbatim from the source file
(sizes taken from offset differences, because gguf-py does not know the size of I2_S); only block_count and the tensor table change.

    dup_layers.py --src ggml-model-i2_s.gguf --out out.gguf --start 10 --end 14 --repeat 2
"""
import argparse
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "3rdparty", "llama.cpp", "gguf-py"))
import gguf  # noqa: E402

ALIGN = 32


def pad(n):
    return (-n) % ALIGN


def build(src, out, a, b, k):
    r = gguf.GGUFReader(src)
    raw = open(src, "rb").read()
    n_layers = int(r.fields["bitnet-b1.58.block_count"].contents())
    assert 0 <= a <= b < n_layers and k >= 1
    seq = list(range(a)) + list(range(a, b + 1)) * k + list(range(b + 1, n_layers))

    tensors = sorted(r.tensors, key=lambda t: t.data_offset)
    sizes = {}
    for i, t in enumerate(tensors):
        end = tensors[i + 1].data_offset if i + 1 < len(tensors) else len(raw)
        sizes[t.name] = (t.data_offset, end - t.data_offset)

    by_layer, other = {}, []
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(.+)", t.name)
        (by_layer.setdefault(int(m.group(1)), []) if m else other).append((m.group(2) if m else t.name, t))

    plan = []  # (new name, tensor)
    for name, t in other:
        plan.append((name, t))
    for v, src_layer in enumerate(seq):
        for suffix, t in by_layer[src_layer]:
            plan.append((f"blk.{v}.{suffix}", t))

    tinfo_start = r.tensors[0].field.offset
    head = bytearray(raw[:tinfo_start])
    struct.pack_into("<Q", head, 8, len(plan))  # tensor_count
    key = b"bitnet-b1.58.block_count"
    pat = struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, n_layers)
    pos = head.find(pat)
    assert pos >= 0 and head.find(pat, pos + 1) < 0, "block_count field not found uniquely"
    struct.pack_into("<I", head, pos + len(pat) - 4, len(seq))

    infos, rel = bytearray(), 0
    offsets = []
    for name, t in plan:
        nb = name.encode()
        infos += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(t.shape))
        for d in t.shape:
            infos += struct.pack("<Q", int(d))
        infos += struct.pack("<IQ", int(t.tensor_type), rel)
        offsets.append(rel)
        rel += sizes[t.name][1]
        assert sizes[t.name][1] % ALIGN == 0 or t is tensors[-1], f"unaligned tensor {t.name}"

    with open(out, "wb") as f:
        f.write(head)
        f.write(infos)
        f.write(b"\0" * pad(len(head) + len(infos)))
        for name, t in plan:
            off, size = sizes[t.name]
            f.write(raw[off:off + size])
    return seq, os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    seq, size = build(a.src, a.out, a.start, a.end, a.repeat)
    print(f"{len(seq)} virtual layers, {size / 1e9:.3f} GB: {seq}")


if __name__ == "__main__":
    main()
