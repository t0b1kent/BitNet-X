#!/usr/bin/env python3
"""Real-weights check of the I2_S layout.

Needs (otherwise exits 77 = SKIP):
  BITNET_GGUF  path to the official ggml-model-i2_s.gguf
  BITNET_BF16  directory with the BF16 master (model.safetensors)

For blk.0.attn_q.weight the packed GGUF trits are decoded with three candidate
layouts and compared with the ternarisation of the BF16 master weights
(s = 1/mean|w|, round, clip). Only the canonical 128/32 layout may match.
The files are opened read-only and never modified.
"""
import json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "3rdparty", "llama.cpp", "gguf-py"))
import i2s_ref

G = os.environ.get("BITNET_GGUF")
B = os.environ.get("BITNET_BF16")
if not G or not B or not os.path.isfile(G) or not os.path.isfile(os.path.join(B, "model.safetensors")):
    print("SKIP: set BITNET_GGUF (official i2_s gguf) and BITNET_BF16 (dir with model.safetensors)")
    sys.exit(77)

from gguf import GGUFReader

r = GGUFReader(G, "r")

def kv(key):
    f = r.fields.get(key)
    return bytes(f.parts[f.data[0]]).decode() if f else None

print("general.architecture =", kv("general.architecture"))
print("tokenizer.ggml.model =", kv("tokenizer.ggml.model"))
print("tokenizer.ggml.pre   =", kv("tokenizer.ggml.pre") or "ABSENT")

t = next(t for t in r.tensors if t.name == "blk.0.attn_q.weight")
K, M = int(t.shape[0]), int(t.shape[1])
n = K * M
raw = np.asarray(t.data).reshape(-1).view(np.uint8)
packed = raw[: n // 4]
scale = float(np.frombuffer(raw[n // 4 : n // 4 + 4].tobytes(), dtype=np.float32)[0])
print(f"{t.name}: type={t.tensor_type.name} shape=({K},{M}) bytes={raw.size} (n/4+32={n // 4 + 32}) scale={scale:.7f}")

with open(os.path.join(B, "model.safetensors"), "rb") as f:
    hl = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(hl))
    info = h["model.layers.0.self_attn.q_proj.weight"]
    a, b = info["data_offsets"]
    assert info["dtype"] == "BF16", info["dtype"]
    f.seek(8 + hl + a)
    w = (np.frombuffer(f.read(b - a), dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
w = w.reshape(M, K)
ref, absmean = i2s_ref.ternarize_absmean(w)
ref = ref.reshape(-1)
print(f"BF16 mean|w| = {absmean:.7f}  (GGUF scale {scale:.7f}, ratio {scale / absmean:.6f})")

res = {}
for layout in ("128/32", "64/16", "sequential"):
    d = i2s_ref.unpack(packed, layout)
    res[layout] = float((d == ref).mean()) * 100
    print(f"layout {layout:10s}: match with absmean-ternary(BF16) = {res[layout]:6.2f} %")

ok = True
if raw.size != n // 4 + 32:
    print("FAIL: tensor byte size != n/4 + 32"); ok = False
# 1.22 % residual is a property of the published checkpoint (microsoft/BitNet#632), not of the reader
if not res["128/32"] > 98.0:
    print("FAIL: canonical 128/32 layout does not reproduce the BF16 ternarisation"); ok = False
if not (res["64/16"] < 70.0 and res["sequential"] < 70.0):
    print("FAIL: a non-canonical layout matches too well"); ok = False
if not np.array_equal(i2s_ref.pack(i2s_ref.unpack(packed)), packed):
    # every code must be 0/1/2; a code 3 would not survive the round trip
    print("FAIL: packed data contains invalid codes"); ok = False
sys.exit(0 if ok else 1)
