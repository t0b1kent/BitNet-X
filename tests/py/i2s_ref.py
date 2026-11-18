"""NumPy reference for the BitNet I2_S tensor format (canonical layout "128/32").

A tensor with n ternary weights is stored as n/4 bytes of packed 2-bit codes
followed by a 32-byte tail whose first 4 bytes are the float32 scale.

Weights are grouped in blocks of 128 over the flattened (row-major) tensor.
Block b occupies bytes [32*b, 32*b+32). Inside a block, byte gp (0..31) holds
the codes of weights {gp, 32+gp, 64+gp, 96+gp} in bits [7:6], [5:4], [3:2],
[1:0]. Codes: 0 -> -1, 1 -> 0, 2 -> +1.

This is the layout of the official microsoft/bitnet-b1.58-2B-4T GGUF, of
dequantize_row_i2_s (ggml-cpu/quants.c) and of quantize_to_i2_s in
utils/convert-hf-to-gguf-bitnet.py.
"""
import numpy as np

SHIFTS = (6, 4, 2, 0)


def pack(t: np.ndarray) -> np.ndarray:
    """ternary int array (values -1/0/+1, size % 128 == 0) -> packed uint8[n/4]"""
    t = np.asarray(t).reshape(-1)
    assert t.size % 128 == 0, "I2_S needs a multiple of 128 weights"
    codes = (t.astype(np.int16) + 1).astype(np.uint8).reshape(-1, 4, 32)
    out = np.zeros((codes.shape[0], 32), dtype=np.uint8)
    for g, sh in enumerate(SHIFTS):
        out |= codes[:, g, :] << sh
    return out.reshape(-1)


def unpack(packed: np.ndarray, layout: str = "128/32") -> np.ndarray:
    """packed uint8 -> ternary int8[n]. Non-canonical layouts are provided only
    to demonstrate what the wrong readers see."""
    p = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if layout == "128/32":
        p = p.reshape(-1, 32)
        out = np.empty((p.shape[0], 4, 32), np.int8)
        for g, sh in enumerate(SHIFTS):
            out[:, g, :] = ((p >> sh) & 3).astype(np.int8) - 1
    elif layout == "64/16":
        p = p.reshape(-1, 16)
        out = np.empty((p.shape[0], 4, 16), np.int8)
        for g, sh in enumerate(SHIFTS):
            out[:, g, :] = ((p >> sh) & 3).astype(np.int8) - 1
    elif layout == "sequential":
        out = np.empty((p.size, 4), np.int8)
        for g, sh in enumerate(SHIFTS):
            out[:, g] = ((p >> sh) & 3).astype(np.int8) - 1
    else:
        raise ValueError(layout)
    return out.reshape(-1)


def ternarize_absmean(w: np.ndarray):
    """BitNet b1.58 weight quantisation: s = 1/mean|w|; round(w*s) clipped to [-1,1].
    Returns (ternary int8, scale = mean|w|)."""
    w = np.asarray(w, dtype=np.float32)
    m = float(np.abs(w).mean())
    s = 1.0 / max(m, 1e-5)
    return np.clip(np.round(w * s), -1, 1).astype(np.int8), m


def quantize_act_i8(x: np.ndarray):
    """Per-row activation quantisation as in quantize_row_i8_s (quants.c):
    s = 127/max|x|; q = clip(round(x*s), -128, 127). Returns (q int8, s, sum(q)).
    C roundf rounds half away from zero (np.round would round half to even)."""
    x = np.asarray(x, dtype=np.float32)
    amax = float(np.abs(x).max())
    s = np.float32(127.0 / amax) if amax > 0 else np.float32(0.0)
    v = x * s
    q = np.clip(np.sign(v) * np.floor(np.abs(v) + np.float32(0.5)), -128, 127).astype(np.int8)
    return q, float(s), int(q.astype(np.int32).sum())


def mul_mat(packed_rows: np.ndarray, n: int, scale: float, x: np.ndarray) -> np.ndarray:
    """Reference for the engine's I2_S mul_mat on one activation vector x[n]:
    out[r] = (sum_i code_ri*q_i - sum_i q_i) * scale / act_scale."""
    w = unpack(packed_rows).reshape(-1, n).astype(np.int32)
    q, s, qsum = quantize_act_i8(x)
    dot_codes = (w + 1) @ q.astype(np.int32)
    return (dot_codes - qsum).astype(np.float32) * np.float32(scale) / np.float32(s)
