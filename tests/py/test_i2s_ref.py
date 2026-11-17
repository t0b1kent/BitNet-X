#!/usr/bin/env python3
"""Self-test of the NumPy I2_S reference (no model files needed)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i2s_ref

rng = np.random.default_rng(1)
fail = 0

def check(name, ok):
    global fail
    print(("ok   " if ok else "FAIL ") + name)
    fail |= (not ok)

# 1) pack/unpack are inverse
t = rng.integers(-1, 2, size=128 * 40).astype(np.int8)
p = i2s_ref.pack(t)
check("unpack(pack(t)) == t", np.array_equal(i2s_ref.unpack(p), t))
check("packed size == n/4", p.size == t.size // 4)

# 2) hand-written example pins down the bit order: weight i -> byte i%32 (of its
#    128-block), bits 7-6 for i//32 == 0 ... bits 1-0 for i//32 == 3
t2 = np.zeros(128, np.int8)            # all zero -> every code is 1 -> byte 0b01010101
t2[0] = 1                              # code 2 in bits 7:6 of byte 0  -> 0b10010101
t2[33] = -1                            # code 0 in bits 5:4 of byte 1  -> 0b01000101
t2[127] = 1                            # code 2 in bits 1:0 of byte 31 -> 0b01010110
p2 = i2s_ref.pack(t2)
exp = np.full(32, 0b01010101, np.uint8); exp[0] = 0b10010101; exp[1] = 0b01000101; exp[31] = 0b01010110
check("bit order of the canonical layout", np.array_equal(p2, exp))

# 3) the non-canonical readers really differ on random data
check("64/16 reader != canonical", not np.array_equal(i2s_ref.unpack(p, "64/16"), t))
check("sequential reader != canonical", not np.array_equal(i2s_ref.unpack(p, "sequential"), t))

# 4) engine arithmetic: (sum code*q - sum q) * ws / as == sum w*q * ws / as
n, rows = 256, 8
w = rng.integers(-1, 2, size=(rows, n)).astype(np.int8)
x = rng.standard_normal(n).astype(np.float32)
q, s, qsum = i2s_ref.quantize_act_i8(x)
ref = (w.astype(np.int32) @ q.astype(np.int32)).astype(np.float32) * np.float32(0.75) / np.float32(s)
got = i2s_ref.mul_mat(i2s_ref.pack(w), n, 0.75, x)
check("mul_mat reference identity", np.array_equal(ref, got))

# 5) absmean ternarisation
wf = np.array([0.9, -0.2, 0.05, -1.4], np.float32)   # mean|w| = 0.6375
tq, m = i2s_ref.ternarize_absmean(wf)
check("ternarize_absmean", np.array_equal(tq, [1, 0, 0, -1]) and abs(m - 0.6375) < 1e-6)

sys.exit(1 if fail else 0)
