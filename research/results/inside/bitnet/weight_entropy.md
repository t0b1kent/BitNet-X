Ternary weights: 2.084 billion. Share -1 / 0 / +1: 0.289 / 0.422 / 0.289.
Entropy per weight H = 1.560 bits (ceiling log2 3 = 1.585). Upper bound of information in the ternary layers: 407 MB (at the ceiling 413 MB).
Storage: I2_S 2 bits/weight = 521 MB; base-3 packing 1.6 bits = 417 MB; ideal entropy coding = 407 MB.

| projection | -1 | 0 | +1 | H, bits |
|---|---|---|---|---|
| down_proj | 0.293 | 0.414 | 0.293 | 1.565 |
| gate_proj | 0.288 | 0.423 | 0.288 | 1.560 |
| k_proj | 0.274 | 0.452 | 0.274 | 1.541 |
| o_proj | 0.292 | 0.416 | 0.292 | 1.564 |
| q_proj | 0.268 | 0.463 | 0.268 | 1.533 |
| up_proj | 0.292 | 0.417 | 0.292 | 1.563 |
| v_proj | 0.313 | 0.375 | 0.313 | 1.580 |

Per layer H: 0:1.567 1:1.431 2:1.509 3:1.545 4:1.560 5:1.561 6:1.567 7:1.572 8:1.571 9:1.570 10:1.561 11:1.568 12:1.559 13:1.564 14:1.555 15:1.563 16:1.563 17:1.567 18:1.567 19:1.566 20:1.554 21:1.569 22:1.563 23:1.569 24:1.564 25:1.570 26:1.570 27:1.563 28:1.566 29:1.557
