#!/usr/bin/env python3
"""E1 phase 1, screening: perplexity of GGUFs with a repeated block of layers (Metal). Compares with the unmodified model on the same texts.

    ppl_grid.py --lengths 3,5,10 --starts 2,10,20 --repeat 2
"""
import argparse, csv, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~/Developer/MicrosoftAI")
SRC = f"{HOME}/BitNet-GGUF/ggml-model-i2_s.gguf"
BIN = "/private/tmp/bitnetx-build-int/bin"
TEXTS = {"small": f"{HOME}/BitNet-X/tests/data/ppl_small.txt", "doc": os.path.join(HERE, "ppl_doc.txt")}
TMP = "/private/tmp/bx-e1"


def ppl(model, text):
    out = subprocess.run([f"{BIN}/llama-perplexity", "-m", model, "-f", text, "-c", "256", "-b", "256", "-ngl", "99", "-t", "4"],
                         capture_output=True, text=True, env=dict(os.environ, DYLD_LIBRARY_PATH=BIN))
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out.stdout + out.stderr)
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="3,5,10")
    ap.add_argument("--starts", default="2,10,20")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "ppl_grid.csv"))
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    rows = []
    t0 = time.time()
    base = {k: ppl(SRC, p) for k, p in TEXTS.items()}
    rows.append(["baseline", 30, 0, 0, 1, base["small"][0], base["doc"][0], base["doc"][1], 0.0])
    print(f"baseline: small {base['small'][0]:.4f}  doc {base['doc'][0]:.4f} +- {base['doc'][1]:.4f}", flush=True)
    for L in map(int, a.lengths.split(",")):
        for s in map(int, a.starts.split(",")):
            e = s + L - 1
            if e > 29:
                continue
            out = f"{TMP}/loop_{s}_{e}_x{a.repeat}.gguf"
            subprocess.run([sys.executable, os.path.join(HERE, "dup_layers.py"), "--src", SRC, "--out", out, "--start", str(s),
                            "--end", str(e), "--repeat", str(a.repeat)], check=True, stdout=subprocess.DEVNULL)
            r = {k: ppl(out, p) for k, p in TEXTS.items()}
            os.remove(out)
            extra = (a.repeat - 1) * L
            rows.append([f"{s}-{e}x{a.repeat}", 30 + extra, s, e, a.repeat, r["small"][0], r["doc"][0], r["doc"][1], round(100 * extra / 30, 1)])
            print(f"layers {s}-{e} x{a.repeat} (+{extra} layers, +{100 * extra / 30:.0f}% FLOPs): small {r['small'][0]:.4f}  doc {r['doc'][0]:.4f} +- {r['doc'][1]:.4f}  "
                  f"(doc vs baseline {100 * (r['doc'][0] / base['doc'][0] - 1):+.1f}%)  [{time.time() - t0:.0f}s]", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "virtual_layers", "start", "end", "repeat", "ppl_small", "ppl_doc", "ppl_doc_se", "extra_flops_pct"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
