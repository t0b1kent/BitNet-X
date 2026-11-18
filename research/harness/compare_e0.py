#!/usr/bin/env python3
"""E0 comparison across models on the same 100 tasks.

For each results dir: greedy pass (mode A), oracle pass@5 and selected@1 (mode B: the first of 5 samples that
passes the 3 public examples, else sample 0), failure categories and finish reasons. Intervals and paired
differences use a cluster bootstrap over families (tasks inside a family are near-duplicates).

  compare_e0.py --tasks ../tasks/dev_100.jsonl --base bitnet=../results/e0_dev \
      bonsai2-2B=../results/cmp/bonsai2-2B/e0 bonsai2-4B=../results/cmp/bonsai2-4B/e0 --out ../results/cmp/e0_compare.md
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade import grade_text  # noqa: E402


def load(path):
    return {r["task"]: r for r in map(json.loads, open(path))}


def per_task(d, tasks):
    A, B = load(os.path.join(d, "mode_A.jsonl")), load(os.path.join(d, "mode_B.jsonl"))
    out = {}
    for tid, t in tasks.items():
        a = A[tid]["attempts"][0]
        atts = B[tid]["attempts"]
        sel = None
        for s in atts:
            if grade_text(s["text"], t, "public")[0] == "pass":
                sel = s
                break
        out[tid] = dict(family=t["family"], A=a["category"] == "pass", pass5=any(s["category"] == "pass" for s in atts),
                        sel=(sel or atts[0])["category"] == "pass", sel_found=sel is not None,
                        catA=a["category"], finA=a.get("finish"), ctokA=a.get("ctoks"))
    return out


def boot(rows_by_fam, key, base=None, n=4000, seed=0):
    fams = sorted(rows_by_fam)
    rng = random.Random(seed)

    def val(fs, src):
        rows = [r for f in fs for r in src[f]]
        return 100.0 * sum(r[key] for r in rows) / len(rows)

    point = val(fams, rows_by_fam)
    dpoint = None if base is None else point - val(fams, base)
    xs, ds = [], []
    for _ in range(n):
        fs = [rng.choice(fams) for _ in fams]
        xs.append(val(fs, rows_by_fam))
        if base is not None:
            ds.append(xs[-1] - val(fs, base))
    xs.sort()
    ds.sort()
    q = lambda v: (v[int(0.025 * n)], v[int(0.975 * n) - 1])
    return point, q(xs), dpoint, (q(ds) if base is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--base", required=True, help="name=dir of the reference model")
    ap.add_argument("others", nargs="+", help="name=dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    tasks = {t["id"]: t for t in map(json.loads, open(a.tasks))}
    specs = [a.base] + a.others
    data = {}
    for s in specs:
        name, d = s.split("=", 1)
        data[name] = per_task(d, tasks)
    base_name = a.base.split("=", 1)[0]
    byfam = {m: collections.defaultdict(list) for m in data}
    for m, rows in data.items():
        for tid in sorted(rows):
            byfam[m][rows[tid]["family"]].append(rows[tid])

    L = ["# E0 comparison (same 100 tasks, 10 families; 95% CI by cluster bootstrap over families)", ""]
    L.append(f"| model | A greedy | B pass@5 (oracle) | B selected@1 (public examples) | Δ A vs {base_name} | Δ selected@1 vs {base_name} |")
    L.append("|---|---|---|---|---|---|")
    for m in data:
        b = None if m == base_name else byfam[base_name]
        pa, ca, da, dca = boot(byfam[m], "A", b)
        p5, c5, _, _ = boot(byfam[m], "pass5", b)
        ps, cs, ds, dcs = boot(byfam[m], "sel", b)
        fmt = lambda p, c: f"{p:.0f}% [{c[0]:.0f}-{c[1]:.0f}]"
        fd = lambda d, c: "" if d is None else f"{d:+.0f} [{c[0]:+.0f}; {c[1]:+.0f}]"
        L.append(f"| {m} | {fmt(pa, ca)} | {fmt(p5, c5)} | {fmt(ps, cs)} | {fd(da, dca)} | {fd(ds, dcs)} |")
    L += ["", "## Per family: A greedy / pass@5 / selected@1 (of 10)", ""]
    fams = sorted(byfam[base_name])
    L.append("| family | " + " | ".join(data) + " |")
    L.append("|---|" + "---|" * len(data))
    for f in fams:
        cells = []
        for m in data:
            r = byfam[m][f]
            cells.append(f"{sum(x['A'] for x in r)} / {sum(x['pass5'] for x in r)} / {sum(x['sel'] for x in r)}")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L += ["", "## Mode A: failure categories and finish reasons", ""]
    for m, rows in data.items():
        cats = collections.Counter(r["catA"] for r in rows.values())
        fins = collections.Counter(r["finA"] for r in rows.values())
        trunc = sum(1 for r in rows.values() if r["finA"] in ("length", "limit"))
        L.append(f"* {m}: {dict(cats.most_common())}; finish {dict(fins)}; truncated {trunc}")
    L += ["", "## Public examples as a filter (mode B, all samples are graded on private tests)", ""]
    for m, rows in data.items():
        found = sum(r["sel_found"] for r in rows.values())
        good = sum(r["sel"] for r in rows.values() if r["sel_found"])
        L.append(f"* {m}: a sample passing public examples found in {found}/100 tasks; of those, hidden tests passed {good}/{found}")
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
