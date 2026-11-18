#!/usr/bin/env python3
"""Summary of an E0 run: pass rates with Wilson 95% intervals, tokens, failure taxonomy, compute-matched baseline.

    summarize.py ../results/e0_dev
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict


def load(d, mode):
    p = os.path.join(d, f"mode_{mode}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:4.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}]" if n else "-"


def main():
    d = sys.argv[1]
    out = []
    P = out.append
    modes = {m: load(d, m) for m in "ABCDE"}
    P(f"# E0 summary: {d}\n")
    meta = os.path.join(d, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta))
        P(f"started {m['started']}, repo {m['repo_commit'][:10]}, machine load at start {m['load_at_start'][0]:.1f}\n")

    P("| mode | tasks | pass (final) | completion tokens / task | s / task |")
    P("|---|---|---|---|---|")
    names = dict(A="A single (greedy)", B="B pass@1 of 5 samples", C="C algorithm hint", D="D execution feedback", E="E reasoning first")
    tok = {}
    for mode, recs in modes.items():
        if not recs:
            continue
        n = len(recs)
        if mode == "B":
            c1 = sum(r["c"] for r in recs)
            P(f"| {names[mode]} | {n} | {fmt(c1, 5 * n)} | {sum(r['ctoks'] for r in recs) / n / 5:.0f} per sample | {sum(r['secs'] for r in recs) / n / 5:.1f} per sample |")
            P(f"| B pass@5 (any of 5) | {n} | {fmt(sum(r['ok'] for r in recs), n)} | {sum(r['ctoks'] for r in recs) / n:.0f} | {sum(r['secs'] for r in recs) / n:.1f} |")
        else:
            P(f"| {names[mode]} | {n} | {fmt(sum(r['ok'] for r in recs), n)} | {sum(r['ctoks'] for r in recs) / n:.0f} | {sum(r['secs'] for r in recs) / n:.1f} |")
        tok[mode] = sum(r["ctoks"] for r in recs) / n

    if modes["A"]:
        P("\n## Per family, mode A (greedy)\n")
        P("| family | skills | pass |")
        P("|---|---|---|")
        fam = defaultdict(list)
        skills = {}
        for r in modes["A"]:
            fam[r["family"]].append(r["ok"])
            skills[r["family"]] = ",".join(r["skills"])
        for f, v in fam.items():
            P(f"| {f} | {skills[f]} | {sum(v)}/{len(v)} |")
        P("\n## Failure taxonomy, mode A (first attempt)\n")
        cats = Counter(r["attempts"][0]["category"] for r in modes["A"])
        P(", ".join(f"{k}: {v}" for k, v in cats.most_common()))

    if modes["D"] and modes["A"]:
        a_ok = {r["task"]: r["ok"] for r in modes["A"]}
        d_ok = {r["task"]: r["ok"] for r in modes["D"]}
        both = [t for t in d_ok if t in a_ok]
        fixed = sum(1 for t in both if not a_ok[t] and d_ok[t])
        broken = sum(1 for t in both if a_ok[t] and not d_ok[t])
        P(f"\n## Mode D vs A: fixed {fixed}, broken {broken} (of {len(both)}); rounds used: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r['rounds'] for r in modes['D']).items())))

    if modes["B"] and "A" in tok:
        P("\n## Compute-matched independent sampling (from the 5 samples of mode B)\n")
        P("| method | tokens / task | pass | independent samples with the same token budget (k) | pass@k |")
        P("|---|---|---|---|---|")
        for mode in "CDE":
            if modes[mode]:
                k = max(1, min(5, round(tok[mode] / tok["A"])))
                pk = sum(pass_at_k(5, r["c"], k) for r in modes["B"]) / len(modes["B"])
                n = len(modes[mode])
                P(f"| {names[mode]} | {tok[mode]:.0f} | {fmt(sum(r['ok'] for r in modes[mode]), n)} | {k} | {100 * pk:.1f}% |")
        P("\nA method is only interesting if it beats the independent-sampling column at the same token budget.")

    text = "\n".join(out)
    print(text)
    open(os.path.join(d, "summary.md"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
