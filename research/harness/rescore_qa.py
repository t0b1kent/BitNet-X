#!/usr/bin/env python3
"""Re-score saved E0b / E0b-3 replies with the current qa_probe.correct (no model calls).

  rescore_qa.py <qa.jsonl or noise.jsonl> [--line first|last]
Prints the table in the same layout as the original summary plus how many verdicts changed.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_probe import correct, wilson  # noqa: E402

TYPES = ("hop1", "hop2", "hop3", "count", "older")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--line", choices=("first", "last"), default="first")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.log)]
    key = "cond" if "cond" in rows[0] else "level"
    res = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    changed = defaultdict(int)
    order = []
    for r in rows:
        ok = correct(r["reply"], r["ans"], r["q"], a.line)
        g = r[key]
        if g not in order:
            order.append(g)
        res[g][r["type"]][0] += ok
        res[g][r["type"]][1] += 1
        if ok != r["ok"]:
            changed[(g, r["type"], "+" if ok else "-")] += 1
    lines = [f"| {key} | " + " | ".join(TYPES) + " | all |", "|---" * (len(TYPES) + 2) + "|"]
    for g in order:
        k = sum(res[g][t][0] for t in TYPES); n = sum(res[g][t][1] for t in TYPES)
        lo, hi = wilson(k, n)
        lines.append(f"| {g} | " + " | ".join(f"{res[g][t][0]}/{res[g][t][1]}" for t in TYPES)
                     + f" | {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}] |")
    print("\n".join(lines))
    print("changed verdicts:", dict(changed) or "none")


if __name__ == "__main__":
    main()
