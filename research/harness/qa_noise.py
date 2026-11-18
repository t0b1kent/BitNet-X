#!/usr/bin/env python3
"""E0b-3: accuracy versus the number of irrelevant facts next to all the needed ones (same made-up worlds as E0b).

For every question: the needed facts + n distractor facts from the same world, shuffled (seeded). n = 0 is the E0b oracle,
n = "all" is the E0b open condition. Measures how precise a fact selector has to be for this model.

    qa_noise.py --questions ../results/e0b_qa/questions.json --out ../results/e0b3_noise/<model> --levels 0,3,10,30,all
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_modes as R  # noqa: E402
from qa_probe import correct, wilson  # noqa: E402

TAIL = "Reply with only the answer."
TAILS = {
    "answer": TAIL,
    # explicit reasoning: does the model fail to select facts, or just not try? (scored with --line last)
    "reason": "First write out the facts you need, one per line, and reason step by step. "
              "Then write the final answer alone on the last line.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="0,3,10,30,all")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--max-tokens", type=int, default=24, help="24 = strict protocol; larger lets a model reason first")
    ap.add_argument("--line", choices=("first", "last"), default="first", help="which reply line is scored")
    ap.add_argument("--tail", choices=sorted(TAILS), default="answer", help="instruction after the question")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (pilot runs)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel requests (server needs -np >= jobs)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.questions))
    if a.limit:
        items = items[:a.limit]
    levels = a.levels.split(",")
    tail = TAILS[a.tail]
    res = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    pool = ThreadPoolExecutor(max_workers=a.jobs)
    with open(os.path.join(a.out, "noise.jsonl"), "w") as log:
        for lv in levels:
            bodies = []
            for i, it in enumerate(items):
                rng = random.Random(f"noise/{i}/{lv}")
                others = [f for f in it["all_facts"] if f not in it["oracle_facts"]]
                n = len(others) if lv == "all" else min(int(lv), len(others))
                facts = it["oracle_facts"] + rng.sample(others, n)
                rng.shuffle(facts)
                bodies.append("\n".join(facts) + f"\n\nQuestion: {it['q']}\n{tail}")
            atts = pool.map(lambda b: R.chat(a.port, R.msgs(b), 0.0, 1.0, a.max_tokens, 0), bodies)
            for it, att in zip(items, atts):
                ok = correct(att["text"], it["ans"], it["q"], a.line)
                res[lv][it["type"]][0] += ok
                res[lv][it["type"]][1] += 1
                log.write(json.dumps(dict(level=lv, type=it["type"], q=it["q"], ans=it["ans"], reply=att["text"][:1500], ok=ok,
                                         ctoks=att["ctoks"], finish=att["finish"])) + "\n")
            log.flush()
            print(f"level {lv} done", flush=True)
    types = ("hop1", "hop2", "hop3", "count", "older")
    lines = ["| distractors | " + " | ".join(types) + " | all |", "|---" * (len(types) + 2) + "|"]
    for lv in levels:
        k = sum(res[lv][t][0] for t in types); n = sum(res[lv][t][1] for t in types)
        lo, hi = wilson(k, n)
        lines.append(f"| {lv} | " + " | ".join(f"{res[lv][t][0]}/{res[lv][t][1]}" for t in types) + f" | {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}] |")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(a.out, "summary.md"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
