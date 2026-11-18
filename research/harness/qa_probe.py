#!/usr/bin/env python3
"""E0b: knowledge vs reasoning. A made-up world (invented names, cities, jobs) whose facts cannot be in any training set.

Conditions for the same questions:
  closed  no facts in the prompt            -> what the weights know about this world (nothing): the guessing floor
  oracle  only the facts the answer needs   -> pure reasoning / reading ability with knowledge supplied for free
  open    all facts of the world (+ noise)  -> reading ability plus finding the relevant facts

Question types: hop1 (one fact), hop2 (two facts chained), hop3 (three chained), count (aggregate), older (compare two numbers).

    qa_probe.py --out ../results/e0b_qa --worlds 5 --per-type 6
"""
import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_modes as R  # noqa: E402

SYL = ["ba", "do", "ki", "lu", "me", "na", "po", "ri", "su", "ta", "ve", "zo", "fe", "gi", "ha", "jo", "ku", "la", "mi", "no"]
JOBS = ["baker", "pilot", "tailor", "farmer", "painter", "doctor", "miner", "teacher", "sailor", "potter"]
PETS = ["cat", "dog", "parrot", "rabbit", "turtle", "goat"]
TAIL = "Reply with only the answer."  # validated on dev questions; adding "Use digits for numbers" made the model answer with numbers
NUMW = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def pname(rng):
    return "".join(rng.choice(SYL) for _ in range(rng.choice([2, 3]))).capitalize()


def make_world(rng, n=24):
    names = set()
    while len(names) < n:
        names.add(pname(rng))
    names = sorted(names)
    rng.shuffle(names)
    cities = []
    while len(cities) < 6:
        c = pname(rng) + rng.choice(["burg", "ton", "ford"])
        if c not in cities:
            cities.append(c)
    ages = rng.sample(range(21, 80), n)
    people = {nm: dict(job=rng.choice(JOBS), city=rng.choice(cities), age=ages[i], pet=rng.choice(PETS), mentor=None)
              for i, nm in enumerate(names)}
    for i, nm in enumerate(names):
        if i >= 3:
            people[nm]["mentor"] = names[rng.randrange(0, i)]
    return names, people


def fact(people, key, p):
    d = people[p]
    return {"job": f"{p} works as a {d['job']}.", "city": f"{p} lives in {d['city']}.", "age": f"{p} is {d['age']} years old.",
            "pet": f"{p} owns a {d['pet']}.", "mentor": f"{p}'s mentor is {d['mentor']}."}[key]


def questions(rng, names, people, per_type):
    qs = []
    with_m = [p for p in names if people[p]["mentor"]]
    with_mm = [p for p in with_m if people[people[p]["mentor"]]["mentor"]]
    for _ in range(per_type):
        p = rng.choice(names)
        qs.append(dict(type="hop1", q=f"What is {p}'s job?", ans=people[p]["job"], need=[("job", p)]))
        p = rng.choice(with_m); m = people[p]["mentor"]
        qs.append(dict(type="hop2", q=f"What is the job of {p}'s mentor?", ans=people[m]["job"], need=[("mentor", p), ("job", m)]))
        p = rng.choice(with_mm); m = people[p]["mentor"]; mm = people[m]["mentor"]
        qs.append(dict(type="hop3", q=f"In which city does the mentor of {p}'s mentor live?", ans=people[mm]["city"],
                       need=[("mentor", p), ("mentor", m), ("city", mm)]))
        counts = defaultdict(list)
        for x in names:
            counts[people[x]["city"]].append(x)
        c = rng.choice([c for c, v in counts.items() if 2 <= len(v) <= 5])
        qs.append(dict(type="count", q=f"How many people live in {c}?", ans=str(len(counts[c])), need=[("city", x) for x in counts[c]]))
        a, b = rng.sample(names, 2)
        older = a if people[a]["age"] > people[b]["age"] else b
        qs.append(dict(type="older", q=f"Who is older, {a} or {b}?", ans=older, need=[("age", a), ("age", b)]))
    return qs


OLDER_RE = re.compile(r"Who is older, (\w+) or (\w+)\?")


def other_name(q, ans):
    """For 'Who is older, A or B?' the name that is not the answer, else None."""
    m = OLDER_RE.search(q)
    if not m:
        return None
    return m.group(2) if m.group(1).lower() == ans.lower() else m.group(1)


def answer_line(reply, line="first"):
    """'first': first line (strict protocol, the model was told to reply with only the answer).
    'last': last non-empty line (for runs with a large token budget, where a model may reason before answering)."""
    rows = [r for r in reply.strip().split("\n") if r.strip()]
    if not rows:
        return ""
    return (rows[0] if line == "first" else rows[-1]).lower()


def correct(reply, ans, q=None, line="first"):
    toks = [NUMW.get(t, t) for t in re.findall(r"[a-z0-9]+", answer_line(reply, line))]
    if ans.isdigit():
        nums = [t for t in toks if t.isdigit()]
        return bool(nums) and nums[0] == ans
    other = other_name(q, ans) if q else None
    if other and other.lower() in toks:
        # both names: only an explicit comparison counts ("A is older than B", "B is younger than A");
        # anything else (e.g. the question restated) is not an answer
        text = answer_line(reply, line)
        a_, o_ = re.escape(ans.lower()), re.escape(other.lower())
        return bool(re.search(rf"\b{a_}\b\W+(?:is\s+)?older\s+than\W+{o_}\b", text)
                    or re.search(rf"\b{o_}\b\W+(?:is\s+)?younger\s+than\W+{a_}\b", text))
    return ans.lower() in toks


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds", type=int, default=5)
    ap.add_argument("--per-type", type=int, default=6)
    ap.add_argument("--seed", default="qa-1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--max-tokens", type=int, default=24, help="24 = strict protocol; larger lets a model reason first")
    ap.add_argument("--line", choices=("first", "last"), default="first", help="which reply line is scored")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    items = []
    for w in range(a.worlds):
        rng = random.Random(f"{a.seed}/{w}")
        names, people = make_world(rng)
        allfacts = [fact(people, k, p) for p in names for k in ("job", "city", "age", "pet", "mentor") if not (k == "mentor" and not people[p]["mentor"])]
        rng.shuffle(allfacts)
        for q in questions(rng, names, people, a.per_type):
            q["world"] = w
            q["oracle_facts"] = [fact(people, k, p) for k, p in q["need"]]
            q["all_facts"] = allfacts
            items.append(q)
    json.dump(items, open(os.path.join(a.out, "questions.json"), "w"), indent=0)

    res = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    log = open(os.path.join(a.out, "qa.jsonl"), "w")
    for cond in ("closed", "oracle", "open"):
        for it in items:
            facts = {"closed": [], "oracle": it["oracle_facts"], "open": it["all_facts"]}[cond]
            body = ("\n".join(facts) + "\n\n" if facts else "") + f"Question: {it['q']}\n{TAIL}"
            att = R.chat(a.port, R.msgs(body), 0.0, 1.0, a.max_tokens, 0)
            ok = correct(att["text"], it["ans"], it["q"], a.line)
            res[it["type"]][cond][0] += ok
            res[it["type"]][cond][1] += 1
            log.write(json.dumps(dict(cond=cond, type=it["type"], world=it["world"], q=it["q"], ans=it["ans"], reply=att["text"][:1500],
                                      ok=ok, ptoks=att["ptoks"], ctoks=att["ctoks"], finish=att["finish"])) + "\n")
        log.flush()
        print(f"condition {cond} done", flush=True)

    lines = ["| type | closed (no facts) | oracle (needed facts only) | open (all facts) |", "|---|---|---|---|"]
    for t in ("hop1", "hop2", "hop3", "count", "older"):
        cells = []
        for cond in ("closed", "oracle", "open"):
            k, n = res[t][cond]
            lo, hi = wilson(k, n)
            cells.append(f"{k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    tot = {c: [sum(res[t][c][0] for t in res), sum(res[t][c][1] for t in res)] for c in ("closed", "oracle", "open")}
    lines.append("| all | " + " | ".join(f"{tot[c][0]}/{tot[c][1]} = {100 * tot[c][0] / tot[c][1]:.0f}%" for c in ("closed", "oracle", "open")) + " |")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(a.out, "summary.md"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
