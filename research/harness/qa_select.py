#!/usr/bin/env python3
"""E0b-2: how should a small model SELECT facts?  Same made-up worlds and questions as E0b (questions.json).

  self     the model copies the facts it needs from all facts, then answers from its own selection
  lexical  one-shot keyword retrieval: the 6 facts sharing most words with the question
  lookup   tool loop: facts about the names/cities in the question are looked up; the model may ask for
           up to 3 more lookups ('LOOKUP: <name>') before answering ('ANSWER: ...')

Reference points from E0b: oracle (only needed facts) 91 %, open (all 117 facts) 39 %.

    qa_select.py --questions ../results/e0b_qa/questions.json --out ../results/e0b2_select
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_modes as R  # noqa: E402
from qa_probe import correct, wilson  # noqa: E402

ANSWER_TAIL = "Reply with only the answer."
STOP = {"what", "is", "the", "of", "in", "which", "does", "live", "who", "how", "many", "people", "or", "a", "job", "s", "older", "mentor"}


def lookup(facts, term):
    return [f for f in facts if re.search(rf"\b{re.escape(term)}\b", f)]


def answer(port, facts, q):
    body = ("\n".join(facts) + "\n\n" if facts else "") + f"Question: {q}\n{ANSWER_TAIL}"
    att = R.chat(port, R.msgs(body), 0.0, 1.0, 24, 0)
    return att["text"], att["ctoks"]


def run_self(port, it):
    allf = it["all_facts"]
    body = "\n".join(allf) + f"\n\nQuestion: {it['q']}\nCopy the facts that are needed to answer this question, one per line, and nothing else."
    att = R.chat(port, R.msgs(body), 0.0, 1.0, 160, 0)
    lines = [l.strip(" -*\t") for l in att["text"].split("\n")]
    chosen = [f for f in allf if f in lines]
    text, ct = answer(port, chosen, it["q"])
    return text, att["ctoks"] + ct, dict(n_selected=len(chosen), recall=sum(f in chosen for f in it["oracle_facts"]) / len(it["oracle_facts"]))


def run_lexical(port, it, k=6):
    qw = {w.lower() for w in re.findall(r"[A-Za-z]+", it["q"])} - STOP
    scored = sorted(it["all_facts"], key=lambda f: -len(qw & {w.lower() for w in re.findall(r"[A-Za-z]+", f)}))
    chosen = scored[:k]
    text, ct = answer(port, chosen, it["q"])
    return text, ct, dict(n_selected=len(chosen), recall=sum(f in chosen for f in it["oracle_facts"]) / len(it["oracle_facts"]))


def run_lookup(port, it, max_lookups=3):
    allf = it["all_facts"]
    terms = [w for w in re.findall(r"\b[A-Z][a-z]+\b", it["q"]) if w.lower() not in STOP]
    facts = []
    for t in terms:
        facts += [f for f in lookup(allf, t) if f not in facts]
    ctoks, lookups, text = 0, 0, None
    while lookups <= max_lookups:
        body = ("\n".join(facts) + "\n\n" if facts else "") + (
            f"Question: {it['q']}\nIf these facts are enough to answer, reply 'ANSWER: <answer>'. "
            "If you need the facts about another person or city, reply 'LOOKUP: <name>'.")
        att = R.chat(port, R.msgs(body), 0.0, 1.0, 24, 0)
        ctoks += att["ctoks"]
        m = re.search(r"LOOKUP:\s*([A-Z][a-z]+)", att["text"])
        a = re.search(r"ANSWER:\s*(.+)", att["text"])
        if a and (not m or a.start() < m.start()):
            text = a.group(1)
            break
        if m and lookups < max_lookups:
            new = [f for f in lookup(allf, m.group(1)) if f not in facts]
            facts += new
            lookups += 1
            continue
        break
    if text is None:
        text, ct = answer(port, facts, it["q"])
        ctoks += ct
    return text, ctoks, dict(n_selected=len(facts), lookups=lookups,
                             recall=sum(f in facts for f in it["oracle_facts"]) / len(it["oracle_facts"]))


METHODS = {"self": run_self, "lexical": run_lexical, "lookup": run_lookup}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="lexical,lookup,self")
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.questions))
    res = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    rec = defaultdict(list)
    with open(os.path.join(a.out, "select.jsonl"), "w") as log:
        for m in a.methods.split(","):
            for it in items:
                text, ct, info = METHODS[m](a.port, it)
                ok = correct(text, it["ans"])
                res[it["type"]][m][0] += ok
                res[it["type"]][m][1] += 1
                rec[m].append(info["recall"])
                log.write(json.dumps(dict(method=m, type=it["type"], q=it["q"], ans=it["ans"], reply=text[:80], ok=ok, ctoks=ct, **info)) + "\n")
            log.flush()
            print(f"method {m} done", flush=True)
    ms = a.methods.split(",")
    lines = ["| type | " + " | ".join(ms) + " |", "|---" * (len(ms) + 1) + "|"]
    for t in ("hop1", "hop2", "hop3", "count", "older"):
        cells = []
        for m in ms:
            k, n = res[t][m]
            lo, hi = wilson(k, n)
            cells.append(f"{k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("| all | " + " | ".join(f"{sum(res[t][m][0] for t in res)}/{sum(res[t][m][1] for t in res)}" for m in ms) + " |")
    lines.append("| recall of needed facts | " + " | ".join(f"{100 * sum(rec[m]) / len(rec[m]):.0f}%" for m in ms) + " |")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(a.out, "summary.md"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
