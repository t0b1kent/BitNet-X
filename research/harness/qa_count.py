#!/usr/bin/env python3
"""E0b-4: counting over many facts. Why models fail and which procedure makes them count.

Diagnosis from E0b-3: BitNet cannot add past 3 even on clean facts (aggregation limit); Bonsai 4B adds perfectly but
misses matching facts in a long list (29/30 undercounts at 117 facts: retrieval under attention dilution); Bonsai 2B
includes non-matching people. Each method removes one or both failure points:

  direct     the model answers with the number (baseline, = E0b-3 lenient)
  enumerate  the model copies every matching fact as a numbered list, then the count on the last line
  chunk      facts split into chunks of --chunk; the model counts each chunk (inside its reliable range); code adds
  judge      the model answers yes/no for each fact separately (no dilution); code adds
  program    the model sees 6 example facts and writes count_people(facts) for this city; the sandbox runs it
  -- round 2 (towards 100 %) --
  judge2     per fact, two independent questions: yes/no, and "which city does this fact say someone lives in?"
             (the city is compared by code). Reported as judge_yn, judge_city and cascade (both must agree)
  program_v  generic count_people(facts, city): greedy + 4 samples; the first one that passes a worked example
             (another question's fact list with its known answer) is used; selected@1 as in E0
  scan       numbered facts, one pass, output forced by a grammar to exactly "1:Y 2:N ..." (one decision per fact,
             no copying, no loops); code counts the Y
Same questions and the same fact lists as qa_noise.py (seed noise/<index>/<level>).
    qa_count.py --questions ../results/e0b_qa/questions.json --out ../results/cmp/count/<model> --levels 0,30,all --jobs 8
Caveat: one fact syntax, so finding "lives in <city>" is lexical here; paraphrased facts are the harder follow-up.
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_modes as R  # noqa: E402
from grade import extract_code, run_tests  # noqa: E402
from qa_probe import NUMW, answer_line  # noqa: E402

METHODS = ("direct", "enumerate", "chunk", "judge", "program", "judge2", "program_v", "scan")
ZERO = re.compile(r"\b(no one|nobody|none|zero|no people|no person)\b")


def parse_num(reply, line="last"):
    text = re.sub(r"\bno one\b", "nobody", answer_line(reply, line))  # "no one" is zero, not 1
    toks = [NUMW.get(t, t) for t in re.findall(r"[a-z0-9]+", text)]
    nums = [t for t in toks if t.isdigit()]
    if nums:
        return int(nums[0])
    return 0 if ZERO.search(text) else None


def fact_list(it, i, lv):
    rng = random.Random(f"noise/{i}/{lv}")
    others = [f for f in it["all_facts"] if f not in it["oracle_facts"]]
    n = len(others) if lv == "all" else min(int(lv), len(others))
    facts = it["oracle_facts"] + rng.sample(others, n)
    rng.shuffle(facts)
    return facts


def city_of(q):
    return re.search(r"How many people live in (\w+)\?", q).group(1)


def first_word(text):
    m = re.search(r"[A-Za-z]+", text)
    return m.group(0).lower() if m else ""


def scan_grammar(n):
    lines = " ".join(f'"{i}:" yn "\\n"' for i in range(1, n + 1))
    return f"root ::= {lines}\nyn ::= \"Y\" | \"N\"\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="0,30,all")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", default="count.jsonl")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allq = json.load(open(a.questions))
    qs = [(i, it) for i, it in enumerate(allq) if it["type"] == "count"]
    if a.limit:
        qs = qs[:a.limit]
    pool = ThreadPoolExecutor(max_workers=a.jobs)

    def ask(body, mt, temp=0.0, seed=0, extra=None):
        return R.chat(a.port, R.msgs(body), temp, 0.95 if temp else 1.0, mt, seed, extra)

    res = defaultdict(lambda: [0, 0])
    extra = defaultdict(lambda: defaultdict(int))
    order = []
    log = open(os.path.join(a.out, a.log), "w")

    def emit(method, lv, rows):
        if method not in order:
            order.append(method)
        for r in rows:
            ok = r["pred"] == r["ans"]
            res[(method, lv)][0] += ok; res[(method, lv)][1] += 1
            out = {k: v for k, v in r.items() if k not in ("it", "facts", "gold", "pf")}
            log.write(json.dumps(dict(method=method, level=lv, q=r["it"]["q"], ok=ok, n_facts=len(r["facts"]), **out)) + "\n")
        log.flush()
        print(f"{method} level {lv}: {res[(method, lv)][0]}/{res[(method, lv)][1]}", flush=True)

    def fact_stats(method, lv, rows, flags):
        """flags[ri] = list of (fact, decided_yes); fills pred / tp / fp / fn and the summary counters."""
        for ri, r in enumerate(rows):
            tp = sum(1 for f, y in flags[ri] if y and f in r["gold"])
            fp = sum(1 for f, y in flags[ri] if y and f not in r["gold"])
            fn = sum(1 for f, y in flags[ri] if not y and f in r["gold"])
            r.update(pred=tp + fp, tp=tp, fp=fp, fn=fn,
                     fp_facts=[f for f, y in flags[ri] if y and f not in r["gold"]][:5],
                     fn_facts=[f for f, y in flags[ri] if not y and f in r["gold"]][:5])
            for k in ("tp", "fp", "fn"):
                extra[(method, lv)][k] += r[k]

    for method in a.methods.split(","):
        for lv in a.levels.split(","):
            rows = []
            for i, it in qs:
                facts = fact_list(it, i, lv)
                rows.append(dict(i=i, it=it, facts=facts, city=city_of(it["q"]), ans=int(it["ans"]),
                                 gold=set(it["oracle_facts"])))
            if method == "direct":
                bodies = ["\n".join(r["facts"]) + f"\n\nQuestion: {r['it']['q']}\nReply with only the answer." for r in rows]
                for r, att in zip(rows, pool.map(lambda b: ask(b, 384), bodies)):
                    r["pred"] = parse_num(att["text"]); r["reply"] = att["text"][:1500]
                emit(method, lv, rows)
            elif method == "enumerate":
                bodies = ["\n".join(r["facts"]) + f"\n\nQuestion: {r['it']['q']}\n"
                          f"Go through the facts above in order. Copy every fact that says someone lives in {r['city']}, "
                          "one per line, numbered 1, 2, 3, ... Copy nothing else. "
                          "Then write the number of people alone on the last line." for r in rows]
                for r, att in zip(rows, pool.map(lambda b: ask(b, 384), bodies)):
                    r["pred"] = parse_num(att["text"]); r["reply"] = att["text"][:1500]
                    listed = [m.group(1).strip() for m in re.finditer(r"^\s*\d+[.)]\s*(.+?)\s*$", att["text"], re.M)]
                    tp = len(set(listed) & r["gold"]); fp = len([x for x in listed if x not in r["gold"]])
                    r["listed"] = len(listed); r["tp"] = tp; r["fp"] = fp; r["fn"] = len(r["gold"]) - tp
                    r["lines_ok"] = len(listed) == r["ans"]
                    for k in ("tp", "fp", "fn", "lines_ok"):
                        extra[(method, lv)][k] += r[k]
                emit(method, lv, rows)
            elif method == "chunk":
                jobs = []
                for ri, r in enumerate(rows):
                    for c in range(0, len(r["facts"]), a.chunk):
                        part = r["facts"][c:c + a.chunk]
                        jobs.append((ri, sum(f in r["gold"] for f in part),
                                     "\n".join(part) + f"\n\nQuestion: {r['it']['q']}\nCount only the facts above. Reply with only the number."))
                outs = list(pool.map(lambda j: ask(j[2], 384), jobs))
                for r in rows:
                    r["pred"] = 0; r["chunk_err"] = 0; r["chunks"] = 0
                for (ri, true_k, _), att in zip(jobs, outs):
                    p = parse_num(att["text"])
                    rows[ri]["pred"] = None if (p is None or rows[ri]["pred"] is None) else rows[ri]["pred"] + p
                    rows[ri]["chunks"] += 1; rows[ri]["chunk_err"] += (p != true_k)
                for r in rows:
                    extra[(method, lv)]["chunks"] += r["chunks"]; extra[(method, lv)]["chunk_err"] += r["chunk_err"]
                emit(method, lv, rows)
            elif method == "judge":
                jobs = [(ri, f, f"Fact: {f}\nDoes this fact say that someone lives in {r['city']}? Reply with only yes or no.")
                        for ri, r in enumerate(rows) for f in r["facts"]]
                outs = list(pool.map(lambda j: ask(j[2], 8), jobs))
                flags = defaultdict(list)
                for (ri, f, _), att in zip(jobs, outs):
                    flags[ri].append((f, att["text"].strip().lower().startswith("yes")))
                fact_stats(method, lv, rows, flags)
                emit(method, lv, rows)
            elif method == "judge2":
                yn_jobs = [(ri, f, f"Fact: {f}\nDoes this fact say that someone lives in {r['city']}? Reply with only yes or no.")
                           for ri, r in enumerate(rows) for f in r["facts"]]
                # direct question; the conditional wording "if ... reply the city, otherwise none" made BitNet answer
                # "none" even for "X lives in Y." (checked by hand, 2026-11-18)
                city_jobs = [(ri, f, f"Fact: {f}\nWhich city does this fact say someone lives in? "
                                     "If it does not mention where someone lives, answer none.\nAnswer with one word.")
                             for ri, r in enumerate(rows) for f in r["facts"]]
                yn = list(pool.map(lambda j: ask(j[2], 8), yn_jobs))
                ct = list(pool.map(lambda j: ask(j[2], 12), city_jobs))
                f_yn, f_ct, f_both = defaultdict(list), defaultdict(list), defaultdict(list)
                for (ri, f, _), a1, a2 in zip(yn_jobs, yn, ct):
                    y1 = a1["text"].strip().lower().startswith("yes")
                    words = re.findall(r"[a-z]+", answer_line(a2["text"], "first").lower())
                    y2 = rows[ri]["city"].lower() in words and "none" not in words[:1]
                    f_yn[ri].append((f, y1)); f_ct[ri].append((f, y2)); f_both[ri].append((f, y1 and y2))
                for name, fl in (("judge_yn", f_yn), ("judge_city", f_ct), ("cascade", f_both)):
                    rs = [dict(r) for r in rows]
                    fact_stats(name, lv, rs, fl)
                    emit(name, lv, rs)
            elif method == "program":
                bodies = []
                for r in rows:
                    ex = random.Random(f"examples/{r['i']}").sample(r["it"]["all_facts"], 6)
                    bodies.append("Each fact is one English sentence. The facts are given as a Python list of strings `facts`. "
                                  "Examples of facts:\n" + "\n".join(ex) +
                                  f"\n\nWrite a Python function `count_people(facts)` that returns how many people live in {r['city']} "
                                  "according to the facts. Reply with only the code.")
                atts = list(pool.map(lambda b: ask(b, 384), bodies))
                codes = [extract_code(att["text"]) for att in atts]
                runs = list(pool.map(lambda x: None if x[0] is None else run_tests(x[0], "count_people", [[[x[1]["facts"]], x[1]["ans"]]]),
                                     zip(codes, rows)))
                for r, att, run in zip(rows, atts, runs):
                    r["reply"] = att["text"][:1500]
                    r["pred"] = r["ans"] if (run and run.get("passed") == 1) else None
                    r["run_status"] = None if run is None else run.get("status")
                emit(method, lv, rows)
            elif method == "program_v":
                n = len(rows)
                jobs = []
                for ri, r in enumerate(rows):
                    ex = random.Random(f"examples/{r['i']}").sample(r["it"]["all_facts"], 6)
                    body = ("Each fact is one English sentence. The facts are given as a Python list of strings `facts`. "
                            "Examples of facts:\n" + "\n".join(ex) +
                            "\n\nWrite a Python function `count_people(facts, city)` that returns how many people live in `city` "
                            "according to the facts. Reply with only the code.")
                    for s in range(5):
                        jobs.append((ri, s, body))
                atts = list(pool.map(lambda j: ask(j[2], 384, 0.0 if j[1] == 0 else 0.8, j[1]), jobs))
                cands = defaultdict(list)
                for (ri, s, _), att in zip(jobs, atts):
                    cands[ri].append(extract_code(att["text"]))
                # worked example: another question's list (next world) with its known answer
                pub = {ri: rows[(ri + 7) % n] for ri in range(n)}

                def check(x):
                    ri, code = x
                    if code is None:
                        return False
                    p = pub[ri]
                    run = run_tests(code, "count_people", [[[p["facts"], p["city"]], p["ans"]]])
                    return run.get("passed") == 1
                checks = list(pool.map(check, [(ri, c) for ri in range(n) for c in cands[ri]]))
                k = 0
                chosen = {}
                for ri in range(n):
                    ok_list = checks[k:k + len(cands[ri])]; k += len(cands[ri])
                    idx = next((j for j, o in enumerate(ok_list) if o), None)
                    chosen[ri] = (idx, cands[ri][idx if idx is not None else 0])
                runs = list(pool.map(lambda ri: None if chosen[ri][1] is None else
                                     run_tests(chosen[ri][1], "count_people", [[[rows[ri]["facts"], rows[ri]["city"]], rows[ri]["ans"]]]),
                                     range(n)))
                for ri, (r, run) in enumerate(zip(rows, runs)):
                    r["pred"] = r["ans"] if (run and run.get("passed") == 1) else None
                    r["passing_candidate"] = chosen[ri][0]
                    r["code"] = (chosen[ri][1] or "")[:600]
                    extra[(method, lv)]["found"] += chosen[ri][0] is not None
                emit(method, lv, rows)
            elif method == "scan":
                bodies, grams = [], []
                for r in rows:
                    numbered = "\n".join(f"[{j}] {f}" for j, f in enumerate(r["facts"], 1))
                    bodies.append(numbered + f"\n\nQuestion: {r['it']['q']}\n"
                                  f"For each fact, in order, write its number, a colon, and Y if the fact says someone lives in "
                                  f"{r['city']}, otherwise N.")
                    grams.append(scan_grammar(len(r["facts"])))
                atts = list(pool.map(lambda x: ask(x[0], 6 * len(x[2]["facts"]) + 16, extra={"grammar": x[1]}),
                                     zip(bodies, grams, rows)))
                flags = defaultdict(list)
                for ri, (r, att) in enumerate(zip(rows, atts)):
                    marks = dict((int(m.group(1)), m.group(2)) for m in re.finditer(r"(\d+):([YN])", att["text"]))
                    for j, f in enumerate(r["facts"], 1):
                        flags[ri].append((f, marks.get(j) == "Y"))
                    r["complete"] = len(marks) == len(r["facts"])
                    extra[(method, lv)]["complete"] += r["complete"]
                fact_stats(method, lv, rows, flags)
                emit(method, lv, rows)
    levels = a.levels.split(",")
    lines = ["| method | " + " | ".join(f"{lv} distractors" for lv in levels) + " | notes |", "|---" * (len(levels) + 2) + "|"]
    for method in order:
        notes = []
        for lv in levels:
            e = extra[(method, lv)]
            nq = res[(method, lv)][1]
            if "tp" in e or "fn" in e:
                s = f"{lv}: found {e['tp']}/{e['tp'] + e['fn']} needed, {e['fp']} wrong"
                if method == "enumerate":
                    s += f", line count right {e['lines_ok']}/{nq}"
                if method == "scan":
                    s += f", complete {e['complete']}/{nq}"
                notes.append(s)
            if method == "chunk" and e["chunks"]:
                notes.append(f"{lv}: wrong chunk counts {e['chunk_err']}/{e['chunks']}")
            if method == "program_v":
                notes.append(f"{lv}: a candidate passed the worked example in {e['found']}/{nq}")
        lines.append(f"| {method} | " + " | ".join(f"{res[(method, lv)][0]}/{res[(method, lv)][1]}" for lv in levels)
                     + " | " + "; ".join(notes) + " |")
    text = "\n".join(lines)
    print(text)
    name = "summary.md" if a.log == "count.jsonl" else a.log.replace(".jsonl", "_summary.md")
    open(os.path.join(a.out, name), "w").write(text + "\n")


if __name__ == "__main__":
    main()
