#!/usr/bin/env python3
"""E0b-5: read once into a table, answer by an exact query (the model parses, code reasons).

Why: E0b-3/E0b-4 show the failures are attention over a long list (missed or extra facts) and adding past a few
items, not understanding one fact. So the model only ever looks at one sentence at a time:
  1. per world, once (cached): each fact -> JSON {person, attr, value}, decoding constrained by a JSON schema;
  2. per question: the question -> a query JSON {op: get | count | older, ...}, schema-constrained, 5 worked examples;
  3. code runs the query over the table (follows mentor chains, counts, compares ages exactly).
All 150 E0b questions, open condition (all 117 facts of the world), on the original wording and on paraphrases whose
wording is not in the worked examples. Extraction and parsing are scored against gold derived from the generator's
fact templates.
    qa_table.py --questions ../results/e0b_qa/questions.json --out ../results/cmp/table/<model> --jobs 8
Caveat: one fact syntax per relation, so fact parsing is easy here; paraphrased facts are the harder follow-up.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_modes as R  # noqa: E402
from qa_probe import correct, wilson  # noqa: E402

ATTRS = ["job", "city", "age", "pet", "mentor"]
FACT_SCHEMA = {"type": "object", "properties": {"person": {"type": "string"}, "attr": {"type": "string", "enum": ATTRS},
                                                "value": {"type": "string"}},
               "required": ["person", "attr", "value"], "additionalProperties": False}
QUERY_SCHEMA = {"type": "object", "properties": {
    "op": {"type": "string", "enum": ["get", "count", "older"]},
    "person": {"type": "string"},
    "path": {"type": "array", "items": {"type": "string", "enum": ATTRS}},
    "attr": {"type": "string", "enum": ATTRS},
    "value": {"type": "string"},
    "a": {"type": "string"},
    "b": {"type": "string"}}, "required": ["op"], "additionalProperties": False}

FACT_PROMPT = """Convert the fact into JSON with the keys person, attr (one of job, city, age, pet, mentor) and value.

Fact: Lena works as a doctor.
JSON: {"person": "Lena", "attr": "job", "value": "doctor"}
Fact: Omar lives in Paris.
JSON: {"person": "Omar", "attr": "city", "value": "Paris"}
Fact: Lena is 31 years old.
JSON: {"person": "Lena", "attr": "age", "value": "31"}
Fact: Omar owns a fish.
JSON: {"person": "Omar", "attr": "pet", "value": "fish"}
Fact: Lena's mentor is Omar.
JSON: {"person": "Lena", "attr": "mentor", "value": "Omar"}

Fact: %s
JSON:"""

QUERY_PROMPT = """Convert the question into a JSON query over a table of people (attributes: job, city, age, pet, mentor).
"get" follows the path of attributes starting from a person; "count" counts people with attr equal to value; "older" compares two people.

Question: What is Lena's job?
JSON: {"op": "get", "person": "Lena", "path": ["job"]}
Question: What is the job of Lena's mentor?
JSON: {"op": "get", "person": "Lena", "path": ["mentor", "job"]}
Question: In which city does the mentor of Lena's mentor live?
JSON: {"op": "get", "person": "Lena", "path": ["mentor", "mentor", "city"]}
Question: How many people live in Paris?
JSON: {"op": "count", "attr": "city", "value": "Paris"}
Question: Who is older, Lena or Omar?
JSON: {"op": "older", "a": "Lena", "b": "Omar"}

Question: %s
JSON:"""

GOLD_FACT = [(re.compile(r"^(\w+) works as an? (\w+)\.$"), "job"), (re.compile(r"^(\w+) lives in (\w+)\.$"), "city"),
             (re.compile(r"^(\w+) is (\d+) years old\.$"), "age"), (re.compile(r"^(\w+) owns an? (\w+)\.$"), "pet"),
             (re.compile(r"^(\w+)'s mentor is (\w+)\.$"), "mentor")]
GOLD_Q = [(re.compile(r"^What is (\w+)'s job\?$"), lambda m: {"op": "get", "person": m[1], "path": ["job"]}),
          (re.compile(r"^What is the job of (\w+)'s mentor\?$"), lambda m: {"op": "get", "person": m[1], "path": ["mentor", "job"]}),
          (re.compile(r"^In which city does the mentor of (\w+)'s mentor live\?$"),
           lambda m: {"op": "get", "person": m[1], "path": ["mentor", "mentor", "city"]}),
          (re.compile(r"^How many people live in (\w+)\?$"), lambda m: {"op": "count", "attr": "city", "value": m[1]}),
          (re.compile(r"^Who is older, (\w+) or (\w+)\?$"), lambda m: {"op": "older", "a": m[1], "b": m[2]})]
PARA = {"hop1": ["Which job does {p} have?", "What does {p} do for a living?", "What work does {p} do?"],
        "hop2": ["What does the mentor of {p} do for a living?", "Which job does {p}'s mentor have?",
                 "{p} has a mentor. What is that mentor's job?"],
        "hop3": ["Where does the mentor of the mentor of {p} live?",
                 "{p}'s mentor has a mentor. In which city does that person live?",
                 "Which city is home to the mentor of {p}'s mentor?"],
        "count": ["What is the number of people living in {p}?", "Count the people who live in {p}.",
                  "How many residents does {p} have?"],
        "older": ["Between {p} and {o}, who is older?", "Which of {p} and {o} is older?",
                  "Is {p} older than {o}, or is {o} older? Name the older one."]}


def copy_schema(fact):
    """FACT_SCHEMA where person and value may only be words copied from this fact (BitNet 'corrects' unfamiliar
    names when it copies them freely: Givedo -> Gideon, Habahaford -> Habahof)."""
    words = sorted({w for w in re.findall(r"[A-Za-z0-9]+", fact) if w != "s"})
    s = json.loads(json.dumps(FACT_SCHEMA))
    s["properties"]["person"] = {"type": "string", "enum": words}
    s["properties"]["value"] = {"type": "string", "enum": words}
    return s


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def gold_fact(f):
    for rx, attr in GOLD_FACT:
        m = rx.match(f)
        if m:
            return (norm(m[1]), attr, norm(m[2]))
    return None


def gold_query(q):
    for rx, mk in GOLD_Q:
        m = rx.match(q)
        if m:
            return mk(m)
    return None


def paraphrase(it, k):
    g = gold_query(it["q"])
    t = PARA[it["type"]][k % 3]
    if it["type"] == "count":
        return t.format(p=g["value"])
    if it["type"] == "older":
        return t.format(p=g["a"], o=g["b"])
    return t.format(p=g["person"])


def same_query(x, g):
    if not isinstance(x, dict) or x.get("op") != g["op"]:
        return False
    if g["op"] == "get":
        return norm(x.get("person")) == norm(g["person"]) and list(x.get("path") or []) == g["path"]
    if g["op"] == "count":
        return x.get("attr") == "city" and norm(x.get("value")) == norm(g["value"])
    return {norm(x.get("a")), norm(x.get("b"))} == {norm(g["a"]), norm(g["b"])}


def execute(qy, table):
    """table: person(norm) -> attr -> value (raw). Returns the answer string or None."""
    if not isinstance(qy, dict):
        return None
    op = qy.get("op")
    if op == "get":
        x = qy.get("person")
        for attr in qy.get("path") or []:
            x = table.get(norm(x), {}).get(attr)
            if x is None:
                return None
        return x
    if op == "count":
        attr, v = qy.get("attr") or "city", norm(qy.get("value"))
        return str(sum(1 for d in table.values() if norm(d.get(attr, "")) == v))
    if op == "older":
        a, b = qy.get("a"), qy.get("b")
        try:
            ya, yb = int(table[norm(a)]["age"]), int(table[norm(b)]["age"])
        except (KeyError, ValueError, TypeError):
            return None
        return a if ya > yb else b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--copy", action="store_true", help="person/value restricted to words of the fact (copy_schema)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.questions))
    pool = ThreadPoolExecutor(max_workers=a.jobs)

    def ask_json(prompt, schema, mt):
        att = R.chat(a.port, R.msgs(prompt), 0.0, 1.0, mt, 0, {"json_schema": schema})
        try:
            return json.loads(att["text"]), att["text"]
        except json.JSONDecodeError:
            return None, att["text"]

    # 1. read every world once
    worlds = {}
    for it in items:
        worlds.setdefault(it["world"], it["all_facts"])
    jobs = [(w, f) for w, facts in sorted(worlds.items()) for f in facts]
    outs = list(pool.map(lambda j: ask_json(FACT_PROMPT % j[1], copy_schema(j[1]) if a.copy else FACT_SCHEMA, 48), jobs))
    tables = defaultdict(lambda: defaultdict(dict))
    ext_ok = ext_n = 0
    ext_err = []
    with open(os.path.join(a.out, "facts.jsonl"), "w") as log:
        for (w, f), (d, raw) in zip(jobs, outs):
            g = gold_fact(f)
            got = None
            if isinstance(d, dict) and {"person", "attr", "value"} <= d.keys():
                tables[w][norm(d["person"])][d["attr"]] = str(d["value"]).strip()
                got = (norm(d["person"]), d["attr"], norm(d["value"]))
            ok = got == g
            ext_ok += ok; ext_n += 1
            if not ok and len(ext_err) < 8:
                ext_err.append((f, raw))
            log.write(json.dumps(dict(world=w, fact=f, raw=raw, ok=ok)) + "\n")
    print(f"fact extraction exact: {ext_ok}/{ext_n}", flush=True)

    # 2-3. parse each question, run the query
    res = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    parse_ok = defaultdict(lambda: [0, 0])
    with open(os.path.join(a.out, "questions.jsonl"), "w") as log:
        for wording in ("original", "paraphrase"):
            qtexts = [it["q"] if wording == "original" else paraphrase(it, k) for k, it in enumerate(items)]
            parsed = list(pool.map(lambda q: ask_json(QUERY_PROMPT % q, QUERY_SCHEMA, 64), qtexts))
            for it, qt, (qy, raw) in zip(items, qtexts, parsed):
                ans = execute(qy, tables[it["world"]])
                ok = ans is not None and correct(str(ans), it["ans"], it["q"])
                pok = same_query(qy, gold_query(it["q"]))
                res[wording][it["type"]][0] += ok; res[wording][it["type"]][1] += 1
                parse_ok[wording][0] += pok; parse_ok[wording][1] += 1
                log.write(json.dumps(dict(wording=wording, type=it["type"], q=qt, ans=it["ans"], query=qy, raw=raw,
                                          got=ans, ok=ok, parse_ok=pok)) + "\n")
            print(f"{wording}: parse exact {parse_ok[wording][0]}/{parse_ok[wording][1]}", flush=True)

    types = ("hop1", "hop2", "hop3", "count", "older")
    lines = [f"Fact extraction exact: {ext_ok}/{ext_n}", "",
             "| wording | query parse exact | " + " | ".join(types) + " | all |", "|---" * (len(types) + 3) + "|"]
    for wording in ("original", "paraphrase"):
        k = sum(res[wording][t][0] for t in types); n = sum(res[wording][t][1] for t in types)
        lo, hi = wilson(k, n)
        lines.append(f"| {wording} | {parse_ok[wording][0]}/{parse_ok[wording][1]} | "
                     + " | ".join(f"{res[wording][t][0]}/{res[wording][t][1]}" for t in types)
                     + f" | {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}] |")
    if ext_err:
        lines += ["", "Extraction errors (first 8):"] + [f"* `{f}` -> `{raw[:120]}`" for f, raw in ext_err]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(a.out, "summary.md"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
