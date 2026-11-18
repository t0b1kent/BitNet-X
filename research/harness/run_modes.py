#!/usr/bin/env python3
"""E0: where does BitNet b1.58 2B-4T fail on small coding tasks?  Modes A..E against bitnet.cpp's llama-server.

  A single   one greedy answer
  B k5       five independent samples (T=0.7, top_p=0.95)
  C hint     greedy, the prompt carries a short algorithm hint
  D feedback greedy, then up to 3 repair rounds with execution feedback from the 3 public examples
  E cot      greedy, the model first writes up to 5 steps of reasoning, then the code

Every attempt is graded on the private tests in a sandbox (grade.py). Resumable: finished (mode, task) pairs are skipped.

    run_modes.py --tasks ../tasks/dev_100.jsonl --modes A,B,C,D,E --out ../results/e0_dev --per-family-limit 10
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grade import grade_text  # noqa: E402

SYSTEM = "You are a helpful AI assistant."
TAIL = "\n\nReply with the complete function in a single ```python code block and nothing else."
TAIL_COT = ("\n\nFirst describe your approach in at most 5 short numbered steps. "
            "Then give the complete function in a single ```python code block.")
FIX_TAIL = " Reply with the complete corrected function in a single ```python code block and nothing else."
MAX_TOKENS = {"A": 600, "B": 600, "C": 600, "D": 600, "E": 900}


def render(messages):
    """The chat format of the original HF model (tokenizer_config.json): 'Role: content<|eot_id|>' per message, then
    'Assistant: '. BOS is added by the server's tokenizer. The template stored inside the official GGUF differs
    ('Human:' / 'BITNETAssistant:' plus an EOS after the generation prompt) and derails the model, so it is not used."""
    out = "".join(f"{m['role'].capitalize()}: {m['content'].strip()}<|eot_id|>" for m in messages)
    return out + "Assistant: "


API = os.environ.get("BX_API", "bitnet")  # "bitnet": raw /completion with the original BitNet format; "chat": model's own template


def chat(port, messages, temperature, top_p, max_tokens, seed, extra=None):
    """extra: additional server fields, e.g. {"grammar": ...} or {"json_schema": ...} (constrained decoding)."""
    if API == "chat":
        # the model's own chat template (server started with --jinja); thinking off, so every model answers directly
        body = {"messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed,
                "stream": False, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}, **(extra or {})}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        text = d["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
        u = d.get("usage", {})
        return dict(text=text, ptoks=u.get("prompt_tokens", 0), ctoks=u.get("completion_tokens", 0),
                    secs=time.time() - t0, finish=d["choices"][0].get("finish_reason"))
    body = {"prompt": render(messages), "n_predict": max_tokens, "temperature": temperature, "top_p": top_p,
            "seed": seed, "stream": False, "cache_prompt": True, **(extra or {})}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return dict(text=d["content"], ptoks=d.get("tokens_evaluated", 0), ctoks=d.get("tokens_predicted", 0),
                secs=time.time() - t0, finish=d.get("stop_type"))


def msgs(user):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def graded(att, task, which="private"):
    cat, res = grade_text(att["text"], task, which)
    att["category"] = cat
    att["passed"] = res["passed"] if res else 0
    att["total"] = res["total"] if res else len(task[which])
    att["first_fail"] = res.get("first_fail") if res else None
    att["error"] = res.get("error") if res else ""
    return att


def finish(mode, task, attempts, ok, extra=None):
    rec = dict(mode=mode, task=task["id"], family=task["family"], skills=task["skills"], ok=bool(ok),
               ctoks=sum(a["ctoks"] for a in attempts), ptoks=sum(a["ptoks"] for a in attempts),
               secs=round(sum(a["secs"] for a in attempts), 2), attempts=attempts)
    if extra:
        rec.update(extra)
    return rec


def mode_A(port, task, **_):
    a = graded(chat(port, msgs(task["prompt"] + TAIL), 0.0, 1.0, MAX_TOKENS["A"], 0), task)
    return finish("A", task, [a], a["category"] == "pass")


def mode_B(port, task, **_):
    atts = [graded(chat(port, msgs(task["prompt"] + TAIL), 0.7, 0.95, MAX_TOKENS["B"], 1000 + s), task) for s in range(5)]
    c = sum(a["category"] == "pass" for a in atts)
    return finish("B", task, atts, c > 0, dict(n=5, c=c))


def mode_C(port, task, **_):
    a = graded(chat(port, msgs(task["prompt"] + f"\n\nHint: {task['hint']}" + TAIL), 0.0, 1.0, MAX_TOKENS["C"], 0), task)
    return finish("C", task, [a], a["category"] == "pass")


def feedback_text(task, att):
    cat, name = att["category"], task["name"]
    if cat == "no_code":
        return "Your reply did not contain a Python code block." + FIX_TAIL
    if cat in ("syntax_error", "import_error"):
        return f"Your code could not be loaded: {att['error']}." + FIX_TAIL
    if cat == "no_entry":
        return f"The function `{name}` is not defined in your code." + FIX_TAIL
    ff = att.get("first_fail") or {}
    call = f"{name}({', '.join(repr(x) for x in ff.get('args', []))})"
    if ff.get("kind") == "timeout":
        return f"The call {call} did not finish in time (is there an infinite loop?)." + FIX_TAIL
    if ff.get("kind") == "error":
        return f"The call {call} raised {ff.get('error')}." + FIX_TAIL
    return f"The call {call} should return {ff.get('expected')!r} but your code returned {ff.get('got')}." + FIX_TAIL


def mode_D(port, task, first=None, **_):
    user = task["prompt"] + TAIL
    conv = msgs(user)
    a = first if first else chat(port, conv, 0.0, 1.0, MAX_TOKENS["D"], 0)
    atts = [a]
    rounds = 0
    while True:
        pub = graded(dict(a, text=a["text"]), task, "public")
        if pub["category"] == "pass" or rounds == 3:
            break
        rounds += 1
        conv = conv + [{"role": "assistant", "content": a["text"]}, {"role": "user", "content": feedback_text(task, pub)}]
        a = chat(port, conv, 0.0, 1.0, MAX_TOKENS["D"], 0)
        atts.append(a)
    for att in atts:
        graded(att, task, "private")
    return finish("D", task, atts, atts[-1]["category"] == "pass", dict(rounds=rounds, public_ok=pub["category"] == "pass"))


def mode_E(port, task, **_):
    a = graded(chat(port, msgs(task["prompt"] + TAIL_COT), 0.0, 1.0, MAX_TOKENS["E"], 0), task)
    return finish("E", task, [a], a["category"] == "pass")


MODES = {"A": mode_A, "B": mode_B, "C": mode_C, "D": mode_D, "E": mode_E}


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--modes", default="A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--per-family-limit", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    seen, tasks = {}, []
    for t in load_jsonl(a.tasks):
        if seen.get(t["family"], 0) < a.per_family_limit:
            seen[t["family"]] = seen.get(t["family"], 0) + 1
            tasks.append(t)

    meta_path = os.path.join(a.out, "meta.json")
    if not os.path.exists(meta_path):
        repo = os.path.join(HERE, "..", "..")
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        json.dump(dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), tasks_file=a.tasks, n_tasks=len(tasks), repo_commit=commit,
                       load_at_start=os.getloadavg(), server_port=a.port, python=sys.version.split()[0]),
                  open(meta_path, "w"), indent=2)

    for mode in a.modes.split(","):
        path = os.path.join(a.out, f"mode_{mode}.jsonl")
        done = {r["task"] for r in load_jsonl(path)}
        a_log = {r["task"]: r for r in load_jsonl(os.path.join(a.out, "mode_A.jsonl"))}
        with open(path, "a", encoding="utf-8") as fh:
            for i, task in enumerate(tasks):
                if task["id"] in done:
                    continue
                first = a_log[task["id"]]["attempts"][0] if mode == "D" and task["id"] in a_log else None
                if first:
                    first = {k: first[k] for k in ("text", "ptoks", "ctoks", "secs", "finish")}
                rec = MODES[mode](a.port, task, first=first)
                rec["load"] = round(os.getloadavg()[0], 2)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                cats = ",".join(x["category"] for x in rec["attempts"])
                print(f"[{mode}] {i + 1:3d}/{len(tasks)} {task['id']:20s} ok={int(rec['ok'])} ctoks={rec['ctoks']:5d} {rec['secs']:6.1f}s {cats}", flush=True)


if __name__ == "__main__":
    main()
