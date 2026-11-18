#!/usr/bin/env python3
"""Look inside BitNet b1.58 2B-4T (MLX, BF16 master with the released online ternary quantization).

attn   Attention from the last prompt position (the one that produces the answer) to every fact, per layer and head.
       For each level of distractors: which heads put their attention on the needed facts, how much of it, and does
       the best head still rank the needed facts on top when the list grows (attention dilution, retrieval heads).
lens   Logit lens: at every layer, project the last hidden state through the final norm and the (tied) embedding and
       read the rank of the correct answer token and of "none", for two wordings of the same request.
       Shows whether the answer exists inside and a late layer overrides it, or it never forms.

    ~/Developer/MicrosoftAI/.venv-train/bin/python inside.py attn --out ../results/inside/bitnet
    ~/Developer/MicrosoftAI/.venv-train/bin/python inside.py lens --out ../results/inside/bitnet
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import bx_qat  # noqa: E402
import mlx_lm.models.bitnet as B  # noqa: E402
from mlx_lm.models.base import create_attention_mask  # noqa: E402

Q = os.path.join(HERE, "..", "results", "e0b_qa", "questions.json")
SYSTEM = "You are a helpful AI assistant."
CAPTURE = {"on": False, "rows": []}
_orig_attn = B.Attention.__call__


TAU = {}  # layer index -> mx.array [n_heads] of logit multipliers (sharpening)


def _attn_capture(self, x, mask=None, cache=None):
    if getattr(self, "_tau", None) is not None:
        return _attn_sharp(self, x, mask, cache)
    if CAPTURE["on"]:
        Bsz, L, _ = x.shape
        q = self.q_proj(x).reshape(Bsz, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(Bsz, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        rep = self.n_heads // self.n_kv_heads
        k = mx.repeat(k, rep, axis=1)
        s = (q[:, :, -1:, :].astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2))[0, :, 0, :] * self.scale
        CAPTURE["rows"].append(mx.softmax(s, axis=-1))  # [heads, L], last query sees every position
    return _orig_attn(self, x, mask, cache)


def _attn_sharp(self, x, mask, cache):
    from mlx_lm.models.base import scaled_dot_product_attention
    Bsz, L, _ = x.shape
    q = self.q_proj(x).reshape(Bsz, L, self.n_heads, -1).transpose(0, 2, 1, 3)
    k = self.k_proj(x).reshape(Bsz, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = self.v_proj(x).reshape(Bsz, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q, k = self.rope(q), self.rope(k)
    q = q * self._tau[None, :, None, None].astype(q.dtype)  # scaling the query scales that head's logits
    o = scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=mask)
    o = o.transpose(0, 2, 1, 3).reshape(Bsz, L, -1)
    return self.o_proj(self.attn_sub_norm(o))


B.Attention.__call__ = _attn_capture


def set_tau(model, heads, tau):
    """heads: list of (layer, head); tau applied to them, 1 elsewhere; tau == 1 removes the hook."""
    for l, layer in enumerate(model.model.layers):
        t = np.ones(layer.self_attn.n_heads, dtype=np.float32)
        for (hl, hh) in heads:
            if hl == l:
                t[hh] = tau
        layer.self_attn._tau = None if (tau == 1 or (t == 1).all()) else mx.array(t)


def first_ids(tok, word):
    return sorted({tok.encode(v, add_special_tokens=False).ids[0] for v in (word, " " + word, word.capitalize(), " " + word.capitalize())})


def cmd_sharpen(a, model, tok):
    items = json.load(open(Q))
    sel = [(i, it) for i, it in enumerate(items) if it["type"] in ("hop2", "hop3", "count")]
    retrieval = [(16, 13), (16, 8), (16, 6), (17, 4), (14, 2), (17, 17), (17, 12)]  # from attn_count.md / attn_hop2.md
    layers_14_17 = [(l, h) for l in range(14, 18) for h in range(20)]
    every = [(l, h) for l in range(30) for h in range(20)]
    sets = {"retrieval heads (7)": retrieval, "all heads, layers 14-17 (80)": layers_14_17, "all 600 heads": every}
    prompts = {}
    for lv in a.levels.split(","):
        for i, it in sel:
            ids, _ = prompt_with_spans(tok, fact_list(it, i, lv), it["q"])
            prompts[(lv, i)] = (ids, first_ids(tok, it["ans"]), it["type"])
    rows = []

    def score():
        res = defaultdict(lambda: [0, 0])
        for (lv, i), (ids, tg, ty) in prompts.items():
            logits = model(mx.array(ids)[None])[0, -1]
            ok = int(mx.argmax(logits)) in tg
            res[(lv, ty)][0] += ok; res[(lv, ty)][1] += 1
        return res
    set_tau(model, [], 1)
    base = score()
    rows.append(("baseline", 1.0, base))
    print("baseline", {k: v[0] for k, v in base.items()}, flush=True)
    for name, heads in sets.items():
        for tau in [float(t) for t in a.taus.split(",")]:
            set_tau(model, heads, tau)
            r = score()
            rows.append((name, tau, r))
            print(name, tau, {k: v[0] for k, v in r.items()}, flush=True)
    set_tau(model, [], 1)
    lvls = a.levels.split(",")
    hdr = "| heads | x logits | " + " | ".join(f"{lv}: {t}" for lv in lvls for t in ("hop2", "hop3", "count")) + " | sum |"
    lines = ["First answer token correct (of 30 per cell), BitNet, prompt 'Reply with only the answer.'", "", hdr,
             "|---|---|" + "---|" * (3 * len(lvls) + 1)]
    for name, tau, r in rows:
        cells = [f"{r[(lv, t)][0]}" for lv in lvls for t in ("hop2", "hop3", "count")]
        lines.append(f"| {name} | {tau} | " + " | ".join(cells) + f" | {sum(v[0] for v in r.values())} |")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(a.out, "sharpen.md"), "w").write(text + "\n")


def prompt_with_spans(tok, facts, question, tail="Reply with only the answer."):
    head = f"System: {SYSTEM}<|eot_id|>User: "
    body = "\n".join(facts)
    text = head + body + f"\n\nQuestion: {question}\n{tail}<|eot_id|>Assistant: "
    enc = tok.encode(text, add_special_tokens=False)
    ids = [bx_qat.BOS] + enc.ids
    offs = [(0, 0)] + enc.offsets
    spans, pos = [], len(head)
    for f in facts:  # char range of each fact -> token positions
        a, b = pos, pos + len(f)
        spans.append([i for i, (s, e) in enumerate(offs) if e > a and s < b and e > s])
        pos = b + 1
    return ids, spans


def fact_list(it, i, lv):
    rng = random.Random(f"noise/{i}/{lv}")
    others = [f for f in it["all_facts"] if f not in it["oracle_facts"]]
    n = len(others) if lv == "all" else min(int(lv), len(others))
    facts = it["oracle_facts"] + rng.sample(others, n)
    rng.shuffle(facts)
    return facts


def cmd_attn(a, model, tok):
    items = json.load(open(Q))
    sel = [(i, it) for i, it in enumerate(items) if it["type"] in a.types.split(",")]
    if a.limit:
        sel = sel[:a.limit]
    out = defaultdict(list)  # level -> per question: gold share matrix [layers, heads], per-fact masses
    for lv in a.levels.split(","):
        for i, it in sel:
            facts = fact_list(it, i, lv)
            gold = [j for j, f in enumerate(facts) if f in it["oracle_facts"]]
            ids, spans = prompt_with_spans(tok, facts, it["q"])
            CAPTURE["rows"] = []; CAPTURE["on"] = True
            mx.eval(model(mx.array(ids)[None]))
            CAPTURE["on"] = False
            A = np.array(mx.stack(CAPTURE["rows"]))  # [layers, heads, L]
            per_fact = np.stack([A[:, :, sp].sum(-1) for sp in spans], -1)  # [layers, heads, n_facts]
            out[lv].append(dict(type=it["type"], q=it["q"], gold=gold, per_fact=per_fact))
        print(f"level {lv}: {len(out[lv])} prompts", flush=True)
    # analysis: pick retrieval heads on level 10 (selection set), evaluate them on every level
    def gold_share(r):
        pf = r["per_fact"]; tot = pf.sum(-1) + 1e-9
        return pf[:, :, r["gold"]].sum(-1) / tot  # [layers, heads]: share of fact-attention on the needed facts
    sel_lv = "10" if "10" in out else list(out)[0]
    mean_share = np.mean([gold_share(r) for r in out[sel_lv]], 0)
    top = np.dstack(np.unravel_index(np.argsort(-mean_share, axis=None)[:5], mean_share.shape))[0]
    lines = [f"Retrieval-head candidates (highest mean share of fact-attention on the needed facts at level {sel_lv}): "
             + ", ".join(f"L{l}H{h} {mean_share[l, h]:.2f}" for l, h in top), "",
             "| distractors | facts | needed share, base rate | best head (fixed): share of fact-attention on needed facts | "
             "same head: needed facts in its top-k (recall) | same head: attention per needed fact | "
             "all 600 heads: max share (mean over questions) |",
             "|---|---|---|---|---|---|---|"]
    l0, h0 = top[0]
    for lv, rows in out.items():
        nf = np.mean([r["per_fact"].shape[-1] for r in rows])
        base = np.mean([len(r["gold"]) / r["per_fact"].shape[-1] for r in rows])
        shares = [gold_share(r)[l0, h0] for r in rows]
        rec, per = [], []
        for r in rows:
            v = r["per_fact"][l0, h0]; k = len(r["gold"])
            rec.append(len(set(np.argsort(-v)[:k]) & set(r["gold"])) / k)
            per.append(v[r["gold"]].mean())
        best_any = np.mean([gold_share(r).max() for r in rows])
        lines.append(f"| {lv} | {nf:.0f} | {base:.2f} | {np.mean(shares):.2f} | {np.mean(rec):.2f} | {np.mean(per):.3f} | {best_any:.2f} |")
    # per-layer profile at the largest level: mean over heads of the best-head share per layer
    last = list(out)[-1]
    prof = np.mean([gold_share(r).max(-1) for r in out[last]], 0)
    lines += ["", f"Per layer, level {last}: best head share on needed facts: "
              + " ".join(f"{l}:{p:.2f}" for l, p in enumerate(prof))]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(a.out, f"attn_{a.types.replace(',', '_')}.md"), "w").write(text + "\n")


def hidden_last(model, ids):
    m = model.model
    h = m.embed_tokens(mx.array(ids)[None])
    mask = create_attention_mask(h, None)
    states = []
    for layer in m.layers:
        h = layer(h, mask, cache=None)
        states.append(h[0, -1])
    return states


def cmd_lens(a, model, tok):
    items = json.load(open(Q))
    facts = sorted({f for it in items for f in it["all_facts"]})
    rng = random.Random("lens")
    city = [f for f in facts if " lives in " in f]
    other = [f for f in facts if " lives in " not in f]
    sample = rng.sample(city, a.n) + rng.sample(other, a.n // 2)
    wordings = {
        "conditional": "Fact: {f}\nIf this fact says which city someone lives in, reply with only the name of that city. "
                       "Otherwise reply with only: none",
        "direct": "Fact: {f}\nWhich city does this fact say someone lives in? If it does not mention where someone lives, "
                  "answer none.\nAnswer with one word."}
    def first_ids(word):  # the answer may start with or without a space, lower or capitalised
        return sorted({tok.encode(v, add_special_tokens=False).ids[0] for v in (word, " " + word, word.capitalize(), " " + word.capitalize())})
    none_ids = first_ids("none")
    W = model.model.embed_tokens.weight.astype(mx.float32)
    res = defaultdict(lambda: defaultdict(list))
    final = defaultdict(lambda: [0, 0])
    for name, tmpl in wordings.items():
        for f in sample:
            ids = bx_qat.chat_ids(tok, tmpl.format(f=f))
            has_city = " lives in " in f
            targets = first_ids(f.split(" lives in ")[1].rstrip(".")) if has_city else none_ids
            states = hidden_last(model, ids)
            for l, h in enumerate(states):
                logits = (model.model.norm(h[None]).astype(mx.float32) @ W.T)[0]
                ranks = np.argsort(np.argsort(-np.array(logits)))
                rank_t = int(min(ranks[t] for t in targets)); rank_n = int(min(ranks[t] for t in none_ids))
                res[(name, has_city)][l].append((rank_t, rank_n))
            final[(name, has_city)][0] += int(mx.argmax(logits)) in targets; final[(name, has_city)][1] += 1
        print(f"{name} done", flush=True)
    lines = ["Logit lens, last prompt position. Cell: share of facts where the correct answer token is top-1 / median rank "
             "of the correct token / share where 'none' is top-1.", ""]
    L = len(res[("direct", True)])
    show = list(range(0, L, 3)) + [L - 1]
    lines.append("| wording, fact | final answer right | " + " | ".join(f"L{l}" for l in show) + " |")
    lines.append("|---|---|" + "---|" * len(show))
    for (name, has_city), per in res.items():
        cells = []
        for l in show:
            v = per[l]
            top1 = np.mean([rt == 0 for rt, _ in v]); med = int(np.median([rt for rt, _ in v])); none1 = np.mean([rn == 0 for _, rn in v])
            cells.append(f"{top1:.2f} / {med} / {none1:.2f}")
        k, n = final[(name, has_city)]
        lines.append(f"| {name}, {'city fact' if has_city else 'other fact'} | {k}/{n} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(a.out, "lens.md"), "w").write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["attn", "lens", "sharpen"])
    ap.add_argument("--taus", default="1.5,2,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="3,10,30,all")  # sharpen: pass --levels 30,all
    ap.add_argument("--types", default="count")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, tok = bx_qat.load()
    {"attn": cmd_attn, "lens": cmd_lens, "sharpen": cmd_sharpen}[a.cmd](a, model, tok)


if __name__ == "__main__":
    main()
