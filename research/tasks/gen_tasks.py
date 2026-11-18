#!/usr/bin/env python3
"""Seeded generator of small Python coding tasks with public examples and hidden tests (BitNet-X research).

No network, no external datasets: every task is built from a seed, so it cannot be in any training set.
Each task has 3 public examples (shown in the prompt, used for execution feedback) and 25 private tests (grading only).

    gen_tasks.py --seed 1 --per-family 10 --out dev_100.jsonl
"""
import argparse
import hashlib
import json
import random

N_PUBLIC = 3
N_PRIVATE = 25
VOWELS = "aeiou"


# ---------------------------------------------------------------------------------------------------------------------
# families: each returns dict(name, params, spec, ref, gen, hint, skills); ref is Python source of the reference solution
# ---------------------------------------------------------------------------------------------------------------------

def fam_digit_arith(rng):
    a, b, m = rng.randint(2, 9), rng.randint(3, 12), rng.choice([97, 101, 211, 1009])
    name = "score_number"
    spec = (f"Given a non-negative integer n, let s be the sum of the decimal digits of n*n. "
            f"Return (s*{a} + {b}) % {m}.")
    ref = (f"def {name}(n):\n"
           f"    s = sum(int(c) for c in str(n * n))\n"
           f"    return (s * {a} + {b}) % {m}\n")
    gen = lambda r: [r.choice([0, 1, 7, 10, 99, r.randint(0, 10**6), r.randint(0, 10**3)])]
    hint = "Square n, turn the square into a string, add up its digits, then apply the formula."
    return dict(name=name, params=["n"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["arithmetic"])


def fam_filter_map(rng):
    p = rng.randint(2, 5)
    r_ = rng.randint(0, p - 1)
    k, l, t = rng.randint(2, 5), rng.randint(1, 9), rng.randint(2, 4)
    name = "filter_map_sum"
    spec = (f"Given a list of integers xs: keep the elements x with x % {p} == {r_} (Python modulo), "
            f"map each kept x to x*{k} - {l}, and return the sum of the first {t} mapped values in the original order "
            f"(all of them if there are fewer than {t}).")
    ref = (f"def {name}(xs):\n"
           f"    ys = [x * {k} - {l} for x in xs if x % {p} == {r_}]\n"
           f"    return sum(ys[:{t}])\n")
    gen = lambda r: [[r.randint(-30, 30) for _ in range(r.choice([0, 1, 2, 5, 8, 14]))]]
    hint = "Filter first, then map, then slice the first values and sum them."
    return dict(name=name, params=["xs"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["composition"])


def fam_string_rule(rng):
    p = rng.randint(2, 4)
    sep = rng.choice(["-", "_", "|", "+"])
    name = "process_words"
    spec = (f"Given a string text of lowercase words separated by single spaces (possibly empty), process each word: "
            f"if its length is a multiple of {p}, reverse it; otherwise, if it starts with a vowel (a, e, i, o, u), "
            f"convert it to upper case; otherwise leave it unchanged. Return the processed words joined with '{sep}'.")
    ref = (f"def {name}(text):\n"
           f"    out = []\n"
           f"    for w in (text.split(' ') if text else []):\n"
           f"        if len(w) % {p} == 0:\n"
           f"            out.append(w[::-1])\n"
           f"        elif w[0] in 'aeiou':\n"
           f"            out.append(w.upper())\n"
           f"        else:\n"
           f"            out.append(w)\n"
           f"    return '{sep}'.join(out)\n")

    def gen(r):
        words = []
        for _ in range(r.choice([0, 1, 3, 5, 7])):
            n = r.randint(1, 7)
            w = "".join(r.choice("abcdefghilmnoprstu") for _ in range(n))
            words.append(w)
        return [" ".join(words)]
    hint = "Loop over the words and apply the first rule that matches: length rule, then vowel rule, else keep."
    return dict(name=name, params=["text"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["instruction-following"])


def fam_longest_run(rng):
    k = rng.randint(-2, 3)
    name = "longest_run"
    spec = (f"Given a list of integers xs, remove every element smaller than {k}, then return the length of the "
            f"longest run of equal adjacent elements in the remaining list (0 if it is empty).")
    ref = (f"def {name}(xs):\n"
           f"    ys = [x for x in xs if x >= {k}]\n"
           f"    best = 0\n"
           f"    cur = 0\n"
           f"    prev = None\n"
           f"    for y in ys:\n"
           f"        if cur > 0 and y == prev:\n"
           f"            cur += 1\n"
           f"        else:\n"
           f"            cur = 1\n"
           f"        prev = y\n"
           f"        best = max(best, cur)\n"
           f"    return best\n")
    gen = lambda r: [[r.randint(-2, 4) for _ in range(r.choice([0, 1, 3, 8, 12, 16]))]]
    hint = "Filter the list, then scan once keeping the current run length and the best run length so far."
    return dict(name=name, params=["xs"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["scanning"])


def fam_prefix_diff(rng):
    mod = rng.choice([2, 3])
    name = "spread_sum"
    spec = (f"Given a non-empty list of integers a, for each index i let d[i] = max(a[0..i]) - min(a[i..end]) "
            f"(both ranges inclusive). Return the sum of those d[i] that are divisible by {mod}.")
    ref = (f"def {name}(a):\n"
           f"    total = 0\n"
           f"    for i in range(len(a)):\n"
           f"        d = max(a[:i + 1]) - min(a[i:])\n"
           f"        if d % {mod} == 0:\n"
           f"            total += d\n"
           f"    return total\n")
    gen = lambda r: [[r.randint(-9, 9) for _ in range(r.choice([1, 2, 4, 7, 12]))]]
    hint = "For each index compute the maximum of the prefix and the minimum of the suffix, subtract, and add if divisible."
    return dict(name=name, params=["a"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["prefix-suffix"])


def fam_subarray_sum(rng):
    k = rng.randint(0, 6)
    name = "count_subarrays"
    spec = f"Given a list of integers xs, return the number of non-empty contiguous subarrays whose sum equals {k}."
    ref = (f"def {name}(xs):\n"
           f"    n = len(xs)\n"
           f"    count = 0\n"
           f"    for i in range(n):\n"
           f"        s = 0\n"
           f"        for j in range(i, n):\n"
           f"            s += xs[j]\n"
           f"            if s == {k}:\n"
           f"                count += 1\n"
           f"    return count\n")
    gen = lambda r: [[r.randint(-3, 5) for _ in range(r.choice([0, 1, 3, 6, 9, 12]))]]
    hint = "Try every start index and extend the end index, keeping a running sum, and count the matches."
    return dict(name=name, params=["xs"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["classic-algorithm"])


def fam_stack_machine(rng):
    c, d = rng.randint(2, 5), rng.randint(1, 4)
    name = "run_machine"
    spec = ("A stack machine starts with an empty stack and executes a list of instruction strings. "
            "'PUSH v' pushes the integer v. 'ADD' pops the top two values and pushes their sum. "
            "'SUBR' pops a (the top) and then b, and pushes a - b. 'DUP' pushes a copy of the top value. "
            f"'SWAP' exchanges the top two values. 'TWIST' pops a and pushes a*{c} + {d}. "
            "Return the final top of the stack, or -1 if the stack is empty at the end. "
            "All programs given are valid (no instruction needs more values than the stack holds).")
    ref = (f"def {name}(program):\n"
           f"    st = []\n"
           f"    for ins in program:\n"
           f"        parts = ins.split()\n"
           f"        op = parts[0]\n"
           f"        if op == 'PUSH':\n"
           f"            st.append(int(parts[1]))\n"
           f"        elif op == 'ADD':\n"
           f"            a = st.pop(); b = st.pop(); st.append(a + b)\n"
           f"        elif op == 'SUBR':\n"
           f"            a = st.pop(); b = st.pop(); st.append(a - b)\n"
           f"        elif op == 'DUP':\n"
           f"            st.append(st[-1])\n"
           f"        elif op == 'SWAP':\n"
           f"            st[-1], st[-2] = st[-2], st[-1]\n"
           f"        elif op == 'TWIST':\n"
           f"            a = st.pop(); st.append(a * {c} + {d})\n"
           f"    return st[-1] if st else -1\n")

    def gen(r):
        prog, depth = [], 0
        for _ in range(r.randint(0, 11)):
            options = ["PUSH"]
            if depth >= 1:
                options += ["DUP", "TWIST"]
            if depth >= 2:
                options += ["ADD", "SUBR", "SWAP"]
            op = r.choice(options)
            if op == "PUSH":
                prog.append(f"PUSH {r.randint(-9, 9)}"); depth += 1
            elif op == "DUP":
                prog.append("DUP"); depth += 1
            elif op in ("ADD", "SUBR"):
                prog.append(op); depth -= 1
            else:
                prog.append(op)
        return [prog]
    hint = "Keep a Python list as the stack and handle each instruction in turn; pop in the exact order the spec gives."
    return dict(name=name, params=["program"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["simulation", "long-dependency"])


def fam_grid_paths(rng):
    dv = rng.randint(2, 5)
    name = "count_paths"
    spec = ("Given a rectangular grid (list of lists of non-negative integers, at least 1x1), count the paths from the "
            "top-left cell to the bottom-right cell that move only right or down and whose sum of visited cell values "
            f"(both end cells included) is divisible by {dv}.")
    ref = (f"def {name}(grid):\n"
           f"    R, C = len(grid), len(grid[0])\n"
           f"    dp = [[None] * C for _ in range(R)]\n"
           f"    for i in range(R):\n"
           f"        for j in range(C):\n"
           f"            cur = {{}}\n"
           f"            if i == 0 and j == 0:\n"
           f"                cur[grid[0][0] % {dv}] = 1\n"
           f"            else:\n"
           f"                for pi, pj in ((i - 1, j), (i, j - 1)):\n"
           f"                    if pi >= 0 and pj >= 0:\n"
           f"                        for rem, cnt in dp[pi][pj].items():\n"
           f"                            nr = (rem + grid[i][j]) % {dv}\n"
           f"                            cur[nr] = cur.get(nr, 0) + cnt\n"
           f"            dp[i][j] = cur\n"
           f"    return dp[R - 1][C - 1].get(0, 0)\n")

    def gen(r):
        R, C = r.randint(1, 4), r.randint(1, 4)
        return [[[r.randint(0, 5) for _ in range(C)] for _ in range(R)]]
    hint = "Dynamic programming: for every cell keep a count of paths per remainder of the sum modulo the divisor."
    return dict(name=name, params=["grid"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["dynamic-programming", "planning"])


def fam_custom_base(rng):
    b = rng.randint(3, 9)
    letters = list("abcdefghij")
    rng.shuffle(letters)
    sym = "".join(letters[:b])
    name = "to_custom_base"
    spec = (f"Convert a non-negative integer n to base {b}. The digit values 0..{b - 1} are written with the symbols "
            f"'{sym}' respectively (the first symbol is the digit 0). Return the string with no leading zeros "
            f"('{sym[0]}' for n = 0).")
    ref = (f"def {name}(n):\n"
           f"    sym = '{sym}'\n"
           f"    if n == 0:\n"
           f"        return sym[0]\n"
           f"    out = ''\n"
           f"    while n > 0:\n"
           f"        out = sym[n % {b}] + out\n"
           f"        n //= {b}\n"
           f"    return out\n")
    gen = lambda r: [r.choice([0, 1, b - 1, b, r.randint(0, 60), r.randint(0, 5000)])]
    hint = "Repeatedly take n modulo the base to get the last digit, map it to its symbol, and divide n by the base."
    return dict(name=name, params=["n"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["arithmetic", "algorithm"])


def fam_bfs_order(rng):
    desc = rng.random() < 0.5
    order = "descending" if desc else "ascending"
    name = "bfs_order"
    spec = ("Given n (the number of nodes, labeled 0..n-1) and a list of undirected edges [u, v], return the list of "
            "nodes in the order visited by a breadth-first search from node 0, where the neighbors of each node are "
            f"visited in {order} order of node label. Nodes not reachable from node 0 are omitted.")
    ref = (f"def {name}(n, edges):\n"
           f"    adj = [set() for _ in range(n)]\n"
           f"    for u, v in edges:\n"
           f"        adj[u].add(v); adj[v].add(u)\n"
           f"    seen = [False] * n\n"
           f"    seen[0] = True\n"
           f"    order = []\n"
           f"    queue = [0]\n"
           f"    while queue:\n"
           f"        x = queue.pop(0)\n"
           f"        order.append(x)\n"
           f"        for y in sorted(adj[x], reverse={desc}):\n"
           f"            if not seen[y]:\n"
           f"                seen[y] = True\n"
           f"                queue.append(y)\n"
           f"    return order\n")

    def gen(r):
        n = r.randint(1, 8)
        edges = []
        if n >= 2:
            for _ in range(r.randint(0, 10)):
                u, v = r.randrange(n), r.randrange(n)
                if u != v:
                    edges.append([u, v])
        return [n, edges]
    hint = "Standard BFS with a queue and a visited set; sort each node's neighbors in the required order before enqueueing."
    return dict(name=name, params=["n", "edges"], spec=spec, ref=ref, gen=gen, hint=hint, skills=["classic-algorithm", "graph"])


FAMILIES = [fam_digit_arith, fam_filter_map, fam_string_rule, fam_longest_run, fam_prefix_diff,
            fam_subarray_sum, fam_stack_machine, fam_grid_paths, fam_custom_base, fam_bfs_order]


# ---------------------------------------------------------------------------------------------------------------------

def fmt_call(name, args):
    return f"{name}({', '.join(repr(a) for a in args)})"


def build_task(fam_fn, seed, idx):
    fam = fam_fn.__name__[4:]
    for attempt in range(40):
        rng = random.Random(f"{seed}/{fam}/{idx}/{attempt}")
        f = fam_fn(rng)
        ns = {}
        exec(f["ref"], ns)
        fn = ns[f["name"]]
        seen, tests = set(), []
        guard = 0
        while len(tests) < N_PUBLIC + N_PRIVATE and guard < 2000:
            guard += 1
            args = f["gen"](rng)
            key = json.dumps(args, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            tests.append([json.loads(key), json.loads(json.dumps(fn(*json.loads(key))))])
        if len(tests) < N_PUBLIC + N_PRIVATE:
            continue
        distinct = len({json.dumps(t[1]) for t in tests})
        if distinct < 5:
            continue
        public, private = tests[:N_PUBLIC], tests[N_PUBLIC:]
        examples = "\n".join(f"{fmt_call(f['name'], a)} == {e!r}" for a, e in public)
        prompt = (f"Write a Python function `{f['name']}({', '.join(f['params'])})`.\n\n{f['spec']}\n\n"
                  f"Examples:\n{examples}")
        return dict(id=f"{fam}-{idx:02d}", family=fam, skills=f["skills"], name=f["name"], params=f["params"],
                    prompt=prompt, hint=f["hint"], public=public, private=private, ref=f["ref"])
    raise RuntimeError(f"could not build task {fam}-{idx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [build_task(f, a.seed, i) for f in FAMILIES for i in range(a.per_family)]
    with open(a.out, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"{len(tasks)} tasks -> {a.out}  sha256={digest}")


if __name__ == "__main__":
    main()
