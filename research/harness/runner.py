#!/usr/bin/env python3
"""Runs one candidate solution against a list of tests. Executed inside grade.py's sandbox, never imported.

argv: <dir with cand.py and tests.json> <entry function name>; writes <dir>/result.json
"""
import json
import os
import signal
import sys


def main():
    d, entry = sys.argv[1], sys.argv[2]
    res = {"status": "ok", "passed": 0, "total": 0, "n_wrong": 0, "n_error": 0, "n_timeout": 0, "first_fail": None, "error": ""}

    def out():
        with open(os.path.join(d, "result.json"), "w") as f:
            json.dump(res, f)

    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    except Exception:
        pass

    src = open(os.path.join(d, "cand.py")).read()
    tests = json.load(open(os.path.join(d, "tests.json")))
    res["total"] = len(tests)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    ns = {"__name__": "cand"}
    try:
        exec(compile(src, "cand.py", "exec"), ns)
    except SyntaxError as e:
        res.update(status="syntax_error", error=str(e)[:200]); out(); return
    except BaseException as e:
        res.update(status="import_error", error=f"{type(e).__name__}: {e}"[:200]); out(); return
    fn = ns.get(entry)
    if not callable(fn):
        res.update(status="no_entry"); out(); return

    class Timeout(Exception):
        pass

    def on_alarm(signum, frame):
        raise Timeout()

    signal.signal(signal.SIGALRM, on_alarm)
    for pos, (args, expected) in enumerate(tests):
        info = None
        try:
            signal.setitimer(signal.ITIMER_REAL, 2.0)
            got = fn(*args)
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                got_n = json.loads(json.dumps(got))
            except Exception:
                got_n = repr(got)
            if got_n == expected:
                res["passed"] += 1
                continue
            res["n_wrong"] += 1
            info = {"kind": "wrong", "got": repr(got)[:120]}
        except Timeout:
            res["n_timeout"] += 1
            info = {"kind": "timeout"}
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            res["n_error"] += 1
            info = {"kind": "error", "error": f"{type(e).__name__}: {e}"[:120]}
        signal.setitimer(signal.ITIMER_REAL, 0)
        if res["first_fail"] is None:
            res["first_fail"] = dict(info, args=args, expected=expected)
        if res["n_timeout"] >= 2:
            res["n_timeout"] += len(tests) - pos - 1  # a looping solution: do not wait for every remaining test
            break
    out()


if __name__ == "__main__":
    main()
