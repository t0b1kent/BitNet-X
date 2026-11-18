#!/usr/bin/env python3
"""Sandboxed grading of model-written Python code (macOS sandbox-exec: no network, writes only to a temp dir).

    python grade.py --selftest
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "runner.py")
CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def extract_code(text):
    m = CODE_RE.search(text)
    if m:
        return m.group(1)
    if re.search(r"^\s*def\s+\w+\(", text, re.M):
        return text
    return None


def _profile(tmp):
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            f"(allow file-write* (subpath \"{tmp}\") (literal \"/dev/null\") (literal \"/dev/tty\"))")


def run_tests(code, entry, tests, timeout=40):
    """-> dict(status, passed, total, n_wrong, n_error, n_timeout, first_fail, error)"""
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="bxgrade-"))
    try:
        with open(os.path.join(tmp, "cand.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmp, "tests.json"), "w") as f:
            json.dump(tests, f)
        cmd = ["/usr/bin/sandbox-exec", "-p", _profile(tmp), sys.executable, "-I", "-B", RUNNER, tmp, entry]
        try:
            subprocess.run(cmd, cwd=tmp, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, start_new_session=True)
        except subprocess.TimeoutExpired:
            return dict(status="hard_timeout", passed=0, total=len(tests), n_wrong=0, n_error=0, n_timeout=len(tests),
                        first_fail=None, error="")
        rp = os.path.join(tmp, "result.json")
        if not os.path.exists(rp):
            return dict(status="runner_crash", passed=0, total=len(tests), n_wrong=0, n_error=0, n_timeout=0,
                        first_fail=None, error="no result.json")
        return json.load(open(rp))
    finally:
        subprocess.run(["/bin/rm", "-rf", tmp], check=False)


def category(res):
    """pass / no_code / syntax_error / import_error / no_entry / timeout / runtime_error / wrong_answer"""
    if res is None:
        return "no_code"
    if res["status"] in ("syntax_error", "import_error", "no_entry"):
        return res["status"]
    if res["status"] in ("hard_timeout", "runner_crash"):
        return "timeout" if res["status"] == "hard_timeout" else "runtime_error"
    if res["passed"] == res["total"]:
        return "pass"
    failed = res["total"] - res["passed"]
    if res["n_timeout"] > 0 and res["n_timeout"] >= failed / 2:
        return "timeout"
    if res["n_error"] > 0 and res["n_error"] >= res["n_wrong"]:
        return "runtime_error"
    return "wrong_answer"


def grade_text(text, task, which="private"):
    """-> (category, result dict or None)"""
    code = extract_code(text)
    if code is None:
        return "no_code", None
    res = run_tests(code, task["name"], task[which])
    return category(res), res


def selftest():
    sys.path.insert(0, os.path.join(HERE, "..", "tasks"))
    tasks = [json.loads(l) for l in open(os.path.join(HERE, "..", "tasks", "dev_100.jsonl"))]
    ok = True
    # 1. every reference solution passes its private tests
    bad = [t["id"] for t in tasks if category(run_tests(t["ref"], t["name"], t["private"])) != "pass"]
    print(f"reference solutions passing private tests: {len(tasks) - len(bad)}/{len(tasks)}", "" if not bad else bad[:5])
    ok &= not bad
    t = tasks[0]
    cases = {
        "wrong answer": (f"def {t['name']}(*a):\n    return 123456789\n", "wrong_answer"),
        "runtime error": (f"def {t['name']}(*a):\n    return 1 // 0\n", "runtime_error"),
        "infinite loop": (f"def {t['name']}(*a):\n    while True:\n        pass\n", "timeout"),
        "syntax error": ("def f(:\n", "syntax_error"),
        "wrong name": ("def other(x):\n    return 0\n", "no_entry"),
    }
    for label, (code, want) in cases.items():
        got = category(run_tests(code, t["name"], t["private"]))
        print(f"{label:14s} -> {got:14s} {'ok' if got == want else 'EXPECTED ' + want}")
        ok &= got == want
    # 2. the sandbox really blocks network and writes outside the temp dir
    import socket
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    probe = os.path.join(tempfile.gettempdir(), "bx_sandbox_probe.txt")
    evil = (f"def {t['name']}(*a):\n"
            f"    import socket\n"
            f"    try:\n"
            f"        socket.create_connection(('127.0.0.1', {port}), timeout=2).close()\n"
            f"        net = 'NETOPEN'\n"
            f"    except Exception:\n"
            f"        net = 'BLOCKED'\n"
            f"    try:\n"
            f"        open({probe!r}, 'w').write('x')\n"
            f"        w = 'WROTE'\n"
            f"    except Exception:\n"
            f"        w = 'DENIED'\n"
            f"    return net + '/' + w\n")
    r = run_tests(evil, t["name"], [t["private"][0]])
    server.close()
    got = (r.get("first_fail") or {}).get("got")
    leaked = os.path.exists(probe)
    print(f"sandbox probe (local socket / write outside temp dir): {got}; file exists: {leaked}")
    ok &= got == "'BLOCKED/DENIED'" and not leaked
    if leaked:
        os.remove(probe)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
