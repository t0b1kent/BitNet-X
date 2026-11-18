#!/usr/bin/env python3
"""Behaviour tests for setup_env.py (no network, no build, nothing is executed:
run_command / shutil.copyfile are replaced by recorders where needed)."""
import argparse, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import setup_env

fail = 0
def check(name, ok, detail=""):
    global fail
    print(("ok   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    fail |= (not ok)

def make_args(**kw):
    d = dict(hf_repo=None, model_dir="models/BitNet-b1.58-2B-4T", log_dir=tempfile.mkdtemp(prefix="bitnetx-setup-"),
             quant_type="i2_s", quant_embd=False, use_pretuned=False)
    d.update(kw)
    return argparse.Namespace(**d)

# ---- run_command -----------------------------------------------------------
setup_env.args = make_args()
try:
    setup_env.run_command([sys.executable, "-c", "pass"])
    check("run_command(ok command, no log_step) returns", True)
except SystemExit as e:
    check("run_command(ok command, no log_step) returns", False, f"SystemExit({e.code})")

try:
    setup_env.run_command([sys.executable, "-c", "import sys; sys.exit(3)"])
    check("run_command(failing command, no log_step) exits 1", False, "returned")
except SystemExit as e:
    check("run_command(failing command, no log_step) exits 1", e.code == 1, f"SystemExit({e.code})")

try:
    setup_env.run_command([sys.executable, "-c", "pass"], log_step="t_ok")
    check("run_command(ok command, log_step) returns", True)
except SystemExit as e:
    check("run_command(ok command, log_step) returns", False, f"SystemExit({e.code})")

# ---- compile(): TL flag must follow --quant-type -----------------------------
def cmake_configure_cmd(arch, quant_type):
    calls = []
    orig_rc, orig_si = setup_env.run_command, setup_env.system_info
    setup_env.run_command = lambda command, shell=False, log_step=None: calls.append(command)
    setup_env.system_info = lambda: ("Linux", arch)
    setup_env.args = make_args(quant_type=quant_type)
    try:
        setup_env.compile()
    finally:
        setup_env.run_command, setup_env.system_info = orig_rc, orig_si
    return next(c for c in calls if "-B" in c)

for arch, qt, want in [
    ("arm64",  "i2_s", "-DBITNET_ARM_TL1=OFF"),
    ("arm64",  "tl1",  "-DBITNET_ARM_TL1=ON"),
    ("x86_64", "i2_s", "-DBITNET_X86_TL2=OFF"),
    ("x86_64", "tl2",  "-DBITNET_X86_TL2=ON"),
]:
    cmd = cmake_configure_cmd(arch, qt)
    flags = [a for a in cmd if a.startswith("-DBITNET_")]
    check(f"compile() {arch} --quant-type {qt}: {want}", flags == [want], " ".join(flags))

# ---- gen_code(): i2_s needs no LUT codegen; pretuned kernels must survive ----
def gen_code_actions(arch, quant_type, use_pretuned, model="bitnet_b1_58-3B"):
    actions = []
    orig = (setup_env.run_command, setup_env.system_info, setup_env.shutil.copyfile)
    setup_env.run_command = lambda command, shell=False, log_step=None: actions.append(("run", os.path.basename(command[1])))
    setup_env.system_info = lambda: ("Linux", arch)
    setup_env.shutil.copyfile = lambda src, dst: actions.append(("copy", os.path.basename(src), dst))
    setup_env.args = make_args(quant_type=quant_type, use_pretuned=use_pretuned, model_dir="models/" + model)
    cwd = os.getcwd(); os.chdir(ROOT)          # preset_kernels/ is looked up relative to the repo root
    try:
        setup_env.gen_code()
    finally:
        os.chdir(cwd)
        setup_env.run_command, setup_env.system_info, setup_env.shutil.copyfile = orig
    return actions

a = gen_code_actions("arm64", "i2_s", False, model="BitNet-b1.58-2B-4T")
check("gen_code() arm64 i2_s: tracked include/bitnet-lut-kernels.h is not regenerated", a == [], str(a))

a = gen_code_actions("arm64", "tl1", True)
check("gen_code() arm64 tl1 --use-pretuned: preset copied and not overwritten by codegen",
      [x[0] for x in a] == ["copy", "copy"] and a[0][1] == "bitnet-lut-kernels-tl1.h", str(a))

a = gen_code_actions("arm64", "tl1", False)
check("gen_code() arm64 tl1: codegen_tl1.py runs", a == [("run", "codegen_tl1.py")], str(a))

a = gen_code_actions("x86_64", "tl2", True)
check("gen_code() x86_64 tl2 --use-pretuned: preset copied and not overwritten by codegen",
      len(a) >= 1 and all(x[0] == "copy" for x in a) and a[0][1] == "bitnet-lut-kernels-tl2.h", str(a))

a = gen_code_actions("x86_64", "tl2", False)
check("gen_code() x86_64 tl2: codegen_tl2.py runs", a == [("run", "codegen_tl2.py")], str(a))

sys.exit(1 if fail else 0)
