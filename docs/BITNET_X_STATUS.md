# BitNet-X: status of the Apple Silicon (arm64) port

Branch `bx/integration` = upstream Microsoft BitNet (`d989e45`) + the test harness + patches 0001-0009
(each also open as a separate draft PR). Nothing here modifies model weights or the pinned
`3rdparty/llama.cpp` commit (`390c307`); fixes to it are patch files in `patches/llama.cpp/`.

## What works (verified on an M1 Pro, macOS, clean build)
`bash scripts/run_tests.sh` -> `PASS=16 FAIL=0 XFAIL=1` (the XFAIL is `src/ggml-bitnet-mad.cpp`, which is not part of the build).
The official `microsoft/bitnet-b1.58-2B-4T-gguf` (`ggml-model-i2_s.gguf`, unchanged) runs on the CPU I2_S kernels:
`The capital of France is` -> ` Paris. Paris is a city that is known for its rich history, culture,`.

## Defects fixed (see the PRs for file:line, math and before/after tests)
build link on macOS (0001), I2_S dot kernel layout + NEON (0002), I2_S GEMM fallback (0003), ReLU^2 for bitnet-b1.58 (0004),
tokenizer pre-type (0005), Metal abort on I2_S (0006), BLAS crash on I2_S (0007), quantize_i2_s layout (0008),
`src1_cont` without llamafile (0009), `setup_env.py` (separate commit).

## Not done / not verified
* There is no Metal kernel for I2_S: matmuls run on the CPU (Metal only stops aborting). Speed was not measured.
* TL1 kernels were not built or run. No quality benchmarks (math / code) yet.
* `quantize_i2_s` end to end on an F32 model was not run (unit test only).

## Reproduce
```bash
git clone --recursive <this repo> && cd BitNet-X && git checkout bx/integration
BITNET_GGUF=/path/ggml-model-i2_s.gguf BITNET_BF16=/path/BitNet-bf16-dir bash scripts/run_tests.sh
```
Details: `tests/README.md`, `patches/README.md`. Experiment plan: `docs/experiments/PLAN.md` (PR branch `bx/experiment-plan`).
