# BitNet-X tests

Correctness tests for the BitNet b1.58 inference path. Everything runs on a
plain checkout: unit tests compile the **real engine sources** directly
(`3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c`, `ggml-cpu-i2s.c`,
`src/ggml-bitnet-mad.cpp`) without a full build; integration tests use a build
in `/private/tmp` and the official model files, which are only read.

## Quick start

```bash
git submodule update --init --recursive

# unit tests only (about 10 s)
scripts/run_tests.sh --no-build

# everything: unit tests, build smoke test (several minutes), integration tests
export BITNET_GGUF=/path/to/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
export BITNET_BF16=/path/to/bitnet-b1.58-2B-4T-bf16        # dir with model.safetensors
scripts/run_tests.sh

# details of every test, or selected tests
scripts/run_tests.sh -v
scripts/run_tests.sh --no-build -v i2s_roundtrip i2s_dot
```

Requirements: clang, cmake, python3 with numpy. Nothing is installed by the
tests. Tests that need files or binaries that are not available are reported
as `SKIP`.

Output: one line per test — `PASS`, `FAIL`, `SKIP`, `XFAIL` (listed in
`tests/KNOWN_FAILURES` and failing: an open, documented defect) or `XPASS`
(listed but passing: the list is stale). The exit status is non-zero if there
is a `FAIL` or an `XPASS`.

## Tests

| name | what it checks | needs |
|---|---|---|
| `patches_idempotent` | `scripts/apply_patches.sh` twice, then `--check` | – |
| `py_i2s_reference` | NumPy reference of the I2_S format (`tests/py/i2s_ref.py`) | numpy |
| `gguf_layout_real` | official GGUF trits decoded with 128/32, 64/16 and sequential layouts vs ternarised BF16 master | `BITNET_GGUF`, `BITNET_BF16` |
| `i2s_roundtrip` | `quantize_i2_s` emits canonical bytes; `dequantize_row_i2_s` inverts it | – |
| `i2s_dot` | `ggml_vec_dot_i2_i8_s` and `ggml_gemv_i2_i8_s` vs integer reference on canonical input | – |
| `i2s_dot_nodotprod` | same, compiled with `-march=armv8-a+nodotprod` | arm64 |
| `i2s_gemm` | `ggml_gemm_i2_i8_s` (4 activation columns) vs integer reference | – |
| `i2s_gemm_nodotprod` | same, compiled with `-march=armv8-a+nodotprod` | arm64 |
| `mad_layout` | `src/ggml-bitnet-mad.cpp` `quantize_i2_s` emits canonical bytes | – |
| `src1_cont` | `ggml-cpu.c` compiles with and without `GGML_USE_LLAMAFILE` | – |
| `setup_env` | `setup_env.py` behaviour (`run_command`, cmake flags, `gen_code`) with recorders | – |
| `build_smoke` | configure with the exact `setup_env.py` flags and build the tools in `/private/tmp/bitnetx-build-smoke` | cmake |
| `it_cpu_generation` | greedy CPU (`-ngl 0`) continuation is not degenerate (`@@@@`) | build, `BITNET_GGUF` |
| `it_relu2` | continuation of "The capital of France is" starts with " Paris"; PPL on `tests/data/ppl_small.txt` < 20 | build, `BITNET_GGUF` |
| `it_tokenizer` | `llama-tokenize` ids == HF `tokenizer.json` ids (`tests/data/tokenizer_expected.json`) | build, `BITNET_GGUF` |
| `it_blas_batch` | a 32+ token prompt on CPU does not crash | build, `BITNET_GGUF` |
| `it_metal_default` | `-ngl 99` does not abort | macOS, build, `BITNET_GGUF` |

Integration tests take their binaries from `BITNETX_BIN`; if it is not set and
`build_smoke` passed in the same run, its `bin/` directory is used.

## The canonical I2_S layout ("128/32")

`n` ternary weights → `n/4` bytes + 32-byte tail (float32 scale first).
Weights are grouped in blocks of 128 over the flattened tensor; block `b` is
bytes `[32b, 32b+32)`. Byte `gp` of a block holds the 2-bit codes of weights
`{gp, 32+gp, 64+gp, 96+gp}` in bits `[7:6] [5:4] [3:2] [1:0]`; codes
`0/1/2` = `-1/0/+1`. Reference implementations: `tests/py/i2s_ref.py`,
`tests/c/i2s_ref.h`. `gguf_layout_real` proves on real weights that this — and
not 64/16 or "4 consecutive weights per byte" — is what the official GGUF
contains (98.78 % agreement with the ternarised BF16 master for
`blk.0.attn_q.weight`; the remaining 1.22 % is a property of the published
checkpoints, see microsoft/BitNet#632, and is not an engine defect).

The CPU kernels return `sum_i code_i * q_i` (codes, not signed weights);
`ggml-cpu.c` then computes `(dot - sum(q)) * weight_scale / act_scale`.

## Patches to the submodule

See `patches/README.md`. The submodule pin never changes; `cmake` and
`tests/build_smoke.sh` run `scripts/apply_patches.sh`, which is idempotent.
`scripts/apply_patches.sh --reverse` restores the pristine submodule.

## Regenerating test data

`tests/data/tokenizer_expected.json` comes from the HF tokenizer:

```bash
python3 -m venv /path/outside/the/repo/venv && /path/outside/the/repo/venv/bin/pip install tokenizers
BITNET_BF16=/path/to/bitnet-b1.58-2B-4T-bf16 /path/outside/the/repo/venv/bin/python tests/py/gen_tokenizer_expected.py
```

`tests/data/ppl_small.txt` is a small original text (about 1 000 tokens)
written for this repository; perplexities on it are only comparable with each
other, not with WikiText numbers.

## Baseline

`tests/BASELINE.md` records the state of `main` (d989e45) on an M1 Pro.
