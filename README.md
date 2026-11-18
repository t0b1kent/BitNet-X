# BitNet-X

**bitnet.cpp running natively on Apple Silicon, plus a research track on making small ternary models smarter.**

BitNet-X is a fork of [microsoft/BitNet](https://github.com/microsoft/BitNet) (bitnet.cpp). It started as a port of
bitnet.cpp to Apple Silicon and grew into two tracks:

* **Engine**: correctness fixes for arm64 and x86, and Metal GPU kernels for the ternary `I2_S` format, so the official
  [BitNet b1.58 2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) model runs on the Mac GPU.
* **Research**: why a 2B ternary model fails where it fails, measured with automatic checking, and what fixes it
  (see [Research](#research) below).

The original upstream README is kept in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

## Engine

All changes to the pinned `3rdparty/llama.cpp` are patch files in [`patches/llama.cpp`](patches/llama.cpp), applied
automatically when `cmake` configures the build. Each patch has a test in [`tests/`](tests) that fails without it.

| patch | what it fixes or adds |
|---|---|
| 0001 | macOS link: `quantize_i2_s` / `dequantize_row_i2_s` moved to ggml-base |
| 0002 | I2_S dot product: canonical 128×32 layout, NEON path |
| 0003 | I2_S GEMM fallback for activation columns |
| 0004 | bitnet-b1.58 FFN uses squared ReLU, not SiLU (upstream issue #588) |
| 0005 | default pre-tokenizer `llama-bpe` when the GGUF has none (upstream issue #684) |
| 0006 | Metal: report I2_S matmul as unsupported instead of aborting (`-ngl > 0` crashed) |
| 0007 | BLAS: same for I2_S (crash on prompts of 32+ tokens) |
| 0008 | `quantize_i2_s` packing layout |
| 0009 | ggml-cpu `src1_cont` without llamafile |
| 0010 | **Metal kernels for I2_S** (int8 mat-vec on the GPU) |
| 0011 | Metal bit-serial popcount kernels (×3 prefill, ×1.3 decode over 0010) |
| 0012 | **Metal matrix prefill** for I2_S (`kernel_mul_mm`, ×3 prefill over 0011; `GGML_METAL_I2S_NO_MM=1` disables it) |

### Speed (`llama-bench -p 512 -n 64 -r 3`, official `ggml-model-i2_s.gguf`)

| platform | build | threads | prefill pp512, tok/s | decode tg64, tok/s |
|---|---|---|---|---|
| M1 Pro, Metal GPU | BitNet-X, patches 0001–0012 | 4 | **710** | **53.0** |
| M1 Pro, Metal GPU | up to 0011 (bit-serial) | 4 | 236 | 52.5 |
| M1 Pro, Metal GPU | up to 0010 (int8) | 4 | 75 | 40.0 |
| M1 Pro, CPU | BitNet-X | 4 | 45 | 27 |
| Ryzen 9 9900X, CPU (Linux) | BitNet-X | 4 / 8 / 12 | 284 / 508 / 668 | 38.7 / 39.6 / 39.3 |

Measured by one operator on one day on machines with background load; re-measure on an idle machine before quoting.
Decode on the Mac reads about 1.18 GB per token (≈63 GB/s, about 31 % of the 200 GB/s peak), so there is headroom.

### Correctness

Perplexity on `tests/data/ppl_small.txt` (`llama-perplexity -c 256`): Metal (all three kernel paths) **6.4292**,
x86 CPU **6.4323**. The unpatched upstream on x86 gives **71.8** (numerically wrong output: "The capital of France
is" → "a small city, and the capital of France is a small city…"); BitNet-X answers "Paris. Paris is a city…".
Which of patches 0001–0009 fixes the x86 result has not been isolated yet.

### Build and run (macOS, Apple Silicon)

```bash
git clone --recursive https://github.com/t0b1kent/BitNet-X && cd BitNet-X
cmake -S . -B build -DBITNET_ARM_TL1=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build -j --target llama-cli llama-server llama-bench llama-perplexity
# model: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf  (ggml-model-i2_s.gguf)
build/bin/llama-cli -m ggml-model-i2_s.gguf -ngl 99 -p "The capital of France is" -n 32
```

Prebuilt macOS arm64 binaries are attached to the [releases](https://github.com/t0b1kent/BitNet-X/releases).

**Chat format.** The chat template embedded in the official GGUF (`Human: … BITNETAssistant:` followed by an extra
end token) does not match the original model; through `/v1/chat/completions` the model loses the format. Use the
original format with `/completion`: `System: …<|eot_id|>User: …<|eot_id|>Assistant: `.

### Tests

```bash
scripts/run_tests.sh --no-build          # unit tests, about 10 s
BITNET_GGUF=/path/ggml-model-i2_s.gguf BITNET_BF16=/path/bf16-dir scripts/run_tests.sh   # everything
```

Unit tests compile the real engine sources; integration tests check generation, perplexity and that the Metal
matrix path gives the same output and perplexity as the other paths. Details: [tests/README.md](tests/README.md).

## Research

Question: why can a small model not hold the strength of a big one, and what puts that strength into it. Everything
is measured with automatic checking (generated tasks, sandboxed grader, private tests, a sealed test set), with
confidence intervals and the failure modes checked by reading the outputs. Full write-up, in Russian, with all
numbers, caveats and sources: [docs/research/RESEARCH_PLAN.md](docs/research/RESEARCH_PLAN.md). Code and raw results:
[`research/`](research).

Main findings so far (toy tasks; one fact syntax; see the write-up for the limits):

* **Training recipe, not the ternary format, limits BitNet.** On the same 100 code tasks, Ternary Bonsai 2 4B
  (PrismML, ternary, made from Qwen3.5-4B; same file size as BitNet, 1.15 GB, same decode speed) solves 76 % vs 37 %
  for BitNet 2B-4T (+39 pp, 95 % CI +15…+65 by task family). Bonsai 2B, with a third fewer non-embedding
  parameters, is not worse than BitNet.
* **Reasoning over given facts is fine; selecting facts is not.** BitNet answers 91 % of 1–3-step questions about a
  made-up world when given only the needed facts, 39 % when given all 117 facts of the world. Three extra facts cost
  14 pp, ten cost 32 pp.
* **Inside the model:** a few attention heads in layers 14–17 (about 1 % of 600) do the retrieval. With 3 extra
  facts they put 95 % of their fact-attention on the needed facts, with 117 facts 31 %: softmax spreads the rest
  over a hundred irrelevant facts. With an awkward wording the right answer exists at layer 27 and the last two layers
  override it ("none"): the knowledge is there, the instruction is applied wrongly.
* **Counting over 117 facts** fails for every model (0–23 %). Per-fact judging with exact adding fixes most of it;
  **reading the facts once into a table** (the model parses each fact and each question, code runs the query) gives
  **150/150** on all question types for BitNet and both Bonsai models, 91–99 % on paraphrased questions. The table
  is ≈0.7 KB versus ≈72 MB of attention cache for the same facts as text.
* **Ternary weights are already used almost fully:** 1.56 bits of entropy per weight out of the log₂3 = 1.585
  ceiling. More useful knowledge per byte has to come from what training puts into the weights and from moving
  arbitrary facts into exact-access memory, not from packing more bits per weight.

Next: training on a Mac with MLX (a trainable ternary BitNet that matches the engine's perplexity is in
[`research/train/bx_qat.py`](research/train/bx_qat.py)): consistency across wordings, learned fact selection, and a
native read-write memory for small on-device models.

## Credits and license

Fork of [microsoft/BitNet](https://github.com/microsoft/BitNet) (MIT) and [llama.cpp](https://github.com/ggml-org/llama.cpp)
(MIT); see [LICENSE](LICENSE). Model weights are not part of this repository. Ternary Bonsai models
([PrismML](https://prismml.com), Apache-2.0) were used only for comparison and are not redistributed.
