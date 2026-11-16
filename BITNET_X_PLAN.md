# BitNet-X Research Project

## Purpose

Develop a correct, reproducible and efficient
Microsoft BitNet b1.58 inference engine for
Apple Silicon, starting with M1 Pro.

The longer-term research goal is to improve
reasoning and coding accuracy of small ternary
models without losing their efficiency.

## Original models

BF16 master weights:
microsoft/bitnet-b1.58-2B-4T-bf16

GGUF inference model:
microsoft/bitnet-b1.58-2B-4T-gguf

These models are stored separately from Git.

## Phase 1: Correctness

1. Document current upstream revision.
2. Inspect ARM64 build configuration.
3. Reproduce any build and runtime failures.
4. Validate I2_S and TL1 ternary kernels.
5. Validate activation functions.
6. Check tokenizer and GGUF metadata.
7. Compare numerical outputs with a reference.
8. Add regression tests.

## Phase 2: Performance

1. Benchmark CPU inference on Apple Silicon.
2. Measure prefill and decode separately.
3. Profile memory usage and latency.
4. Investigate NEON and Metal optimizations.
5. Validate correctness after every change.

## Phase 3: Research

1. Establish math and coding baselines.
2. Investigate recursive computation.
3. Investigate adaptive numerical precision.
4. Evaluate small trainable reasoning modules.
5. Compare accuracy per unit of compute.

## Engineering rules

- Preserve original Microsoft attribution and license.
- Never modify the original model weights.
- Never commit model weights or credentials.
- Work in feature branches and submit pull requests.
- Do not merge without review.
- Do not claim speedups without measurements.
- Keep correctness tests reproducible.
- Document assumptions and unresolved problems.
