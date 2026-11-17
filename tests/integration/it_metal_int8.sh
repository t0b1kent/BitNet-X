#!/usr/bin/env bash
# Same as it_metal_default but with the bit-serial path switched off (GGML_METAL_I2S_NO_BITS=1): the int8 Metal kernels
# (used when the weights cannot be repacked, e.g. private buffers) must give the correct answer too.
set -uo pipefail
export GGML_METAL_I2S_NO_BITS=1
exec bash "$(dirname "${BASH_SOURCE[0]}")/it_metal_default.sh"
