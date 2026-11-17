// ggml symbols referenced by src/ggml-bitnet-mad.cpp that the unit test needs.
#include <stddef.h>
#include <stdint.h>
size_t ggml_row_size(int type, int64_t ne) { (void)type; return (size_t)ne; }
