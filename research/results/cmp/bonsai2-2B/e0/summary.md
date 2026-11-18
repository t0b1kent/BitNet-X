# E0 summary: research/results/cmp/bonsai2-2B/e0

started 2026-11-17 21:54:12, repo 99f1f8ef99, machine load at start 11.9

| mode | tasks | pass (final) | completion tokens / task | s / task |
|---|---|---|---|---|
| A single (greedy) | 100 | 46/100 = 46.0% [36.6-55.7] | 235 | 2.4 |
| B pass@1 of 5 samples | 100 | 214/500 = 42.8% [38.5-47.2] | 231 per sample | 2.2 per sample |
| B pass@5 (any of 5) | 100 | 63/100 = 63.0% [53.2-71.8] | 1155 | 11.2 |

## Per family, mode A (greedy)

| family | skills | pass |
|---|---|---|
| digit_arith | arithmetic | 10/10 |
| filter_map | composition | 10/10 |
| string_rule | instruction-following | 10/10 |
| longest_run | scanning | 10/10 |
| prefix_diff | prefix-suffix | 0/10 |
| subarray_sum | classic-algorithm | 0/10 |
| stack_machine | simulation,long-dependency | 0/10 |
| grid_paths | dynamic-programming,planning | 0/10 |
| custom_base | arithmetic,algorithm | 6/10 |
| bfs_order | classic-algorithm,graph | 0/10 |

## Failure taxonomy, mode A (first attempt)

pass: 46, wrong_answer: 22, runtime_error: 20, syntax_error: 10, timeout: 2

## Compute-matched independent sampling (from the 5 samples of mode B)

| method | tokens / task | pass | independent samples with the same token budget (k) | pass@k |
|---|---|---|---|---|

A method is only interesting if it beats the independent-sampling column at the same token budget.
