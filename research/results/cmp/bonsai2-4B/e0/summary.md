# E0 summary: research/results/cmp/bonsai2-4B/e0

started 2026-11-17 22:26:45, repo 99f1f8ef99, machine load at start 4.6

| mode | tasks | pass (final) | completion tokens / task | s / task |
|---|---|---|---|---|
| A single (greedy) | 100 | 76/100 = 76.0% [66.8-83.3] | 148 | 3.2 |
| B pass@1 of 5 samples | 100 | 378/500 = 75.6% [71.6-79.2] | 155 per sample | 3.0 per sample |
| B pass@5 (any of 5) | 100 | 86/100 = 86.0% [77.9-91.5] | 775 | 14.8 |

## Per family, mode A (greedy)

| family | skills | pass |
|---|---|---|
| digit_arith | arithmetic | 10/10 |
| filter_map | composition | 10/10 |
| string_rule | instruction-following | 10/10 |
| longest_run | scanning | 10/10 |
| prefix_diff | prefix-suffix | 10/10 |
| subarray_sum | classic-algorithm | 1/10 |
| stack_machine | simulation,long-dependency | 1/10 |
| grid_paths | dynamic-programming,planning | 4/10 |
| custom_base | arithmetic,algorithm | 10/10 |
| bfs_order | classic-algorithm,graph | 10/10 |

## Failure taxonomy, mode A (first attempt)

pass: 76, wrong_answer: 24

## Compute-matched independent sampling (from the 5 samples of mode B)

| method | tokens / task | pass | independent samples with the same token budget (k) | pass@k |
|---|---|---|---|---|

A method is only interesting if it beats the independent-sampling column at the same token budget.
