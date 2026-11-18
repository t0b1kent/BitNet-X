# E0 summary: results/e0_dev

started 2026-11-17 18:56:51, repo 99f1f8ef99, machine load at start 5.2

| mode | tasks | pass (final) | completion tokens / task | s / task |
|---|---|---|---|---|
| A single (greedy) | 100 | 37/100 = 37.0% [28.2-46.8] | 118 | 4.0 |
| B pass@1 of 5 samples | 100 | 142/500 = 28.4% [24.6-32.5] | 109 per sample | 2.1 per sample |
| B pass@5 (any of 5) | 100 | 47/100 = 47.0% [37.5-56.7] | 544 | 10.7 |
| C algorithm hint | 100 | 44/100 = 44.0% [34.7-53.8] | 134 | 3.9 |
| D execution feedback | 100 | 37/100 = 37.0% [28.2-46.8] | 368 | 35.7 |
| E reasoning first | 100 | 34/100 = 34.0% [25.5-43.7] | 228 | 4.5 |

## Per family, mode A (greedy)

| family | skills | pass |
|---|---|---|
| digit_arith | arithmetic | 10/10 |
| filter_map | composition | 0/10 |
| string_rule | instruction-following | 10/10 |
| longest_run | scanning | 10/10 |
| prefix_diff | prefix-suffix | 3/10 |
| subarray_sum | classic-algorithm | 1/10 |
| stack_machine | simulation,long-dependency | 0/10 |
| grid_paths | dynamic-programming,planning | 0/10 |
| custom_base | arithmetic,algorithm | 3/10 |
| bfs_order | classic-algorithm,graph | 0/10 |

## Failure taxonomy, mode A (first attempt)

wrong_answer: 42, pass: 37, runtime_error: 17, syntax_error: 4

## Mode D vs A: fixed 0, broken 0 (of 100); rounds used: 0: 46, 3: 54

## Compute-matched independent sampling (from the 5 samples of mode B)

| method | tokens / task | pass | independent samples with the same token budget (k) | pass@k |
|---|---|---|---|---|
| C algorithm hint | 134 | 44/100 = 44.0% [34.7-53.8] | 1 | 28.4% |
| D execution feedback | 368 | 37/100 = 37.0% [28.2-46.8] | 3 | 40.4% |
| E reasoning first | 228 | 34/100 = 34.0% [25.5-43.7] | 2 | 35.3% |

A method is only interesting if it beats the independent-sampling column at the same token budget.
