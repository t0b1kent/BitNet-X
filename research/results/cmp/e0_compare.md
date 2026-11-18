# E0 comparison (same 100 tasks, 10 families; 95% CI by cluster bootstrap over families)

| model | A greedy | B pass@5 (oracle) | B selected@1 (public examples) | Δ A vs bitnet | Δ selected@1 vs bitnet |
|---|---|---|---|---|---|
| bitnet | 37% [12-65] | 47% [20-74] | 43% [17-70] |  |  |
| bonsai2-2B | 46% [16-76] | 63% [39-86] | 60% [36-83] | +9 [-7; +33] | +17 [-2; +41] |
| bonsai2-4B | 76% [52-100] | 86% [68-100] | 84% [65-100] | +39 [+15; +65] | +41 [+19; +66] |

## Per family: A greedy / pass@5 / selected@1 (of 10)

| family | bitnet | bonsai2-2B | bonsai2-4B |
|---|---|---|---|
| bfs_order | 0 / 0 / 0 | 0 / 6 / 4 | 10 / 10 / 10 |
| custom_base | 3 / 9 / 7 | 6 / 8 / 8 | 10 / 10 / 10 |
| digit_arith | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 10 / 10 |
| filter_map | 0 / 0 / 0 | 10 / 10 / 10 | 10 / 10 / 10 |
| grid_paths | 0 / 0 / 0 | 0 / 0 / 0 | 4 / 6 / 6 |
| longest_run | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 10 / 10 |
| prefix_diff | 3 / 5 / 5 | 0 / 3 / 3 | 10 / 10 / 10 |
| stack_machine | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 1 / 1 |
| string_rule | 10 / 10 / 10 | 10 / 10 / 9 | 10 / 10 / 10 |
| subarray_sum | 1 / 3 / 1 | 0 / 6 / 6 | 1 / 9 / 7 |

## Mode A: failure categories and finish reasons

* bitnet: {'wrong_answer': 42, 'pass': 37, 'runtime_error': 17, 'syntax_error': 4}; finish {'eos': 96, 'limit': 4}; truncated 4
* bonsai2-2B: {'pass': 46, 'wrong_answer': 22, 'runtime_error': 20, 'syntax_error': 10, 'timeout': 2}; finish {'stop': 87, 'length': 13}; truncated 13
* bonsai2-4B: {'pass': 76, 'wrong_answer': 24}; finish {'stop': 100}; truncated 0

## Public examples as a filter (mode B, all samples are graded on private tests)

* bitnet: a sample passing public examples found in 64/100 tasks; of those, hidden tests passed 43/64
* bonsai2-2B: a sample passing public examples found in 66/100 tasks; of those, hidden tests passed 60/66
* bonsai2-4B: a sample passing public examples found in 93/100 tasks; of those, hidden tests passed 84/93
