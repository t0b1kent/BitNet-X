First answer token correct (of 30 per cell), BitNet, prompt 'Reply with only the answer.'

| heads | x logits | 30: hop2 | 30: hop3 | 30: count | all: hop2 | all: hop3 | all: count | sum |
|---|---|---|---|---|---|---|---|---|
| baseline | 1.0 | 4 | 13 | 4 | 2 | 15 | 3 | 41 |
| retrieval heads (7) | 1.5 | 5 | 13 | 6 | 2 | 14 | 3 | 43 |
| retrieval heads (7) | 2.0 | 5 | 14 | 5 | 2 | 13 | 3 | 42 |
| retrieval heads (7) | 3.0 | 4 | 15 | 4 | 2 | 12 | 2 | 39 |
| all heads, layers 14-17 (80) | 1.5 | 4 | 14 | 3 | 2 | 12 | 3 | 38 |
| all heads, layers 14-17 (80) | 2.0 | 7 | 14 | 3 | 2 | 13 | 3 | 42 |
| all heads, layers 14-17 (80) | 3.0 | 7 | 14 | 6 | 2 | 12 | 4 | 45 |
| all 600 heads | 1.5 | 5 | 7 | 0 | 2 | 11 | 0 | 25 |
| all 600 heads | 2.0 | 4 | 9 | 0 | 2 | 7 | 0 | 22 |
| all 600 heads | 3.0 | 0 | 6 | 1 | 1 | 3 | 0 | 11 |
