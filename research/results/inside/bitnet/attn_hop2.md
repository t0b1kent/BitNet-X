Retrieval-head candidates (highest mean share of fact-attention on the needed facts at level 10): L17H17 0.83, L16H13 0.73, L17H12 0.73, L17H4 0.71, L16H8 0.70

| distractors | facts | needed share, base rate | best head (fixed): share of fact-attention on needed facts | same head: needed facts in its top-k (recall) | same head: attention per needed fact | all 600 heads: max share (mean over questions) |
|---|---|---|---|---|---|---|
| 3 | 5 | 0.40 | 0.96 | 0.90 | 0.454 | 0.99 |
| 10 | 12 | 0.17 | 0.83 | 0.72 | 0.405 | 0.93 |
| 30 | 32 | 0.06 | 0.64 | 0.60 | 0.319 | 0.78 |
| all | 117 | 0.02 | 0.28 | 0.23 | 0.140 | 0.50 |

Per layer, level all: best head share on needed facts: 0:0.07 1:0.04 2:0.05 3:0.10 4:0.10 5:0.08 6:0.11 7:0.10 8:0.08 9:0.06 10:0.07 11:0.07 12:0.09 13:0.06 14:0.29 15:0.07 16:0.29 17:0.40 18:0.18 19:0.12 20:0.20 21:0.14 22:0.11 23:0.18 24:0.16 25:0.07 26:0.06 27:0.06 28:0.05 29:0.05
