Retrieval-head candidates (highest mean share of fact-attention on the needed facts at level 10): L16H13 0.82, L16H8 0.81, L16H6 0.76, L17H4 0.70, L14H2 0.69

| distractors | facts | needed share, base rate | best head (fixed): share of fact-attention on needed facts | same head: needed facts in its top-k (recall) | same head: attention per needed fact | all 600 heads: max share (mean over questions) |
|---|---|---|---|---|---|---|
| 3 | 7 | 0.54 | 0.95 | 0.99 | 0.188 | 0.99 |
| 10 | 14 | 0.26 | 0.82 | 0.82 | 0.150 | 0.93 |
| 30 | 34 | 0.11 | 0.62 | 0.70 | 0.069 | 0.77 |
| all | 117 | 0.03 | 0.31 | 0.38 | 0.043 | 0.44 |

Per layer, level all: best head share on needed facts: 0:0.09 1:0.07 2:0.19 3:0.10 4:0.11 5:0.11 6:0.11 7:0.11 8:0.12 9:0.07 10:0.07 11:0.12 12:0.12 13:0.08 14:0.21 15:0.09 16:0.38 17:0.28 18:0.22 19:0.13 20:0.18 21:0.15 22:0.14 23:0.12 24:0.12 25:0.13 26:0.09 27:0.08 28:0.07 29:0.06
