# Hand signals on HaGRID (test)

`python scripts/eval_hands.py --split test`

split=test view=full photo (live path); landmarker median 24.5 ms/photo

| class | expected | n | correct | accuracy | errors |
|---|---|---|---|---|---|
| fist | none | 24 | 24 | 100% |  |
| one | fingers:1 | 18 | 13 | 72% | fingers:2 x3, none x2 |
| peace | fingers:2 | 19 | 16 | 84% | none x2, fingers:6 x1 |
| two_up | fingers:2 | 23 | 19 | 83% | none x3, fingers:3 x1 |
| peace_inverted | fingers:2 | 18 | 16 | 89% | none x1, fingers:3 x1 |
| call | fingers:2 | 21 | 14 | 67% | none x3, fingers:5 x2, fingers:4 x1 |
| three | fingers:3 | 22 | 15 | 68% | none x5, fingers:4 x2 |
| three2 | fingers:3 | 17 | 15 | 88% | fingers:2 x1, none x1 |
| four | fingers:4 | 21 | 18 | 86% | none x2, fingers:5 x1 |
| palm | fingers:5 | 17 | 14 | 82% | none x3 |
| stop | fingers:5 | 12 | 8 | 67% | none x2, fingers:4 x2 |
| like | thumb:up | 26 | 16 | 62% | none x10 |
| dislike | thumb:down | 17 | 12 | 71% | none x5 |
| **all** | | 255 | 200 | **78.4%** | |

| label confidence | labels reported | correct |
|---|---|---|
| 0.00-0.25 | 18 | 61.1% |
| 0.25-0.50 | 19 | 68.4% |
| 0.50-0.75 | 52 | 94.2% |
| 0.75-1.00 | 103 | 100.0% |
| **>= 0.25** | 174 (68% of photos) | **94.8%** |
| **>= 0.5** | 155 (61% of photos) | **98.1%** |
| **>= 0.75** | 103 (40% of photos) | **100.0%** |

harmful errors: false thumb:up (a "yes") on 0 photos, false thumb:down (a "no") on 0; fist -> thumb:up 0/24, fist -> any label 0/24
