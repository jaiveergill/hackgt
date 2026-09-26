# Hand signals on HaGRID (test_rot-40)

`python scripts/eval_hands.py --split test --rotate -40`

split=test view=full photo (live path), rotated -40 deg; landmarker median 29.0 ms/photo

| class | expected | n | correct | accuracy | errors |
|---|---|---|---|---|---|
| fist | none | 24 | 24 | 100% |  |
| one | fingers:1 | 18 | 13 | 72% | none x3, fingers:2 x2 |
| peace | fingers:2 | 19 | 13 | 68% | none x4, fingers:4 x1, fingers:3 x1 |
| two_up | fingers:2 | 23 | 16 | 70% | none x6, fingers:3 x1 |
| peace_inverted | fingers:2 | 18 | 16 | 89% | none x2 |
| call | fingers:2 | 21 | 15 | 71% | none x2, fingers:5 x2, fingers:3 x1 |
| three | fingers:3 | 22 | 12 | 55% | none x5, fingers:8 x2, fingers:4 x2 |
| three2 | fingers:3 | 17 | 17 | 100% |  |
| four | fingers:4 | 21 | 16 | 76% | none x3, fingers:5 x2 |
| palm | fingers:5 | 17 | 12 | 71% | none x4, fingers:10 x1 |
| stop | fingers:5 | 12 | 9 | 75% | fingers:4 x2, none x1 |
| like | thumb:up | 26 | 8 | 31% | none x17, fingers:2 x1 |
| dislike | thumb:down | 17 | 5 | 29% | none x11, fingers:3 x1 |
| **all** | | 255 | 176 | **69.0%** | |

| label confidence | labels reported | correct |
|---|---|---|
| 0.00-0.25 | 21 | 42.9% |
| 0.25-0.50 | 13 | 84.6% |
| 0.50-0.75 | 44 | 88.6% |
| 0.75-1.00 | 95 | 97.9% |
| **>= 0.25** | 152 (60% of photos) | **94.1%** |
| **>= 0.5** | 139 (55% of photos) | **95.0%** |
| **>= 0.75** | 95 (37% of photos) | **97.9%** |

harmful errors: false thumb:up (a "yes") on 1 photos, false thumb:down (a "no") on 0; fist -> thumb:up 0/24, fist -> any label 0/24
