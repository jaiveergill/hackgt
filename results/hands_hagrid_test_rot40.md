# Hand signals on HaGRID (test_rot40)

`python scripts/eval_hands.py --split test --rotate 40`

split=test view=full photo (live path), rotated 40 deg; landmarker median 25.5 ms/photo

| class | expected | n | correct | accuracy | errors |
|---|---|---|---|---|---|
| fist | none | 24 | 24 | 100% |  |
| one | fingers:1 | 18 | 9 | 50% | point:right x3, none x3, fingers:2 x3 |
| peace | fingers:2 | 19 | 13 | 68% | none x4, fingers:3 x1, fingers:6 x1 |
| two_up | fingers:2 | 23 | 19 | 83% | none x3, fingers:3 x1 |
| peace_inverted | fingers:2 | 18 | 16 | 89% | none x1, fingers:3 x1 |
| call | fingers:2 | 21 | 14 | 67% | none x3, fingers:3 x2, fingers:4 x1 |
| three | fingers:3 | 22 | 13 | 59% | none x6, fingers:5 x1, fingers:4 x1 |
| three2 | fingers:3 | 17 | 16 | 94% | none x1 |
| four | fingers:4 | 21 | 15 | 71% | none x4, fingers:5 x1, fingers:3 x1 |
| palm | fingers:5 | 17 | 13 | 76% | none x4 |
| stop | fingers:5 | 12 | 7 | 58% | none x3, fingers:4 x2 |
| like | thumb:up | 26 | 5 | 19% | none x19, fingers:6 x1, fingers:2 x1 |
| dislike | thumb:down | 17 | 1 | 6% | none x16 |
| **all** | | 255 | 165 | **64.7%** | |

| label confidence | labels reported | correct |
|---|---|---|
| 0.00-0.25 | 17 | 29.4% |
| 0.25-0.50 | 12 | 58.3% |
| 0.50-0.75 | 48 | 93.8% |
| 0.75-1.00 | 87 | 96.6% |
| **>= 0.25** | 147 (58% of photos) | **92.5%** |
| **>= 0.5** | 135 (53% of photos) | **95.6%** |
| **>= 0.75** | 87 (34% of photos) | **96.6%** |

harmful errors: false thumb:up (a "yes") on 0 photos, false thumb:down (a "no") on 0; fist -> thumb:up 0/24, fist -> any label 0/24
