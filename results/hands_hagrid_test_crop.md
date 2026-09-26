# Hand signals on HaGRID (test_crop)

`python scripts/eval_hands.py --split test --crop`

split=test view=ORACLE crop 3.0x ground-truth hand box; landmarker median 29.9 ms/photo

| class | expected | n | correct | accuracy | errors |
|---|---|---|---|---|---|
| fist | none | 24 | 24 | 100% |  |
| one | fingers:1 | 18 | 15 | 83% | fingers:2 x2, none x1 |
| peace | fingers:2 | 19 | 18 | 95% | none x1 |
| two_up | fingers:2 | 23 | 19 | 83% | none x3, fingers:3 x1 |
| peace_inverted | fingers:2 | 18 | 17 | 94% | fingers:3 x1 |
| call | fingers:2 | 21 | 17 | 81% | none x2, fingers:4 x1, fingers:5 x1 |
| three | fingers:3 | 22 | 18 | 82% | none x3, fingers:5 x1 |
| three2 | fingers:3 | 17 | 17 | 100% |  |
| four | fingers:4 | 21 | 19 | 90% | none x1, fingers:5 x1 |
| palm | fingers:5 | 17 | 17 | 100% |  |
| stop | fingers:5 | 12 | 9 | 75% | fingers:4 x3 |
| like | thumb:up | 26 | 19 | 73% | none x7 |
| dislike | thumb:down | 17 | 16 | 94% | none x1 |
| **all** | | 255 | 225 | **88.2%** | |

| label confidence | labels reported | correct |
|---|---|---|
| 0.00-0.25 | 18 | 72.2% |
| 0.25-0.50 | 17 | 70.6% |
| 0.50-0.75 | 58 | 98.3% |
| 0.75-1.00 | 119 | 100.0% |
| **>= 0.25** | 194 (76% of photos) | **96.9%** |
| **>= 0.5** | 177 (69% of photos) | **99.4%** |
| **>= 0.75** | 119 (47% of photos) | **100.0%** |

harmful errors: false thumb:up (a "yes") on 0 photos, false thumb:down (a "no") on 0; fist -> thumb:up 0/24, fist -> any label 0/24
