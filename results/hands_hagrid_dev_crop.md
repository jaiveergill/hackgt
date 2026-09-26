# Hand signals on HaGRID (dev_crop)

`python scripts/eval_hands.py --split dev --crop`

split=dev view=ORACLE crop 3.0x ground-truth hand box; landmarker median 25.6 ms/photo

| class | expected | n | correct | accuracy | errors |
|---|---|---|---|---|---|
| fist | none | 16 | 15 | 94% | thumb:up x1 |
| one | fingers:1 | 22 | 18 | 82% | fingers:2 x3, none x1 |
| peace | fingers:2 | 21 | 15 | 71% | none x3, fingers:3 x2, fingers:1 x1 |
| two_up | fingers:2 | 17 | 15 | 88% | fingers:3 x1, none x1 |
| peace_inverted | fingers:2 | 22 | 22 | 100% |  |
| call | fingers:2 | 19 | 14 | 74% | thumb:up x3, fingers:5 x1, none x1 |
| three | fingers:3 | 18 | 16 | 89% | none x1, fingers:4 x1 |
| three2 | fingers:3 | 23 | 22 | 96% | fingers:4 x1 |
| four | fingers:4 | 19 | 18 | 95% | none x1 |
| palm | fingers:5 | 23 | 23 | 100% |  |
| stop | fingers:5 | 28 | 20 | 71% | fingers:4 x7, none x1 |
| like | thumb:up | 14 | 9 | 64% | none x3, fingers:2 x1, fingers:3 x1 |
| dislike | thumb:down | 23 | 16 | 70% | none x7 |
| **all** | | 265 | 223 | **84.2%** | |

| label confidence | labels reported | correct |
|---|---|---|
| 0.00-0.25 | 25 | 44.0% |
| 0.25-0.50 | 13 | 69.2% |
| 0.50-0.75 | 66 | 95.5% |
| 0.75-1.00 | 127 | 98.4% |
| **>= 0.25** | 206 (78% of photos) | **95.6%** |
| **>= 0.5** | 193 (73% of photos) | **97.4%** |
| **>= 0.75** | 127 (48% of photos) | **98.4%** |

harmful errors: false thumb:up (a "yes") on 4 photos, false thumb:down (a "no") on 0; fist -> thumb:up 1/16, fist -> any label 1/16
