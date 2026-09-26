# Hand signals on HaGRID (dev)

`python scripts/eval_hands.py --split dev`

split=dev view=full photo (live path); landmarker median 24.5 ms/photo

| class | expected | n | correct | accuracy | errors |
|---|---|---|---|---|---|
| fist | none | 16 | 15 | 94% | thumb:up x1 |
| one | fingers:1 | 22 | 15 | 68% | none x4, fingers:2 x2, fingers:5 x1 |
| peace | fingers:2 | 21 | 17 | 81% | none x3, fingers:3 x1 |
| two_up | fingers:2 | 17 | 14 | 82% | fingers:3 x2, none x1 |
| peace_inverted | fingers:2 | 22 | 18 | 82% | none x2, fingers:3 x1, fingers:1 x1 |
| call | fingers:2 | 19 | 12 | 63% | none x4, thumb:up x2, fingers:3 x1 |
| three | fingers:3 | 18 | 13 | 72% | fingers:4 x3, none x2 |
| three2 | fingers:3 | 23 | 21 | 91% | none x1, fingers:1 x1 |
| four | fingers:4 | 19 | 18 | 95% | none x1 |
| palm | fingers:5 | 23 | 22 | 96% | none x1 |
| stop | fingers:5 | 28 | 19 | 68% | none x6, fingers:4 x3 |
| like | thumb:up | 14 | 9 | 64% | none x5 |
| dislike | thumb:down | 23 | 13 | 57% | none x9, fingers:2 x1 |
| **all** | | 265 | 206 | **77.7%** | |

| label confidence | labels reported | correct |
|---|---|---|
| 0.00-0.25 | 25 | 48.0% |
| 0.25-0.50 | 17 | 76.5% |
| 0.50-0.75 | 57 | 98.2% |
| 0.75-1.00 | 112 | 98.2% |
| **>= 0.25** | 186 (70% of photos) | **96.2%** |
| **>= 0.5** | 169 (64% of photos) | **98.2%** |
| **>= 0.75** | 112 (42% of photos) | **98.2%** |

harmful errors: false thumb:up (a "yes") on 3 photos, false thumb:down (a "no") on 0; fist -> thumb:up 1/16, fist -> any label 1/16
