# Hand signals on HaGRID (test_rot40_noroll)

`python scripts/eval_hands.py --split test --rotate 40 --no-roll`

split=test view=crop 3.0x hand box rotate 40 deg, roll from face False (no face found in 112 photos) landmarker latency median 24.2 ms/image

| class | expected | n | hand found | correct | accuracy | errors |
|---|---|---|---|---|---|---|
| fist | fingers:0 | 24 | 22 | 19 | 79% | thumb:up x2, no hand x2, fingers:1 x1 |
| one | fingers:1 | 18 | 18 | 12 | 67% | point:right x4, fingers:2 x1, fingers:0 x1 |
| peace | fingers:2 | 19 | 17 | 16 | 84% | no hand x2, fingers:3 x1 |
| two_up | fingers:2 | 23 | 20 | 19 | 83% | no hand x3, fingers:3 x1 |
| peace_inverted | fingers:2 | 18 | 17 | 17 | 94% | no hand x1 |
| call | fingers:2 | 21 | 21 | 17 | 81% | fingers:4 x2, fingers:3 x1, fingers:1 x1 |
| three | fingers:3 | 22 | 18 | 14 | 64% | no hand x4, fingers:4 x3, fingers:2 x1 |
| three2 | fingers:3 | 17 | 16 | 16 | 94% | no hand x1 |
| four | fingers:4 | 21 | 19 | 17 | 81% | no hand x2, fingers:5 x2 |
| palm | fingers:5 | 17 | 17 | 17 | 100% |  |
| stop | fingers:5 | 12 | 11 | 9 | 75% | fingers:4 x2, no hand x1 |
| like | thumb:up | 26 | 19 | 6 | 23% | fingers:1 x10, no hand x7, fingers:2 x2 |
| dislike | thumb:down | 17 | 17 | 3 | 18% | fingers:1 x14 |
| **all** | | 255 | 232 | 182 | **71.4%** | |

confidence >= 0.5: 84.4% correct of 192 found hands  confidence < 0.5: 50.0% correct of 40 found hands
