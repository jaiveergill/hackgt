# Hand signals on HaGRID (test_rot-40_noroll)

`python scripts/eval_hands.py --split test --rotate -40 --no-roll`

split=test view=crop 3.0x hand box rotate -40 deg, roll from face False (no face found in 117 photos) landmarker latency median 25.7 ms/image

| class | expected | n | hand found | correct | accuracy | errors |
|---|---|---|---|---|---|---|
| fist | fingers:0 | 24 | 22 | 17 | 71% | thumb:up x3, fingers:1 x2, no hand x2 |
| one | fingers:1 | 18 | 18 | 17 | 94% | fingers:2 x1 |
| peace | fingers:2 | 19 | 18 | 18 | 95% | no hand x1 |
| two_up | fingers:2 | 23 | 20 | 19 | 83% | no hand x3, fingers:3 x1 |
| peace_inverted | fingers:2 | 18 | 17 | 17 | 94% | no hand x1 |
| call | fingers:2 | 21 | 21 | 18 | 86% | thumb:up x1, fingers:4 x1, fingers:5 x1 |
| three | fingers:3 | 22 | 17 | 14 | 64% | no hand x5, fingers:4 x2, fingers:5 x1 |
| three2 | fingers:3 | 17 | 16 | 16 | 94% | no hand x1 |
| four | fingers:4 | 21 | 21 | 20 | 95% | fingers:5 x1 |
| palm | fingers:5 | 17 | 17 | 17 | 100% |  |
| stop | fingers:5 | 12 | 12 | 10 | 83% | fingers:4 x2 |
| like | thumb:up | 26 | 20 | 12 | 46% | no hand x6, fingers:1 x5, fingers:2 x2 |
| dislike | thumb:down | 17 | 17 | 14 | 82% | fingers:1 x3 |
| **all** | | 255 | 236 | 209 | **82.0%** | |

confidence >= 0.5: 93.4% correct of 198 found hands  confidence < 0.5: 63.2% correct of 38 found hands
