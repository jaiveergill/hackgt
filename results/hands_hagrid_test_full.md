# Hand signals on HaGRID (test_full)

`python scripts/eval_hands.py --split test --full-frame`

split=test view=full frame landmarker latency median 26.4 ms/image

| class | expected | n | hand found | correct | accuracy | errors |
|---|---|---|---|---|---|---|
| fist | fingers:0 | 24 | 22 | 20 | 83% | thumb:up x2, no hand x2 |
| one | fingers:1 | 18 | 16 | 13 | 72% | fingers:2 x3, no hand x2 |
| peace | fingers:2 | 19 | 17 | 17 | 89% | no hand x2 |
| two_up | fingers:2 | 23 | 20 | 19 | 83% | no hand x3, fingers:3 x1 |
| peace_inverted | fingers:2 | 18 | 17 | 16 | 89% | no hand x1, fingers:3 x1 |
| call | fingers:2 | 21 | 19 | 14 | 67% | no hand x2, fingers:5 x2, fingers:4 x1 |
| three | fingers:3 | 22 | 18 | 15 | 68% | no hand x4, fingers:4 x2, fingers:2 x1 |
| three2 | fingers:3 | 17 | 16 | 15 | 88% | fingers:2 x1, no hand x1 |
| four | fingers:4 | 21 | 19 | 18 | 86% | no hand x2, fingers:5 x1 |
| palm | fingers:5 | 17 | 15 | 14 | 82% | no hand x2, fingers:2 x1 |
| stop | fingers:5 | 12 | 11 | 8 | 67% | fingers:4 x2, no hand x1, fingers:3 x1 |
| like | thumb:up | 26 | 17 | 16 | 62% | no hand x9, fingers:0 x1 |
| dislike | thumb:down | 17 | 16 | 12 | 71% | fingers:1 x3, no hand x1, fingers:0 x1 |
| **all** | | 255 | 223 | 197 | **77.3%** | |

confidence >= 0.5: 95.0% correct of 180 found hands  confidence < 0.5: 60.5% correct of 43 found hands
