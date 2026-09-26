# Hand signals on HaGRID (test)

`python scripts/eval_hands.py --split test`

split=test view=crop 3.0x hand box landmarker latency median 25.5 ms/image

| class | expected | n | hand found | correct | accuracy | errors |
|---|---|---|---|---|---|---|
| fist | fingers:0 | 24 | 23 | 21 | 88% | thumb:up x2, no hand x1 |
| one | fingers:1 | 18 | 17 | 15 | 83% | fingers:2 x2, no hand x1 |
| peace | fingers:2 | 19 | 18 | 18 | 95% | no hand x1 |
| two_up | fingers:2 | 23 | 20 | 19 | 83% | no hand x3, fingers:3 x1 |
| peace_inverted | fingers:2 | 18 | 18 | 17 | 94% | fingers:3 x1 |
| call | fingers:2 | 21 | 20 | 17 | 81% | thumb:up x1, fingers:4 x1, no hand x1 |
| three | fingers:3 | 22 | 19 | 18 | 82% | no hand x3, fingers:5 x1 |
| three2 | fingers:3 | 17 | 17 | 17 | 100% |  |
| four | fingers:4 | 21 | 20 | 19 | 90% | no hand x1, fingers:5 x1 |
| palm | fingers:5 | 17 | 17 | 17 | 100% |  |
| stop | fingers:5 | 12 | 12 | 9 | 75% | fingers:4 x3 |
| like | thumb:up | 26 | 20 | 19 | 73% | no hand x6, fingers:1 x1 |
| dislike | thumb:down | 17 | 17 | 16 | 94% | fingers:1 x1 |
| **all** | | 255 | 238 | 222 | **87.1%** | |

confidence >= 0.5: 98.5% correct of 198 found hands  confidence < 0.5: 67.5% correct of 40 found hands
