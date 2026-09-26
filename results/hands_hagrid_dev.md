# Hand signals on HaGRID (dev)

`python scripts/eval_hands.py --split dev`

split=dev view=crop 3.0x hand box landmarker latency median 35.5 ms/image

| class | expected | n | hand found | correct | accuracy | errors |
|---|---|---|---|---|---|---|
| fist | fingers:0 | 16 | 14 | 13 | 81% | no hand x2, thumb:up x1 |
| one | fingers:1 | 22 | 21 | 18 | 82% | fingers:2 x3, no hand x1 |
| peace | fingers:2 | 21 | 19 | 15 | 71% | fingers:3 x2, no hand x2, fingers:0 x1 |
| two_up | fingers:2 | 17 | 16 | 15 | 88% | fingers:3 x1, no hand x1 |
| peace_inverted | fingers:2 | 22 | 22 | 22 | 100% |  |
| call | fingers:2 | 19 | 18 | 14 | 74% | thumb:up x3, fingers:5 x1, no hand x1 |
| three | fingers:3 | 18 | 17 | 16 | 89% | no hand x1, fingers:4 x1 |
| three2 | fingers:3 | 23 | 23 | 22 | 96% | fingers:4 x1 |
| four | fingers:4 | 19 | 18 | 18 | 95% | no hand x1 |
| palm | fingers:5 | 23 | 23 | 23 | 100% |  |
| stop | fingers:5 | 28 | 27 | 20 | 71% | fingers:4 x7, no hand x1 |
| like | thumb:up | 14 | 12 | 9 | 64% | no hand x2, fingers:0 x1, fingers:2 x1 |
| dislike | thumb:down | 23 | 19 | 16 | 70% | no hand x4, fingers:1 x2, fingers:0 x1 |
| **all** | | 265 | 249 | 221 | **83.4%** | |

confidence >= 0.5: 95.2% correct of 209 found hands  confidence < 0.5: 55.0% correct of 40 found hands
