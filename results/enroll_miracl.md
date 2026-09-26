# Enrollment eval: miracl

8 speakers, 10-phrase inventory, test = takes > 3 of every phrase (160 clips, identical for every K). Generic = VSR log-likelihood only.

| similarity | K takes | generic top-1 | enrolled top-1 (CV weight) | prototype only | weights chosen per fold |
|---|---|---|---|---|---|
| dtw | 1 | 0.769 | 0.994 | 0.981 | [150, 150] |
| dtw | 2 | 0.769 | 0.994 | 0.988 | [150, 150] |
| dtw | 3 | 0.769 | 0.988 | 0.994 | [150, 80] |
| meanpool | 1 | 0.769 | 0.938 | 0.919 | [150, 80] |
| meanpool | 2 | 0.769 | 0.969 | 0.969 | [80, 80] |
| meanpool | 3 | 0.769 | 0.975 | 0.981 | [150, 1000] |

Top-1 over all speakers by weight (DTW):

| K | w=0 | w=2 | w=5 | w=10 | w=15 | w=20 | w=30 | w=50 | w=80 | w=150 | w=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.769 | 0.800 | 0.806 | 0.869 | 0.887 | 0.900 | 0.944 | 0.963 | 0.969 | 0.994 | 0.981 |
| 2 | 0.769 | 0.800 | 0.812 | 0.875 | 0.912 | 0.912 | 0.956 | 0.975 | 0.981 | 0.994 | 0.988 |
| 3 | 0.769 | 0.800 | 0.812 | 0.875 | 0.912 | 0.919 | 0.963 | 0.975 | 0.988 | 0.994 | 0.994 |

Per speaker (DTW, K = 3, w = 150: the best weight over ALL speakers, i.e. in-sample; the CV column above is the honest number):

| speaker | test clips | generic | enrolled |
|---|---|---|---|
| F01 | 20 | 0.500 | 0.950 |
| F02 | 20 | 0.750 | 1.000 |
| F04 | 20 | 0.700 | 1.000 |
| F05 | 20 | 0.850 | 1.000 |
| F06 | 20 | 0.900 | 1.000 |
| M01 | 20 | 0.850 | 1.000 |
| M02 | 20 | 0.750 | 1.000 |
| M04 | 20 | 0.850 | 1.000 |

Confidence (top-1 softmax, K = 3): server thresholds are 0.5 for a critical alert and 0.6 for history.

| | mean conf when right | mean conf when wrong | wrong with conf >= 0.5 | wrong with conf >= 0.6 |
|---|---|---|---|---|
| generic | 0.92 (123) | 0.68 (37) | 27 | 21 |
| enrolled w=150 | 1.00 (159) | 1.00 (1) | 1 | 1 |

Real-path check (engine.score_phrases with an active Profile, K=3, w=150): 0 disagreements with the sweep over 160 clips. Prototype scoring cost: median 5.3 ms per utterance (10 phrases x 3 takes).
Scaling (synthetic features): 120 templates, 60-frame (2.4 s) utterance and takes: 23.5 ms.
Scaling (synthetic features): 408 templates, 60-frame (2.4 s) utterance and takes: 85.3 ms.
