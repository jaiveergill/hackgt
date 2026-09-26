# Enrollment eval: miracl, 8 speakers x 10 phrases x 5 takes

Data: the Kaggle MIRACL-VC1 subset (blueguydeez8974/miracl-vc1), 400 clips, every take of a speaker from one recording session (so cross-session drift is not measured). Test = takes 4-5 of every phrase; profiles use earlier takes.

Evidence for an enrolled phrase = SLOPE * (best DTW similarity - the profile's impostor mean) - OFFSET, added to the VSR log-likelihood; unenrolled phrases get 0. (SLOPE, OFFSET) are fitted by minimising the log-loss of the decoder's posterior on the other 7 speakers' all K=1, all K=2, all K=3, half: enrolled phrases, half: unenrolled phrases instances; every number below is on the held-out speaker.

Fitted (SLOPE, OFFSET) per held-out speaker: F01 (80.4, 12.3), F02 (85.6, 12.2), F04 (83.2, 12.3), F05 (82.7, 11.9), F06 (84.7, 12.1), M01 (77.6, 10.9), M02 (102.2, 15.8), M04 (88.1, 12.4)
Fitted on all speakers (the defaults in enroll.py): SLOPE 85.1, OFFSET 12.4 (enroll.py has SLOPE 85.1, OFFSET 12.4).

| protocol (held out) | n | generic top-1 | enrolled top-1 | ECE generic / enrolled | wrong with conf >= 0.6: generic / enrolled | wrong with conf >= 0.9: generic / enrolled | decisions within 2 nats: generic / enrolled |
|---|---|---|---|---|---|---|---|
| all K=1 | 160 | 0.769 | **0.969** | 0.098 / 0.026 | 21/37 / 5/5 | 7 / 2 | 0.31 / 0.03 |
| all K=2 | 160 | 0.769 | **0.981** | 0.098 / 0.017 | 21/37 / 3/3 | 7 / 2 | 0.31 / 0.02 |
| all K=3 | 160 | 0.769 | **0.988** | 0.098 / 0.014 | 21/37 / 2/2 | 7 / 1 | 0.31 / 0.02 |
| half: enrolled phrases | 160 | 0.769 | **0.950** | 0.098 / 0.035 | 21/37 / 7/8 | 7 / 2 | 0.31 / 0.07 |
| half: unenrolled phrases | 160 | 0.769 | **0.812** | 0.098 / 0.094 | 21/37 / 20/30 | 7 / 11 | 0.31 / 0.23 |
| one phrase not enrolled | 160 | 0.769 | **0.931** | 0.098 / 0.046 | 21/37 / 11/11 | 7 / 6 | 0.31 / 0.07 |
| other speaker's profile | 1120 | 0.769 | **0.924** | 0.098 / 0.055 | 147/259 / 74/85 | 49 / 52 | 0.31 / 0.06 |

Reliability of the confidence (held out, own-profile protocols pooled):

| confidence | generic: n, accuracy, mean conf | enrolled: n, accuracy, mean conf |
|---|---|---|
| 0.0-0.5 | 78, 0.231, 0.45 | 8, 0.250, 0.45 |
| 0.5-0.6 | 48, 0.250, 0.54 | 9, 0.444, 0.54 |
| 0.6-0.9 | 216, 0.611, 0.75 | 61, 0.607, 0.78 |
| 0.9-1.0 | 618, 0.932, 0.98 | 882, 0.973, 1.00 |

Nurse question (real ContextStore prior, gamma 1), held out, own-profile protocols: "Is it <runner-up>?" flips the decision / "Is it <true phrase>?" fixes a wrong decision:

- generic: flips 23.1% of 960 decisions; fixes 54 of 222 wrong decisions
- enrolled: flips 4.0% of 960 decisions; fixes 16 of 59 wrong decisions

Take gate (a new take of phrase X is rejected if the profile's evidence for X is < 0; needs an earlier take of X). Profile = take 1 of every phrase, candidate = take 2, final parameters:

- genuine take 2 rejected: 2/80
- take of another phrase, labelled as X: rejected 707/720
- unrelated speech (2 TED clips x every phrase label): rejected 160/160
- the "no mouth movement" gate (empty CTC greedy transcript), which every take passes first: rejects 22/400 genuine clips
- a first take of a phrase has nothing to compare with and is only checked for mouth movement (use /api/enroll/undo).

Real-path check (PhraseDecoder.decode with a real Profile, all-enrolled and half-enrolled, enroll.py parameters): 0 disagreements with the sweep over 320 decisions, max confidence difference 0.0000.

Large inventory (213 phrases = the 204 ICU phrases + these; CTC prefilter + attention on the top 48 and on every phrase with positive evidence), profile = takes 1-3 of these 10 phrases only, test takes 4-5, real PhraseDecoder, enroll.py parameters (fitted on all speakers, so not held out):

| | n | top-1 | ECE | wrong with conf >= 0.6 | wrong with conf >= 0.9 |
|---|---|---|---|---|---|
| generic | 160 | 0.569 | 0.102 | 24/69 | 3 |
| enrolled | 160 | **0.919** | 0.061 | 8/13 | 2 |

Latency, real features (MIRACL takes as templates, slices of a TED encoding as queries), median of 5, load 5.3 at start:

| inventory | templates | query | evidence ms | decode ms without profile | decode ms with profile |
|---|---|---|---|---|---|
| 40 phrases | 120 | 1 s (25 frames) | 8 | 122 | 128 |
| 40 phrases | 120 | 2 s (50 frames) | 14 | 140 | 156 |
| 40 phrases | 120 | 4 s (100 frames) | 22 | 184 | 205 |
| 40 phrases | profile save (per take) | 4 MB | 14 ms | | |
| 204 phrases | 408 | 1 s (25 frames) | 31 | 113 | 143 |
| 204 phrases | 408 | 2 s (50 frames) | 32 | 164 | 196 |
| 204 phrases | 408 | 4 s (100 frames) | 46 | 206 | 249 |
| 204 phrases | profile save (per take) | 16 MB | 50 ms | | |

Load at end: 5.3.
