# Plan 6: never put the wrong words in the patient's mouth (2026-09-27)

Goal from the team: Open Mode is decent, the 204-phrase list is too small, and an LLM should bring patient context.
Constraint: a wrong phrase is worse than no phrase. Every utterance must end in one of: spoken, asked ("Sounds like: X?"), or "say again".

## What the data said (glasses session log, 2026-09-26)

- Frames were not the problem: 21-29 fps and the face tracked 72-100% while listening.
- 10 of 14 attempts were Open Mode; half the raw outputs were close ("IING WATER", "HELP I AM NOT"), half were far off.
- Phrase Mode: 4 attempts, 13-57% confidence. "I need my medicine" read as "NINEEN FOR" -> "Good night" 13%: no list fixes that clip.
  "I NEED MY AMAZING" -> "I need a break" 40% with "I need medication" 3rd: a re-ranker with context fixes that one.
- Long utterances: a 5 s attempt produced 14 words of nonsense; the attention decoder drifts and loops ("I CAN'T STOP" x14 on a 15 s TED clip).

## Where a prior can act (strongest first)

1. Inside the beam search (shallow fusion). The shipped RNN LM is TED; it produces "hand gestures" for ICU patients.
   -> `silent_running/icu_lm.py`: Witten-Bell subword trigram trained on `data/icu/corpus.txt` (6,141 lines), plugged in as the "lm"
   scorer with weight 0.3 (`--lm-weight`, `SR_LM_WEIGHT`), and added to every candidate score. Sanity: ppl("I need my medication") = 6,
   ppl("I need my amazing") = 77, ppl("mastermind nineteen yahoo") = 35,760. It does not override strong lips: the TED clip still decodes
   as "hand gestures" at weight 0.4, which is correct behaviour.
2. A generated utterance bank as candidates. `scripts/gen_icu_corpus.py`: slot grammar (890 utterances, coverage we can reason about)
   + LLM variety (3 rounds x 10 categories) -> `data/icu/bank.txt`, 4,083 utterances beyond the curated 204. The server decodes against
   all 4,287 (`--no-bank` to disable); the UI and voice bank keep the curated 204. Cost: CTC prefilter over 4,287 = 77 ms, full stage 240 ms in-server.
3. LLM re-ranker (`OpenAIChooser.pick_phrase`, `server._llm_rerank`): consulted only when the lips alone would not speak (confidence
   < 0.85); gets the raw transcript, the top-10 candidates with posteriors, and the context; returns an index or -1 with high/medium/low.
   Its pick is accepted only if the lips score it within 8 nats of the visual top. high -> speak (conf >= 0.75); medium -> becomes the
   first question of the confirm loop; low or -1 -> the visual decision stands. Every pick is broadcast (`llm_pick`) and logged.
4. Long utterances: `VSREngine.segment_by_blanks` cuts at >= 400 ms of CTC blank; `beam_search_segments` decodes each sentence;
   `collapse_loops` + `sane_hypothesis` fall back to the CTC transcript when the attention decoder loops or runs long.

## What the LLM experiments showed (scratch harness, simulated lip noise at 3 levels, 120 trials, gpt-4.1-mini)

| prompt | LLM correct when truth in top-10 | lip scorer top-1 | wrong picks at "high" | abstained when truth absent | latency p50 |
|---|---|---|---|---|---|
| plain "pick one" | 71% | 80% | 3.3% | 10/55 | 0.80 s |
| rules (lip confusions, prefer rank, urgent guard) | 75% | 80% | 8.3% | 7/55 | 0.88 s |
| rules x3 votes | 74% | 80% | 8.3% | 8/55 | 2.80 s |
| rules, no context | 74% | 80% | 8.3% | 7/55 | 0.92 s |

Conclusions that shaped the design: the LLM is NOT a better picker than the lip scorer; it almost never abstains on its own; so it
re-ranks and grades only, the visual-margin check and the confirm loop are the safety, and the candidate list has to be good
first (layers 1-2). On the four real attempts it returned "medium" every time, i.e. it would have asked, not guessed.

## Still to do

- Label captures from the glasses (`data/captures`, one-tap corrections in the UI) and re-run `scripts/captures.py rescore` after each change;
  track sentence accuracy and wrong-at-high rate (must be 0).
- Tune `--lm-weight` and the bank size on labelled clips; grow the grammar where captures show gaps.
- Enrollment through the glasses (main already has it) for the clips where the model reads nothing.
- Sensor windowing on the ESP32 for 4x mouth pixels (Hriday's zoom commit) is the other half of the "read nothing" problem.
