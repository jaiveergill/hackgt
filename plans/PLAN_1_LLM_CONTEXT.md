# Plan 1: Real-life context with an LLM, visual model stays authoritative

**Goal.** Use conversation and situational context to resolve visually ambiguous words, in both modes, without letting the LLM invent content.

**Principle.** The LLM proposes, the visual model verifies. Every LLM output is scored against the video with the same teacher-forced log-likelihood we already use for the phrase bank. A proposal is accepted only if the video supports it nearly as well as the raw top hypothesis.

## Status (2026-09-24 evening): A measured, B built and integrated

Measured on the two captioned TED clips (`scripts/exp_lm_llm.py`, `results/exp_lm_llm.json`), gpt-4o-mini:

| Config | 12 s clip WER (raw beam) | after propose+verify | beam time |
|---|---|---|---|
| beam 10, no LM | 0.273 | **0.227** (fixed "USE JUST MY GLASSES" to "ADJUST MY GLASSES", gap -2.4 nats) | 8.2 s |
| beam 10, LM 0.3 | 0.273 | 0.273 | 12.3 s |
| beam 20, LM 0.3 | 0.273 | 0.250 | 19.3 s |
| beam 20, LM 0.5 | 0.295 | 0.341 (a bad proposal slipped through a length-scaled margin; margin is now a fixed 3 nats) | 19.4 s |

The 2.6 s clip is exact in every config; bad LLM proposals there ("I WANT TO", "GRAND GESTURES") were rejected by verification at -3.5 and -11.3 nats.

Decisions: **LM off** (no gain, 1.5 to 2.4x slower), beam 10, propose-and-verify with a fixed 3-nat margin. Integrated in
`server.py::_bg_llm` and shown in the UI as proposal, verdict, and gap. Next: measure on real silent recordings from the Plan 2 session.

## Current state

- Phrase Mode: transparent additive prior from nurse prompt, category, notes keywords, and recent history. No LLM on this path. Works, sub-second.
- Open Mode: beam-10 n-best, OpenAI `gpt-4o-mini` picks an index, guardrail allows only hypotheses within 3 nats of the best. Correction limited to one word edit. Chooser latency about 2 s, asynchronous.
- Subword RNN language model is downloaded but unused. The published 19.1% number used it.

## Changes

### A. Better raw hypotheses (no LLM)

1. Enable the RNN-LM in beam search (`lm_weight` 0.3, then 0.5) and widen the beam to 20 for Open Mode. Keep beam 10 without LM as the fast path if latency grows past 2 s.
2. Return the top 10 hypotheses to the LLM instead of 5.

Engine change: `VSREngine(lm_weight=..., lm_dir=models/lm_en_subword)` passes `rnnlm`/`rnnlm_conf` into the existing ESPnet decoder builder. Already prototyped in the diff that was rejected; re-apply.

### B. Propose and verify (Open Mode)

1. `OpenAIChooser.propose(candidates, context) -> {sentence, reason}`. Prompt explains the viseme confusion sets (p/b/m, t/d/n, k/g, f/v, s/z, vowels), asks for one sentence with the same word count and rhythm as the top hypotheses, upper case, no new ideas.
2. Server scores `[proposal, top1]` with `engine.score_phrases`. Accept if `score(proposal) >= score(top1) - margin`, margin 3 nats scaled by sentence length (`3 * max(1, n_words/10)`).
3. UI shows both: the raw top hypothesis and the accepted or rejected proposal, with the score gap, so a judge sees exactly what the LLM changed and why it was allowed.
4. Rejected proposals are still displayed, greyed, labelled "not supported by the video".

### C. Richer context object

Sent to the LLM and to the Phrase Mode prior:

- `nurse_prompt`: what was just said or asked (typed, later: live ASR of the nurse's speech through the laptop mic, which is allowed since it is the nurse, not the patient).
- `notes`: free text about the situation.
- `category`: patient-selected need.
- `history`: last 6 confirmed utterances with timestamps.
- `time_of_day`, `since_last_meal` if entered.
- `speaker_profile`: name, and frequent phrases learned from confirmations (count per phrase, feeds the prior as log frequency).

### D. Nurse-side ASR (stretch)

Add a "Nurse speaks" button: record 5 s from the mic, transcribe with OpenAI `gpt-4o-transcribe`, drop into `nurse_prompt`. Makes the context flow natural in the demo. The patient side stays silent and camera-only, so the "no hidden audio" claim holds; state this on screen.

## Evaluation

Script `scripts/exp_lm_llm.py` (prototyped) measures on clips with references:

- TED clips with YouTube captions: WER for beam-10, beam-10 + LM, beam-20 + LM, and each with propose-and-verify.
- Real recordings from Plan 2's session (voiced pass has Whisper references; silent pass has prompt references): same table, plus how often the proposal was accepted and whether acceptance helped or hurt.
- Phrase Mode: top-1 and top-3 with and without context, using the eval harness, with `nurse_prompt` set to a matching question for half the clips and a mismatching one for the other half. The mismatching half checks that context cannot overturn a clear visual result.

Log everything to `results/*.jsonl`.

## Risks

- LLM fluency bias: it prefers common sentences. The verification margin is the control; tune it on the mismatching-context set, not the matching one.
- Latency: LM beam search on CPU may reach 3 s for long utterances. Keep the greedy transcript and phrase result instant, let the beam and LLM update the panel afterwards, as now.
- Overtrust: never auto-speak an accepted proposal in Open Mode unless its gap to top1 is small; otherwise speak the raw hypothesis and show the proposal as a suggestion the user can tap.

**Time estimate.** Half a day for A and B with measurements, 2 hours for C, 2 hours for D.
