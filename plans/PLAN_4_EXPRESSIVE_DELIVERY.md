# Plan 4: Face-driven expressive delivery in the user's own voice (ElevenLabs)

**Decision (2026-09-24).** Voice generation is ElevenLabs: instant clone of the user, `eleven_v3` with audio tags for emotion,
`eleven_flash_v2_5` for the neutral fast path. Per-word timing is done locally on any returned audio. IndexTTS-2 stays a
documented local alternative only (`RESEARCH_EXPRESSIVE_VOICE.md`).

**Success criterion.** The same sentence mouthed angry and mouthed warm produces two clearly different, recognizable deliveries
in the user's cloned voice, and the audio tracks the pace and pauses of the mouthing. End to end under 3 s after release of Listen.

**Measured so far.** v3 with a tag on a stock voice: 0.7 to 1.4 s per short phrase. Flash: 0.25 to 0.4 s. Lip reading: 0.2 to 0.5 s.
Budget holds with about 1 s spare.

## Status (2026-09-25 01:00): steps 1 to 4 built and verified on a stock voice; clone verified on the Creator key

- Camera process runs MediaPipe Face Landmarker with blendshapes at 26 to 30 fps; live expression meter in the UI; each utterance
  carries `{emotion, intensity, scores, baseline}` (`silent_running/expression.py`).
- Video word timing: CTC forced alignment on the lip-reading model, 17 ms, 9/9 words on the TED clip (`silent_running/prosody.py`).
- ElevenLabs mapping: emotion + intensity -> v3 audio tag + stability; rate -> speed; neutral -> Flash. Cache keyed on all of it;
  phrase-bank prewarm covers neutral + angry/warm/sad at two intensities (`silent_running/tts.py`).
- `/api/say`: synth -> MMS alignment of the audio -> per-word stretch with clamped global factor -> WAV; broadcasts a delivery
  report. UI: delivery panel, mouthed-vs-delivered strips, `Replay as` buttons, expressive toggle, emotion override.
- Measured (TED clip "I'm going to make a lot of hand gestures", mouthed span 1.36 s): neutral delivery 1.34 s after retiming;
  v3 angry/warm cached 1.1 to 1.3 s wall including 1.1 s alignment+stretch; first alignment call 7 s (model load, now done when
  "match my pace" is switched on). Uncached v3 synth varied 0.7 to 7.7 s across the evening; the pre-synthesized phrase bank is the mitigation.
- Cloning: works via REST (SDK 2.69 `labels` bug); test clone synthesized in 0.8 s and was deleted.
- Remaining: recording session -> real clone -> expression calibration on the expression pass -> blind test (step 5).

## Pipeline

```
capture (camera process)
  ├─ mouth crops ──────────────► VSR ──► words + CTC frame log-probs
  │                                          │
  │                                          ├─► forced alignment ──► per-word start/end, pauses (40 ms grid)
  └─ Face Landmarker blendshapes ──► utterance expression ──► emotion tag + intensity
                                                                   │
words + tag + intensity + global rate ──► ElevenLabs v3 (cloned voice) ──► mp3
                                                                   │
mp3 ──► MMS forced alignment of the audio ──► per-word stretch + pause insertion ──► play
```

The words stay the lip reader's (or the verified LLM correction). Emotion and timing are controls layered on top; the raw
transcript, the chosen tag, the intensity, and the rate factor are all shown in the UI so the demo is legible.

## Components

### A. Expression from the face (new, in the camera process)

- Switch the camera process from `mp.solutions.face_detection` to the tasks-API **Face Landmarker** with `output_face_blendshapes=True`.
  It returns the 4 keypoints we already use for the mouth crop (derive from the 478 landmarks: eye centers, nose tip, mouth
  center) plus 52 blendshape scores per frame. One model call per frame instead of one; CPU real time.
- Per frame, keep the coefficients that survive mouthing: `browDownL/R`, `browInnerUp`, `browOuterUpL/R`, `eyeSquintL/R`,
  `eyeWideL/R`, `cheekSquintL/R`, `noseSneerL/R`, `mouthSmileL/R`, `mouthFrownL/R`, `mouthPressL/R`. Mouth-shape coefficients get
  half weight because the mouth is busy forming words.
- Rule scores per frame (each 0 to 1):
  - angry = mean(browDown) * 0.5 + mean(eyeSquint) * 0.25 + max(noseSneer, mouthPress) * 0.25
  - warm = mean(mouthSmile) * 0.6 + mean(cheekSquint) * 0.4, penalized by browDown
  - sad = browInnerUp * 0.5 + mean(mouthFrown) * 0.5
  - surprised = mean(browOuterUp) * 0.5 + mean(eyeWide) * 0.5
  - neutral = 1 minus max of the above
- Per utterance: median of the top quartile of frames per emotion, subtract the user's resting baseline (captured over the first
  5 s after the face is found and refreshed when idle), pick the max. Intensity = that score clipped to 0 to 1. Below 0.25 is
  neutral.
- Emit `{"emotion": "angry", "intensity": 0.7, "scores": {...}}` alongside the mouth crops on stop. Also stream the live
  scores in `meta` so the UI can show a small expression meter while the user mouths.
- Optional cross-check: EmotiEffLib (Apache-2.0, ONNX, about 10 ms) on the face crop; if it disagrees strongly, lower intensity.

### B. Timing from the video (new, in the server)

- After decoding, run `torchaudio.functional.forced_align` on the CTC log-probs (T' frames at 25 fps) with the recognized token
  ids. Group tokens at SentencePiece word boundaries into words: start frame, end frame, gap to next word.
- Produce: `mouthed_duration`, per-word durations, and pauses (gaps over 250 ms).
- Global rate factor for ElevenLabs `speed` = clamp(natural_estimate / mouthed_duration, 0.7, 1.2), where natural_estimate is
  words times 0.32 s plus pauses. Silent mouthing runs slower than speech, so this mostly lands between 0.8 and 1.0.

### C. Synthesis (ElevenLabs, `silent_running/tts.py`)

- Emotion tag from the label: angry to `[angry]`, warm to `[warm]`, sad to `[sad]`, surprised to `[surprised]`, neutral to no tag.
  Intensity to `stability`: 0.75 at low intensity, 0.35 at high (lower is more expressive); above 0.8 intensity also prefix a
  stronger tag (`[shouting]` for angry, `[cheerful]` for warm). `similarity_boost` 0.8 to protect the clone.
- Model choice: v3 whenever a tag is present, Flash v2.5 for neutral (4x faster, no tag support).
- Cache key includes voice, model, text, tag, and stability bucket. Phrase Mode bank is pre-synthesized at neutral, angry,
  warm, sad, each at two intensity buckets, per voice: 30 phrases times 7 variants, about 210 short calls, done once after cloning.
- Clone: `POST /api/voice/clone?speaker=<name>` on the voiced pass of the recording session (Plan 3). Requires the upgraded plan.

### D. Per-word timing on the audio (new, `silent_running/prosody.py`)

- Decode the mp3, align to its own words with `torchaudio.pipelines.MMS_FA` (installed; about 0.2 s on CPU for a short phrase).
- For each word, stretch ratio = mouthed word duration / synthesized word duration, clamped to 0.75 to 1.35, smoothed across
  neighbors so no single word jumps. Apply with WSOLA (`audiotsm`) or Rubber Band (`pyrubberband` + `brew install rubberband`, better quality).
  Insert measured pauses as silence. Skip the step entirely if the total mismatch is under 10 percent.
- Output a wav to the UI; the UI plays it and shows a timeline strip: mouthed word bars over synthesized word bars.

### E. Server and UI

- `run_decode` gains `expression` and `timing` inputs; result event carries `emotion`, `intensity`, `rate`, `word_timing`, `tag`.
- New `GET /api/say` that takes text plus controls and returns the timed wav; the UI calls it instead of the plain `/api/tts` when
  expression is on.
- UI: expression meter next to the tracking badge during Listen; on result, a pill "delivered angry 70% · rate 0.9x", the
  audio tags actually sent, and a compact word-timing strip. Toggle "expressive delivery" in settings; an emotion override
  dropdown for the demo so the judge can compare the same sentence at forced angry vs warm.
- Live comparison: a "Replay as ..." row with buttons neutral / angry / warm that resynthesizes the last sentence from cache.

## Recording session additions (Plan 3 script)

Add a short **expression pass**: mouth four phrase-bank phrases twice each, once angry, once warm, with the prompt telling them
which. Used to calibrate the blendshape thresholds per person and as the demo's test clips.

## Evaluation

- Expression: on the expression pass, label accuracy of the rule mapping (target 80 percent on angry vs warm vs neutral) and
  a plot of intensity over time for one angry and one warm clip.
- Delivery: blind test with three teammates, six clips (2 sentences times 3 emotions), ask them to label the emotion and
  whether it is the same speaker. Target: 5 of 6 emotions right, all "same speaker".
- Timing: mean absolute error between mouthed and delivered word durations before and after stretching; total duration within
  10 percent of the mouthed duration.
- Latency: per-stage timings logged in the result event; median end to end under 3 s in Open Mode, under 1 s in Phrase Mode
  from the pre-synthesized bank.

## Risks

- v3 latency spikes: keep Flash as fallback when v3 exceeds 2 s, dropping emotion for that utterance and saying so in the UI.
- Tag over-acting: v3 can overshoot on `[angry]`; stability and the intensity buckets are the controls, tuned in the blind test.
- Instant clone quality on v3: ElevenLabs notes professional clones are not yet tuned for v3; instant clones are fine. Record
  the clone audio clean and 1 to 2 minutes long.
- Blendshapes during mouthing: brow and eye channels are the reliable ones; verify on the expression pass before trusting the mouth channels at all.
- Stretch artifacts: clamp ratios, prefer the global speed setting, stretch only the residual.
- Network dependence: every utterance calls the API. Pre-synthesized Phrase Mode bank removes this from the primary demo.

## Order of work

1. Face Landmarker with blendshapes in the camera process, live expression meter. Verify angry vs warm separation on yourself. (2 h)
2. Forced alignment and word timing from the video; show the strip in the UI. (1.5 h)
3. ElevenLabs tag and stability mapping, v3/Flash selection, cache keys, `Replay as ...` buttons. Works with stock voices before the upgrade. (1.5 h)
4. Audio alignment and per-word stretch. (2 h)
5. After the plan upgrade: clone, pre-synthesize the phrase bank, run the blind test. (1 h plus recording)

Steps 1 to 4 need no upgrade and no recording session; they can be built and tested on stock voices today.
