# Plan 5: latency, end of mouthing to first audio

Plan and research by Jaiveer Gill (branches `fast-path` / `latency`, 2026-09-25, measured on an M3 Pro). Ported to main on
2026-09-26 (branch `fast-path-port`) with the numbers below re-measured on the demo laptop (Apple M2, 8 GB). The integrator
review of the original branches is in `.conductor/reviews/fast-path.md` (local to the integrator workspace).

## Measured: main vs fast-path-port

Live path, end to end: `scripts/eval_latency.py` plays 40 MIRACL clips (speakers F01 + M01, takes 1-2, resampled to 25 fps,
each between still stretches of face) as the live video source, with hands-free listening on, the real UI in headless Chrome and
the ElevenLabs voice Bella; MIRACL's phrases were added to `phrases.txt` for the run. End of mouthing = the wall time the source
delivered a clip's last frame. Second run of each (warm TTS cache), under the machine-wide lock; the load average (1 min) came
from macOS background daemons and is given per run.

| end of mouthing to ... (median, p90) | main (fcde3db) | fast-path-port |
|---|---|---|
| result / decision event (35 clips with a result) | 1.19 s (1.48) | **0.76 s (0.98)** |
| first audio of a spoken decision (18 clips) | 1.68 s (5.50) | **0.82 s (1.12)** |
| server stages: crop / encode / phrase | 15 / 75 / 221 ms | 3 / 71 / 80 ms |
| load average during the run | 5.4-9.7 | 2.7-6.3 |

With "match my pace" on (the port with main's retiming): first audio 1.47 s (p90 4.35 s); the server spent 0.37-1.79 s per
phrase aligning and retiming the audio (`delivery` events).

Where the 0.43 s to the decision comes from:
- **Hands-free end of utterance, 0.60 -> 0.35 s** (`camera_proc.AUTO_HANG`, `SR_AUTO_HANG`): ~0.28 s. `scripts/eval_hang.py`
  (the real capture worker over 400 MIRACL clips and 60 s of TED speech): the utterance ends 0.60 s after the last mouthing
  frame instead of 0.88 s (median). No phrase split, no tail lost, at any hang from 0.20 to 1.00 s: the motion detector sees
  no pause inside any of these phrases longer than 0.15 s. A frozen mouth of 0.6 s inserted in the middle of a phrase stays
  one utterance at 0.35 s and splits at 0.20 s. Phrases missed (swallowed by a false trigger from the clip-to-clip jump): 3
  at 0.35 s, 16 at 0.60 s. **Not measured: silent mouthing by a patient**, whose pauses may be longer; raise `SR_AUTO_HANG`
  if phrases split.
- **Phrase scoring on the GPU**: ~0.14 s. The shortlist (48 phrases after the CTC prefilter) is scored by a copy of the
  attention decoder on the GPU instead of the CPU (+258 MB of GPU memory). `scripts/check_scorer.py`: same phrase selected
  on 400/400 MIRACL clips without a profile and 240/240 with an enrolled profile (including phrases outside the shortlist that
  the profile supports), scores within 2e-5 nats; decode 98 -> 68 ms offline. Every new shape on the GPU compiles kernels
  (0.1-0.7 s once), so calls are padded to a few shapes that `warmup()` compiles at startup (+8 s): no first call over 100 ms slower than its repeat in 128.
  The prefilter stays at 48: fast-path's 16 cost 3.5 points of top-1 (review, 397 MIRACL clips).
- **Mouth crops computed while mouthing** (`camera_proc.IncrementalCropper`): ~0.012 s. `scripts/check_cropper.py`: identical
  crops to the old batch crop, pixel for pixel, on 405 clips x 5 cases (manual, hands-free with frames past the end, landmark
  gaps, no face, too short); the batch crop took 8 ms median (MIRACL) and grows with the utterance (70-100 ms for 2.6-15 s),
  the cropper's finish 1.9 ms.

The first audio drops further because of the voice path: neutral speech at the natural pace plays from a bank of decoded
Web Audio buffers (every phrase the server has cached for the voice, loaded when the voice is picked), other text streams
from ElevenLabs (`/api/tts_stream`, which caches a clip only once its stream completed: `scripts/check_tts_stream.py`), the
ElevenLabs client is created once and warmed at startup, and `/api/say` returns the synthesized mp3 when there is nothing to
retime. Retiming the voice to the mouthing ("match my pace") is now a separate option, off by default, because it costs
0.4-1.8 s per phrase; the face still sets the emotion.

Accuracy: unchanged. MIRACL through the camera path (FaceLandmarker, auto-listen window, 213-phrase inventory): top-1
238/397 on main and on the port, the same phrase on 397/397.

## What was not ported, and why

- **Deciding while the mouth is still moving** (fast-path's streaming commit). Offline on the same 397 clips (the review's
  replay of fast-path's own `_score_partial`), it committed early on 39 clips, 4 of them before the mouth stopped (2 wrong:
  "Yes" in the middle of "Excuse me", "No" in "I love this game"), and a longer phrase that starts with a shorter one ("I need
  help" / "I need help right now") can be cut. With the fix the review requires (commit only once the mouth is still), it
  fires on 16 of 394 clips at a 0.35 s hang (2 wrong), 0.29 s before the final decode on those: **median gain over all
  utterances 0.00 s, mean 10 ms**, for a second decoding path with a race that decoded one utterance twice. The shorter hang
  above gives the same kind of gain to every utterance.
- **Dropping expression tracking**: under 1 ms per frame, and the face emotion is part of the event contract and the voice.
- **Prefilter 48 -> 16**: see above.
- **Token cache**: tokenizing all 204 phrases takes 0.4 ms.

## Claims in the original plan, checked

| original claim | measured on the M2 |
|---|---|
| mouth crop 0.38 s after release, saves ~0.35 s | the batch crop took 11-19 ms for 1.6-2.5 s utterances (review) and 8 ms median on MIRACL; saves ~12 ms |
| phrase scoring 0.45 s | 221 ms live on main (decode + free-transcript score), 80 ms on the GPU |
| "it answers as soon as it is sure, often before you finish" | before the last mouthing frame on 4/397 clips offline (2 wrong) and 0/40 live; not ported |
| steps 1 + 2 + 4 ~ 0.6 s; with step 3 ~ 0.3-0.4 s | 0.76 s to the decision, 0.82 s to first audio (medians above); step 4 (CTC-blank endpointing) not built |

## Ranked plan (original), status

1. Preload phrases as Web Audio buffers, skip decode/retime on cache hits: **done** (cached phrases only; the bank no longer
   synthesizes misses at page load, which spent ElevenLabs credits on every visit).
2. Crop per frame as frames arrive: **done**, ~12 ms.
3. Score while speaking, commit early: **not ported** (above).
4. CTC-blank endpointing instead of the motion hang: not built. The hang was shortened to 0.35 s instead, on evidence.
5. Speculative pre-synthesis of top-2 over a persistent ElevenLabs WebSocket: not built.
6. Per-patient head on frozen encoder output: not built (enrollment, `silent_running/enroll.py`, covers the per-patient part).

## Research notes (with sources)

- No public visual-only streaming sentence model exists. Meta streaming AV-ASR (Ma et al. 2023) has no code/weights: https://ar5iv.labs.arxiv.org/html/2211.02133. torchaudio Emformer AV-ASR is audio+visual fused only: https://docs.pytorch.org/audio/2.8/tutorials/device_avsr.html
- Auto-AVSR conformer (non-causal conv k=31, full-context attention) cannot be chunked without dynamic-chunk retraining: https://speechbrain.readthedocs.io/en/v1.0.3/tutorials/nn/conformer-streaming-asr.html
- Incremental CTC prefix scoring / commit frontier: https://arxiv.org/html/2605.18222, https://arxiv.org/pdf/2006.14941
- Visual KWS (Transpotter, code public, offline only): https://github.com/prajwalkr/transpotter
- Personalized lip reading with <1M trainable params: https://arxiv.org/abs/2409.00986
- Lightweight word classifiers (TCN, MobileNetV4): https://arxiv.org/abs/2001.08702, https://arxiv.org/html/2508.17894v1
- Web Audio latency: https://padenot.github.io/web-audio-perf/
- ElevenLabs latency guidance and WebSocket stream-input: https://elevenlabs.io/docs/eleven-api/guides/how-to/best-practices/latency-optimization, https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/realtime-tts
- Model-driven end-of-utterance: https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1, https://arxiv.org/abs/2409.19990
