# Research: face-driven expressive delivery in the user's own voice

Date: 2026-09-24. Scope: open-source models and tools only; no code yet.

**Target.** Lip reader gives the words. The video gives per-word timing (rate, pauses). Face landmarks give an emotion label and intensity per utterance. An expressive zero-shot TTS renders the words in the user's cloned voice under those controls. Same sentence mouthed angry vs warm must sound clearly different, end to end under about 3 s.

## Bottom line

- **ElevenLabs is not needed for the core.** Open models now do zero-shot cloning plus explicit emotion and rate control, on the Mac. ElevenLabs remains a quality fallback only; its emotion-capable model (v3) is not real-time and its real-time model (Flash v2.5) has no emotion control, and neither gives per-word timing.
- **Primary TTS: IndexTTS-2.5** (Bilibili). It is the only open model whose control surface matches the spec exactly: an 8-dimensional emotion vector with per-dimension intensity, emotion disentangled from speaker timbre, and a duration factor. MLX port runs faster than real time on Apple Silicon.
- **Fallback TTS: Chatterbox-Turbo** (Resemble, MIT). Fastest and cleanest license, but emotion is a single "exaggeration" knob plus text cues, so angry vs warm is less controllable.
- **Timing** comes from CTC forced alignment on the lip-reading model we already run, then per-word time-stretch of the TTS output. All tools are local and already installed or pip-installable.
- **Expression** comes from MediaPipe Face Landmarker blendshapes (52 ARKit-style coefficients, real time, already installed) mapped by rules to emotion and intensity, with EmotiEffLib as an optional learned cross-check.

## 1. Expressive cloned TTS candidates

| Model | Clone | Emotion control | Rate / duration control | Apple Silicon | Latency (short sentence) | License |
|---|---|---|---|---|---|---|
| **IndexTTS-2 / 2.5** (Bilibili, Aug 2026) | zero-shot, few seconds | 8-dim vector `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` each 0 to 1, `emo_alpha` intensity, emotion from text description, or emotion reference audio; timbre and emotion disentangled by design | `duration_factor` 0.5 to 2.0, plus explicit token-count mode for exact duration | `index-tts-2.5-mlx` (int8 GPT, RTF about 0.45, torch-free); `mlx-indextts` for v2.0 (RTF about 1.3 on M2 Max); PyTorch MPS works but 2.4x slower | about 1 s for a 2 s sentence on MLX int8 | code Apache-2.0; weights under the Bilibili Model Use License (non-commercial without contacting them). Same situation as our VSR checkpoint |
| **Chatterbox-Turbo / Nano / Flash** (Resemble, MIT, updated Jul 2026) | zero-shot, 5 s clip | `exaggeration` 0 to 1 and `cfg_weight`; delivery steered by text (CAPS, punctuation, ellipses); paralinguistic tags `[laugh] [sigh] [chuckle] [gasp]`; no discrete emotion label | none explicit; pacing via punctuation | `chatterbox-mlx` / hybrid-mlx (2.4x faster than MPS), CoreML for Flash, Nano runs 3x real time on CPU | sub-second | MIT, watermark baked in |
| **CosyVoice 3** (Alibaba, Apache-2.0) | zero-shot, cross-lingual | natural-language instruct: emotion, speed, dialect | instruct "speak faster/slower" | PyTorch only; MPS unverified, likely slow | 150 ms streaming on A100; unknown on Mac | Apache-2.0 |
| **Zonos v0.1** (Zyphra, Apache-2.0) | zero-shot | 8-dim emotion vector (happiness, sadness, disgust, fear, surprise, anger, other, neutral) | `speaking_rate`, `pitch_std` conditioning | Linux + NVIDIA only per repo | fast on GPU | Apache-2.0; emotion vector is documented as entangled with other conditioning |
| **OpenAudio S1-mini** (Fish Audio) | zero-shot, 10 to 30 s | 50+ inline emotion tags `(angry) (sad) (whispering) (shouting)` | none explicit | 4B model, GPU class | GPU | Fish Audio Research License, non-commercial |
| **Qwen3-TTS** (Apache-2.0, Jan 2026) | Base model clones | emotion instructions only in CustomVoice, which uses preset voices; clone and instruction cannot be combined | instruct | mlx-audio supports it | fast | Apache-2.0 but wrong control surface for us |
| **ElevenLabs** | IVC needs Starter tier | v3 audio tags `[angry] [warm] [whispers]`; Flash v2.5 has none | `voice_settings.speed` 0.7 to 1.2 only | API | v3 several seconds, Flash 75 ms | proprietary |

Why IndexTTS-2.5 first: the face mapping produces exactly an emotion label plus an intensity; IndexTTS takes that directly as a vector and keeps the speaker's timbre fixed while varying emotion, which is the demo's success criterion. Duration control removes most of the post-hoc stretching. Risk: the MLX 2.5 port must be checked for emotion-vector and duration-factor support (the v2.0 MLX port exposes `emotion` and `emo_alpha`; duration was not documented). If the 2.5 port lacks them, run IndexTTS-2.5 in PyTorch on MPS (about 2.4x slower, still near the budget) or use v2.0 MLX.

Why Chatterbox second: MIT license, sub-second on the Mac, 5 s clone. To get angry vs warm we would combine `exaggeration` (0.3 warm to 0.9 angry), `cfg_weight`, and rewriting the text with caps and punctuation. It will sound different, but less reliably "angry" than a labelled emotion.

Not recommended for this: Zonos (Linux/NVIDIA only, entangled controls), OpenAudio S1 (too large for the laptop), Qwen3-TTS (clone and emotion are separate models), Kokoro and similar (no cloning).

## 2. Per-word timing from the video

We already have CTC log-probabilities from the visual model at 25 fps (40 ms resolution). Tools:

- `torchaudio.functional.forced_align` (confirmed present in torchaudio 2.11 in our venv) aligns the recognized token sequence to the CTC frames. Grouping tokens by the SentencePiece word boundary marker gives per-word start and end frames and inter-word gaps. Cost: about 10 ms.
- Silent mouthing is typically slower than voiced speech and pauses are exaggerated, so the mapping to TTS should be relative: rate factor = mouthed duration / TTS natural duration, clamped to 0.7 to 1.4, and pauses inserted where the gap exceeds about 250 ms.
- Getting the TTS audio to follow the per-word pattern, model-agnostic:
  1. Global: pass the rate factor as IndexTTS `duration_factor` (or exact token count), or as text pacing for Chatterbox.
  2. Fine: align the synthesized audio to its words with `torchaudio.pipelines.MMS_FA` (installed) or WhisperX, then time-stretch each word with WSOLA (`audiotsm`, pure Python, tens of ms) or Rubber Band (`pyrubberband` plus `brew install rubberband`, better quality) and insert the measured silences. Neither is installed yet.
- Per-word stretching beyond about 30 percent sounds unnatural; the global factor should carry most of the rate and per-word stretch only the residual.

## 3. Facial expression to emotion and intensity

- **MediaPipe Face Landmarker** (tasks API in the installed mediapipe 0.10.21; `output_face_blendshapes` confirmed) returns 478 landmarks and 52 blendshape scores per frame at real time on CPU. Relevant coefficients: `browDownLeft/Right`, `browInnerUp`, `browOuterUpLeft/Right`, `eyeSquintLeft/Right`, `eyeWideLeft/Right`, `cheekSquintLeft/Right`, `noseSneerLeft/Right`, `mouthSmileLeft/Right`, `mouthFrownLeft/Right`, `mouthPressLeft/Right`, `jawOpen`.
- Rule mapping, tuned live: valence from smile minus frown minus sneer; arousal from brow movement, eye widening or squint, and jaw activity; anger = brows down plus squint plus sneer or press; warm = smile plus cheek squint with relaxed brows; sad = inner brow up plus frown; surprise = brow outer up plus eyes wide plus jaw open. Intensity = magnitude of the winning combination. Aggregate over the utterance with a robust statistic (median of the top quartile of frames), and weight mouth-region coefficients down because the mouth is busy forming words; brows, eyes, cheeks and nose carry the signal during silent speech.
- Output feeds the TTS directly: IndexTTS vector with the winning dimension set to intensity (and `calm` to 1 minus intensity), or Chatterbox `exaggeration` plus a text cue.
- Learned alternatives, for a cross-check or a fallback if rules are too twitchy: **EmotiEffLib** (formerly HSEmotion, Apache-2.0, ONNX or PyTorch, 8 classes with scores, valence and arousal models, about 10 ms per face); **py-feat** (research-grade AU and emotion detection, better offline than live); **LibreFace** (real-time AU and expression, C# and Python).
- Recommendation: blendshape rules as primary because they are transparent and adjustable in front of judges, EmotiEffLib as a secondary vote to smooth the label.

## 4. Latency budget (target under 3 s from release of Listen)

| Stage | Estimate |
|---|---|
| Lip reading to text (current) | 0.2 to 0.5 s |
| CTC forced alignment for timing | 0.01 s |
| Expression aggregation | 0 (computed during capture) |
| IndexTTS-2.5 MLX int8 for a 2 to 3 s sentence | 1.0 to 1.5 s (Chatterbox-Turbo: about 0.5 s) |
| Align TTS audio (MMS_FA, CPU) and per-word stretch | 0.2 to 0.3 s |
| Total | about 1.5 to 2.5 s |

Phrase Mode shortcut: pre-synthesize every phrase at 4 emotions times 3 intensities once per voice; live work is then only the timing stretch, so delivery is near instant.

## 5. Memory on the 18 GB laptop

VSR model about 1 GB, IndexTTS-2.5 int8 a few GB, MediaPipe negligible. Fits. Run TTS in its own process (MLX), as we already do for the camera, to avoid GIL and Metal contention with the VSR encoder.

## 6. What to verify by hand before committing (next step, not now)

1. `index-tts-2.5-mlx`: install, confirm emotion vector and duration factor are exposed, measure seconds per sentence on this M3 Pro, listen to angry vs calm on a 10 s clone of one of us.
2. Chatterbox-Turbo MLX: same measurement as the fallback.
3. Blendshape stream during silent mouthing: confirm brow and eye coefficients separate an angry face from a warm one while the mouth is moving.
4. Forced alignment on a real silent clip: check word boundaries look sane at 40 ms resolution.

## 7. Licensing summary

- IndexTTS-2.5 weights: Bilibili Model Use License, non-commercial without permission. Acceptable for the hackathon; would need a license for a product, same as the VSR checkpoint.
- Chatterbox: MIT. CosyVoice 3, Zonos, Qwen3-TTS, EmotiEffLib: Apache-2.0. OpenAudio S1-mini: non-commercial research license.
- MediaPipe, torchaudio: Apache or BSD.
- Voice cloning consent: only clone people who agree; keep clone reference audio local; delete after the event.

Sources: IndexTTS repo and paper (index-tts/index-tts, arXiv 2506.21619), mlx-indextts and index-tts-2.5-mlx, Resemble Chatterbox model cards and the Nano/Flash announcement, CosyVoice 3 paper and Fun-CosyVoice3 card, Zyphra Zonos conditioning README, fishaudio/openaudio-s1-mini card, Qwen3-TTS discussions on cloning plus emotion, ElevenLabs v3 audio tags and voice settings docs, MediaPipe Face Landmarker guide, EmotiEffLib, py-feat, LibreFace repos.
