# Plan 5: near-instant latency (2026-09-25)

## Measured today (live server, M3 Pro, 2.6 s utterance, warm GPU, cloned voice)

| Stage | Measured | Note |
|---|---|---|
| Endpointing hang | 0.75 s | HANG 0.6 + POST_ROLL 0.15 in auto mode (camera_proc.py) |
| Mouth crop | 0.38 s | batch VideoProcess after release |
| Encode | 0.07 s | warm MPS |
| Phrase scoring | 0.45 s | CTC prefilter on 204 + attention decoder on top 48 (CPU) |
| TTS cached (/api/say) | 0.15 s | /api/tts cached is 0.02 s; mp3->wav + align costs the rest |
| TTS uncached | 0.4-0.5 s | eleven_flash_v2_5; eleven_v3 emotional 1.85 s |
| Browser audio start | 0.1-0.2 s | <audio> element, fresh fetch |

Perceived: ~1.5-2.5 s from mouth stopping to hearing the phrase.

## Ranked plan

1. Preload all 204 phrases as Web Audio AudioBuffers at page load; play with AudioBufferSourceNode (~10 ms). Skip wav decode/retime on cache hits. 2 h, saves 0.3-0.5 s.
2. Crop per frame in camera_proc as frames arrive (small smoother lag), not batch after release. 2-3 h, saves ~0.35 s.
3. Score while speaking: re-encode growing buffer every ~250 ms on MPS; batched CTC scores for all 204 phrases on GPU; commit on margin + last token emitted; attention rescoring only when margin is small. 5-6 h, saves ~0.5 s and can commit before the mouth stops.
4. CTC-blank endpointing (blanks dominate ~150 ms after last token) instead of the 0.6 s motion hang. 2 h, saves ~0.5 s.
5. Speculative pre-synthesis of top-2 for uncached/emotional lines over a persistent ElevenLabs WebSocket. 1-2 h.
6. Stretch: linear/MLP head on frozen encoder output, 5-10 recordings per phrase per speaker. Sub-5 ms scoring, likely better closed-set accuracy; doubles as the per-patient adapter. 3-4 h.

Target: steps 1+2+4 ≈ 0.6 s; with step 3 ≈ 0.3-0.4 s.

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
