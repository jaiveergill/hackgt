# Model research (2026-09-24)

Goal: strongest *practical* open-weight visual-only speech recognition for a laptop demo.

| Candidate | LRS3 WER (V-only) | Weights | Backbone / cost | License | Verdict |
|---|---|---|---|---|---|
| **Auto-AVSR / VSR-multi `LRS3_V_WER19.1`** | 19.1% | HF mirror + Google Drive | ResNet18-3D + 12-layer Conformer (768d) + 6-layer Transformer decoder, ~250M params, joint CTC/attention | non-commercial research | **Selected.** Runs on CPU/MPS, ESPnet beam search exposes n-best and per-token scores; decoder can be teacher-forced to score arbitrary candidate phrases. Chaplin proves the webcam path. |
| Auto-AVSR `vsr_trlrs2lrs3vox2avsp_base.pth` | 20.3% | Google Drive | same family, Apache-2.0 code | checkpoint terms unclear | Backup. Downloaded. |
| Auto-AVSR `vsr_trlrs3vox2_base.pth` | 24.6% | Google Drive | same | same | Weaker. |
| VALLR (ICCV 2025) | 18.7% | Google Drive | phoneme CTC + fine-tuned LLM | CC BY-NC 4.0 | Marginal WER gain, LLM reconstruction stage adds latency and complexity; no n-best/phrase-scoring path documented. Skipped. |
| DLLM-VSR (2026) | 19.4% (USR 2.0 enc.) | HF jh-y/dllm-vsr | Dream-7B diffusion LM + USR-2.0 Huge encoder; eval quoted at 15-25 min on 8 GPUs | MIT code | Too heavy for laptop / low latency. |
| Llama-AVSR (ICASSP 2025/26) | 34-44% V-only | released | AV-HuBERT Large + Llama-3.1-8B | not stated | Worse V-only WER and 8B LLM. |
| MMS-LLaMA / Zero-AVSR / Omni-AVSR | AV-focused | partial | LLM-based | various | Audio-visual focus; visual-only numbers not competitive or weights unclear. |
| BRAVEn / RAVEn | ~23-24% (Large) | pretrained, needs fine-tune ckpt | ViT-style | research | Auto-AVSR already stronger; no ready inference pipeline. |
| AV-HuBERT | 26-28% | yes | fairseq | research | Older, fairseq dependency pain. |

## Primitive proof (real video, audio stripped)

TED clip `data/samples/ted1_12s.mp4` (12 s, 300 frames @25fps, speaker on stage, 480p):

* Reference captions: "...maybe you will feel like you've learned something. Now I'm going to get started with the opening. I'm going to make a lot of hand gestures. I'm going to do this with my right hand. I'm going to do this with my left. I'm going to adjust my glasses."
* Raw visual-only output: `YOU'VE HEARD SOMETHING I'M GOING TO MAKE A LOT OF HAND GESTURES I'M GOING TO DO THIS WITH MY RIGHT HAND I'M GOING TO DO THIS WITH MY LEFT I'M GOING TO USE JUST MY GLASSES`
* 5th-best hypothesis contained the correct "ADJUST MY GLASSES".

2.6 s excerpt: `I'M GOING TO MAKE A LOT OF HAND GESTURES` (exact). Teacher-forced phrase scoring of 34 candidates reproduced the beam score of the true sentence (-1.91) and ranked it first; the two closest distractors were its 3rd/4th beam hypotheses.

## Latency (M3 Pro, ~2.5 s / 55-65-frame utterance)

Standalone, warm: landmarks 0.7 s (now done live during capture), crop 0.05-0.2 s, encoder **0.05-0.1 s on MPS** (0.9-1.2 s CPU),
CTC greedy <10 ms, phrase scoring 0.2-0.3 s (CPU), beam-10 1.4 s (CPU; MPS is slower for beam search: many tiny ops).

Three things made the *server* path 10-100x slower than standalone until fixed:
1. Capture thread + MediaPipe in the model process starved the thousands of tiny MPS ops of the GIL (encode 1-15 s).
   Fix: camera, tracking and mouth cropping run in a separate process (`camera_proc.py`), only 96x96 crops cross the pipe.
2. The Apple GPU clocks down after ~1-3 s idle; the next encode costs 0.5-1.5 s. Fix: keep-warm dummy encode every 0.4 s.
3. MPS from two threads at once crashes Metal. Fix: a lock around encode.

Result: live end-to-end after releasing Listen (Phrase Mode): **~0.2 s to raw transcript, ~0.4-0.5 s to spoken phrase**
(measured over the websocket with a real webcam capture). Open Mode adds ~1.4 s beam search; the LLM chooser (OpenAI gpt-4o-mini, ~1-2 s) runs afterwards asynchronously and only updates the display.
