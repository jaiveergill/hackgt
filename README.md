# Silent Running

Laptop-based assistive communication: the user faces the webcam, **silently mouths a phrase**, and the laptop
recognizes it with a pretrained visual speech recognition (VSR) model and speaks it aloud.

```
webcam → MediaPipe face tracking → mouth ROI (96×96, 25 fps) → Auto-AVSR visual conformer (LRS3, 19.1% WER)
      → { CTC greedy | beam-search n-best | exact log-likelihood of every inventory phrase }
      → contextual reranking (transparent prior; optional local LLM chooser) → selected text → TTS
```

No audio is captured or used anywhere. No lip-reading model was trained; see `RESEARCH.md` for the model
selection and `LICENSES.md` for provenance and restrictions (the checkpoint is research / non-commercial).

## Quick start (Apple Silicon Mac)

```bash
cd lipread
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# weights (≈1.2 GB): see models/ — downloaded from HF mirror Amanvir/LRS3_V_WER19.1 (+ Auto-AVSR checkpoint via gdown)
python -m silent_running.server            # http://127.0.0.1:8000
```

Open Mode's LLM uses the OpenAI API (`gpt-4o-mini`); put `OPENAI_API_KEY=...` in `.env`. It never runs on the primary Phrase Mode path.
Voice output: browser voices by default; with `ELEVEN_LABS_API_KEY` in `.env` the UI lists ElevenLabs stock voices (`--voice Bella`), and cloned voices once the account tier allows Instant Voice Cloning (see `plans/PLAN_3_VOICE_CLONING.md`).

## Using the UI

* **Phrase Mode** (primary demo): hold **HOLD TO LISTEN** (or the space bar), silently mouth one of the phrases
  in `silent_running/phrases.txt`, release. Within ~0.5 s the top phrase appears in large type and is spoken.
  The candidate list shows, per phrase, the visual-only probability (grey bar) and the context-adjusted
  probability (green bar) with the reasons for any adjustment. A "weak match" pill appears when the model's own
  free transcript fits the video much better than any inventory phrase.
* **Open Mode**: same capture, but the open-vocabulary beam search n-best is shown with probabilities. Then the
  OpenAI model *proposes* one corrected sentence from those hypotheses plus the context, and the visual model *verifies*
  it by scoring the proposal against the video; it is accepted only within 3 nats of the raw top hypothesis. The UI shows
  the proposal, the verdict and the score gap. Raw model output is always displayed. Measured on a captioned TED clip:
  WER 0.273 raw, 0.227 after verified correction (`plans/PLAN_1_LLM_CONTEXT.md`).
* **Expressive delivery** (`plans/PLAN_4_EXPRESSIVE_DELIVERY.md`): with an ElevenLabs voice selected, your face sets the emotion
  (MediaPipe blendshapes -> angry / warm / sad / surprised + intensity -> v3 audio tag + stability) and your mouth sets the
  timing (CTC forced alignment on the lip-reading model -> per-word durations and pauses -> speed setting + per-word retiming of
  the returned audio). The delivery panel shows mouthed-vs-delivered word strips and "replay as" buttons for side-by-side judging.
  Clone your own voice from a recording session with `POST /api/voice/clone?speaker=<name>` (Creator tier or higher).
* **Context panel**: what the nurse just asked (yes/no questions boost Yes/No), a patient category, free-text
  notes (keyword overlap boosts phrases), and recent history. Context only re-weights VSR-supported candidates.
* **Eval drawer**: save the last 6 s of webcam as a labelled sample for `scripts/eval.py`.

## Tips that matter for accuracy

* Face the camera squarely, ~40-60 cm away, mouth well lit from the front (no backlight).
* Mouth at normal or slightly slower pace with clear articulation; hold Listen a beat before and after.
* Utterances of 1-3 s work best; the model saw 25 fps TED talks, so keep the head reasonably still.

## Evaluation

```bash
python scripts/record_samples.py --speaker alice --reps 2       # prompts each phrase, saves data/eval/...
python scripts/eval.py --tag baseline --device mps               # results/baseline.jsonl + results/summary.jsonl
```
Each result line has the raw transcript, n-best, the phrase ranking, top-1/top-3 correctness and per-stage latency.

## Layout

* `silent_running/vsr.py` – engine: preprocessing, encoder (MPS), CTC greedy, beam search, **phrase scoring**
* `silent_running/decoder.py` – phrase-constrained decoder combining VSR log-likelihood with the context prior
* `silent_running/context.py` – context store, transparent prior, OpenAI chooser
* `silent_running/camera.py` – capture thread with per-frame face tracking and 25 fps utterance resampling
* `silent_running/server.py` – FastAPI app (MJPEG preview, websocket events, `/api/decode_file`)
* `silent_running/static/index.html` – bedside UI
* `scripts/` – `prove_primitive.py`, `record_samples.py`, `eval.py`, `test_phrase_scoring.py`, `exp_lm_llm.py` (LM / LLM WER experiment),
  `record_session.py` + `prep_session.py` + `adapt.py` (speaker adaptation and voice-clone data, see `plans/`)
* `silent_running/tts.py` – ElevenLabs TTS with cache and voice cloning
* `plans/` – the three plan documents (LLM context, speaker adaptation, voice cloning) with measured status
* `third_party/` – Chaplin (pipeline + vendored ESPnet, MPS patch), Auto-AVSR (SentencePiece model), VSR-multi
