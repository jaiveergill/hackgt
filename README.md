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

## ESP32-CAM glasses: use the USB cable, not WiFi

Over WiFi the board's PCB antenna sits against the wearer's head: the link's capacity drops below what the stream needs,
TCP stalls on lost packets and the picture freezes for seconds. If the board is wired to the laptop anyway (ESP32-CAM-MB USB
board), send the frames down the cable: no radio, nothing to shadow.

1. Flash `firmware/usb_cam/usb_cam.ino` once (it replaces the WiFi sketch; WiFi stays off). Arduino IDE: board package
   "esp32" by Espressif, board "AI Thinker ESP32-CAM", the `/dev/cu.usbserial-*` port, Upload. Or:
   `arduino-cli compile --fqbn esp32:esp32:esp32cam firmware/usb_cam && arduino-cli upload -p /dev/cu.usbserial-XXXX --fqbn esp32:esp32:esp32cam firmware/usb_cam`
   (if the upload can't connect, hold IO0 and tap RST on the board).
2. Run `python -m silent_running.server --source serial` (the one USB-serial port; or `serial:/dev/cu.usbserial-XXXX`).

Measured on the board (ESP32-CAM-MB on macOS, `scripts/bench_link.py serial`): 20-27 fps with no freezes at all (longest
gap 0.05-0.11 s over 4 x 15 s runs; WiFi logs showed 1-3.4 s freezes), 5-7 KB per HVGA frame depending on the scene. The
cable carries ~150 KB/s at 1.5 Mbaud, so bigger frames lower the frame rate instead of freezing it. 2 Mbaud lost a byte in
60% of frames, 1.5 Mbaud 2%, 1 Mbaud 1%: a damaged frame is dropped by its CRC-32, never read smeared.
This board's camera is an OV3660 (PID 0x3660 at 0x3C), not an OV2640: its `set_res_raw` takes the array window and timing
registers, so the zoom is computed for it (`sources.ov3660_window`; the board answers `sensor` with its PID, or `/greg` over
WiFi). 2x reads the centre half of the view binned 1:1 instead of scaled by half: 2x the pixels across the mouth at the same
timing, measured 29-32 fps on the board, switched live in 0.2-0.3 s by the **1x / 2x** toggle on the camera view
(`POST /api/zoom?zoom=2`). Byte loss on the cable is host-load dependent: 0% with the camera alone, 2-6% of frames with the
full app running (the bytes never reach macOS's serial buffer: the CH340 or Apple's driver drops them); each such frame is
dropped by its CRC-32.
## Session logs (read these when something lagged)

Every server start writes `logs/session_<timestamp>.jsonl` (`logs/latest.jsonl` points at the newest). It records, every 5 s,
the capture process's frame stats (fps, p95 interval, gaps over 0.3 s, worst gap, face rate), a 3-packet ping to a network
camera, and host load; plus every decode (latency breakdown, selection, confidence, early commit), streaming partial, voice
delivery, alert, stall/reopen and error. Summarize a session with:

```bash
python scripts/session_report.py              # latest session
python scripts/session_report.py --events     # plus a timeline of decodes, stalls and errors
```
For a network camera it also shows the stream's demand (Mbit/s, KB per frame): freezes start where the WiFi link's capacity
drops below it (e.g. a head shadowing the ESP32's antenna). To compare stream settings on the real link, wear the glasses
and run `python scripts/bench_link.py 172.20.10.2`: current vs `window=2x` vs `window=2x&quality=20`, in alternating
rounds, with frame rate, freezes per minute, Mbit/s, pings and mouth pixels per setting.

## Capture log: shared recognition data (commit this one)

Every live camera utterance (webcam, USB, ESP32 `stream:`) is saved to `data/captures/`: the 96x96 mouth crops the model read
(lossless, ~170 KB each) plus the ranking, decision and context, one JSONL file per server run so teammates never conflict.
Under each result the UI asks **Was this right?** (✓, or type what was actually said). Push it so everyone's sessions add up:

```bash
git add data/captures && git commit -m "captures" && git push
python scripts/captures.py stats      # accuracy at the time, top confusions
python scripts/captures.py rescore    # re-read every labelled clip with this checkout: what a change fixed / broke
```
`SR_CAPTURE=0` turns it off; `SR_CAPTURE=all` also captures `--source file:...` playback (e.g. a phone recording).

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
  ("match my face": MediaPipe blendshapes -> angry / warm / sad / surprised + intensity -> v3 audio tag + stability), and with
  "match my pace" (off by default: aligning the audio delays the voice, `plans/PLAN_5_LATENCY.md`; its 1.5 GB voice aligner loads
  only when you switch it on) your mouth sets the timing
  (CTC forced alignment on the lip-reading model -> per-word durations and pauses -> speed setting + per-word retiming of the
  returned audio). Neutral speech at the natural pace plays from a voice bank the page decodes when the voice is picked (every
  phrase the server has cached); other text streams from ElevenLabs as it is generated (`/api/tts_stream`).
  The delivery panel shows mouthed-vs-delivered word strips and "replay as" buttons for side-by-side judging.
  Clone your own voice from a recording session with `POST /api/voice/clone?speaker=<name>` (Creator tier or higher).
* **Context panel**: what the nurse just asked (yes/no questions boost Yes/No), a patient category, free-text
  notes (keyword overlap boosts phrases), and recent history. Context only re-weights VSR-supported candidates.
* **Eval drawer**: save the last 6 s of webcam as a labelled sample for `scripts/eval.py`.
* **Hands-free listening** ends an utterance once the mouth has been still for 0.35 s (`SR_AUTO_HANG` to change it;
  `scripts/eval_hang.py` measures splits and false triggers per value). Its frames are fixed 0.15 s into that stillness,
  so the model reads it during the rest of the hang and the result shows as the hang runs out (`scripts/check_ahead.py`).
  Latency, measured end to end with `scripts/eval_latency.py`: `plans/PLAN_5_LATENCY.md`.
* **Nurse** (mic button): records until the nurse pauses for 0.7 s (5 s at most), then transcribes.

* **Busy laptop?** `SR_HANDS=0` turns hand gestures off. They cost 13.5 ms on the frames they run on (every 2nd), 6.7 ms per
  frame on average (face tracking: 7 ms), so they only matter for the frame rate when the CPU is saturated; the ESP32's
  ~18 fps is set by the WiFi link, not the laptop.

## Tips that matter for accuracy

* Face the camera squarely, ~40-60 cm away, mouth well lit from the front (no backlight).
* Watch **mouth N/45 px** on the camera badge: the mouth's width in camera pixels vs the 45 px the model's crop reads.
  Amber (under 45) means the crop is upsampled, i.e. blurred: move closer or zoom in. Half the pixels cut Phrase Mode
  top-1 from 64% to 38% on MIRACL clips. Each result, the session log and the capture log record it too.
* ESP32-CAM zoom: the **1x / 2x** toggle on the camera view, or `window=2x` in the source spec (`serial?window=2x`,
  `stream:172.20.10.2?window=2x`). 2x makes the OV3660 read only the centre half of its view, 2x the pixels across the
  mouth at the same frame rate; that is the most detail it gives at this frame rate (beyond 2x frames would only be
  scaled up, so the server refuses). `scripts/check_stream_window.py` and `scripts/check_serial_source.py` check our side
  against a fake OV3660 board.
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
* `scripts/` – `prove_primitive.py`, `record_samples.py`, `eval.py`, `exp_lm_llm.py` (LM / LLM WER experiment),
  `record_session.py` + `prep_session.py` + `adapt.py` (speaker adaptation and voice-clone data, see `plans/`)
* `silent_running/tts.py` – ElevenLabs TTS with cache and voice cloning
* `plans/` – the three plan documents (LLM context, speaker adaptation, voice cloning) with measured status
* `third_party/` – Chaplin (pipeline + vendored ESPnet, MPS patch), Auto-AVSR (SentencePiece model), VSR-multi
