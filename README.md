# Silent Running

Laptop-based assistive communication: the user faces the webcam, **silently mouths a phrase**, and the laptop
recognizes it with a pretrained visual speech recognition (VSR) model and speaks it aloud.

```
webcam → MediaPipe face tracking → mouth ROI (96×96, 25 fps) → Auto-AVSR visual conformer (LRS3, 19.1% WER)
      → { CTC greedy | beam-search n-best } → the LLM (Grok) proposes what was meant, with the context
      → the visual model verifies each proposal against the video → the verified sentence → TTS
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

The interpreter is Grok by default (`XAI_API_KEY=...` in `.env`, model `grok-4.20-0309-non-reasoning`); `--llm-provider openai` uses `OPENAI_API_KEY` and `gpt-4o-mini`. It proposes up to 3 sentences the patient most plausibly meant, the lip model scores each against the video, and the most likely one within 3 nats of the raw reading is used (none: the raw reading stands).
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
full app running; each such frame is dropped by its CRC-32. Root cause: the CH340 buffers 32 bytes (~0.2 ms at 1.5 Mbaud) and
macOS drains it from a user-space driver (com.apple.DriverKit-AppleUSBCHCOM) that competes for the CPU; when it runs late,
bytes are lost, and sometimes it wedges until the port is reopened (a standalone reader under the same load saw both).
The server runs at lower priority with `--source serial` so the driver wins; the reopen is detected after 0.5 s. WCH's
own driver made it worse (24% damaged). A CP2102/FTDI adapter (hundreds of bytes of buffer) would remove it at the source.
## Charge nurse unit board (`/dashboard`)

The provider side: the **Nurse board** tab in the bedside UI (`http://127.0.0.1:8000/`), or `http://127.0.0.1:8000/dashboard` on its own. One tile per bed
with the bed number, patient initials, a status colour (green calm, yellow request, red urgent), the last thing the patient
mouthed and how long ago; the live bed's tile is the glasses camera. **Bed 4 is the real patient; the other six beds are
simulated** so the unit looks like a unit, and the board says so. Requests are listed by urgency with an **Acknowledge** button
and the time-to-acknowledge on every one; the top bar shows open requests, average response time and requests per bed today.
Clicking a bed opens its detail: the live camera, the laptop's ambient overview, and today's transcript at that bedside, the
patient's words and the nurse's (typed in the bedside UI, transcribed from the mic, or a note added on the board), timestamped:
the bedside conversation documents itself.

* `--bed 4 --initials J.G.` name the live bed; `--no-simulate` turns the simulated beds' activity off (they still exist).
* `--ambient auto|<index>|none`: the ambient overview is the laptop's own camera (`/ambient`), preview only, no recognition.
  It picks the first camera that delivers frames (with an iPhone paired, index 0 can be a Continuity Camera that never does).
* Everything the board shows is appended to `data/unit/<date>.jsonl` and replayed at startup, so "today" survives a restart.
  Delete the day's file to start the demo clean. `GET /api/unit` is the snapshot; `POST /api/unit/ack?alert=N` acknowledges.

### The Impiricus layer (HackGT 13 "Invent the next way we engage HCPs")

Impiricus has no public API, so the Impiricus side is **simulated and labelled so on every screen**; the shapes follow their own
products (`silent_running/ascend.py`):

* **Spark trigger out.** Spark runs engagement journeys off real-world events. Every request on the board becomes a de-identified
  `bedside_request` trigger (unit, bed, category, urgency; then `bedside_request_acknowledged` with the time to acknowledge).
  `--ascend-webhook URL` POSTs them for real; without it they are journaled as simulated. `GET /api/ascend` is the journey.
* **Ascend resource in.** On a bed's detail the nurse gets one Ascend-style resource for the open request's category (treatment
  information, dosing calculator, patient resource, or a Wallet card to forward to the family), medical-affairs material only,
  never in the alert path. Opening it, asking a medical science liaison, or sending the Wallet card is the engagement
  (`POST /api/ascend/engage`), journaled and documented on the bed. The top bar counts engagements.
* **ION next best action.** A transparent rule over today's requests at the bed (most frequent category, at least twice).

The pitch: Impiricus reaches the prescriber (DocUpdate is prescriber-only, outpatient). The board reaches the bedside care team
at the moment the patient asks, and hands Impiricus the event that started it.

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

* Hold **HOLD TO LISTEN** (or the space bar), silently mouth anything, release (or turn on hands-free listening).
  The lip-reading model's beam search n-best appears with probabilities; the LLM (Grok) *proposes* up to 3 sentences the
  patient most plausibly meant from those hypotheses plus the context, and the visual model *verifies* them by scoring each
  against the video: the most likely one within 10 nats of the raw top hypothesis is spoken, else the raw top. The UI shows
  the proposals, the verdict, the score gaps and the LLM's reason. Raw model output is always displayed. Measured on a
  captioned TED clip: WER 0.273 raw, 0.227 after verified correction (`plans/PLAN_1_LLM_CONTEXT.md`).
* **Critical phrases**: when what the patient is about to be heard saying is one of the critical phrases in
  `silent_running/phrases.txt` (marked ` !`, e.g. "I can't breathe"), a full-screen alert announces it twice, urgently.
* **Expressive delivery** (`plans/PLAN_4_EXPRESSIVE_DELIVERY.md`): with an ElevenLabs voice selected, your face sets the emotion
  ("match my face": MediaPipe blendshapes -> angry / warm / sad / surprised + intensity -> v3 audio tag + stability), and with
  "match my pace" (off by default: aligning the audio delays the voice, `plans/PLAN_5_LATENCY.md`; its 1.5 GB voice aligner loads
  only when you switch it on) your mouth sets the timing
  (CTC forced alignment on the lip-reading model -> per-word durations and pauses -> speed setting + per-word retiming of the
  returned audio). Neutral speech at the natural pace plays from a voice bank the page decodes when the voice is picked (every
  phrase the server has cached); other text streams from ElevenLabs as it is generated (`/api/tts_stream`).
  The delivery panel shows mouthed-vs-delivered word strips and "replay as" buttons for side-by-side judging.
  Clone your own voice from a recording session with `POST /api/voice/clone?speaker=<name>` (Creator tier or higher).
* **Context panel**: what the nurse just asked, a topic, free-text notes, and recent history, all sent to the LLM with the
  lip readings. The LLM only proposes; the visual model decides what the video supports.
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
  Amber (under 45) means the crop is upsampled, i.e. blurred: move closer or zoom in. Half the pixels cut phrase
  top-1 from 64% to 38% on MIRACL clips. Each result, the session log and the capture log record it too.
* ESP32-CAM zoom: the **1x / 2x** toggle on the camera view, or `window=2x` in the source spec (`serial?window=2x`,
  `stream:172.20.10.2?window=2x`). 2x makes the OV3660 read only the centre half of its view, 2x the pixels across the
  mouth at the same frame rate; that is the most detail it gives at this frame rate (beyond 2x frames would only be
  scaled up, so the server refuses).
* Camera mounted sideways: the **↻** button on the camera view turns the image 90° clockwise per click (live, on the
  laptop: any camera), or `rotate=90|180|270` in the source spec (`serial?rotate=90`). Tracking and crops see it upright. `scripts/check_stream_window.py` and `scripts/check_serial_source.py` check our side
  against a fake OV3660 board.
* Mouth at normal or slightly slower pace with clear articulation; hold Listen a beat before and after.
* Utterances of 1-3 s work best; the model saw 25 fps TED talks, so keep the head reasonably still.

## Evaluation

```bash
python scripts/record_samples.py --speaker alice --reps 2       # prompts each phrase, saves data/eval/...
python scripts/eval.py --tag baseline --device mps               # results/baseline.jsonl + results/summary.jsonl
```
Each result line has the raw transcript (beam top), n-best, top-1/top-3 correctness (in the beam's top 3) and per-stage latency.

## Layout

* `silent_running/vsr.py` – engine: preprocessing, encoder (MPS), CTC greedy, beam search, scoring of texts (verification)
* `silent_running/context.py` – context store and the LLM interpreter (Grok / OpenAI)
* `silent_running/camera.py` – capture thread with per-frame face tracking and 25 fps utterance resampling
* `silent_running/server.py` – FastAPI app (MJPEG preview, websocket events, `/api/decode_file`)
* `silent_running/static/index.html` – bedside UI
* `scripts/` – `prove_primitive.py`, `record_samples.py`, `eval.py`, `exp_lm_llm.py` (LM / LLM WER experiment),
  `record_session.py` + `prep_session.py` + `adapt.py` (speaker adaptation and voice-clone data, see `plans/`)
* `silent_running/tts.py` – ElevenLabs TTS with cache and voice cloning
* `plans/` – the three plan documents (LLM context, speaker adaptation, voice cloning) with measured status
* `third_party/` – Chaplin (pipeline + vendored ESPnet, MPS patch), Auto-AVSR (SentencePiece model), VSR-multi
