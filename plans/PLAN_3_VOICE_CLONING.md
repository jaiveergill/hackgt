# Plan 3: Personal voice output via ElevenLabs (do first)

**Goal.** The laptop speaks the recognized phrase in the patient's own voice. Independent of recognition accuracy, so it ships first.

**Order note.** The 1 to 2 minutes of audio ElevenLabs needs comes out of the same recording session that produces speaker-adaptation data (Plan 2). Run `scripts/record_session.py` once; it writes both.

## Status (2026-09-24 evening)

- `.env` has `ELEVEN_LABS_API_KEY`; TTS verified with the stock voice Bella: 0.37 s per phrase, instant from cache.
- **Blocker for cloning:** the account is pay-as-you-go and the API returns `ivc_not_permitted` / `can_use_instant_voice_cloning: false`. Instant Voice Cloning needs the Starter tier or higher. Until upgraded, the UI offers ElevenLabs stock voices (already wired) and the recording session still captures the audio so cloning is a one-command step later (`POST /api/voice/clone?speaker=<name>`).

## What we use

- ElevenLabs **Instant Voice Cloning** (IVC). Needs roughly 1 to 2 minutes of clean speech; works from 30 s with lower fidelity. Requires a Starter plan or higher.
- TTS model `eleven_flash_v2_5` for low latency (about 75 ms model time, 300 to 600 ms end to end over the API). `eleven_multilingual_v2` if quality matters more than speed.
- Python SDK `elevenlabs`. Key goes in `.env` as `ELEVENLABS_API_KEY` (not present yet).

## Recording (shared with Plan 2)

`scripts/record_session.py --speaker <name>` prompts sentences on screen and records webcam video (25 fps, 640x480) and microphone audio (16 kHz mono WAV) per sentence via ffmpeg avfoundation. Two passes:

1. **Voiced pass**, about 40 sentences, roughly 3 minutes. Read aloud, normal pace, quiet room, 40 to 60 cm from the laptop. Audio feeds the voice clone and Whisper labels; video feeds adaptation.
2. **Silent pass**, the 30 phrase-bank phrases, mouthed only. Feeds adaptation and evaluation. Nothing here goes to ElevenLabs.

Voice clone input: concatenate the voiced-pass WAVs, trim silences over 0.5 s, normalize to -3 dBFS, export one MP3. Skip any take Whisper marks as low confidence or that contains a false start.

## Implementation

Files:

- `silent_running/tts.py`
  - `create_voice(speaker, wav_paths) -> voice_id` using `client.voices.ivc.create(name=..., files=[...])`. Store `{speaker, voice_id, created}` in `data/voices/<speaker>.json`.
  - `synth(text, voice_id) -> mp3 bytes` via `client.text_to_speech.convert(voice_id, text=..., model_id="eleven_flash_v2_5", output_format="mp3_44100_128")`. On-disk cache keyed by sha1 of `(voice_id, text)` in `data/tts_cache/`.
  - `prewarm_phrase_bank(voice_id)`: synthesize all 30 phrases once after cloning. Phrase Mode then plays from cache with zero API latency.
- `silent_running/server.py`
  - `GET /api/tts?text=...&voice=<speaker>` returns cached or fresh MP3.
  - `POST /api/voice/clone` runs `create_voice` on the last session's audio.
  - `STATE["tts"]` gains a third option `"elevenlabs"` and `STATE["voice"]`.
- `static/index.html`
  - Voice selector lists cloned voices plus browser voices. When an ElevenLabs voice is selected, `speak()` plays `<audio src="/api/tts?...">` and falls back to `speechSynthesis` on error.
  - Result panel shows a small pill "spoken in <name>'s voice".
- `scripts/clone_voice.py --speaker <name>` for the command line.

## Steps

1. Add `ELEVENLABS_API_KEY` to `.env`, install `elevenlabs` and `sounddevice` (or use ffmpeg for audio).
2. Write `record_session.py`; verify audio and video land in `data/session/<speaker>/` with a manifest.
3. Record one speaker.
4. Clone, prewarm the phrase bank, listen to three phrases, confirm it sounds like the person.
5. Wire the server route and UI; confirm Phrase Mode plays from cache within about 50 ms of the result.
6. Open Mode: synth on demand; measure API latency and log it in the latency line.

## Evaluation

- Blind check: play two clips, one real recording and one synthesized, to a teammate; they should identify the speaker as the same person.
- Latency: Phrase Mode from cache under 100 ms; Open Mode API call under 1 s.
- Failure mode handled: API down or quota exceeded falls back to the browser voice with a visible pill, never silence.

## Risks

- Consent: only clone people who explicitly agree, and delete the voice from ElevenLabs after the event (`client.voices.delete`). Document in `LICENSES.md`.
- Background noise in the recording room degrades the clone. Record in the quietest spot available, and use a headset mic if one exists.
- IVC on a free tier is not available; confirm the plan before the session.

**Time estimate.** 2 to 3 hours including the recording script that Plan 2 reuses.
