"""ElevenLabs voice cloning + TTS with an on-disk cache (Plan 3).

  create_voice(speaker, wav_paths)    -> voice_id   (Instant Voice Clone; stored in data/voices/<speaker>.json)
  synth(text, voice_id)               -> mp3 bytes  (cached by sha1(voice_id, text) in data/tts_cache/)
  synth_stream(text, voice_id, ...)   -> the same, chunk by chunk as it is generated
  prewarm_phrase_bank(voice_id)       -> synthesizes every phrase in phrases.txt so Phrase Mode plays instantly
"""
import os, json, hashlib, subprocess, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(ROOT, "data", "voices")
CACHE_DIR = os.path.join(ROOT, "data", "tts_cache")
TTS_MODEL = "eleven_flash_v2_5"
V3_MODEL = "eleven_v3"
TAGS = {"angry": ("[angry]", "[shouting]"), "warm": ("[warm]", "[cheerful]"), "sad": ("[sad]", "[sad] [whispers]"), "surprised": ("[surprised]", "[gasps] [surprised]")}


_CLIENT = None


def _client():
    """One client per process: its HTTPS connection pool stays warm. A client per request paid a TLS handshake each time,
    and the first request after startup paid the SDK import too (the server warms it at startup)."""
    global _CLIENT
    if _CLIENT is None:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
        key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_LABS_API_KEY", "")
        if not key:
            return None
        from elevenlabs.client import ElevenLabs
        _CLIENT = ElevenLabs(api_key=key)
    return _CLIENT


def available():
    return _client() is not None


def list_voices():
    out = []
    if os.path.isdir(VOICES_DIR):
        for fn in sorted(os.listdir(VOICES_DIR)):
            if fn.endswith(".json"):
                out.append(json.load(open(os.path.join(VOICES_DIR, fn))))
    return out


def voice_for(speaker):
    """speaker -> voice_id. Accepts a cloned speaker name, a stock voice name (e.g. 'Bella'), or a raw voice_id."""
    p = os.path.join(VOICES_DIR, f"{speaker}.json")
    if os.path.exists(p):
        return json.load(open(p))["voice_id"]
    for v in stock_voices():
        if v["name"].lower() == speaker.lower() or v["voice_id"] == speaker:
            return v["voice_id"]
    return None


_stock_cache = None
def stock_voices():
    """ElevenLabs premade voices (available on every plan), cached for the process."""
    global _stock_cache
    if _stock_cache is None:
        c = _client()
        if c is None:
            return []
        try:
            vs = c.voices.search(page_size=50).voices
            _stock_cache = [{"name": v.name.split(" - ")[0], "voice_id": v.voice_id, "stock": True,
                             "labels": getattr(v, "labels", None) or {}} for v in vs if v.category == "premade"]
        except Exception as e:
            print("[tts] stock voice list failed:", e); return []
    return _stock_cache


def prepare_clone_audio(wav_paths, out_mp3):
    """Concatenate takes, trim silences > 0.5 s, normalize, export one MP3 for cloning."""
    lst = out_mp3 + ".txt"
    with open(lst, "w") as f:
        for w in wav_paths:
            f.write(f"file '{os.path.abspath(w)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-af", "silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB,loudnorm=I=-16:TP=-1.5",
                    "-ar", "44100", "-ac", "1", "-b:a", "128k", out_mp3], check=True)
    os.remove(lst)
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out_mp3]))
    return out_mp3, dur


def create_voice(speaker, wav_paths, description="Silent Running patient voice"):
    c = _client()
    if c is None:
        raise RuntimeError("ELEVENLABS_API_KEY not set in .env")
    os.makedirs(VOICES_DIR, exist_ok=True)
    mp3, dur = prepare_clone_audio(wav_paths, os.path.join(VOICES_DIR, f"{speaker}_clone_input.mp3"))
    # The python SDK (2.69) sends a `labels` field the API rejects ("Labels must be serialized dictionary object"),
    # so call the REST endpoint directly.
    import httpx
    key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_LABS_API_KEY", "")
    with open(mp3, "rb") as f:
        r = httpx.post("https://api.elevenlabs.io/v1/voices/add", headers={"xi-api-key": key}, timeout=120,
                       data={"name": f"SilentRunning-{speaker}", "description": description, "remove_background_noise": "true"},
                       files=[("files", (os.path.basename(mp3), f, "audio/mpeg"))])
    if r.status_code >= 300:
        raise RuntimeError(f"voice add failed {r.status_code}: {r.text[:300]}")
    voice_id = r.json()["voice_id"]
    rec = {"speaker": speaker, "voice_id": voice_id, "created": time.time(), "audio_seconds": round(dur, 1), "n_takes": len(wav_paths)}
    json.dump(rec, open(os.path.join(VOICES_DIR, f"{speaker}.json"), "w"), indent=1)
    return rec


def plan_delivery(text, emotion="neutral", intensity=0.0, rate=1.0):
    """Map (emotion, intensity, rate) -> ElevenLabs request. Neutral uses Flash (fast, no tags); emotions use v3 with audio tags.
    intensity 0..1 -> stability (lower = more expressive) in two buckets; >0.8 adds the stronger tag."""
    emotion = emotion if emotion in TAGS and intensity > 0 else "neutral"
    if emotion == "neutral":
        return {"model": TTS_MODEL, "text": text, "tag": "", "settings": {"stability": 0.5, "similarity_boost": 0.8, "speed": float(rate)}, "bucket": "n"}
    mild, strong = TAGS[emotion]
    tag = strong if intensity > 0.8 else mild
    stab = 0.65 if intensity < 0.5 else 0.35
    return {"model": V3_MODEL, "text": f"{tag} {text}", "tag": tag, "settings": {"stability": stab, "similarity_boost": 0.8, "speed": float(rate)}, "bucket": f"{emotion}{'2' if intensity > 0.8 else ('1' if intensity >= 0.5 else '0')}"}


def _cache_path(voice_id, text, model=TTS_MODEL, bucket="n", speed=1.0):
    h = hashlib.sha1(f"{voice_id}|{model}|{bucket}|{round(speed, 1)}|{text.strip().lower()}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.mp3")


def _plan(text, voice_id, emotion, intensity, rate):
    """-> (ElevenLabs request plan, its cache file)"""
    plan = plan_delivery(text, emotion, intensity, rate)
    return plan, _cache_path(voice_id, text, plan["model"], plan["bucket"], plan["settings"]["speed"])


def _store(path, audio):
    """Write a cache file atomically: a concurrent reader never sees half a clip."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".part")
    with os.fdopen(fd, "wb") as f:
        f.write(audio)
    os.replace(tmp, path)


def cached(text, voice_id, emotion="neutral", intensity=0.0, rate=1.0):
    """The cached mp3 for this request, or None."""
    p = _plan(text, voice_id, emotion, intensity, rate)[1]
    return open(p, "rb").read() if os.path.exists(p) else None


def synth(text, voice_id, use_cache=True, emotion="neutral", intensity=0.0, rate=1.0):
    """Returns (mp3_bytes, from_cache, seconds). Emotion/intensity/rate map to a v3 audio tag + stability + speed."""
    plan, p = _plan(text, voice_id, emotion, intensity, rate)
    if use_cache and os.path.exists(p):
        return open(p, "rb").read(), True, 0.0
    c = _client()
    if c is None:
        raise RuntimeError("ELEVENLABS_API_KEY not set in .env")
    t0 = time.time()
    audio = b"".join(c.text_to_speech.convert(voice_id=voice_id, text=plan["text"], model_id=plan["model"], output_format="mp3_44100_128",
                                              voice_settings=plan["settings"]))
    _store(p, audio)
    return audio, False, time.time() - t0


def synth_stream(text, voice_id, emotion="neutral", intensity=0.0, rate=1.0):
    """synth(), yielded chunk by chunk as ElevenLabs generates it, for a request that is not cached. v3 (the emotions)
    streams too: "[angry] I can't breathe" started 0.58 s after the request, and the whole clip took 0.85-1.0 s. The clip is
    cached only once the stream has completed: one the client dropped or that failed upstream is not."""
    plan, p = _plan(text, voice_id, emotion, intensity, rate)
    c = _client()
    if c is None:
        raise RuntimeError("ELEVENLABS_API_KEY not set in .env")
    chunks = []
    for chunk in c.text_to_speech.stream(voice_id=voice_id, text=plan["text"], model_id=plan["model"], output_format="mp3_44100_128",
                                         voice_settings=plan["settings"]):
        chunks.append(chunk)
        yield chunk
    _store(p, b"".join(chunks))


def prewarm_phrase_bank(voice_id, phrases=None, expressive=True):
    """Pre-synthesize every phrase: neutral + (angry, warm, sad) x (mild, strong) so Phrase Mode plays instantly."""
    if phrases is None:
        phrases = [l.strip() for l in open(os.path.join(ROOT, "silent_running", "phrases.txt")) if l.strip() and not l.startswith("#")]
    variants = [("neutral", 0.0)] + ([(e, i) for e in ("angry", "warm", "sad") for i in (0.6, 0.9)] if expressive else [])
    done = 0
    for ph in phrases:
        for e, i in variants:
            try:
                _, cached, _ = synth(ph, voice_id, emotion=e, intensity=i)
                done += 0 if cached else 1
            except Exception as ex:
                print("[tts] prewarm failed", ph, e, ex)
    return done
