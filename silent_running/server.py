"""Silent Running server: webcam -> face tracking -> pretrained VSR -> constrained/contextual decoding -> UI.

  python -m silent_running.server [--camera 0] [--port 8000] [--device mps] [--llm gpt-4o-mini]
"""
import os, sys, json, time, asyncio, threading, argparse, subprocess, re
sys.setswitchinterval(0.0005)  # many tiny torch ops on MPS/CPU must not wait 5 ms behind the camera thread each
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.vsr import VSREngine, load_phrases, load_phrase_table, _read_video
from silent_running.camera_proc import CameraProcess
from silent_running.context import ContextStore, OpenAIChooser
from silent_running.decoder import PhraseDecoder
from silent_running import tts as eltts
from silent_running import prosody

STATIC = os.path.join(ROOT, "silent_running", "static")
app = FastAPI(title="Silent Running")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

LLM_MARGIN = 3.0  # nats; an LLM proposal is accepted only if the visual model scores it within this of the raw top hypothesis
PHRASE_GAP_THRESHOLD = 6.0  # nats; free transcript beating every phrase by more than this => "no phrase matched"
STATE = {"mode": "phrase", "auto_listen": False, "auto_speak": True, "tts": "browser", "voice": None, "expressive": True, "emotion_override": None, "llm_enabled": True, "status": "idle", "utt_id": 0}
clients = set()
loop = None
engine = camera = context = phrase_decoder = llm = None
phrases = load_phrases()
PHRASE_TABLE = load_phrase_table()
CRITICAL = {r["phrase"].lower() for r in PHRASE_TABLE if r["critical"]}
CATEGORY_OF = {r["phrase"].lower(): r["category"] for r in PHRASE_TABLE}
CRITICAL_CONF = 0.5
LOG = []  # conversation log: [{ts, who, text, ...}]
work_lock = threading.Lock()


def broadcast(msg):
    data = json.dumps(msg)
    async def _send():
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
    if loop is not None:
        asyncio.run_coroutine_threadsafe(_send(), loop)


def set_status(s, **kw):
    STATE["status"] = s
    broadcast({"type": "status", "status": s, **kw})


def speak_backend(text):
    subprocess.Popen(["say", "-v", "Samantha", text])


# ----------------------------------------------------------------------------- decoding pipeline
def run_decode(rois, n_face, n_total, duration, source="webcam", label=None, t_crop=0.0, expression=None):
    """Full pipeline on an utterance's mouth crops. Emits incremental events so the UI can show progress."""
    with work_lock:
        STATE["utt_id"] += 1
        uid = STATE["utt_id"]
        t0 = time.time() - t_crop
        set_status("processing", utt_id=uid, stage="encode")
        try:
            x = engine.to_model_input(rois)
        except Exception as e:
            set_status("idle")
            broadcast({"type": "error", "utt_id": uid, "message": f"Mouth crop failed: {e}"})
            return None
        t1 = time.time()
        set_status("processing", utt_id=uid, stage="encode")
        enc = engine.encode(x)
        t2 = time.time()
        greedy = engine.ctc_greedy(enc)
        t3 = time.time()
        broadcast({"type": "raw", "utt_id": uid, "stage": "greedy", "text": greedy, "n_frames": int(x.shape[1]), "duration": duration,
                   "latency": {"crop": t1 - t0, "encode": t2 - t1, "greedy": t3 - t2}})
        result = {"utt_id": uid, "mode": STATE["mode"], "raw_greedy": greedy, "n_frames": int(x.shape[1]), "duration": duration, "source": source, "label": label,
                  "expression": expression or {"emotion": "neutral", "intensity": 0.0}}
        LAST["enc"] = enc; LAST["utt_id"] = uid
        if not greedy.strip():
            # CTC saw no speech-like mouth movement at all. The attention decoder would hallucinate fluent text here.
            set_status("idle")
            broadcast({"type": "error", "utt_id": uid, "message": "No speech detected (mouth did not move enough). Mouth the phrase clearly while holding Listen."})
            result.update({"selected": None, "no_speech": True})
            return result
        if STATE["mode"] == "phrase":
            pd = phrase_decoder.decode(enc)
            # how well does the best inventory phrase explain the video compared with the model's own free transcript?
            greedy_score = engine.score_phrases(enc, [greedy])[0]["score"] if greedy.strip() else None
            best_vsr = max(r["vsr_score"] for r in pd["ranking"])
            gap = (greedy_score - best_vsr) if greedy_score is not None else 0.0
            t4 = time.time()
            result.update({"selected": pd["selected"], "confidence": pd["confidence"], "margin": pd["margin"], "visual_top": pd["visual_top"],
                           "context_changed_choice": pd["context_changed_choice"], "ranking": pd["ranking"][:6],
                           "greedy_score": greedy_score, "best_phrase_score": best_vsr, "phrase_gap": gap,
                           "in_inventory": gap < PHRASE_GAP_THRESHOLD,
                           "context": context.snapshot(), "latency_total": t4 - t0})
            result["timing"] = _safe_timing(enc, pd["selected"]); LAST["timing"] = result["timing"]
            set_status("idle")
            broadcast({"type": "result", **result, "latency": {"crop": t1 - t0, "encode": t2 - t1, "phrase": t4 - t3, "total": t4 - t0}})
            sel = pd["selected"]
            result["category"] = CATEGORY_OF.get(sel.lower())
            result["critical"] = sel.lower() in CRITICAL and pd["confidence"] >= CRITICAL_CONF
            if result["critical"]:
                broadcast({"type": "alert", "utt_id": uid, "text": sel, "confidence": pd["confidence"], "ts": time.time()})
            if pd["confidence"] >= 0.6 and gap < PHRASE_GAP_THRESHOLD:
                context.add_history(sel)
                _log("patient", sel, confidence=pd["confidence"], critical=result["critical"], emotion=(expression or {}).get("emotion"))
            if STATE["auto_speak"] and STATE["tts"] == "backend":
                speak_backend(pd["selected"])
            # background: full beam n-best so the UI can show what the open-vocab decoder thought
            threading.Thread(target=_bg_nbest, args=(enc, uid), daemon=True).start()
        else:
            set_status("processing", utt_id=uid, stage="beam")
            nbest = engine.beam_search(enc, 5)
            t4 = time.time()
            probs = _softmax([h["score"] for h in nbest])
            for h, p in zip(nbest, probs):
                h["prob"] = p
            result.update({"nbest": nbest, "selected": _pretty(nbest[0]["text"]), "confidence": probs[0], "context": context.snapshot(), "latency_total": t4 - t0})
            result["timing"] = _safe_timing(enc, nbest[0]["text"]); LAST["timing"] = result["timing"]
            set_status("idle")
            broadcast({"type": "result", **result, "latency": {"crop": t1 - t0, "encode": t2 - t1, "beam": t4 - t3, "total": t4 - t0}})
            context.add_history(nbest[0]["text"])
            _log("patient", _pretty(nbest[0]["text"]), confidence=probs[0], mode="open", emotion=(expression or {}).get("emotion"))
            if STATE["auto_speak"] and STATE["tts"] == "backend":
                speak_backend(nbest[0]["text"])
            if STATE["llm_enabled"] and llm is not None:
                threading.Thread(target=_bg_llm, args=(nbest, uid, enc), daemon=True).start()
        return result


def _bg_nbest(enc, uid):
    try:
        nbest = engine.beam_search(enc, 5)
        probs = _softmax([h["score"] for h in nbest])
        for h, p in zip(nbest, probs):
            h["prob"] = p
        broadcast({"type": "nbest", "utt_id": uid, "nbest": nbest})
    except Exception as e:
        broadcast({"type": "nbest", "utt_id": uid, "error": str(e)})


def _bg_llm(nbest, uid, enc):
    """Open Mode contextual decoding: the LLM PROPOSES one corrected sentence from the n-best + context, the visual
    model VERIFIES it by scoring the proposal against the video. Accepted only if within LLM_MARGIN nats of the raw
    top hypothesis. Both the proposal and the verdict are broadcast so the UI shows exactly what happened."""
    t0 = time.time()
    ctx = context.snapshot()
    ctx["recent_utterances"] = ctx.pop("history", [])
    out = llm.propose([{"text": h["text"], "score": h["score"]} for h in nbest], ctx)
    dt = time.time() - t0
    if not out or "error" in out:
        broadcast({"type": "llm", "utt_id": uid, "error": (out or {}).get("error", "no response"), "latency": dt}); return
    top = nbest[0]["text"].strip().upper()
    prop = out["sentence"]
    if prop == top:
        broadcast({"type": "llm", "utt_id": uid, "proposal": _pretty(prop), "reason": out["reason"], "gap": 0.0, "accepted": True, "changed": False,
                   "corrected": _pretty(top), "latency": dt, "model": llm.model}); return
    with work_lock:
        sc = {r["phrase"]: r["score"] for r in engine.score_phrases(enc, [prop, top])}
    gap = sc[prop] - sc[top]
    accepted = gap >= -LLM_MARGIN
    broadcast({"type": "llm", "utt_id": uid, "proposal": _pretty(prop), "reason": out["reason"], "gap": gap, "accepted": accepted, "changed": True,
               "corrected": _pretty(prop if accepted else top), "latency": time.time() - t0, "model": llm.model})


def _word_edits(a, b):
    a, b = a.lower().split(), b.lower().split()
    d = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, wb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (wa != wb))
    return d[len(b)]


LAST = {"enc": None, "utt_id": 0}


def _log(who, text, **kw):
    rec = {"ts": time.time(), "who": who, "text": text, **kw}
    LOG.append(rec); del LOG[:-200]
    broadcast({"type": "log", "entry": rec})


def nurse_listen(seconds=5.0):
    """Record the laptop mic (the caregiver speaking), transcribe with OpenAI, and drop it into context as the nurse prompt."""
    import tempfile
    set_status("nurse_listening")
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation", "-i", ":0", "-t", str(seconds), "-ar", "16000", "-ac", "1", wav])
    set_status("idle")
    if not os.path.exists(wav):
        broadcast({"type": "error", "message": "microphone capture failed"}); return
    try:
        from openai import OpenAI
        with open(wav, "rb") as f:
            text = OpenAI().audio.transcriptions.create(model="gpt-4o-transcribe", file=f, language="en").text.strip()
    except Exception as e:
        broadcast({"type": "error", "message": f"transcription failed: {e}"}); return
    finally:
        try: os.remove(wav)
        except Exception: pass
    if not text:
        broadcast({"type": "error", "message": "nurse: nothing heard"}); return
    context.update(last_prompt=text)
    _log("nurse", text)
    broadcast({"type": "context", "context": context.snapshot()})


def _safe_timing(enc, text):
    try:
        return prosody.video_word_timing(engine, enc, text)
    except Exception as e:
        print("[timing]", e); return None


def _keep_warm():
    """Apple GPU clocks down after ~1 s idle and the next encode costs 0.5-1.5 s instead of 0.05 s.
    A tiny dummy encode (24 frames) every 0.4 s (0.2 s while listening) keeps the demo path fast."""
    dummy = torch.zeros(1, 24, 88, 88)
    while True:
        try:
            if not work_lock.locked():
                engine.encode(dummy)
        except Exception as e:
            print("[keepwarm]", e)
        time.sleep(0.2 if (camera and camera.listening) else 0.4)


def _softmax(xs):
    xs = np.array(xs, dtype=np.float64)
    e = np.exp(xs - xs.max())
    return (e / e.sum()).tolist()


def _pretty(s):
    s = s.strip().lower()
    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"\bi'", "I'", s)
    return s[:1].upper() + s[1:] if s else s


# ----------------------------------------------------------------------------- HTTP
@app.get("/")
def index():
    return HTMLResponse(open(os.path.join(STATIC, "index.html")).read())


def mjpeg():
    boundary = b"--frame"
    last = None
    while True:
        jpg = camera.preview_jpeg if camera else None
        if jpg is not None and jpg is not last:
            last = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        time.sleep(1 / 20)


@app.get("/stream")
def stream():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/tts")
def api_tts(text: str, voice: str = None):
    """Cloned-voice speech (Plan 3). Returns MP3; 503 if no ElevenLabs key or voice so the UI falls back to the browser voice."""
    from fastapi.responses import Response
    speaker = voice or STATE.get("voice")
    vid = eltts.voice_for(speaker) if speaker else None
    if not vid or not eltts.available():
        return JSONResponse({"error": "no cloned voice available"}, status_code=503)
    try:
        audio, cached, secs = eltts.synth(text, vid)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return Response(content=audio, media_type="audio/mpeg", headers={"X-TTS-Cached": str(cached), "X-TTS-Seconds": f"{secs:.2f}"})


@app.get("/api/say")
def api_say(text: str, emotion: str = "neutral", intensity: float = 0.0, rate: float = 1.0, voice: str = None, retime: int = 1, utt_id: int = 0):
    """Expressive delivery: ElevenLabs (tag/stability/speed from emotion+intensity+rate) then per-word retiming to the
    mouthed word durations of utterance `utt_id` (if it is the last one). Returns WAV + X-Delivery header with the report."""
    from fastapi.responses import Response
    speaker = voice or STATE.get("voice")
    vid = eltts.voice_for(speaker) if speaker else None
    if not vid or not eltts.available():
        return JSONResponse({"error": "no ElevenLabs voice available"}, status_code=503)
    if STATE.get("emotion_override"):
        emotion, intensity = STATE["emotion_override"], max(intensity, 0.7)
    if not STATE.get("expressive", True):
        emotion, intensity = "neutral", 0.0
    t0 = time.time()
    try:
        mp3, cached, secs = eltts.synth(text, vid, emotion=emotion, intensity=intensity, rate=rate)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    plan = eltts.plan_delivery(text, emotion, intensity, rate)
    report = {"emotion": emotion, "intensity": intensity, "rate": rate, "tag": plan["tag"], "model": plan["model"], "stability": plan["settings"]["stability"],
              "cached": cached, "t_synth": round(time.time() - t0, 2), "retime": {"applied": False}}
    wav, sr = prosody.mp3_to_wav(mp3)
    timing = LAST.get("timing") if utt_id and utt_id == LAST.get("utt_id") else None
    if retime and timing and timing.get("words"):
        t1 = time.time()
        try:
            aw = prosody.align_audio_words(wav, sr, [w["word"] for w in timing["words"]])
            wav, rep = prosody.retime(wav, sr, aw, timing["words"], timing["pauses"])
            rep["audio_words"] = aw; rep["t"] = round(time.time() - t1, 2)
            report["retime"] = rep
        except Exception as e:
            report["retime"] = {"applied": False, "reason": str(e)}
    report["total"] = round(time.time() - t0, 2)
    broadcast({"type": "delivery", "utt_id": utt_id, **report})
    return Response(content=prosody.wav_bytes(wav, sr), media_type="audio/wav", headers={"X-Delivery": json.dumps({k: v for k, v in report.items() if k != "retime"})})


@app.post("/api/voice/clone_upload")
async def api_voice_clone_upload(speaker: str, request: Request):
    """Clone from audio recorded in the browser (webm/opus or wav body). Converts with ffmpeg, clones, prewarms in background."""
    body = await request.body()
    if len(body) < 20000:
        return JSONResponse({"error": "recording too short"}, status_code=400)
    d = os.path.join(ROOT, "data", "session", speaker, "voiced"); os.makedirs(d, exist_ok=True)
    raw = os.path.join(d, f"browser_{int(time.time())}.webm"); wav = raw[:-5] + ".wav"
    open(raw, "wb").write(body)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-ar", "16000", "-ac", "1", wav])
    if not os.path.exists(wav):
        return JSONResponse({"error": "could not decode recording"}, status_code=400)
    try:
        rec = eltts.create_voice(speaker, [wav])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    STATE["voice"] = speaker
    broadcast({"type": "state", "state": STATE})
    def _pw():
        n = eltts.prewarm_phrase_bank(rec["voice_id"], phrases)
        broadcast({"type": "prewarmed", "speaker": speaker, "n": n})
    threading.Thread(target=_pw, daemon=True).start()
    return {**rec, "prewarming": True}


@app.get("/api/voices")
def api_voices():
    return {"available": eltts.available(), "can_clone": eltts.can_clone(), "voices": eltts.list_voices(),
            "stock": eltts.stock_voices()[:12], "selected": STATE.get("voice")}


@app.post("/api/voice/clone")
def api_voice_clone(speaker: str):
    """Clone from the voiced pass of data/session/<speaker> and prewarm the phrase bank."""
    sdir = os.path.join(ROOT, "data", "session", speaker, "manifest.jsonl")
    if not os.path.exists(sdir):
        return JSONResponse({"error": "no session for speaker"}, status_code=404)
    wavs = [os.path.join(ROOT, r["wav"]) for r in map(json.loads, open(sdir)) if r.get("wav")]
    if not wavs:
        return JSONResponse({"error": "no voiced takes"}, status_code=400)
    try:
        rec = eltts.create_voice(speaker, wavs)
        STATE["voice"] = speaker
        broadcast({"type": "state", "state": STATE})
        threading.Thread(target=lambda: print("[tts] prewarmed", eltts.prewarm_phrase_bank(rec["voice_id"], phrases)), daemon=True).start()
        n = 0
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    STATE["voice"] = speaker
    broadcast({"type": "state", "state": STATE})
    return {**rec, "prewarmed": n}


@app.get("/api/state")
def api_state():
    return {"state": STATE, "phrases": phrases, "phrase_table": PHRASE_TABLE, "log": LOG[-50:], "context": context.snapshot(), "camera": camera.meta if camera else None,
            "engine": {"device": str(engine.device), "decode_device": str(engine.decode_device), "model": "Auto-AVSR " + os.path.basename(STATE.get("model_dir", "LRS3_V_WER19.1"))},
            "llm": {"model": llm.model if llm else None, "available": bool(llm and llm.available())}}


@app.post("/api/decode_file")
def api_decode_file(path: str, label: str = None):
    """Run the identical pipeline on a video file (for evaluation / fallback demos)."""
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    frames = _read_video(p)[0].numpy()
    t0 = time.time()
    lms = engine.landmarks_for_frames(frames)
    n_face = sum(l is not None for l in lms)
    if n_face < max(4, len(lms) // 4):
        return JSONResponse({"error": f"face not tracked ({n_face}/{len(lms)} frames)"}, status_code=400)
    try:
        rois = engine.mouth_rois(frames, lms)
    except Exception as e:
        return JSONResponse({"error": f"crop failed: {e}"}, status_code=400)
    res = run_decode(rois, n_face, len(lms), len(frames) / 25.0, source=os.path.relpath(p, ROOT), label=label, t_crop=time.time() - t0)
    return res or {"error": "decode failed"}


# ----------------------------------------------------------------------------- WebSocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "state": STATE, "phrases": phrases, "phrase_table": PHRASE_TABLE, "context": context.snapshot(), "log": LOG[-30:]}))
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start":
                try:
                    camera.start_listening()
                    set_status("listening")
                except Exception as e:
                    set_status("idle")
                    await ws.send_text(json.dumps({"type": "error", "message": f"camera error: {e}"}))
            elif cmd == "stop":
                set_status("processing", stage="crop")
                threading.Thread(target=_stop_and_decode, daemon=True).start()
            elif cmd == "mode":
                STATE["mode"] = msg.get("mode", "phrase")
                broadcast({"type": "state", "state": STATE})
            elif cmd == "settings":
                for k in ("auto_speak", "tts", "llm_enabled", "voice", "expressive", "emotion_override"):
                    if k in msg: STATE[k] = msg[k]
                if "auto_listen" in msg and camera:
                    STATE["auto_listen"] = bool(msg["auto_listen"]); camera.set_auto(STATE["auto_listen"])
                broadcast({"type": "state", "state": STATE})
            elif cmd == "context":
                context.update(notes=msg.get("notes"), category=msg.get("category"), last_prompt=msg.get("last_prompt"))
                broadcast({"type": "context", "context": context.snapshot()})
            elif cmd == "speak":
                speak_backend(msg.get("text", ""))
            elif cmd == "nurse":
                threading.Thread(target=nurse_listen, args=(float(msg.get("seconds", 5)),), daemon=True).start()
            elif cmd == "nurse_text":
                t = (msg.get("text") or "").strip()
                if t:
                    context.update(last_prompt=t); _log("nurse", t); broadcast({"type": "context", "context": context.snapshot()})
            elif cmd == "confirm":  # user/nurse confirmed a phrase (adds to history)
                context.add_history(msg.get("text", ""))
                broadcast({"type": "context", "context": context.snapshot()})
            elif cmd == "save_sample":
                _save_sample(msg.get("phrase", ""), msg.get("speaker", "unknown"))
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


def _stop_and_decode():
    try:
        u = camera.stop_listening()
    except Exception as e:
        set_status("idle"); broadcast({"type": "error", "message": f"camera error: {e}"}); return
    _decode_utterance(u)


def _on_auto_utterance(u):
    set_status("processing", stage="auto")
    threading.Thread(target=_decode_utterance, args=(u, "webcam-auto"), daemon=True).start()


def _decode_utterance(u, source="webcam"):
    if u["rois"] is None:
        set_status("idle")
        msgs = {"too short": "Utterance too short. Hold Listen while you mouth the phrase.", "face not tracked": f"Face not tracked well enough ({u['n_face']}/{u['n_total']} frames). Face the camera and try again."}
        broadcast({"type": "error", "message": msgs.get(u["error"], u["error"] or "capture failed")})
        return
    run_decode(u["rois"], u["n_face"], u["n_total"], u["duration"], source=source, t_crop=u["t_crop"], expression=u.get("expression"))


def _save_sample(phrase, speaker):
    """Save the last utterance window as an eval sample (same format as scripts/record_samples.py)."""
    import cv2
    frames = camera.snapshot_last(6.0)
    if frames is None:
        return
    d = os.path.join(ROOT, "data", "eval", speaker)
    os.makedirs(d, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_") or "sample"
    k = 0
    while os.path.exists(os.path.join(d, f"{slug}__{k}.mp4")):
        k += 1
    fn = os.path.join(d, f"{slug}__{k}.mp4")
    tmp = fn + ".raw.mp4"
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -an -r 25 -c:v libx264 -crf 18 -pix_fmt yuv420p "{fn}" && rm "{tmp}"')
    with open(os.path.join(ROOT, "data", "eval", "manifest.jsonl"), "a") as f:
        f.write(json.dumps({"speaker": speaker, "phrase": phrase, "file": os.path.relpath(fn, ROOT), "n_frames": len(frames), "fps": 25, "ts": time.time(), "source": "ui"}) + "\n")
    broadcast({"type": "saved", "file": os.path.relpath(fn, ROOT), "phrase": phrase})


@app.websocket("/ws_meta")
async def ws_meta(ws: WebSocket):
    """Lightweight 10 Hz tracking metadata stream."""
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps({"type": "meta", **(camera.meta if camera else {}), "status": STATE["status"]}))
            await asyncio.sleep(0.1)
    except Exception:
        pass


@app.on_event("startup")
async def _startup():
    global loop
    loop = asyncio.get_event_loop()


def main():
    global engine, camera, context, phrase_decoder, llm
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=-1, help="camera index; -1 = auto-detect first live camera")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--decode-device", default="cpu")
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--llm", default="gpt-4o-mini")
    ap.add_argument("--model-dir", default=None, help="VSR checkpoint dir (default models/LRS3_V_WER19.1; use models/adapted_<name> after scripts/adapt.py)")
    ap.add_argument("--voice", default=None, help="default TTS voice: cloned speaker name or stock ElevenLabs voice name (e.g. Bella)")
    ap.add_argument("--no-camera", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    engine = VSREngine(device=args.device, decode_device=args.decode_device, beam_size=10, ctc_weight=args.ctc_weight, **({"model_dir": args.model_dir} if args.model_dir else {}))
    STATE["model_dir"] = os.path.relpath(args.model_dir, ROOT) if args.model_dir else "models/LRS3_V_WER19.1"
    if args.voice:
        STATE["voice"] = args.voice
    engine.warmup()
    print(f"[engine] loaded + warmed in {time.time()-t0:.1f}s (encoder on {engine.device}, decoder on {engine.decode_device})")
    context = ContextStore()
    phrase_decoder = PhraseDecoder(engine, phrases, context, gamma=args.gamma)
    llm = OpenAIChooser(model=args.llm)
    print(f"[llm] openai {args.llm} available={llm.available()}")
    if not args.no_camera:
        camera = CameraProcess(index=args.camera)
        camera.on_auto_utterance = _on_auto_utterance
        for _ in range(400):  # child imports torch/mediapipe first (~10-20 s)
            if camera.opened: break
            time.sleep(0.1)
        print(f"[camera] opened={camera.opened} index={camera.index}")
    threading.Thread(target=_keep_warm, daemon=True).start()
    def _prewarm_aligner():
        try:
            t = time.time(); prosody._aligner(); print(f"[prosody] MMS aligner ready in {time.time()-t:.1f}s")
        except Exception as e:
            print("[prosody] aligner prewarm failed:", e)
    threading.Thread(target=_prewarm_aligner, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
