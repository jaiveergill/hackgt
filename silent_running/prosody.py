"""Timing: per-word durations from the video (CTC forced alignment) and per-word time-stretch of TTS audio (Plan 4, B + D).

  video_word_timing(engine, enc, text)  -> {"words": [{"word", "start", "end"}], "pauses": [...], "duration": s, "rate": float}
  align_audio_words(wav, sr, words)      -> [{"word", "start", "end"}] on the synthesized audio (MMS forced aligner)
  retime(wav, sr, audio_words, video_words) -> wav stretched per word with video pauses inserted
"""
import io, re, subprocess, tempfile, os
import numpy as np
import torch, torchaudio
import torch.nn.functional as F

FPS = 25.0
WORD_S = 0.32          # rough natural spoken duration per word, for the global rate estimate
PAUSE_MIN = 0.25       # gaps longer than this become explicit pauses
STRETCH_MIN, STRETCH_MAX = 0.8, 1.25   # per-word residual
GLOBAL_MIN, GLOBAL_MAX = 0.7, 1.4      # whole-utterance factor


def video_word_timing(engine, enc, text):
    """Force-align the recognized text to the visual CTC frames (40 ms each). Returns word boundaries in seconds."""
    toks = engine.tokenize(text)
    if not toks:
        return None
    logp = engine.model.ctc.log_softmax(enc.unsqueeze(0)).cpu().float()  # (1, T', V)
    T = logp.size(1)
    tgt = torch.tensor([toks], dtype=torch.int32)
    if len(toks) > T:
        return None
    try:
        ali, scores = torchaudio.functional.forced_align(logp, tgt, blank=0)
    except Exception:
        return None
    ali = ali[0].tolist()
    # frames where each token is emitted (first frame of each non-blank run)
    emit = []
    prev = 0
    for t, a in enumerate(ali):
        if a != 0 and a != prev:
            emit.append(t)
        prev = a
    if len(emit) != len(toks):
        return None
    pieces = [engine.token_list[i] for i in toks]
    words = []
    for i, (p, f) in enumerate(zip(pieces, emit)):
        if p.startswith("▁") or not words:
            words.append({"word": p.replace("▁", ""), "start_f": f, "end_f": f})
        else:
            words[-1]["word"] += p
            words[-1]["end_f"] = f
    # a word ends where the next begins (minus a frame), last word ends at its last emission + 3 frames
    out = []
    for i, w in enumerate(words):
        end_f = words[i + 1]["start_f"] - 1 if i + 1 < len(words) else min(w["end_f"] + 3, T - 1)
        out.append({"word": w["word"], "start": round(w["start_f"] / FPS, 3), "end": round(max(end_f, w["start_f"] + 1) / FPS, 3)})
    pauses = []
    for a, b in zip(out, out[1:]):
        gap = b["start"] - a["end"]
        if gap >= PAUSE_MIN:
            pauses.append({"after": a["word"], "seconds": round(gap, 3)})
    duration = out[-1]["end"] - out[0]["start"]
    natural = len(out) * WORD_S + sum(p["seconds"] for p in pauses)
    rate = float(np.clip(natural / max(duration, 0.2), 0.7, 1.2))
    return {"words": out, "pauses": pauses, "duration": round(duration, 3), "rate": round(rate, 3), "n_frames": T}


_mms = None
def _aligner():
    global _mms
    if _mms is None:
        bundle = torchaudio.pipelines.MMS_FA
        _mms = (bundle.get_model(), bundle.get_dict(), bundle.get_labels())
    return _mms


def mp3_to_wav(mp3_bytes, sr=16000):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes); p = f.name
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", p, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout
    os.remove(p)
    return np.frombuffer(out, dtype=np.float32).copy(), sr


def align_audio_words(wav, sr, words):
    """MMS forced aligner: word boundaries (seconds) of `words` in the synthesized audio."""
    model, dictionary, labels = _aligner()
    norm = [re.sub(r"[^a-z']", "", w.lower()) for w in words]
    norm = [w if w else "a" for w in norm]
    with torch.no_grad():
        x = torch.from_numpy(wav).unsqueeze(0)
        if sr != 16000:
            x = torchaudio.functional.resample(x, sr, 16000)
        emission, _ = model(x)
    tokens = [dictionary.get(c, dictionary.get("*", 0)) for w in norm for c in w]
    lens = [len(w) for w in norm]
    try:
        ali, _ = torchaudio.functional.forced_align(emission, torch.tensor([tokens], dtype=torch.int32), blank=0)
    except Exception:
        return None
    ali = ali[0].tolist()
    ratio = x.size(1) / emission.size(1) / 16000.0  # seconds per emission frame
    starts = []
    prev = 0
    for t, a in enumerate(ali):
        if a != 0 and a != prev:
            starts.append(t)
        prev = a
    if len(starts) != len(tokens):
        return None
    out, k = [], 0
    for w, n in zip(words, lens):
        s = starts[k] * ratio; e = (starts[k + n - 1] + 1) * ratio
        out.append({"word": w, "start": round(s, 3), "end": round(e, 3)}); k += n
    for a, b in zip(out, out[1:]):
        a["end"] = max(a["end"], min(b["start"], a["end"] + 0.08))
    return out


def _stretch(seg, sr, ratio):
    if abs(ratio - 1.0) < 0.03 or len(seg) < int(0.05 * sr):
        return seg
    try:
        import pyrubberband as prb
        return prb.time_stretch(seg, sr, 1.0 / ratio).astype(np.float32)  # rubberband rate>1 = faster
    except Exception:
        from audiotsm import wsola
        from audiotsm.io.array import ArrayReader, ArrayWriter
        r = ArrayReader(seg.reshape(1, -1)); w = ArrayWriter(1)
        wsola(1, speed=1.0 / ratio).run(r, w)
        return w.data[0].astype(np.float32)


def retime(wav, sr, audio_words, video_words, pauses):
    """Stretch each synthesized word toward its mouthed duration (clamped, smoothed) and insert video pauses.
    Returns (wav, report) where report lists per-word ratios."""
    if not audio_words or not video_words or len(audio_words) != len(video_words):
        return wav, {"applied": False, "reason": "word count mismatch"}
    ratios = []
    for a, v in zip(audio_words, video_words):
        da, dv = max(a["end"] - a["start"], 0.05), max(v["end"] - v["start"], 0.05)
        ratios.append(dv / da)
    # global factor (mouthed vs synthesized total, clamped) is applied too: the TTS speed setting maxes out at 1.2x and
    # rarely closes the gap. Per-word residuals are clamped again so no single word is mangled.
    g_raw = float(np.exp(np.mean(np.log(ratios))))
    g = float(np.clip(g_raw, GLOBAL_MIN, GLOBAL_MAX))
    ratios = [float(np.clip(r / g_raw, STRETCH_MIN, STRETCH_MAX)) * g for r in ratios]
    ratios = [float(np.clip(r, GLOBAL_MIN * STRETCH_MIN, GLOBAL_MAX * STRETCH_MAX)) for r in ratios]
    sm = [ratios[0]] + [(ratios[i - 1] + 2 * ratios[i] + ratios[i + 1]) / 4 for i in range(1, len(ratios) - 1)] + ([ratios[-1]] if len(ratios) > 1 else [])
    if max(abs(r - 1) for r in sm) < 0.10 and not pauses:
        return wav, {"applied": False, "reason": "within 10%", "ratios": [round(r, 2) for r in sm]}
    pause_after = {p["after"]: p["seconds"] for p in pauses}
    pieces = [wav[: int(audio_words[0]["start"] * sr)]]
    for i, (a, r) in enumerate(zip(audio_words, sm)):
        s, e = int(a["start"] * sr), int((audio_words[i + 1]["start"] if i + 1 < len(audio_words) else a["end"]) * sr)
        pieces.append(_stretch(wav[s:e], sr, r))
        if a["word"] in pause_after and i + 1 < len(audio_words):
            pieces.append(np.zeros(int(min(pause_after[a["word"]], 1.5) * sr), dtype=np.float32))
    pieces.append(wav[int(audio_words[-1]["end"] * sr):])
    return np.concatenate(pieces), {"applied": True, "ratios": [round(r, 2) for r in sm], "global": round(g, 2), "global_raw": round(g_raw, 2)}


def wav_bytes(wav, sr):
    import soundfile as sf
    buf = io.BytesIO(); sf.write(buf, wav, sr, format="WAV", subtype="PCM_16"); return buf.getvalue()
