"""Check /api/tts_stream: it streams a clip that is not cached and caches it only once its generation has completed. No
network, no model: the ElevenLabs client is a fake whose stream yields 5 chunks of 1000 bytes, and the cache goes to a temp dir.

  python scripts/check_tts_stream.py

Cases (a truncated clip in the cache would be what every later playback of the phrase used):
  client stops after 1 chunk        (the UI replaced the audio source)  -> the generation goes on: cached whole, 5000 bytes
  upstream fails after 2 chunks     (ElevenLabs error mid-stream)       -> not cached, an `error` event, the stream aborts
  full clip                                                             -> cached, 5000 bytes; the next request is served whole
  prefetched, then requested        (Open Mode: the voice starts while the LLM decides) -> the request joins the running
                                    generation: one upstream call, all 5 chunks
Prints expected vs actual; exit 1 on a failure.
"""
import asyncio, os, shutil, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import silent_running.server as srv
from silent_running import tts as eltts

eltts.CACHE_DIR = tempfile.mkdtemp()
_path = eltts._cache_path
eltts._cache_path = lambda *a, **k: os.path.join(eltts.CACHE_DIR, os.path.basename(_path(*a, **k)))
eltts.voice_for = lambda speaker: "VOICE"
eltts.available = lambda: True
EVENTS = []
srv.broadcast = EVENTS.append


CALLS = []


class FakeElevenLabs:
    class text_to_speech:
        @staticmethod
        def stream(**kw):
            CALLS.append(kw["text"])
            for i in range(5):
                time.sleep(0.02)
                if kw["text"].startswith("ERR") and i == 2:
                    raise RuntimeError("ElevenLabs 500 mid-stream")
                yield bytes([65 + i]) * 1000


eltts._client = lambda: FakeElevenLabs
failures = []


def check(ok, what, expected, actual):
    print(f"{'PASS' if ok else 'FAIL'} {what}: expected {expected}, got {actual}")
    if not ok:
        failures.append(what)


async def consume(text, n):
    """Read n chunks of the streamed response, then drop it as a browser would (or until it ends / fails)."""
    r = srv.api_tts_stream(text=text, voice="x")
    got, raised = 0, None
    try:
        async for _ in r.body_iterator:
            got += 1
            if got == n:
                break
    except RuntimeError as e:
        raised = str(e)
    await r.body_iterator.aclose()
    return got, raised


def generated():
    """Wait for the generations in progress to finish (they run on their own threads)."""
    while eltts._generating:
        time.sleep(0.01)


for text, n, want_cached in (("Client stops after 1 chunk", 1, True), ("ERR upstream fails after 2 chunks", 99, False), ("Full clip", 99, True)):
    EVENTS.clear()
    got, raised = asyncio.run(consume(text, n))
    generated()
    audio = eltts.cached(text, "VOICE")
    print(f"{text!r}: read {got} chunks, stream raised {raised!r}")
    check((audio is not None) == want_cached and (audio is None or len(audio) == 5000), f"{text!r} cached",
          "the full 5000 bytes" if want_cached else "nothing", f"{len(audio)} bytes" if audio is not None else "nothing")
    r2 = srv.api_tts_stream(text=text, voice="x")
    served_whole = r2.headers.get("x-tts-cached") == "True"
    check(served_whole == want_cached, f"{text!r} next request", "served from the cache" if want_cached else "streamed again",
          "served from the cache" if served_whole else "streamed again")
    if text.startswith("ERR"):
        errs = [e["message"] for e in EVENTS if e.get("type") == "error"]
        check(bool(errs) and raised is not None, "upstream failure is shown and aborts the stream", "an error event and a raised stream",
              f"events {errs}, raised {raised!r}")
    else:
        check(not EVENTS, f"{text!r} raises no error event", "none", EVENTS)

generated()
CALLS.clear()
eltts.prefetch("Prefetched", "VOICE")
time.sleep(0.05)  # the request comes while the clip is being generated
got, raised = asyncio.run(consume("Prefetched", 99))
generated()
check((CALLS, got, raised) == (["Prefetched"], 5, None), "a request joins the prefetched generation",
      "one upstream call, 5 chunks, no error", f"calls {CALLS}, {got} chunks, raised {raised!r}")
check(len(eltts.cached("Prefetched", "VOICE") or b"") == 5000, "the prefetched clip is cached", "5000 bytes",
      f"{len(eltts.cached('Prefetched', 'VOICE') or b'')} bytes")

shutil.rmtree(eltts.CACHE_DIR)
print("CHECK_TTS_STREAM " + ("PASSED" if not failures else f"FAILED: {failures}"))
sys.exit(1 if failures else 0)
