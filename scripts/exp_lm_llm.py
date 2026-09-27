"""Experiment: does (a) the subword RNN-LM in beam search and (b) LLM propose + VSR verify reduce WER?
Uses the TED clips (audio stripped) with their YouTube reference captions."""
import os, sys, time, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
from silent_running.vsr import VSREngine, ROOT
from silent_running.context import LLMInterpreter

REFS = {
    "data/samples/ted1_short.mp4": "I'm going to make a lot of hand gestures",
    "data/samples/ted1_12s.mp4": "you've learned something now I'm going to get started with the opening I'm going to make a lot of hand gestures I'm going to do this with my right hand I'm going to do this with my left I'm going to adjust my glasses",
}
def norm(s): return re.sub(r"[^a-z' ]", "", s.lower()).split()
def wer(h, r): h, r = norm(h), norm(r); return torchaudio.functional.edit_distance(h, r) / max(len(r), 1)

llm = LLMInterpreter(model=next((a for a in sys.argv[1:] if not a.startswith("--")), None))
rows = []
for lm_w, beam in ([(0.0, 10)] if "--fast" in sys.argv else [(0.0, 10), (0.3, 10), (0.3, 20), (0.5, 20)]):
    eng = VSREngine(device="mps", decode_device="cpu", beam_size=beam, lm_weight=lm_w)
    eng.warmup()
    for path, ref in REFS.items():
        x, lm, rois = eng.preprocess_video(os.path.join(ROOT, path))
        enc = eng.encode(x)
        t0 = time.time(); nb = eng.beam_search(enc, 10); t_beam = time.time() - t0
        w_beam = wer(nb[0]["text"], ref)
        # LLM propose + VSR verify
        t1 = time.time()
        prop = llm.propose([{"text": h["text"], "score": h["score"]} for h in nb], {"notes": "speaker giving a talk on stage"})
        t_llm = time.time() - t1
        w_llm = None; accepted = None; sc = None
        if prop and "error" not in prop:
            sc = eng.score_phrases(enc, [prop["sentences"][0], nb[0]["text"]])
            s_prop = next(r["score"] for r in sc if r["phrase"] == prop["sentences"][0]); s_top = next(r["score"] for r in sc if r["phrase"] == nb[0]["text"])
            accepted = s_prop >= s_top - 3.0  # fixed margin (length-scaled margin let a bad proposal through)
            w_llm = wer(prop["sentences"][0] if accepted else nb[0]["text"], ref)
        rows.append(dict(lm=lm_w, beam=beam, clip=os.path.basename(path), t_beam=round(t_beam, 2), wer_beam=round(w_beam, 3), top=nb[0]["text"],
                         llm=(prop.get("sentences") or [None])[0] if prop else None, llm_vs_top=(round(s_prop - s_top, 2) if sc else None), accepted=accepted, wer_final=(round(w_llm, 3) if w_llm is not None else None), t_llm=round(t_llm, 1)))
        print(json.dumps(rows[-1]))
json.dump(rows, open(os.path.join(ROOT, "results", "exp_lm_llm" + ("_fast" if "--fast" in sys.argv else "") + ".json"), "w"), indent=1)
