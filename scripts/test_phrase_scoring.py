import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from silent_running.vsr import VSREngine, load_phrases, ROOT
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
eng = VSREngine(device=dev, beam_size=10)
print(f"loaded in {eng.load_time:.1f}s on {dev}")
vid = os.path.join(ROOT, "data/samples/ted1_short.mp4")
t0 = time.time(); x, lm, rois = eng.preprocess_video(vid); t1 = time.time()
enc = eng.encode(x); 
if dev != "cpu": torch.mps.synchronize()
t2 = time.time()
print(f"preprocess {t1-t0:.2f}s encode {t2-t1:.2f}s  input {tuple(x.shape)} enc {tuple(enc.shape)}")
print("ctc greedy:", eng.ctc_greedy(enc))
t3 = time.time(); hyps = eng.beam_search(enc, 5); t4 = time.time()
print(f"beam {t4-t3:.2f}s")
for h in hyps: print(f"   {h['score']:8.2f} {h['text']}")
# sanity: tokenizer round-trip of the top hypothesis should match beam's own ids
print("tokenize check:", eng.tokenize(hyps[0]['text']))
cands = load_phrases() + ["I'm going to make a lot of hand gestures", "I'm going to make a lot of head gestures", "I want to make a lot of hand gestures", "I'm going to make a lot of hand"]
t5 = time.time(); res = eng.score_phrases(enc, cands); t6 = time.time()
print(f"score {len(cands)} phrases in {t6-t5:.2f}s")
for r in res[:8]: print(f"   {r['score']:8.2f}  att={r['att']:8.2f} ctc={r['ctc']:8.2f} ntok={r['n_tok']:2d}  {r['phrase']}")
