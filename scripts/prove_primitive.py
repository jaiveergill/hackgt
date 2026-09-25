"""Milestone 1: video of a person speaking -> pretrained VSR (Auto-AVSR LRS3 19.1% WER) -> text.
Uses Chaplin's vendored pipeline (mediapipe face detection -> mouth ROI -> conformer -> beam search).
"""
import os, sys, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPLIN = os.path.join(ROOT, "third_party", "chaplin")
sys.path.insert(0, CHAPLIN)
sys.argv = [sys.argv[0]] + [os.path.abspath(a) if os.path.exists(a) else a for a in sys.argv[1:]]
os.chdir(CHAPLIN)  # config paths are relative to chaplin root

import torch, torchvision, av, numpy as np

def _read_video(filename, pts_unit="sec"):
    """Replacement for torchvision.io.read_video (removed in recent torchvision). Returns (T,H,W,3) uint8 RGB tensor."""
    frames = []
    with av.open(filename) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    return torch.from_numpy(np.stack(frames)), None, {}
torchvision.io.read_video = _read_video

from pipelines.pipeline import InferencePipeline

p = argparse.ArgumentParser()
p.add_argument("videos", nargs="+")
p.add_argument("--device", default="cpu")
p.add_argument("--beam", type=int, default=40)
args = p.parse_args()

cfg = os.path.join(ROOT, "configs", f"LRS3_V_WER19.1_beam{args.beam}.ini")
os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
with open(cfg, "w") as f:
    f.write(f"""[input]
modality=video
v_fps=25

[model]
v_fps=25
model_path={ROOT}/models/LRS3_V_WER19.1/model.pth
model_conf={ROOT}/models/LRS3_V_WER19.1/model.json
rnnlm=
rnnlm_conf=

[decode]
beam_size={args.beam}
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.1
lm_weight=0.0
""")

t0 = time.time()
pipe = InferencePipeline(cfg, device=torch.device(args.device), detector="mediapipe", face_track=True)
print(f"[load] {time.time()-t0:.1f}s device={args.device} beam={args.beam}")
for v in args.videos:
    v = os.path.abspath(v)
    t0 = time.time()
    lm = pipe.landmarks_detector(v)
    t1 = time.time()
    data = pipe.dataloader.load_data(v, lm)
    t2 = time.time()
    with torch.no_grad():
        enc = pipe.model.model.encode(data.to(pipe.model.device))
        if args.device != "cpu": torch.mps.synchronize()
        t3 = time.time()
        hyps = pipe.model.beam_search(enc)
        t4 = time.time()
    from espnet.asr.asr_utils import add_results_to_json
    out = add_results_to_json([h.asdict() for h in hyps[:1]], pipe.model.token_list).replace("▁", " ").replace("<eos>", "").strip()
    print(f"[{os.path.basename(v)}] frames={len(lm)} detected={sum(l is not None for l in lm)} landmarks={t1-t0:.2f}s crop={t2-t1:.2f}s encode={t3-t2:.2f}s beam={t4-t3:.2f}s")
    print(f"  RAW: {out}")
    for h in hyps[:5]:
        print(f"    {float(h.score):8.2f}  {add_results_to_json([h.asdict()], pipe.model.token_list).replace('▁',' ').replace('<eos>','').strip()}")
