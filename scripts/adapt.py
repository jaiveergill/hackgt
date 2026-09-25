"""Speaker adaptation: fine-tune the pretrained visual model on one person's session (Plan 2).

  python scripts/adapt.py --speaker alice [--epochs 10] [--lr 2e-5] [--freeze-encoder-below 6] [--anchor]

Loads models/LRS3_V_WER19.1 into Auto-AVSR's trainable E2E (joint CTC + attention loss), freezes the visual frontend,
trains on data/session/<speaker>/prep.jsonl (split=train) with the pretraining augmentations, selects the epoch by
held-out silent accuracy, and saves models/adapted_<speaker>/{model.pth,model.json} in the same layout as the base so the
server can load it with --model-dir.

Runs in a process with ONLY Auto-AVSR's espnet on the path (Chaplin's clashes). Uses MPS if available.
"""
import os, sys, json, argparse, time, random, re, copy
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "auto_avsr"))
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E  # noqa: E402
from datamodule.transforms import TextTransform, AdaptiveTimeMask  # noqa: E402
from lightning import get_beam_search_decoder  # noqa: E402
import torchvision  # noqa: E402

BASE = os.path.join(ROOT, "models", "LRS3_V_WER19.1")
ANCHOR = [("data/samples/ted1_short.mp4", "I'M GOING TO MAKE A LOT OF HAND GESTURES")]


def to_trainer_keys(sd):
    return {k.replace("encoder.frontend.", "frontend.").replace("encoder.embed.0.", "proj_encoder."): v for k, v in sd.items()}


def to_inference_keys(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("frontend."):
            k = "encoder." + k
        elif k.startswith("proj_encoder."):
            k = k.replace("proj_encoder.", "encoder.embed.0.")
        out[k] = v
    return out


class Aug:
    def __init__(self, train):
        self.train = train
        self.mask = AdaptiveTimeMask(10, 25)

    def __call__(self, rois):  # (T,96,96) uint8 -> (T,1,88,88) float
        x = torch.from_numpy(rois).float() / 255.0
        T = x.shape[0]
        if self.train:
            i, j = random.randint(0, 8), random.randint(0, 8)
            x = x[:, i:i + 88, j:j + 88]
            if random.random() < 0.5:
                x = x.flip(-1)
            x = x * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)
            x = x.clamp(0, 1)
        else:
            x = x[:, 4:92, 4:92]
        x = (x - 0.421) / 0.165
        x = x.unsqueeze(1)
        if self.train:
            x = self.mask(x)
        return x


def load_model():
    tt = TextTransform()
    model = E2E(len(tt.token_list), "video", ctc_weight=0.1)
    sd = torch.load(os.path.join(BASE, "model.pth"), map_location="cpu")
    missing, unexpected = model.load_state_dict(to_trainer_keys(sd), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model, tt


@torch.no_grad()
def decode(model, tt, x, dev, beam=5):
    model.eval()
    feat = model.encoder(model.proj_encoder(model.frontend(x.unsqueeze(0).to(dev))), None)[0].squeeze(0)
    bs = get_beam_search_decoder(model, tt.token_list, beam_size=beam)
    hyp = bs(feat)[0]
    return tt.post_process(torch.tensor(hyp.yseq[1:].tolist())).replace("<eos>", "").strip()


@torch.no_grad()
def score_phrases(model, tt, x, phrases, dev):
    """Same phrase-constrained scoring the server uses (attention + CTC), for held-out Phrase Mode top-1."""
    import torch.nn.functional as F
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    model.eval()
    enc = model.encoder(model.proj_encoder(model.frontend(x.unsqueeze(0).to(dev))), None)[0]  # (1,T',D)
    toks = [tt.tokenize(p.upper()).tolist() for p in phrases]
    B, L = len(toks), max(len(t) for t in toks) + 1
    ys_in = torch.full((B, L), model.eos, dtype=torch.long); ys_out = torch.full((B, L), -1, dtype=torch.long)
    for b, t in enumerate(toks):
        ys_in[b, 0] = model.sos; ys_in[b, 1:len(t) + 1] = torch.tensor(t); ys_out[b, :len(t)] = torch.tensor(t); ys_out[b, len(t)] = model.eos
    ys_in, ys_out = ys_in.to(dev), ys_out.to(dev)
    logits, _ = model.decoder(ys_in, subsequent_mask(L, device=dev).unsqueeze(0).expand(B, -1, -1), enc.expand(B, -1, -1), None)
    lp = F.log_softmax(logits, -1).gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1).masked_fill(ys_out < 0, 0).sum(-1)
    ctc_lp = model.ctc.log_softmax(enc)[0].cpu().float()
    tgt = torch.cat([torch.tensor(t) for t in toks]); tl = torch.tensor([len(t) for t in toks]); il = torch.full((B,), ctc_lp.size(0))
    ctc = -F.ctc_loss(ctc_lp.unsqueeze(1).expand(-1, B, -1), tgt, il, tl, blank=0, reduction="none", zero_infinity=True)
    return (0.9 * lp.cpu() + 0.1 * ctc).tolist()


def norm(s):
    return re.sub(r"[^a-z' ]", "", s.lower()).split()


def evaluate(model, tt, items, phrases, dev):
    """Returns dict(top1, wer) on held-out items."""
    import torchaudio
    aug = Aug(False)
    top1 = n_phr = 0; ed = nw = 0
    for it in items:
        x = aug(np.load(os.path.join(ROOT, it["npy"])))
        if it["pass"] == "silent":
            sc = score_phrases(model, tt, x, phrases, dev)
            top1 += phrases[int(np.argmax(sc))].upper() == it["label"]; n_phr += 1
        hyp = decode(model, tt, x, dev)
        ed += torchaudio.functional.edit_distance(norm(hyp), norm(it["label"])); nw += len(norm(it["label"]))
    return {"top1": top1 / max(n_phr, 1), "n_phrase": n_phr, "wer": ed / max(nw, 1), "n_words": nw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--freeze-encoder-below", type=int, default=6, help="freeze conformer blocks [0, n)")
    ap.add_argument("--silent-weight", type=float, default=2.0, help="oversample silent takes by this factor")
    ap.add_argument("--anchor", action="store_true", help="mix in TED anchor clips against forgetting")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    dev = torch.device(args.device)
    sdir = os.path.join(ROOT, "data", "session", args.speaker)
    items = [json.loads(l) for l in open(os.path.join(sdir, "prep.jsonl")) if l.strip()]
    train = [it for it in items if it["split"] == "train"]; test = [it for it in items if it["split"] == "test"]
    phrases = [l.strip() for l in open(os.path.join(ROOT, "silent_running", "phrases.txt")) if l.strip() and not l.startswith("#")]
    print(f"train {len(train)} takes, test {len(test)} takes ({sum(t['pass']=='silent' for t in test)} silent phrases)")
    model, tt = load_model()
    model.to(dev)
    for p in model.frontend.parameters():
        p.requires_grad = False
    for i, blk in enumerate(model.encoder.encoders):
        if i < args.freeze_encoder_below:
            for p in blk.parameters():
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params)/1e6:.0f}M params on {dev}")

    base_eval = evaluate(model, tt, test, phrases, dev) if test else None
    print("baseline held-out:", base_eval)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    steps_per_epoch = max(1, int(len(train) * 1.5 / args.batch))
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    aug = Aug(True)
    weights = [args.silent_weight if it["silent"] else 1.0 for it in train]
    anchor = []
    if args.anchor:
        sys.path.insert(0, ROOT)
        # anchor crops are produced by prep of the TED clips if present
        for path, text in ANCHOR:
            npy = os.path.join(ROOT, path[:-4] + ".npy")
            if os.path.exists(npy):
                anchor.append({"npy": path[:-4] + ".npy", "label": text, "silent": False, "pass": "anchor"})
        print(f"anchor clips: {len(anchor)}")
    best, best_state, hist = -1, None, []
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); tot = 0.0; nb = 0
        for _ in range(steps_per_epoch):
            batch = random.choices(train, weights=weights, k=args.batch)
            if anchor and random.random() < 0.15:
                batch[-1] = random.choice(anchor)
            xs = [aug(np.load(os.path.join(ROOT, it["npy"]))) for it in batch]
            ys = [tt.tokenize(it["label"]) for it in batch]
            lengths = torch.tensor([x.shape[0] for x in xs]); T = int(lengths.max())
            xb = torch.zeros(len(xs), T, 1, 88, 88)
            for i, x in enumerate(xs):
                xb[i, :x.shape[0]] = x
            L = max(len(y) for y in ys)
            yb = torch.full((len(ys), L), -1, dtype=torch.long)
            for i, y in enumerate(ys):
                yb[i, :len(y)] = y
            loss, loss_ctc, loss_att, acc = model(xb.to(dev), lengths.to(dev), yb.to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            tot += float(loss); nb += 1
        ev = evaluate(model, tt, test, phrases, dev) if test else {"top1": 0, "wer": 0}
        hist.append({"epoch": ep, "loss": tot / nb, **ev, "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        print(f"epoch {ep}: loss {tot/nb:.3f}  held-out top1 {ev['top1']:.2%}  wer {ev['wer']:.3f}  ({time.time()-t0:.0f}s)")
        key = ev["top1"] - 0.2 * ev["wer"]
        if key > best:
            best, best_state = key, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    out = args.out or os.path.join(ROOT, "models", f"adapted_{args.speaker}")
    os.makedirs(out, exist_ok=True)
    torch.save(to_inference_keys(best_state), os.path.join(out, "model.pth"))
    import shutil
    shutil.copy(os.path.join(BASE, "model.json"), os.path.join(out, "model.json"))
    json.dump({"speaker": args.speaker, "baseline": base_eval, "history": hist, "args": vars(args)}, open(os.path.join(out, "adapt_log.json"), "w"), indent=1)
    print(f"saved {out}  baseline top1 {base_eval['top1'] if base_eval else 'n/a'} -> best {max(h['top1'] for h in hist):.2%}")


if __name__ == "__main__":
    main()
