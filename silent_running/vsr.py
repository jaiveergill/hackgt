"""Silent Running - visual speech recognition engine.

Wraps the pretrained Auto-AVSR visual-only conformer (LRS3 19.1% WER checkpoint, Imperial College /
Pingchuan Ma; weights mirrored on HF by Amanvir) via Chaplin's vendored ESPnet pipeline.

Exposes three levels of output from the *same* video encoding:
  * beam_search(enc)          -> open-vocabulary n-best hypotheses with model scores
  * score_phrases(enc, list)  -> exact model log-likelihood of each candidate phrase
                                 (teacher-forced attention decoder + CTC forward), for
                                 constrained decoding against a phrase inventory
  * ctc_greedy(enc)           -> instant, cheap transcript
"""
import os, sys, time, json, threading
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPLIN = os.path.join(ROOT, "third_party", "chaplin")
if CHAPLIN not in sys.path:
    sys.path.insert(0, CHAPLIN)

# --- torchvision.io.read_video was removed in recent torchvision; chaplin's detector uses it ---
import torchvision, av
def _read_video(filename, pts_unit="sec"):
    frames = []
    with av.open(filename) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    return torch.from_numpy(np.stack(frames)), None, {}
torchvision.io.read_video = _read_video

from pipelines.model import AVSR  # noqa: E402
from pipelines.detectors.mediapipe.detector import LandmarksDetector  # noqa: E402
from pipelines.detectors.mediapipe.video_process import VideoProcess  # noqa: E402
from pipelines.data.transforms import VideoTransform  # noqa: E402
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask  # noqa: E402
import sentencepiece  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "LRS3_V_WER19.1")
SPM_MODEL = os.path.join(ROOT, "third_party", "auto_avsr", "spm", "unigram", "unigram5000.model")
LM_DIR = os.path.join(ROOT, "models", "lm_en_subword")


class VSREngine:
    def __init__(self, device="mps", decode_device="cpu", beam_size=10, ctc_weight=0.1, model_dir=MODEL_DIR, half=False, lm_weight=0.0, lm_dir=LM_DIR):
        """device: where the visual frontend + conformer encoder run (mps is ~15x faster than cpu on Apple Silicon).
        decode_device: where the transformer decoder / CTC head / beam search run (cpu is fastest: many tiny ops)."""
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.decode_device = torch.device(decode_device)
        self.ctc_weight = ctc_weight
        self.lm_weight = lm_weight
        t0 = time.time()
        self.avsr = AVSR("video", os.path.join(model_dir, "model.pth"), os.path.join(model_dir, "model.json"),
                         rnnlm=os.path.join(lm_dir, "model.pth") if lm_weight > 0 else None,
                         rnnlm_conf=os.path.join(lm_dir, "model.json") if lm_weight > 0 else None,
                         penalty=0.0, ctc_weight=ctc_weight, lm_weight=lm_weight,
                         beam_size=beam_size, device=self.decode_device)
        self.model = self.avsr.model
        # hybrid placement: encoder on `device`, decoder+ctc on `decode_device`
        self.model.encoder.to(self.device)
        self.enc_dtype = torch.float16 if (half and self.device.type == "mps") else torch.float32
        self.model.encoder.to(self.enc_dtype)
        self.model.decoder.to(self.decode_device)
        self.model.ctc.to(self.decode_device)
        self.avsr.beam_search.to(self.decode_device)
        self.token_list = self.avsr.token_list
        self.tok2id = {t: i for i, t in enumerate(self.token_list)}
        self.sos = self.eos = self.model.odim - 1
        self.spm = sentencepiece.SentencePieceProcessor(model_file=SPM_MODEL)
        self.detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=True)
        self.video_transform = VideoTransform(speed_rate=1)
        self.load_time = time.time() - t0
        self.mps_lock = threading.Lock()  # MPS is not safe to use from two threads at once (Metal command buffer assertion)

    # ------------------------------------------------------------------ preprocessing
    def landmarks_for_frames(self, frames_rgb):
        """frames_rgb: (T,H,W,3) uint8 numpy. Returns list of 4x2 landmark arrays (or None per frame)."""
        lm = self.detector.detect(frames_rgb, self.detector.full_range_detector)
        if all(l is None for l in lm):
            lm = self.detector.detect(frames_rgb, self.detector.short_range_detector)
        return lm

    def mouth_rois(self, frames_rgb, landmarks):
        """Aligned 96x96 grayscale mouth crops (T,96,96) uint8, or None if no face."""
        if all(l is None for l in landmarks):
            return None
        return self.video_process(frames_rgb, list(landmarks))

    def to_model_input(self, rois):
        """(T,96,96) uint8 -> (1,T,88,88) normalized float tensor expected by encoder."""
        return self.video_transform(torch.tensor(rois))

    def preprocess_video(self, path):
        frames = _read_video(path)[0].numpy()
        lm = self.landmarks_for_frames(frames)
        rois = self.mouth_rois(frames, lm)
        return self.to_model_input(rois), lm, rois

    # ------------------------------------------------------------------ model
    @torch.no_grad()
    def encode(self, x):
        """(1,T,88,88) -> (T', 768) encoder output, returned on decode_device in fp32."""
        with self.mps_lock:
            x = x.to(self.device, self.enc_dtype)
            enc = self.model.encode(x)  # (T', 768)
            enc = enc.float().to(self.decode_device)
            if self.device.type == "mps":
                torch.mps.synchronize()
        return enc

    def warmup(self, n_frames=50):
        x = torch.zeros(1, n_frames, 88, 88)
        enc = self.encode(x)
        self.ctc_greedy(enc)
        self.score_phrases(enc, ["warm up"])

    def _ids_to_text(self, ids):
        return "".join(self.token_list[i] for i in ids if i not in (0, self.eos)).replace("▁", " ").strip()

    @torch.no_grad()
    def beam_search(self, enc, nbest=5):
        hyps = self.avsr.beam_search(enc)
        out = []
        for h in hyps[:nbest]:
            ids = h.yseq.tolist()[1:]
            out.append({"text": self._ids_to_text(ids), "score": float(h.score)})
        return out

    @torch.no_grad()
    def ctc_greedy(self, enc):
        lp = self.model.ctc.log_softmax(enc.unsqueeze(0))[0]  # (T', V)
        ids = lp.argmax(-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                out.append(i)
            prev = i
        return self._ids_to_text(out)

    def tokenize(self, text):
        pieces = self.spm.EncodeAsPieces(text.upper())
        return [self.tok2id.get(p, self.tok2id["<unk>"]) for p in pieces]

    @torch.no_grad()
    def ctc_scores(self, enc, phrases):
        """CTC log-likelihood of each phrase (cheap, vectorized). Used to prefilter big inventories."""
        toks = [self.tokenize(p) for p in phrases]
        ctc_lp = self.model.ctc.log_softmax(enc.unsqueeze(0))[0].cpu().float()
        B, T = len(toks), ctc_lp.size(0)
        targets = torch.cat([torch.tensor(t) for t in toks]); tgt_lens = torch.tensor([len(t) for t in toks]); in_lens = torch.full((B,), T, dtype=torch.long)
        ctc = -F.ctc_loss(ctc_lp.unsqueeze(1).expand(-1, B, -1), targets, in_lens, tgt_lens, blank=0, reduction="none", zero_infinity=True)
        return ctc.tolist()

    @torch.no_grad()
    def score_phrases(self, enc, phrases, prefilter=48):
        """Exact log p(phrase | video) under the pretrained model for every candidate phrase.

        Returns list of dicts sorted by combined score (descending):
          {phrase, att: attention-decoder log-lik, ctc: CTC log-lik, score: (1-w)*att + w*ctc, n_tok}
        Same weighting the beam search uses, so scores are comparable to beam hypotheses.
        With a big inventory, only the top `prefilter` phrases by CTC score get the (expensive) attention decoder;
        the rest are returned with a CTC-only estimate flagged `prefiltered_out`.
        """
        if prefilter and len(phrases) > prefilter:
            ctc_all = self.ctc_scores(enc, phrases)
            order = sorted(range(len(phrases)), key=lambda i: -ctc_all[i])
            keep = [phrases[i] for i in order[:prefilter]]
            res = self._score_phrases_full(enc, keep)
            floor = min(r["score"] for r in res)
            for i in order[prefilter:]:
                res.append({"phrase": phrases[i], "att": None, "ctc": ctc_all[i], "score": min(ctc_all[i], floor) - 1.0, "n_tok": 0, "prefiltered_out": True})
            return res
        return self._score_phrases_full(enc, phrases)

    @torch.no_grad()
    def _score_phrases_full(self, enc, phrases):
        dev = self.decode_device
        toks = [self.tokenize(p) for p in phrases]
        B = len(toks)
        L = max(len(t) for t in toks) + 1
        ys_in = torch.full((B, L), self.eos, dtype=torch.long)
        ys_out = torch.full((B, L), -1, dtype=torch.long)
        for b, t in enumerate(toks):
            ys_in[b, 0] = self.sos
            ys_in[b, 1:len(t) + 1] = torch.tensor(t)
            ys_out[b, :len(t)] = torch.tensor(t)
            ys_out[b, len(t)] = self.eos
        ys_in, ys_out = ys_in.to(dev), ys_out.to(dev)
        ys_mask = subsequent_mask(L, device=dev).unsqueeze(0).expand(B, -1, -1)
        memory = enc.unsqueeze(0).expand(B, -1, -1)
        logits, _ = self.model.decoder(ys_in, ys_mask, memory, None)
        logp = F.log_softmax(logits, dim=-1)
        tgt = ys_out.clamp(min=0)
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp.masked_fill(ys_out < 0, 0.0)
        att = tok_lp.sum(-1)  # (B,)

        # CTC forward log-likelihood per phrase
        ctc_lp = self.model.ctc.log_softmax(enc.unsqueeze(0))[0]  # (T', V)
        T = ctc_lp.size(0)
        log_probs = ctc_lp.unsqueeze(1).expand(-1, B, -1)  # (T', B, V)
        targets = torch.cat([torch.tensor(t) for t in toks]).to(dev)
        tgt_lens = torch.tensor([len(t) for t in toks])
        in_lens = torch.full((B,), T, dtype=torch.long)
        ctc = -F.ctc_loss(log_probs.cpu().float(), targets.cpu(), in_lens, tgt_lens, blank=0, reduction="none", zero_infinity=True)
        att = att.cpu()
        score = (1 - self.ctc_weight) * att + self.ctc_weight * ctc
        res = []
        for b, p in enumerate(phrases):
            res.append({"phrase": p, "att": float(att[b]), "ctc": float(ctc[b]), "score": float(score[b]), "n_tok": len(toks[b])})
        res.sort(key=lambda r: -r["score"])
        return res


def load_phrase_table(path=os.path.join(ROOT, "silent_running", "phrases.txt")):
    """-> list of {phrase, category, critical}. '## name' lines start a category; trailing ' !' marks critical phrases."""
    out, cat = [], "general"
    for l in open(path):
        l = l.strip()
        if not l or l.startswith("# ") or l == "#":
            continue
        if l.startswith("## "):
            cat = l[3:].strip().lower(); continue
        crit = l.endswith(" !")
        ph = l[:-2].strip() if crit else l
        out.append({"phrase": ph, "category": cat, "critical": crit})
    return out


def load_phrases(path=os.path.join(ROOT, "silent_running", "phrases.txt")):
    return [r["phrase"] for r in load_phrase_table(path)]
