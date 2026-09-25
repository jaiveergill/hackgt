# Plan 2: Speaker adaptation, the model learns one person's mouth in ten minutes

**Goal.** Fine-tune the pretrained visual model on a few minutes of one person's webcam video so silent mouthing by that person is recognized far more reliably. Data comes from the same session as the voice clone (Plan 3).

**Constraint check.** We are adapting a pretrained checkpoint, not training a lip reader. Allowed by the brief.

## Why it should work

- The gap between the benchmark (TED, studio light, voiced) and our setting (laptop webcam, silent, one angle) is a domain gap, and a few minutes of in-domain data closes most of it. Speaker-adaptive lip reading papers report 20 to 40 percent relative WER reduction from 1 to 5 minutes of adaptation data.
- Both checkpoints (`LRS3_V_WER19.1`, `vsr_trlrs2lrs3vox2avsp_base`) are the same 250M-parameter conformer with 767 identical tensors; only key prefixes differ (`encoder.frontend.*` vs `frontend.*`). So the stronger 19.1% weights load into Auto-AVSR's clean Lightning training code (`third_party/auto_avsr/lightning.py`, joint CTC + attention loss) after a key rename.
- Memory: 250M params fp32 is 1 GB; with gradients and AdamW state about 4 GB, plus activations for 2 to 4 s clips. Fits the 18 GB M3 Pro on MPS. Frozen frontend cuts it further.

## Data (from `scripts/record_session.py`, shared with Plan 3)

Per speaker, in `data/session/<speaker>/`:

| Pass | Content | Label source | Use |
|---|---|---|---|
| Voiced | about 40 sentences read aloud: 20 Harvard sentences (phonetically balanced) + 20 hospital sentences | OpenAI `gpt-4o-transcribe` on the WAV, keep takes with confidence above threshold | train |
| Silent | the 30 phrase-bank phrases mouthed, 2 takes each | the prompt text | 1 take train, 1 take held-out test |
| Silent free | 10 arbitrary short sentences shown on screen, mouthed | prompt text | test only (open vocabulary) |

Total about 12 minutes on camera. The silent takes matter most: people articulate differently when silent, and that is the deployment condition.

Preprocessing: existing pipeline (MediaPipe landmarks, affine alignment, 96x96 grayscale mouth crops at 25 fps) via `VSREngine.preprocess_video`, saved as `.npy` per clip plus a CSV in Auto-AVSR's `path,length,token_ids` format so its `DataModule` can read it directly.

## Training

`scripts/adapt.py --speaker <name> --base models/LRS3_V_WER19.1 --out models/adapted_<name>`

1. Build the Auto-AVSR `ModelModule` (video modality), load the 19.1 weights with the key rename, verify by running the TED clip through it and matching the known transcript.
2. Freeze the 3D-conv + ResNet frontend. Train encoder blocks 6 to 12, the decoder, and the CTC head. Option `--lora` trains rank-8 adapters on attention projections only (fewer trainable params, safer with little data).
3. AdamW, lr 2e-5 (full) or 2e-4 (LoRA), warmup 50 steps, cosine to zero, batch of 4 clips, 8 to 15 epochs, label smoothing 0.1, CTC weight 0.1 as in pretraining.
4. Augmentation from the pretraining recipe: random 88x88 crop, horizontal flip, adaptive time masking, plus mild brightness jitter for webcam lighting.
5. Anchor set against forgetting: include the two TED clips with references at 10 percent of each epoch. If TED WER rises by more than 5 points, stop.
6. Select the epoch by held-out silent phrase accuracy (Phrase Mode top-1), not by loss.
7. Save `model.pth` + `model.json` in the same layout as the base so `--model-dir` in the server switches models with no other change.

Expected wall time on MPS: 5 to 15 minutes for 40 clips and 10 epochs. If a conformer op is unsupported on MPS, fall back to CPU (slower, about 45 minutes) or run the adaptation on a rented GPU and copy the checkpoint back.

## Server and UI

- `--model-dir models/adapted_<name>` at launch, and a "Speaker" dropdown that hot-swaps the state dict (2 s).
- Result panel pill: "adapted to <name>" or "generic model".
- A/B toggle in the eval drawer for live comparison during the demo.

## Evaluation (the headline number)

Run `scripts/eval.py` twice on the held-out silent takes, `--tag generic` and `--tag adapted_<name>`:

- Phrase Mode top-1 and top-3 accuracy on 30 phrases.
- Open-vocabulary WER on the 10 silent free sentences and on the voiced pass held-out takes.
- TED anchor WER, must not regress.
- Latency unchanged (same architecture).

Also test a second speaker with the generic model and with the first speaker's adapted model, to show the adaptation is specific and does not break others.

## Risks

- Overfitting with 40 clips: mitigated by freezing the frontend, low lr, augmentation, anchor set, and early stopping on held-out silent takes.
- Silent vs voiced mismatch: the silent pass is in training for this reason; if voiced-only training helps less, weight silent clips 2x.
- Whisper labels wrong: drop low-confidence takes; the silent takes have exact labels anyway.
- MPS training bugs: `torch.backends.mps` handles conv3d and attention; relative positional attention in the conformer is plain matmul. Verify one training step early, before the recording session.
- Time: the person must be available for 12 minutes on camera; schedule it at the same time as the voice recording.

**Time estimate.** 1 hour to verify a training step on MPS with the loaded weights, 2 hours for data tooling, 15 minutes recording, 1 hour training and A/B eval.
