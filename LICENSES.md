# Licensing and provenance

Silent Running is a non-commercial hackathon prototype. This file tracks every third-party
component so we know what could and could not be commercialized later.

| Component | Source | License | Notes |
|---|---|---|---|
| **VSR checkpoint `LRS3_V_WER19.1`** (visual-only conformer, 19.1% WER on LRS3, ~250M params) | Pingchuan Ma / Imperial College, released in [Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) model zoo; HF mirror [Amanvir/LRS3_V_WER19.1](https://huggingface.co/Amanvir/LRS3_V_WER19.1) | Custom BSD-style license with **non-commercial clauses**: "individual research and private study" and "research prototype ... only be used for comparative or benchmarking purposes"; content must not be used to train commercial technology | **Blocks commercial use.** Fine for the hackathon. Trained on LRS3 + VoxCeleb2 + AVSpeech (~3,448 h). |
| Subword LM `lm_en_subword` | same model zoo; HF mirror [Amanvir/lm_en_subword](https://huggingface.co/Amanvir/lm_en_subword) | same as above | Downloaded but **not used** (Chaplin's config leaves `rnnlm` empty; we do our own constrained decoding). |
| **Auto-AVSR checkpoint `vsr_trlrs2lrs3vox2avsp_base.pth`** (20.3% WER, 3,291 h) | [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) model zoo (Google Drive) | Code Apache-2.0; README says checkpoints "may have their own licenses or terms and conditions derived from the dataset used for training" | Downloaded as a backup/alternative. Same architecture family. |
| Auto-AVSR code (`third_party/auto_avsr`) | [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) | Apache-2.0 | We use its SentencePiece `unigram5000` model for tokenizing candidate phrases. |
| Chaplin pipeline code (`third_party/chaplin`) | [amanvirparhar/chaplin](https://github.com/amanvirparhar/chaplin) | MIT (Chaplin); vendored `pipelines/` and `espnet/` files carry Apache-2.0 headers (Imperial College / ESPnet) | Reference webcam → MediaPipe → mouth ROI → ESPnet beam search pipeline. We patched `espnet/nets/ctc_prefix_score.py` for MPS. |
| `Visual_Speech_Recognition_for_Multiple_Languages` (`third_party/vsr_multi`) | mpc001 | Custom non-commercial license (see above) | Cloned for reference only. |
| MediaPipe face detection (0.10.21) | Google | Apache-2.0 | Legacy `solutions` API, needed by the crop pipeline. |
| ESPnet | espnet | Apache-2.0 | Vendored subset inside Chaplin. |
| PyTorch, torchvision, torchaudio, OpenCV, PyAV, scikit-image, sentencepiece | various | BSD/Apache/MIT/LGPL (PyAV, FFmpeg) | Standard. |
| macOS `say` / browser `speechSynthesis` | Apple / browser vendor | OS-provided | TTS. |
| OpenAI API (`gpt-4o-mini`) for contextual reranking | OpenAI | OpenAI terms of use | Used only to *choose among* VSR-supported candidates; key in `.env` (not committed). |

## Dataset-derived restrictions

* **LRS3** (TED/TEDx talks) is distributed under a research-only license; TED content itself is CC BY-NC-ND. Models trained on it inherit non-commercial restrictions in practice.
* **VoxCeleb2** is CC BY 4.0 but derived from YouTube; **AVSpeech** is YouTube-derived with no redistribution license.
* Consequence: **every strong public VSR checkpoint we found (Auto-AVSR, VSR-multi, VALLR CC BY-NC 4.0, Llama-AVSR built on AV-HuBERT) is non-commercial in practice.** A commercial product would need to retrain on licensed data or license the model from the authors.

## Sample media used for testing

* `data/samples/ted*.mp4`: short excerpts of public TED talks downloaded from YouTube, used solely as local test inputs to prove the pipeline (video track only, audio stripped). Not redistributed.
