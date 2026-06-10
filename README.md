# frisian-asr

Open, end-to-end Frisian (West Frisian, `fy-NL`) speech recognition: build a CC0 dataset
from Common Voice, then fine-tune NVIDIA's streaming Nemotron 3.5 ASR into a dedicated
Frisian model — with a leak-free, before/after WER showcase.

Two public artifacts come out of this repo:

| Artifact | Hugging Face | What it is |
|---|---|---|
| 📚 Dataset | [`LokaalHub/frisian-asr-cv22`](https://huggingface.co/datasets/LokaalHub/frisian-asr-cv22) | CC0, 49.6 h, speaker- & sentence-disjoint splits |
| 🗣️ Model | [`LokaalHub/frisian-asr-streaming-0.6b`](https://huggingface.co/LokaalHub/frisian-asr-streaming-0.6b) | Streaming Frisian ASR, fine-tuned from Nemotron 3.5 ASR |

## Headline result

Frisian is **not** one of Nemotron 3.5 ASR's 40 locales, so the base model can't transcribe
it (~83 % WER — garbled, Dutch-like text). A single full fine-tune on ~40 h of Frisian,
conditioned on the nearest supported locale slot (`nl-NL`), flips that:

**Raw WER (%), held-out test (3,173 clips), cache-aware streaming — same eval for base & fine-tuned:**

| Latency (`att_context_size`) | Base | Fine-tuned | Rel. improvement |
|---|---|---|---|
| 80 ms `[56,0]` | 82.87 % | **20.36 %** | **75.4 %** |
| 1120 ms `[56,13]` | 81.69 % | **18.01 %** | **77.9 %** |

Numbers are measured with NVIDIA's own streaming inference script and raw WER (NeMo
`word_error_rate`, no normalization), matching NVIDIA's reporting convention. Full method and
caveats: [`MODEL_CARD.md`](MODEL_CARD.md).

## Pipeline & repo layout

```
1. Build dataset   →  hf_job.py + launch_job.py   →  LokaalHub/frisian-asr-cv22 (published)
                      build_dataset.py + config.yaml  (local CV25 variant — see note below)
2. Baseline (before)  →  baseline_job.py
3. Fine-tune          →  finetune_job.py
4. Streaming eval     →  streaming_eval_job.py
   (all launched via launch_showcase.py; check_job.py / diag_job.py are helpers)
5. Publish            →  MODEL_CARD.md  →  LokaalHub/frisian-asr-streaming-0.6b
```

Everything heavy runs on **Hugging Face Jobs** (GPU), billed to your account. `launch_*.py`
build the `run_uv_job` commands; the `*_job.py` files are the remote scripts.

> **Note on dataset versions.** The *published* dataset (`frisian-asr-cv22`) was built by
> `hf_job.py` from `fsicoli/common_voice_22_0`. `build_dataset.py` + `config.yaml` are a
> later **local** pipeline variant targeting Common Voice 25.0 with a wav2vec2 CER filter;
> it is kept for reference and produces a larger (~130–170 h) build, but was not the source
> of the published artifact.

## Quickstart

```bash
pip install -r requirements.txt
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"   # a write token

# --- dataset (remote, the published path) ---
python launch_job.py smoke      # 30 clips/split sanity build
python launch_job.py full       # full CV22 build → LokaalHub/frisian-asr-cv22

# --- showcase (baseline → fine-tune → streaming eval), de-risking ladder ---
python launch_showcase.py baseline        # base WER on test ("before")
python launch_showcase.py finetune-smoke  # 20 steps / 30 clips, validates A100 pipeline
python launch_showcase.py finetune        # full fine-tune (A100) → model + before/after WER
python launch_showcase.py stream-eval     # NVIDIA-format streaming WER tables
```

`python -m pytest tests/` runs the dataset-pipeline unit tests (no data/GPU needed).

## Fine-tuning recipe (NVIDIA-faithful)

The fine-tune follows NVIDIA's official
[Nemotron 3.5 ASR fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb):
`speech_to_text_finetune.py` + `fastconformer_transducer_bpe_streaming_prompt`, AdamW, base LR
0.1 (NoamAnnealing), `batch_duration=200`, bf16, tokenizer reused from base (data < 50 h),
dual-tag (`lang` + `target_lang`) manifests. Frisian rides the `nl-NL` prompt slot because it
isn't in the 40 locales. The model is **monolingual Frisian by design** (no replay → the other
locales are not preserved). Full rationale, deviations, and limitations in
[`MODEL_CARD.md`](MODEL_CARD.md).

## Dataset datasheet (CV22, published)

- **Source / license:** Mozilla Common Voice `fy-NL`, **CC0**. Speaker IDs are anonymized hashes.
- **Splits:** `test`/`dev` from CV's high-confidence splits; `train` = validated-rest + a
  CER-filtered slice of the `other` bucket. Train is disjoint from eval by speaker **and** sentence.
- **`other`-bucket QC:** each clip CTC-decoded with a Frisian-trained wav2vec2 model
  (`greenw0lf/wav2vec2-large-xls-r-1b-frisian`); kept if CER vs. prompt ≤ 0.25. A per-clip
  `quality_score` is exposed.
- **Known limitations:** read speech only (no spontaneous/dialectal); heavy sentence repetition;
  demographic skew; residual label noise in the `other` slice (mitigated, not eliminated).

## Coding guidelines

`CLAUDE.md` and `skills/karpathy-guidelines/` govern changes here: think before coding,
simplicity first, surgical changes, goal-driven execution.
