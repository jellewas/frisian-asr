---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
language:
- fy
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
datasets:
- LokaalHub/frisian-asr-cv22
tags:
- automatic-speech-recognition
- speech
- audio
- frisian
- fy
- nemo
- fastconformer
- rnnt
- cache-aware-streaming
- nemotron
metrics:
- wer
model-index:
- name: frisian-asr-streaming-0.6b
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: frisian-asr-cv22 (test)
      type: LokaalHub/frisian-asr-cv22
      split: test
      args:
        language: fy
    metrics:
    - type: wer
      value: 20.36
      name: Raw WER (streaming, 80 ms / att_context_size=[56,0])
    - type: wer
      value: 18.01
      name: Raw WER (streaming, 1120 ms / att_context_size=[56,13])
---

# frisian-asr-streaming-0.6b

A streaming Frisian (West Frisian, `fy-NL`) ASR model, fine-tuned from
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
on the [`LokaalHub/frisian-asr-cv22`](https://huggingface.co/datasets/LokaalHub/frisian-asr-cv22) dataset.

> **This is a community fine-tune, not an NVIDIA model.** It is a derivative of NVIDIA's
> Nemotron 3.5 ASR. NVIDIA did not produce, endorse, or review this model. "Nemotron" is a
> trademark of NVIDIA and is used here only to identify the base model.

## TL;DR

Frisian is **not** one of Nemotron 3.5 ASR's 40 supported locales, so the base model cannot
transcribe it — under any language prompt it produces garbled, Dutch-like text (≈83% WER). A
single full fine-tune on ≈40 h of Frisian, conditioned on the closest supported locale slot
(`nl-NL`, Dutch), takes it to **≈20% raw WER at 80 ms streaming** — a **75% relative
reduction** — on a leak-free, speaker- and sentence-disjoint test set.

**Raw WER (%) on held-out Frisian test (3,173 clips), cache-aware streaming. Same evaluation
for both the base and the fine-tuned model. Lowest-latency streaming, `att_context_size=[56,0]` (80 ms):**

| Language        | Base Model WER | Fine-tuned WER | Relative Improvement |
|-----------------|----------------|----------------|----------------------|
| Frisian (nl-NL) | 82.87%         | **20.36%**     | **75.4%**            |

## ⚠️ Important limitations — read before use

- **This model is monolingual Frisian.** It was fully fine-tuned on Frisian-only data with **no
  multilingual replay**. As a result it has **catastrophically forgotten the other 39 locales** of
  the base model. It will transcribe (almost) any input as Frisian. Do **not** use it as a
  drop-in replacement for the multilingual base model. (NVIDIA's fine-tuning guidance recommends
  mixing in replay data to preserve other languages; that was intentionally omitted here because
  the goal was a dedicated Frisian model — see [Relationship to NVIDIA's recipe](#relationship-to-nvidias-fine-tuning-recipe).)
- **Frisian rides the `nl-NL` (Dutch) prompt slot.** You must pass `target_lang=nl-NL` at
  inference. The tag `nl-NL` no longer means "Dutch" for this checkpoint — it means "Frisian."
- **Domain is read speech.** Training and test data are Common Voice (read sentences, varied
  consumer mics). Expect degradation on spontaneous, conversational, noisy, or far-field audio.
- **Reported WER is raw** (no text normalization beyond removing the language tag): casing and
  punctuation count as errors. This matches NVIDIA's reporting convention and is *stricter* than
  a normalized WER (which is ≈2 points lower, see [Evaluation](#evaluation)).

## How to use

This is a NeMo checkpoint (`.nemo`). It requires NeMo (≥ 26.06, for the prompt-conditioned model).

```python
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("LokaalHub/frisian-asr-streaming-0.6b",
                       "nemotron-3.5-frisian.nemo")  # .nemo filename retained from training
model = nemo_asr.models.ASRModel.restore_from(ckpt)
```

Offline / full-context transcription:

```python
# Frisian rides the Dutch (nl-NL) prompt slot — this is required.
hyps = model.transcribe(["clip.wav"])  # ensure manifests/inputs carry target_lang=nl-NL
```

Cache-aware **streaming** transcription (the condition the model is benchmarked in), using
NVIDIA's reference script from the NeMo repo:

```bash
python NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path=nemotron-3.5-frisian.nemo \
    dataset_manifest=test.json \
    target_lang=nl-NL \
    att_context_size="[56,0]" \   # 80 ms (lowest latency). Use [56,13] for 1120 ms / best accuracy.
    decoder_type=rnnt \
    pad_and_drop_preencoded=true \
    strip_lang_tags=true \
    batch_size=32
```

Each manifest line: `{"audio_filepath": "...wav", "duration": 1.23, "text": "...", "lang": "nl-NL", "target_lang": "nl-NL"}`.

## Model description

- **Base model:** [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
  — Cache-Aware FastConformer-RNNT, 600M params, 24 encoder layers, `d_model=512`,
  language-ID prompt conditioning over 40 locales.
- **Adaptation strategy:** Frisian is not in the 40 locales. The language prompt is a fixed
  128-way embedding, so an invented `<fy-NL>` tag would map to an *untrained* slot. Instead,
  every training clip is tagged `target_lang=nl-NL` — Dutch is Frisian's closest high-resource
  neighbour (orthographically and phonetically), giving a warm start. Fine-tuning then
  specialises the `nl-NL` slot (and the shared weights) to Frisian.
- **Tokenizer:** reused unchanged from the base model. NeMo guidance recommends reusing the
  pretrained tokenizer when the fine-tuning set is small (< 50 h); ours is ≈40 h.

## Training data

[`LokaalHub/frisian-asr-cv22`](https://huggingface.co/datasets/LokaalHub/frisian-asr-cv22)
(public, CC0), derived from Common Voice 22.0 Frisian (`fy-NL`):

| Split | Hours | Notes |
|-------|-------|-------|
| train | 40.4 h | 3,924 validated clips + 26,005 quality-filtered Common Voice `other` clips |
| dev   | 4.6 h  | validation during training |
| test  | 4.7 h  | 3,173 clips, held out |

Splits are **speaker- and sentence-disjoint** — no speaker or sentence appears in more than one
split — so the before/after comparison is leak-free.

## Training procedure

Full fine-tune with NeMo's `speech_to_text_finetune.py` and the prompt-conditioned
cache-aware streaming config, initialised from the base checkpoint via `init_from_nemo_model`.
This follows NVIDIA's official
[Nemotron 3.5 ASR fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb).

```bash
python NeMo/examples/asr/speech_to_text_finetune.py \
  --config-path=NeMo/examples/asr/conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming_prompt \
  +init_from_nemo_model=base.nemo \
  model.train_ds.manifest_filepath=train.json \
  model.validation_ds.manifest_filepath=dev.json \
  model.train_ds.is_tarred=false \
  model.train_ds.batch_duration=200 \
  model.optim.name=adamw \
  model.optim.lr=0.1 \
  model.optim.weight_decay=0.001 \
  model.optim.sched.warmup_steps=500 \
  model.optim.sched.d_model=512 \
  trainer.max_steps=8000 \
  trainer.devices=1 \
  trainer.precision=bf16 \
  trainer.accelerator=gpu
```

**Hyperparameters**

| | Value | Source |
|---|---|---|
| script / config | `speech_to_text_finetune.py` / `fastconformer_transducer_bpe_streaming_prompt` | NVIDIA notebook |
| optimizer | AdamW, weight_decay 1e-3 | NVIDIA notebook |
| base LR | 0.1 (NoamAnnealing) | NVIDIA notebook |
| `sched.d_model` | **512** (matches this checkpoint's real encoder dim) | model card |
| `warmup_steps` | 500 | this work |
| `batch_duration` | 200 (dynamic batching by audio seconds) | NVIDIA notebook |
| schedule | `max_steps=8000` (step budget) | NVIDIA blog (recommended for streaming/iterable data) |
| precision | bf16 | NVIDIA notebook |
| tokenizer | reused from base (no vocab change) | NeMo < 50 h guidance |

With NoamAnnealing the effective peak LR is `lr · d_model^-0.5 · warmup^-0.5 ≈ 0.1 · 512^-0.5 · 500^-0.5 ≈ 2.0e-4`.

**Hardware:** 1× NVIDIA A100 80GB (Hugging Face Jobs). Wall-clock ≈ 1 h 35 m. Best validation
WER (greedy, dev) reached 0.233.

### Relationship to NVIDIA's fine-tuning recipe

The **training recipe is faithful** to NVIDIA's official notebook on every load-bearing axis:
script, config, `init_from_nemo_model`, AdamW + base LR 0.1, `batch_duration=200`, bf16, dual-tag
(`lang` + `target_lang`) manifests with `lang_field=target_lang`, and tokenizer reuse.

Three documented choices differ from the notebook, all defensible:

1. **`sched.d_model=512` (notebook uses 1024).** The notebook's `1024` is a generic template
   value; the official model card states this 0.6B checkpoint has `D=512` / 24 encoder layers.
   512 is the architecturally-correct value for the NoamAnnealing constant here.
2. **`warmup_steps=500` (notebook: 100).** A gentler warmup; benign free hyperparameter.
3. **Step budget `max_steps=8000` (notebook demo: epochs).** NVIDIA's blog states a fixed step
   budget is "the right way to schedule with streaming/iterable data."

Two aspects are **extensions of**, not part of, NVIDIA's documented guidance:

- **Riding `nl-NL` for an out-of-set language.** NVIDIA's published examples (Greek, Bulgarian)
  fine-tune locales that already exist in the 40. Frisian does not, so we reuse the nearest
  supported slot. This is a deliberate workaround, not an NVIDIA-endorsed procedure.
- **No replay → monolingual result.** NVIDIA recommends replay to protect other languages; we
  omitted it on purpose to produce a dedicated Frisian model (see [Limitations](#️-important-limitations--read-before-use)).

## Evaluation

Evaluated with NVIDIA's own cache-aware streaming script
(`speech_to_text_cache_aware_streaming_infer.py`), `target_lang=nl-NL`, `decoder_type=rnnt`,
`strip_lang_tags=true`. Metric is **raw WER** (NeMo `word_error_rate`, no normalization). The
base and fine-tuned models are evaluated identically.

**Latency ladder (fine-tuned, raw WER %, cache-aware streaming):**

| `att_context_size` | Latency | WER    |
|--------------------|---------|--------|
| `[56, 0]`          | 80 ms   | 20.36% |
| `[56, 1]`          | 160 ms  | 19.78% |
| `[56, 3]`          | 320 ms  | 18.70% |
| `[56, 6]`          | 560 ms  | 18.40% |
| `[56, 13]`         | 1120 ms | 18.01% |

**Base vs fine-tuned (raw WER %):**

| `att_context_size` | Base   | Fine-tuned | Rel. improvement |
|--------------------|--------|------------|------------------|
| `[56, 0]` (80 ms)  | 82.87% | 20.36%     | 75.4%            |
| `[56, 13]` (1120 ms) | 81.69% | 18.01%   | 77.9%            |

The ≈2.3-point spread across the latency ladder (20.4% @ 80 ms → 18.0% @ 1120 ms) is the
expected accuracy-vs-latency tradeoff of cache-aware streaming: less look-ahead means lower
latency and slightly higher WER. One checkpoint serves the whole ladder.
For reference, in **offline / full-context** decoding the fine-tuned model scores ≈18.7% raw
(≈17.4% normalized) — essentially equal to the 1120 ms streaming number, as expected. The base
model's own published Dutch (`nl-NL`) WER is 11.46% @ 1120 ms; a lower-resource language riding
that slot landing at 18.0% @ 1120 ms is a credible result.

## License

This model is a derivative of `nvidia/nemotron-3.5-asr-streaming-0.6b` and is released under the
same **[OpenMDW-1.1](https://openmdw.ai/license/1-1/)** terms as the base model. The training
data (`LokaalHub/frisian-asr-cv22`) is CC0. You are responsible for complying with the base
model's license and Common Voice's terms for any downstream use.

## Citation

```bibtex
@misc{frisian-asr-streaming-0.6b,
  title  = {frisian-asr-streaming-0.6b: a streaming Frisian ASR fine-tune of Nemotron 3.5 ASR},
  author = {LokaalHub},
  year   = {2026},
  note   = {Fine-tune of nvidia/nemotron-3.5-asr-streaming-0.6b on LokaalHub/frisian-asr-cv22},
  url    = {https://huggingface.co/LokaalHub/frisian-asr-streaming-0.6b}
}
```

**Base model:** [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) ·
**Recipe:** [NVIDIA Riva fine-tuning notebook](https://github.com/nvidia-riva/tutorials/blob/main/asr-finetune-nemotron-3.5-asr-streaming-prompt.ipynb) ·
**Framework:** [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
