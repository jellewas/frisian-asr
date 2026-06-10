"""HF Job: measure the BASE Nemotron 3.5 ASR WER on Frisian (the 'before' number).

Runs remotely via run_uv_job. Loads our published test split, materializes 16 kHz
mono WAVs + a NeMo manifest, loads nvidia/nemotron-3.5-asr-streaming-0.6b, transcribes
under an existing language tag (nl-NL — Frisian's closest trained locale, since fy-NL
is not one of the 40), and reports WER at deployment streaming latency.

Frisian isn't in the model's 40 locales and the language prompt is a fixed embedding,
so a brand-new tag is untrained — we deliberately ride the nl-NL (Dutch) slot. Expect
a high 'before' WER; that is the point of the showcase.

Config via env:
  HF_TOKEN (secret, required)   DATASET   default LokaalHub/frisian-asr-cv22
  MODEL    default nvidia/nemotron-3.5-asr-streaming-0.6b
  SPLIT    default test          TARGET_LANG default nl-NL
  ATT_CONTEXT default [56,0] (80 ms chunk, deployment latency)
  LIMIT    >0 => smoke test on N clips     BATCH default 16
"""
import json
import os
import re

SR = 16000
TOKEN = os.environ["HF_TOKEN"]
DATASET = os.environ.get("DATASET", "LokaalHub/frisian-asr-cv22")
MODEL = os.environ.get("MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
SPLIT = os.environ.get("SPLIT", "test")
TARGET_LANG = os.environ.get("TARGET_LANG", "nl-NL")
ATT_CONTEXT = os.environ.get("ATT_CONTEXT", "[56,0]")
LIMIT = int(os.environ.get("LIMIT", "0")) or None
BATCH = int(os.environ.get("BATCH", "16"))


# --- text normalization for fair WER (model emits caps+punct; refs are lower) ---
def norm(s):
    s = re.sub(r"<[^>]+>", " ", s)  # drop <nl-NL> style language tags the model appends
    s = s.lower().strip()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)  # drop punctuation, keep apostrophes
    return " ".join(s.split())


# --- 1. materialize WAVs + NeMo manifest from our HF dataset --------------------
import io

import numpy as np
import soundfile as sf
import librosa
from datasets import Audio, load_dataset

print(f"loading {DATASET}:{SPLIT} (limit={LIMIT}) ...", flush=True)
ds = load_dataset(DATASET, split=SPLIT, token=TOKEN)
ds = ds.cast_column("audio", Audio(decode=False))  # avoid torchcodec; decode bytes ourselves
if LIMIT:
    ds = ds.select(range(min(LIMIT, len(ds))))


def decode_audio(a):
    src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
    arr, sr = sf.read(src, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


os.makedirs("wav", exist_ok=True)
MANIFEST = "manifest.json"
refs, files = [], []
with open(MANIFEST, "w") as f:
    for i, ex in enumerate(ds):
        arr, sr = decode_audio(ex["audio"])
        if sr != SR:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
        p = os.path.abspath(f"wav/{i:05d}.wav")
        sf.write(p, arr, SR)
        text = ex["text"]
        # NeMo's lhotse adapter maps manifest 'lang' -> supervision.language, which the
        # prompt-index dataloader reads to pick the language prompt. This is THE wiring.
        f.write(json.dumps({"audio_filepath": p, "duration": round(len(arr) / SR, 3),
                            "text": text, "lang": TARGET_LANG}, ensure_ascii=False) + "\n")
        refs.append(text)
        files.append(p)
print(f"materialized {len(files)} clips", flush=True)

# --- 2. transcribe via NVIDIA's offline script (reads manifest 'lang') ----------
import subprocess

if not os.path.isdir("NeMo"):
    subprocess.run("git clone --depth 1 https://github.com/NVIDIA/NeMo.git", shell=True, check=True)

PREDS = os.path.abspath("preds.json")
cmd = (
    f"python NeMo/examples/asr/transcribe_speech.py "
    f"pretrained_name={MODEL} "
    f"dataset_manifest={os.path.abspath(MANIFEST)} "
    f"output_filename={PREDS} "
    f"batch_size={BATCH}"
)  # lang tags (<nl-NL>) are stripped in norm(); transcribe_speech has no strip option
print("+", cmd, flush=True)
subprocess.run(cmd, shell=True, check=True)

# --- 3. parse predictions (aligned by audio_filepath) --------------------------
preds = {}
for line in open(PREDS):
    r = json.loads(line)
    preds[r["audio_filepath"]] = r.get("pred_text", "")
hyps = [preds.get(p, "") for p in files]

# --- 4. WER --------------------------------------------------------------------
import jiwer

wer_raw = jiwer.wer(refs, hyps)
wer_norm = jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])

print("\n================ BASELINE RESULT ================", flush=True)
print(f"model           : {MODEL}", flush=True)
print(f"dataset/split   : {DATASET}:{SPLIT}  ({len(files)} clips)", flush=True)
print(f"target_lang     : {TARGET_LANG}", flush=True)
print(f"WER (raw)       : {wer_raw*100:.2f}%", flush=True)
print(f"WER (normalized): {wer_norm*100:.2f}%", flush=True)
print("--- 5 sample (ref | hyp) ---", flush=True)
for r, h in list(zip(refs, hyps))[:5]:
    print(f"  REF: {r}\n  HYP: {h}\n", flush=True)
print("=================================================", flush=True)

with open("baseline_result.json", "w") as f:
    json.dump({"model": MODEL, "dataset": DATASET, "split": SPLIT, "n": len(files),
               "target_lang": TARGET_LANG,
               "wer_raw": wer_raw, "wer_norm": wer_norm}, f, indent=2)
