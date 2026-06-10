"""HF Jobs build script: assemble the filtered Frisian ASR dataset on HF GPU infra.

Runs remotely via `huggingface_hub.run_uv_job` (see launch_job.py). Loads Common Voice
fy-NL straight from a HF script-mirror (no manual tarball), filters the `other` bucket
with a Frisian wav2vec2 model on GPU, and pushes a private DatasetDict to LokaalHub.

Config via env:
  HF_TOKEN     (secret, required)   CV_DATASET   default mozilla-foundation/common_voice_17_0
  HF_REPO      target repo          ASR_MODEL    default greenw0lf/wav2vec2-large-xls-r-1b-frisian
  MAX_CER      default 0.25         LIMIT        >0 => streaming smoke test of N clips/split
"""
import os
import unicodedata
from pathlib import Path

SR = 16000
TOKEN = os.environ["HF_TOKEN"]
SRC = os.environ.get("CV_DATASET", "fsicoli/common_voice_22_0")
REPO = os.environ.get("HF_REPO", "LokaalHub/frisian-asr-cv22")
MODEL = os.environ.get("ASR_MODEL", "greenw0lf/wav2vec2-large-xls-r-1b-frisian")
MAX_CER = float(os.environ.get("MAX_CER", "0.25"))
LIMIT = int(os.environ.get("LIMIT", "0")) or None


# --- pure helpers (mirrored from build_dataset.py) -------------------------
def normalize_text(text):
    s = unicodedata.normalize("NFC", str(text)).strip()
    for bad, good in {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}.items():
        s = s.replace(bad, good)
    return " ".join(s.split())


def is_valid_text(text):
    return any(ch.isalpha() for ch in text)


def char_error_rate(reference, hypothesis):
    ref, hyp = reference.strip(), hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cost = 0 if rc == hc else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / len(ref)


# --- load CV fy-NL from the HF mirror --------------------------------------
import itertools

from datasets import Audio, Dataset, DatasetDict, load_dataset


def load_split(name):
    ds = load_dataset(SRC, "fy-NL", split=name, trust_remote_code=True,
                      token=TOKEN, streaming=bool(LIMIT))
    if LIMIT:
        ds = list(itertools.islice(ds, LIMIT))
    return ds


def col(rows, key):
    return [r[key] for r in rows] if isinstance(rows, list) else rows[key]


print(f"loading {SRC} fy-NL (limit={LIMIT}) ...", flush=True)
# fsicoli exposes train/dev/test/other (no combined 'validated' split). train/dev/test
# are validated clips; 'other' is the unvalidated bucket we filter. The ~41k leftover
# validated clips have no audio in any HF mirror, so train-core = the official train split.
train_core = load_split("train")
other = load_split("other")
dev = load_split("validation")  # fsicoli exposes the dev split as 'validation'
test = load_split("test")

held_clients = set(col(dev, "client_id")) | set(col(test, "client_id"))
held_sents = {normalize_text(s) for s in list(col(dev, "sentence")) + list(col(test, "sentence"))}

# --- GPU filter model ------------------------------------------------------
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
proc = AutoProcessor.from_pretrained(MODEL, token=TOKEN)
model = Wav2Vec2ForCTC.from_pretrained(MODEL, token=TOKEN).to(device).eval()


def quality_score(arr, sentence):
    inp = proc(arr, sampling_rate=SR, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inp).logits
    hyp = proc.decode(torch.argmax(logits, dim=-1)[0])
    return 1.0 - char_error_rate(normalize_text(sentence).lower(), hyp.lower())


# --- materialize -----------------------------------------------------------
import soundfile as sf
import librosa

OUT = Path("wav")
rows = {"train": [], "dev": [], "test": []}


def to_16k(ex):
    a = ex["audio"]
    arr, sr = a["array"], a["sampling_rate"]
    return librosa.resample(arr, orig_sr=sr, target_sr=SR) if sr != SR else arr


def emit(ex, split, source, qs=1.0):
    text = normalize_text(ex["sentence"])
    if not is_valid_text(text):
        return
    arr = to_16k(ex)
    dur = len(arr) / SR
    if not (1.0 <= dur <= 20.0):
        return
    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    p = d / (Path(ex["path"]).stem + ".wav")
    sf.write(str(p), arr, SR)
    rows[split].append({
        "audio": str(p), "duration": round(dur, 3), "text": text, "lang": "fy-NL",
        "client_id": ex["client_id"], "split": split, "source": source,
        "quality_score": float(qs),
    })


def iter_rows(ds):
    return ds if isinstance(ds, list) else ds


for ex in iter_rows(test):
    emit(ex, "test", "validated")
for ex in iter_rows(dev):
    emit(ex, "dev", "validated")
for ex in iter_rows(train_core):
    if ex["client_id"] in held_clients or normalize_text(ex["sentence"]) in held_sents:
        continue
    emit(ex, "train", "validated")

kept = total = 0
for ex in iter_rows(other):
    if ex["client_id"] in held_clients or normalize_text(ex["sentence"]) in held_sents:
        continue
    total += 1
    qs = quality_score(to_16k(ex), ex["sentence"])
    if qs < 1.0 - MAX_CER:
        continue
    kept += 1
    emit(ex, "train", "other", qs)
    if kept % 500 == 0:
        print(f"  filtered other: kept {kept}/{total}", flush=True)

print(f"filter 'other': kept {kept}/{total}", flush=True)
for s, rs in rows.items():
    print(f"  {s}: {len(rs)} clips, {sum(r['duration'] for r in rs)/3600:.1f} h", flush=True)

# --- push ------------------------------------------------------------------
dd = DatasetDict({
    s: Dataset.from_list(rs).cast_column("audio", Audio(sampling_rate=SR))
    for s, rs in rows.items() if rs
})
dd.push_to_hub(REPO, private=True, token=TOKEN)
print("pushed (private):", REPO, flush=True)
