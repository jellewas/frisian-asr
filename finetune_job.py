"""HF Job (A100): fine-tune Nemotron 3.5 ASR on Frisian, then report before/after WER.

Pipeline:
  1. materialize train/dev/test WAVs + NeMo manifests from LokaalHub/frisian-asr-cv22
     (every clip tagged target_lang=nl-NL — Frisian rides Dutch, its closest trained slot)
  2. clone NeMo (for examples/asr/speech_to_text_finetune.py + the prompt config) and
     download the base .nemo checkpoint
  3. fine-tune with init_from_nemo_model, step-budget driven, bf16, single GPU
  4. evaluate base vs fine-tuned on the held-out test split at deployment latency
  5. push the fine-tuned model to HF (private)

Tokenizer is reused from the base model (data < 50 h) — no vocab surgery.

Config via env (all optional except HF_TOKEN):
  HF_TOKEN(secret)  DATASET=LokaalHub/frisian-asr-cv22  MODEL=nvidia/nemotron-3.5-asr-streaming-0.6b
  TARGET_LANG=nl-NL  MAX_STEPS=8000  LR=2e-4  WARMUP=500  BATCH_DURATION=200
  ATT_CONTEXT=[56,0]  OUT_REPO=LokaalHub/nemotron-3.5-frisian  LIMIT(>0 smoke)
"""
import glob
import json
import os
import re
import subprocess
import sys


def set_cuda_home():
    """Point numba's libnvvm loader at the nvidia-cuda-nvcc pip wheel (RNNT loss JIT).

    The wheel ships only versioned libnvvm.so.N; numba does CDLL('libnvvm.so'), so we
    create the unversioned symlink and put the dir on LD_LIBRARY_PATH for the child.
    """
    for d in sys.path:
        hits = glob.glob(os.path.join(d, "nvidia", "cuda_nvcc", "nvvm", "lib64", "libnvvm.so*"))
        if not hits:
            continue
        home = os.path.join(d, "nvidia", "cuda_nvcc")
        libdir = os.path.join(home, "nvvm", "lib64")
        os.environ["CUDA_HOME"] = home
        os.environ["LD_LIBRARY_PATH"] = libdir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        unversioned = os.path.join(libdir, "libnvvm.so")
        if not os.path.exists(unversioned):
            try:
                os.symlink(hits[0], unversioned)
            except OSError as e:
                print("WARN symlink:", e, flush=True)
        print("CUDA_HOME ->", home, "| libnvvm:", os.path.basename(hits[0]), flush=True)
        return
    print("WARN libnvvm.so not found on sys.path", flush=True)


set_cuda_home()

SR = 16000
TOKEN = os.environ["HF_TOKEN"]
DATASET = os.environ.get("DATASET", "LokaalHub/frisian-asr-cv22")
MODEL = os.environ.get("MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
TARGET_LANG = os.environ.get("TARGET_LANG", "nl-NL")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8000"))
# NoamAnnealing base LR: effective peak = LR * d_model^-0.5 * warmup^-0.5. With d_model=512
# and warmup=500, base 0.1 -> peak ~2e-4. (Matches NVIDIA's notebook; not a toy value.)
LR = float(os.environ.get("LR", "0.1"))
WARMUP = int(os.environ.get("WARMUP", "500"))
BATCH_DURATION = int(os.environ.get("BATCH_DURATION", "200"))
ATT_CONTEXT = os.environ.get("ATT_CONTEXT", "[56,0]")
OUT_REPO = os.environ.get("OUT_REPO", "LokaalHub/nemotron-3.5-frisian")
LIMIT = int(os.environ.get("LIMIT", "0")) or None


def norm(s):
    s = re.sub(r"<[^>]+>", " ", s)  # drop <nl-NL> style language tags
    s = s.lower().strip()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# --- 1. materialize WAVs + manifests ------------------------------------------
import io

import numpy as np
import soundfile as sf
import librosa
from datasets import Audio, load_dataset


def decode_audio(a):
    src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
    arr, sr = sf.read(src, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def build_manifest(split, path):
    ds = load_dataset(DATASET, split=split, token=TOKEN)
    ds = ds.cast_column("audio", Audio(decode=False))  # avoid torchcodec
    if LIMIT:
        ds = ds.select(range(min(LIMIT, len(ds))))
    d = f"wav/{split}"
    os.makedirs(d, exist_ok=True)
    refs, files = [], []
    with open(path, "w") as f:
        for i, ex in enumerate(ds):
            arr, sr = decode_audio(ex["audio"])
            if sr != SR:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
            p = os.path.abspath(f"{d}/{i:05d}.wav")
            sf.write(p, arr, SR)
            # training config reads lang_field='target_lang'; transcribe_speech reads 'lang'.
            # Write both so the same manifests serve training, validation, and eval.
            f.write(json.dumps({"audio_filepath": p, "duration": round(len(arr) / SR, 3),
                                "text": ex["text"], "lang": TARGET_LANG, "target_lang": TARGET_LANG},
                               ensure_ascii=False) + "\n")
            refs.append(ex["text"])
            files.append(p)
    print(f"{split}: {len(files)} clips -> {path}", flush=True)
    return files, refs


print("=== 1. data prep ===", flush=True)
build_manifest("train", "train.json")
build_manifest("dev", "dev.json")
test_files, test_refs = build_manifest("test", "test.json")

# --- 2. NeMo source + base checkpoint -----------------------------------------
print("=== 2. fetch NeMo + base checkpoint ===", flush=True)
if not os.path.isdir("NeMo"):
    sh("git clone --depth 1 https://github.com/NVIDIA/NeMo.git")

import torch
import nemo.collections.asr as nemo_asr
import jiwer

base = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
base.save_to("base.nemo")
del base
print("saved base.nemo", flush=True)


def eval_nemo(model_path, tag):
    """Transcribe the test split via NVIDIA's offline script (reads manifest 'lang')."""
    preds = os.path.abspath(f"preds_{tag}.json")
    sh(f"python NeMo/examples/asr/transcribe_speech.py model_path={model_path} "
       f"dataset_manifest={os.path.abspath('test.json')} output_filename={preds} "
       f"batch_size=16")  # <nl-NL> tags stripped in norm()
    pmap = {}
    for line in open(preds):
        r = json.loads(line)
        pmap[r["audio_filepath"]] = r.get("pred_text", "")
    hyps = [pmap.get(p, "") for p in test_files]
    wer = jiwer.wer([norm(r) for r in test_refs], [norm(h) for h in hyps])
    print(f"[{tag}] WER (normalized) = {wer*100:.2f}%", flush=True)
    return wer, hyps


print("=== 3a. BEFORE (base on test) ===", flush=True)
wer_before, _ = eval_nemo(os.path.abspath("base.nemo"), "base")

# --- 3. fine-tune --------------------------------------------------------------
print("=== 3b. fine-tune ===", flush=True)
CFG_DIR = "NeMo/examples/asr/conf/fastconformer/cache_aware_streaming"
CFG_NAME = "fastconformer_transducer_bpe_streaming_prompt"
sh(
    f"python NeMo/examples/asr/speech_to_text_finetune.py "
    f"--config-path={os.path.abspath(CFG_DIR)} --config-name={CFG_NAME} "
    f"+init_from_nemo_model={os.path.abspath('base.nemo')} "
    f"model.train_ds.manifest_filepath={os.path.abspath('train.json')} "
    f"model.validation_ds.manifest_filepath={os.path.abspath('dev.json')} "
    f"model.train_ds.is_tarred=false model.train_ds.batch_duration={BATCH_DURATION} "
    f"model.optim.name=adamw model.optim.lr={LR} model.optim.weight_decay=0.001 "
    f"model.optim.sched.warmup_steps={WARMUP} model.optim.sched.d_model=512 "
    f"trainer.max_steps={MAX_STEPS} trainer.devices=1 trainer.precision=bf16 "
    f"trainer.accelerator=gpu "
    f"exp_manager.exp_dir={os.path.abspath('exp')} "
    f"exp_manager.name=frisian_ft "
)

# locate the trained .nemo
import glob
cands = sorted(glob.glob("exp/**/*.nemo", recursive=True), key=os.path.getmtime)
if not cands:
    sys.exit("no trained .nemo produced — inspect training logs above")
trained_path = cands[-1]
print("trained checkpoint:", trained_path, flush=True)

# --- 4. AFTER + report ---------------------------------------------------------
print("=== 4. AFTER (fine-tuned on test) ===", flush=True)
wer_after, _ = eval_nemo(trained_path, "finetuned")

print("\n================ SHOWCASE RESULT ================", flush=True)
print(f"test clips        : {len(test_files)}", flush=True)
print(f"target_lang       : {TARGET_LANG}", flush=True)
print(f"WER before (base) : {wer_before*100:.2f}%", flush=True)
print(f"WER after (FT)    : {wer_after*100:.2f}%", flush=True)
rel = (wer_before - wer_after) / wer_before * 100 if wer_before else 0.0
print(f"relative WER drop : {rel:.1f}%", flush=True)
print("=================================================", flush=True)

with open("showcase_result.json", "w") as f:
    json.dump({"model": MODEL, "dataset": DATASET, "target_lang": TARGET_LANG,
               "max_steps": MAX_STEPS, "lr": LR,
               "wer_before": wer_before, "wer_after": wer_after, "rel_drop_pct": rel}, f, indent=2)

# --- 5. push fine-tuned model --------------------------------------------------
if OUT_REPO:
    print("=== 5. push fine-tuned model ===", flush=True)
    from huggingface_hub import HfApi
    api = HfApi(token=TOKEN)
    api.create_repo(OUT_REPO, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=trained_path, path_in_repo="nemotron-3.5-frisian.nemo",
                    repo_id=OUT_REPO)
    api.upload_file(path_or_fileobj="showcase_result.json", path_in_repo="showcase_result.json",
                    repo_id=OUT_REPO)
    print("pushed (private):", OUT_REPO, flush=True)
