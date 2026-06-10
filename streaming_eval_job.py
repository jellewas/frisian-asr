"""HF Job (L4): re-evaluate base vs Frisian-fine-tuned Nemotron in NVIDIA's
cache-aware STREAMING mode and report RAW WER (%) per latency.

Why this exists: the showcase run measured WER with transcribe_speech.py, which
runs the encoder in full-context (offline) mode — the easiest condition. Every
number NVIDIA publishes uses cache-aware *streaming* at a fixed att_context_size
(their headline is [56,0] = 80 ms, the most demanding). This job uses NVIDIA's own
speech_to_text_cache_aware_streaming_infer.py so our numbers sit next to theirs
without an asterisk. Metric = NeMo's word_error_rate (RAW, no normalization),
the same one the script logs as "WER% of streaming mode".

Reports the NVIDIA table format:
  | Language | Base Model WER | Fine-tuned WER | Relative Improvement |
plus a fine-tuned latency ladder.
"""
import glob
import json
import os
import re
import subprocess
import sys


def set_cuda_home():
    """Point numba's libnvvm loader at the nvidia-cuda-nvcc wheel (RNNT decode JIT)."""
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
OUT_REPO = os.environ.get("OUT_REPO", "LokaalHub/nemotron-3.5-frisian")
FT_FILE = os.environ.get("FT_FILE", "nemotron-3.5-frisian.nemo")
LIMIT = int(os.environ.get("LIMIT", "0")) or None

# att_context_size -> end-to-end latency (from the model card's streaming table)
LAT = {"[56, 0]": "80 ms", "[56, 1]": "160 ms", "[56, 3]": "320 ms",
       "[56, 6]": "560 ms", "[56, 13]": "1120 ms"}

# Smoke validates the wiring on a handful of clips at the headline latency only.
# Full run: base at the two anchor latencies, fine-tuned across the whole ladder.
if LIMIT:
    BASE_ATTS = [[56, 0]]
    FT_ATTS = [[56, 0]]
else:
    BASE_ATTS = [[56, 0], [56, 13]]
    FT_ATTS = [[56, 0], [56, 1], [56, 3], [56, 6], [56, 13]]


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# --- 1. materialize test WAVs + manifest --------------------------------------
import io

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
    ds = ds.cast_column("audio", Audio(decode=False))
    if LIMIT:
        ds = ds.select(range(min(LIMIT, len(ds))))
    d = f"wav/{split}"
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        for i, ex in enumerate(ds):
            arr, sr = decode_audio(ex["audio"])
            if sr != SR:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=SR)
            p = os.path.abspath(f"{d}/{i:05d}.wav")
            sf.write(p, arr, SR)
            f.write(json.dumps({"audio_filepath": p, "duration": round(len(arr) / SR, 3),
                                "text": ex["text"], "lang": TARGET_LANG, "target_lang": TARGET_LANG},
                               ensure_ascii=False) + "\n")
    print(f"{split}: manifest -> {path}", flush=True)


print("=== 1. data prep (test split) ===", flush=True)
build_manifest("test", "test.json")
N = sum(1 for _ in open("test.json"))

# --- 2. NeMo source + both checkpoints ----------------------------------------
print("=== 2. fetch NeMo + checkpoints ===", flush=True)
if not os.path.isdir("NeMo"):
    sh("git clone --depth 1 https://github.com/NVIDIA/NeMo.git")

import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download

base = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
base.save_to("base.nemo")
del base
base_path = os.path.abspath("base.nemo")
print("saved base.nemo", flush=True)

ft_path = hf_hub_download(repo_id=OUT_REPO, filename=FT_FILE, token=TOKEN)
print("downloaded fine-tuned:", ft_path, flush=True)

INFER = "NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py"


def stream_wer(model_path, att, tag):
    """Run NVIDIA's streaming eval; return the RAW WER% it logs."""
    att_str = "[" + ",".join(str(x) for x in att) + "]"
    lat = LAT.get(f"[{att[0]}, {att[1]}]", "?")
    out = os.path.abspath(f"out_{tag}_{att[0]}_{att[1]}")
    cmd = (f"python {INFER} model_path={model_path} "
           f"dataset_manifest={os.path.abspath('test.json')} output_path={out} "
           f"target_lang={TARGET_LANG} att_context_size=\"{att_str}\" decoder_type=rnnt "
           f"pad_and_drop_preencoded=true batch_size=32 strip_lang_tags=true cuda=0")
    print("+", cmd, flush=True)
    # WER is emitted via logging (stderr); merge so we can parse it.
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    print(res.stdout[-4000:], flush=True)
    if res.returncode != 0:
        sys.exit(f"streaming infer failed for {tag} {att_str} (rc={res.returncode})")
    m = re.findall(r"WER% of streaming mode:\s*([0-9.]+)", res.stdout)
    if not m:
        sys.exit(f"could not parse WER for {tag} {att_str} — see log above")
    wer = float(m[-1])
    print(f"[{tag}] att={att_str} ({lat}) RAW WER = {wer:.2f}%", flush=True)
    return wer


# --- 3. run base + fine-tuned across latencies --------------------------------
print("=== 3. streaming eval ===", flush=True)
base_wer = {f"[{a[0]}, {a[1]}]": stream_wer(base_path, a, "base") for a in BASE_ATTS}
ft_wer = {f"[{a[0]}, {a[1]}]": stream_wer(ft_path, a, "finetuned") for a in FT_ATTS}

# --- 4. report in NVIDIA's format ---------------------------------------------
HEAD = "[56, 0]"  # NVIDIA's headline condition: lowest-latency streaming, 80 ms
b0, f0 = base_wer[HEAD], ft_wer[HEAD]
rel = (b0 - f0) / b0 * 100 if b0 else 0.0

print("\n" + "=" * 66, flush=True)
print(f"RAW WER (%) on held-out Frisian test ({N} clips), cache-aware streaming.", flush=True)
print("Same evaluation for both the base and the fine-tuned models.", flush=True)
print(f"Lowest-latency streaming: att_context_size={HEAD} ({LAT[HEAD]}).\n", flush=True)
print(f"| Language        | Base Model WER | Fine-tuned WER | Relative Improvement |", flush=True)
print(f"|-----------------|----------------|----------------|----------------------|", flush=True)
print(f"| Frisian (nl-NL) | {b0:>13.0f}% | {f0:>13.0f}% | {rel:>19.0f}% |", flush=True)

print("\nFine-tuned latency ladder (RAW WER %, cache-aware streaming):", flush=True)
print(f"| att_context_size | Latency | WER  |", flush=True)
print(f"|------------------|---------|------|", flush=True)
for a in FT_ATTS:
    k = f"[{a[0]}, {a[1]}]"
    print(f"| {k:>16} | {LAT[k]:>7} | {ft_wer[k]:>4.1f}% |", flush=True)
print("=" * 66, flush=True)

result = {
    "model": MODEL, "dataset": DATASET, "target_lang": TARGET_LANG,
    "metric": "raw WER (NeMo word_error_rate), cache-aware streaming",
    "test_clips": N,
    "headline": {"att_context_size": HEAD, "latency": LAT[HEAD],
                 "base_wer": b0, "finetuned_wer": f0, "relative_improvement_pct": rel},
    "base_wer_by_att": base_wer, "finetuned_wer_by_att": ft_wer,
}
with open("streaming_result.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# --- 5. push the streaming numbers next to the model --------------------------
if OUT_REPO and not LIMIT:
    from huggingface_hub import HfApi
    HfApi(token=TOKEN).upload_file(
        path_or_fileobj="streaming_result.json",
        path_in_repo="streaming_result.json", repo_id=OUT_REPO)
    print("pushed streaming_result.json ->", OUT_REPO, flush=True)
