#!/usr/bin/env python3
"""Build an open Frisian ASR dataset from Common Voice 25.0 (fy-NL, CC0).

Pipeline:
  1. Read CV release tsvs (validated / other / dev / test).
  2. Build speaker- AND sentence-disjoint splits (clean eval, noisy data in train only).
  3. Auto-filter the unvalidated "other" bucket via MMS CTC-decode vs known prompt.
  4. Normalize text (preserve case + punctuation to match the Nemotron base).
  5. Transcode mp3 -> 16 kHz mono wav.
  6. Emit NeMo manifests (train/dev/test) + an HF dataset, optionally pushed to the Hub.

Heavy stages (filter, transcode, push) are guarded; pure helpers are unit-tested.
Run `python build_dataset.py --help` for usage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_pipeline.py)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize a transcript while preserving casing and punctuation.

    The Nemotron base emits punctuation + capitalization, so we keep them.
    We only NFC-normalize, fix common quote/dash variants, and collapse spaces.
    """
    import unicodedata

    s = unicodedata.normalize("NFC", str(text)).strip()
    replacements = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return " ".join(s.split())


def is_valid_text(text: str) -> bool:
    """Reject empty / punctuation-only transcripts that would poison training."""
    return any(ch.isalpha() for ch in text)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein char distance / len(reference). 0.0 == perfect, 1.0 == ref length."""
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


def build_splits(
    validated: pd.DataFrame,
    other: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Return test / dev / train_core / train_extra, speaker- and sentence-disjoint.

    Eval (dev/test) comes from CV's official high-confidence splits and is never
    diluted. Train rows that share a speaker (client_id) or a sentence with eval
    are dropped to prevent leakage. `other` (unvalidated) only ever feeds train.
    """
    held_clients = set(dev["client_id"]) | set(test["client_id"])
    held_sentences = {normalize_text(s) for s in pd.concat([dev["sentence"], test["sentence"]])}

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        norm = df["sentence"].map(normalize_text)
        keep = ~df["client_id"].isin(held_clients) & ~norm.isin(held_sentences)
        return df[keep].copy()

    return {
        "test": test.copy(),
        "dev": dev.copy(),
        "train_core": clean(validated),
        "train_extra": clean(other),
    }


def format_report(rows_by_split: dict, filter_before: int, filter_after: int) -> str:
    """One-page build summary: filter yield + per-split hours + 'other' CER spread."""
    lines = ["=== build report ==="]
    if filter_before:
        pct = 100 * filter_after / filter_before
        lines.append(f"filter 'other': kept {filter_after}/{filter_before} ({pct:.1f}%)")
    for split, rows in sorted(rows_by_split.items()):
        hrs = sum(r["duration"] for r in rows) / 3600
        lines.append(f"{split:6s}: {len(rows):>6d} clips  {hrs:6.1f} h")
        scores = sorted(r["quality_score"] for r in rows if r["source"] == "other")
        if scores:
            n = len(scores)
            pick = lambda q: scores[min(n - 1, int(q * n))]
            lines.append(f"        other quality_score p10/p50/p90: "
                         f"{pick(0.1):.2f}/{pick(0.5):.2f}/{pick(0.9):.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IO + heavy stages
# ---------------------------------------------------------------------------


def read_cv_tsvs(cv_dir: Path) -> dict:
    """Load the CV release tsvs we need. `other.tsv` is required for the extra hours."""
    needed = ["validated", "other", "dev", "test"]
    out = {}
    for name in needed:
        path = cv_dir / f"{name}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Download the official Common Voice 25.0 fy-NL "
                f"tarball (it ships {name}.tsv) and point cv_dir at the fy-NL folder."
            )
        out[name] = pd.read_csv(path, sep="\t", low_memory=False)
    return out


def transcode_to_wav(src: Path, dst: Path, sample_rate: int) -> Optional[float]:
    """mp3 -> 16 kHz mono PCM16 wav. Returns duration seconds, or None on failure."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src), "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(dst),
    ]
    if subprocess.run(cmd).returncode != 0 or not dst.exists():
        return None
    import soundfile as sf

    info = sf.info(str(dst))
    return info.frames / info.samplerate


def filter_other(df: pd.DataFrame, clips_dir: Path, cfg: dict) -> pd.DataFrame:
    """Keep `other` rows whose CTC decode matches the prompt (CER <= max_cer).

    Uses a Frisian-specialized wav2vec2 model for a sharper QC signal than generic MMS.
    Adds a `quality_score` (= 1 - CER) column. Slow on CPU; GPU strongly recommended.
    """
    import librosa
    import torch
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(cfg["asr_model"])
    model = Wav2Vec2ForCTC.from_pretrained(cfg["asr_model"]).to(device).eval()

    scores = []
    for path_rel, sentence in zip(df["path"], df["sentence"]):
        wav, _ = librosa.load(str(clips_dir / path_rel), sr=16000, mono=True)
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        hyp = processor.decode(ids[0])
        scores.append(1.0 - char_error_rate(normalize_text(sentence).lower(), hyp.lower()))

    df = df.copy()
    df["quality_score"] = scores
    return df[df["quality_score"] >= (1.0 - cfg["max_cer"])].copy()


def materialize_split(
    df: pd.DataFrame, split: str, source: str, clips_dir: Path, cfg: dict
) -> list:
    """Transcode each clip and return manifest rows for clips passing duration bounds."""
    wav_dir = Path(cfg["out_dir"]) / "wav" / split
    rows = []
    for rec in df.itertuples(index=False):
        text = normalize_text(rec.sentence)
        if not is_valid_text(text):
            continue
        dst = wav_dir / (Path(rec.path).stem + ".wav")
        dur = transcode_to_wav(clips_dir / rec.path, dst, cfg["sample_rate"])
        if dur is None or not (cfg["min_duration"] <= dur <= cfg["max_duration"]):
            continue
        rows.append({
            "audio_filepath": str(dst.resolve()),
            "duration": round(dur, 3),
            "text": text,
            "lang": cfg["target_lang"],
            "client_id": rec.client_id,
            "split": split,
            "source": source,
            "quality_score": float(getattr(rec, "quality_score", 1.0)),
        })
    return rows


def write_nemo_manifest(rows: list, path: Path) -> None:
    keys = ["audio_filepath", "duration", "text", "lang"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")


def _dataset_card(by_split: dict) -> str:
    """CC0 dataset card with HF metadata so the Hub page renders license/language/task."""
    stats = "\n".join(
        f"- `{split}`: {len(rows)} clips, "
        f"{sum(r['duration'] for r in rows) / 3600:.1f} h"
        for split, rows in sorted(by_split.items())
    )
    return f"""---
license: cc0-1.0
language:
- fy
task_categories:
- automatic-speech-recognition
pretty_name: Frisian ASR (Common Voice 25.0, filtered)
tags:
- frisian
- fy-NL
- common-voice
---

# Frisian ASR (Common Voice 25.0, filtered)

Open Standard-Frisian speech for ASR, built from Mozilla Common Voice 25.0 (`fy-NL`, CC0).
Validated split plus the `other` bucket auto-filtered by CTC agreement (Frisian wav2vec2) with the prompt.

## Splits
{stats}

`test`/`dev` are CV's official high-confidence splits; `train` is the validated remainder
plus filtered `other`, guaranteed disjoint from eval by speaker and by sentence.

## Limitations
Read speech only; heavy sentence repetition; demographic skew; `other`-bucket labels carry
residual noise (see `quality_score`). License CC0; speaker IDs are anonymized hashes.
"""


def push_to_hub(all_rows: list, cfg: dict) -> None:
    """Build an HF Audio dataset and push it publicly to LokaalHub with a CC0 card."""
    from datasets import Audio, Dataset, DatasetDict
    from huggingface_hub import DatasetCard

    by_split: dict = {}
    for r in all_rows:
        by_split.setdefault(r["split"], []).append(r)
    dd = DatasetDict({
        split: Dataset.from_list(
            [{**{k: v for k, v in r.items() if k != "audio_filepath"},
              "audio": r["audio_filepath"]} for r in rows]
        ).cast_column("audio", Audio(sampling_rate=cfg["sample_rate"]))
        for split, rows in by_split.items()
    })
    dd.push_to_hub(cfg["hf_repo"], private=False)
    DatasetCard(_dataset_card(by_split)).push_to_hub(cfg["hf_repo"], repo_type="dataset")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["filter"] = cfg.get("filter", {})
    return cfg


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per split (smoke test)")
    ap.add_argument("--no-filter", action="store_true",
                    help="skip the filter; keep all of 'other'")
    ap.add_argument("--push", action="store_true",
                    help="push the built dataset to the Hub (needs valid HF_TOKEN)")
    ap.add_argument("--report", action="store_true",
                    help="print a one-page build summary (filter yield, hours, CER spread)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cv_dir = Path(cfg["cv_dir"])
    clips_dir = cv_dir / "clips"
    fcfg = cfg["filter"]

    print("Plan:")
    print("  1. read tsvs -> 2. disjoint splits -> 3. filter 'other' -> "
          "4. transcode -> 5. manifests" + (" -> 6. push" if args.push else ""))

    tsvs = read_cv_tsvs(cv_dir)
    splits = build_splits(tsvs["validated"], tsvs["other"], tsvs["dev"], tsvs["test"])
    for name, df in splits.items():
        print(f"  split {name:12s}: {len(df):>7d} clips")

    if args.limit:
        splits = {k: v.head(args.limit) for k, v in splits.items()}

    filt_before = filt_after = 0
    if fcfg.get("enabled", True) and not args.no_filter and len(splits["train_extra"]):
        filt_before = len(splits["train_extra"])
        splits["train_extra"] = filter_other(splits["train_extra"], clips_dir, fcfg)
        filt_after = len(splits["train_extra"])
        print(f"  filter 'other': kept {filt_after}/{filt_before}")

    plan = [
        ("test", splits["test"], "validated"),
        ("dev", splits["dev"], "validated"),
        ("train", splits["train_core"], "validated"),
        ("train", splits["train_extra"], "other"),
    ]
    rows_by_split: dict = {}
    for split, df, source in plan:
        rows = materialize_split(df, split, source, clips_dir, cfg)
        rows_by_split.setdefault(split, []).extend(rows)
        print(f"  materialized {split}/{source}: {len(rows)} clips")

    out = Path(cfg["out_dir"])
    all_rows = []
    for split, rows in rows_by_split.items():
        write_nemo_manifest(rows, out / "manifests" / f"{split}.jsonl")
        total_h = sum(r["duration"] for r in rows) / 3600
        print(f"  wrote manifests/{split}.jsonl: {len(rows)} clips, {total_h:.1f} h")
        all_rows.extend(rows)

    if args.report:
        print(format_report(rows_by_split, filt_before, filt_after))

    if args.push:
        push_to_hub(all_rows, cfg)
        print(f"  pushed to {cfg['hf_repo']}")


if __name__ == "__main__":
    main()
