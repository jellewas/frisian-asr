"""Unit tests for the pure pipeline helpers (no data/GPU needed)."""
import pandas as pd

from build_dataset import (
    build_splits, char_error_rate, format_report, is_valid_text, normalize_text,
)


def test_format_report_counts_and_percentiles():
    rows = {
        "train": [
            {"duration": 3600, "source": "validated", "quality_score": 1.0},
            {"duration": 1800, "source": "other", "quality_score": 0.9},
        ],
        "test": [{"duration": 3600, "source": "validated", "quality_score": 1.0}],
    }
    out = format_report(rows, filter_before=100, filter_after=70)
    assert "kept 70/100 (70.0%)" in out
    assert "train :      2 clips     1.5 h" in out
    assert "test  :      1 clips     1.0 h" in out
    assert "quality_score" in out  # 'other' present in train -> spread line shown


def test_is_valid_text_rejects_empty_and_punct_only():
    assert is_valid_text("Goeie moarn")
    assert not is_valid_text("")
    assert not is_valid_text("...")
    assert not is_valid_text("123 !?")


def test_normalize_preserves_case_and_punct():
    assert normalize_text("  Hoi,   wrâld!  ") == "Hoi, wrâld!"
    assert normalize_text("“dei”") == '"dei"'  # curly quotes -> straight
    assert normalize_text("oan–elkoar") == "oan-elkoar"  # en-dash -> hyphen


def test_cer_bounds():
    assert char_error_rate("kat", "kat") == 0.0
    assert char_error_rate("kat", "kot") == 1 / 3
    assert char_error_rate("", "") == 0.0
    assert char_error_rate("", "x") == 1.0


def _df(rows):
    return pd.DataFrame(rows, columns=["client_id", "sentence", "path"])


def test_splits_are_speaker_and_sentence_disjoint():
    validated = _df([
        ["spk_eval", "shared sentence", "a.mp3"],   # speaker overlaps eval -> drop
        ["spk_x", "Goeie moarn", "b.mp3"],           # sentence overlaps eval -> drop
        ["spk_train", "Unike sin hjir", "c.mp3"],    # clean -> keep
    ])
    other = _df([["spk_y", "Wat oars", "d.mp3"]])     # clean -> keep
    dev = _df([["spk_eval", "Goeie moarn", "e.mp3"]])
    test = _df([["spk_eval", "Oare sin", "f.mp3"]])

    s = build_splits(validated, other, dev, test)

    train_clients = set(s["train_core"]["client_id"]) | set(s["train_extra"]["client_id"])
    eval_clients = set(dev["client_id"]) | set(test["client_id"])
    assert train_clients.isdisjoint(eval_clients)

    train_sents = {normalize_text(x) for x in
                   pd.concat([s["train_core"]["sentence"], s["train_extra"]["sentence"]])}
    eval_sents = {normalize_text(x) for x in pd.concat([dev["sentence"], test["sentence"]])}
    assert train_sents.isdisjoint(eval_sents)

    assert list(s["train_core"]["path"]) == ["c.mp3"]
    assert list(s["train_extra"]["path"]) == ["d.mp3"]
