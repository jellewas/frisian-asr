"""Launch the Nemotron-3.5 Frisian showcase as HF Jobs.

Usage:
  python launch_showcase.py baseline-smoke   # ~30 clips, cheap, validates NeMo install
  python launch_showcase.py baseline         # full test split, the 'before' WER
  python launch_showcase.py finetune         # fine-tune + 'after' WER  (A100)

Reads the HF token from ~/.cache/huggingface/token and HARD-REFUSES the token leaked
in chat (fingerprint below). Rotate first:  python -c "from huggingface_hub import login; login('hf_NEW')"
"""
import hashlib
import os
import sys

from huggingface_hub import run_uv_job

LEAKED_FP = "45b8a91c40b4"  # sha256[:12] of the token exposed in chat — must never run


def resolve_token():
    path = os.path.expanduser("~/.cache/huggingface/token")
    if not os.path.exists(path):
        sys.exit("No ~/.cache/huggingface/token. Run: python -c \"from huggingface_hub import login; login('hf_NEW')\"")
    tok = open(path).read().strip()
    if hashlib.sha256(tok.encode()).hexdigest()[:12] == LEAKED_FP and os.environ.get("ALLOW_LEAKED") != "1":
        sys.exit("REFUSING: cached token is the LEAKED one. Revoke it at hf.co/settings/tokens, "
                 "create a new write token, then: python -c \"from huggingface_hub import login; login('hf_NEW')\". "
                 "To proceed on it anyway, set ALLOW_LEAKED=1.")
    return tok


# NeMo from git main (>= 26.06 required for the prompt-conditioned model). Cython +
# packaging are build deps. No '<'/'>' in pins (shell-redirect trap).
# no spaces anywhere: run_uv_job builds a shell command, so ' @ ' would split into
# stray argv ('@' got spawned as a command). pkg@git+url (no spaces) stays one token
# and resolves to git main — the release that supports the prompt-conditioned model.
# Pin torch/torchaudio to a CUDA-12 build (2.7.1 -> cu126): NeMo git main otherwise
# pulls a cu13 torch that the HF nodes' 12.9 driver can't run ('driver too old').
DEPS_NEMO = [
    "nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo.git",
    "torch==2.7.1", "torchaudio==2.7.1",
    "nvidia-cuda-nvcc-cu12==12.6.85",  # libnvvm for numba RNNT JIT; matches torch's cu126
    "numba-cuda",  # numba 0.65 moved CUDA support here; without it the arch query fails
    "Cython", "packaging", "jiwer", "librosa", "soundfile", "datasets",
]

MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline-smoke"
TOKEN = resolve_token()

common = {"DATASET": "LokaalHub/frisian-asr-cv22", "TARGET_LANG": "nl-NL"}

if MODE == "baseline-smoke":
    script, env = "baseline_job.py", {**common, "SPLIT": "test", "LIMIT": "30"}
    flavor, timeout = "t4-small", "1h"
elif MODE == "baseline":
    script, env = "baseline_job.py", {**common, "SPLIT": "test"}
    flavor, timeout = "l4x1", "1h"
elif MODE == "diag":
    script, env = "diag_job.py", {}
    flavor, timeout = "t4-small", "30m"
elif MODE == "finetune-smoke":
    script, env = "finetune_job.py", {**common, "LIMIT": "30", "MAX_STEPS": "20", "WARMUP": "5",
                                      "OUT_REPO": ""}
    flavor, timeout = "a100-large", "1h"
elif MODE == "finetune":
    script, env = "finetune_job.py", {**common}
    flavor, timeout = "a100-large", "8h"
elif MODE == "stream-eval-smoke":
    script, env = "streaming_eval_job.py", {**common, "LIMIT": "30"}
    flavor, timeout = "l4x1", "1h"
elif MODE == "stream-eval":
    script, env = "streaming_eval_job.py", {**common}
    flavor, timeout = "l4x1", "2h"
else:
    sys.exit(f"unknown mode: {MODE}")

job = run_uv_job(
    script=script,
    dependencies=DEPS_NEMO,
    flavor=flavor,
    env=env,
    secrets={"HF_TOKEN": TOKEN},
    timeout=timeout,
    token=TOKEN,
)
print("MODE:", MODE, "| flavor:", flavor, "| timeout:", timeout)
print("job id:", job.id)
print("url:", getattr(job, "url", None) or f"https://huggingface.co/jobs/jellewas/{job.id}")
