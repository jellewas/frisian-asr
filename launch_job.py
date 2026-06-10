"""Launch the Frisian dataset build as an HF Job. Usage: python launch_job.py [smoke|full]"""
import sys

from huggingface_hub import run_uv_job

MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
TOKEN = __import__("os").environ["HF_TOKEN"]

# NOTE: no '<' or '>' in specs — the job bootstrap builds a shell command and
# unquoted version operators become shell redirects (cause of the first failure).
DEPS = [
    "datasets==2.20.0", "transformers==4.44.2", "torch", "soundfile",
    "librosa", "pandas", "huggingface_hub==0.36.0", "numpy==1.26.4",
]

SRC = "fsicoli/common_voice_22_0"
if MODE == "smoke":
    env = {"LIMIT": "30", "CV_DATASET": SRC, "HF_REPO": "LokaalHub/frisian-asr-smoketest"}
    flavor, timeout = "t4-small", "30m"
else:
    env = {"CV_DATASET": SRC, "HF_REPO": "LokaalHub/frisian-asr-cv22"}
    flavor, timeout = "l4x1", "6h"

job = run_uv_job(
    script="hf_job.py",
    dependencies=DEPS,
    flavor=flavor,
    env=env,
    secrets={"HF_TOKEN": TOKEN},
    timeout=timeout,
    token=TOKEN,
)
print("MODE:", MODE, "| flavor:", flavor)
print("job id:", job.id)
print("url:", getattr(job, "url", None) or f"https://huggingface.co/jobs/jellewas/{job.id}")
