"""Check an HF Job: python check_job.py <job_id> [n_log_lines]"""
import os
import sys

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
jid = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
j = api.inspect_job(job_id=jid, namespace="jellewas")
stage = j.status.stage if hasattr(j.status, "stage") else j.status
print("STATUS:", stage)
try:
    lines = list(api.fetch_job_logs(job_id=jid, namespace="jellewas"))
    for line in lines[-n:]:
        print(line)
except Exception as e:
    print("logs err:", e)
