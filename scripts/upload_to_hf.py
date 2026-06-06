import os
from pathlib import Path

from huggingface_hub import HfApi


HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "oovanger/MCP_datset"
DATA_DIR = Path("historic_tmax_market_data")

api = HfApi()

paths = list(DATA_DIR.rglob("*"))
files = [p for p in paths if p.is_file()]

print(f"Uploading {len(files)} files to {REPO_ID} ...")

for local_path in sorted(files):
    path_in_repo = "data/" + str(local_path.relative_to(DATA_DIR))
    print(f"  {local_path} -> {path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )

print("Done. Verify at:")
print(f"  https://huggingface.co/datasets/{REPO_ID}")
