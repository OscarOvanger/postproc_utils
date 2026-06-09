from __future__ import annotations

from pathlib import Path


HF_REPO_ID = "oovanger/MCP_datset"
LOCAL_CACHE_DIR = Path("data/cache/hf_raw")


def _hf_client():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for src.trackj.hf_data_store. "
            "Install the repo requirements before calling HF data-store functions."
        ) from exc
    return HfApi, hf_hub_download


def _repo_path(city: str, file_type: str, filename: str) -> str:
    return f"trackj_raw/{city}/{file_type}/{filename}"


def upload_raw_file(local_path: Path, city: str, file_type: str, token: str | None = None) -> str:
    local_path = Path(local_path)
    hf_path = _repo_path(city, file_type, local_path.name)
    HfApi, _ = _hf_client()
    HfApi().upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hf_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=token,
    )
    return hf_path


def fetch_raw_file(city: str, file_type: str, filename: str, token: str | None = None) -> Path:
    cache_path = LOCAL_CACHE_DIR / city / file_type / filename
    if cache_path.exists():
        return cache_path
    hf_path = _repo_path(city, file_type, filename)
    _, hf_hub_download = _hf_client()
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=hf_path,
        local_dir=LOCAL_CACHE_DIR,
        token=token,
    )
    return Path(downloaded)


def list_raw_files(city: str, file_type: str, token: str | None = None) -> list[str]:
    prefix = f"trackj_raw/{city}/{file_type}/"
    HfApi, _ = _hf_client()
    files = HfApi().list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=token)
    return sorted(path.removeprefix(prefix) for path in files if path.startswith(prefix))


def sync_city_to_hf(city: str, local_raw_dir: Path, token: str | None = None) -> None:
    city_dir = Path(local_raw_dir) / city
    if not city_dir.exists():
        print(f"uploaded 0 files, skipped 0 files (missing local raw dir: {city_dir})")
        return

    uploaded = 0
    skipped = 0
    known_by_type: dict[str, set[str]] = {}
    for path in sorted(p for p in city_dir.rglob("*") if p.is_file()):
        file_type = path.parent.relative_to(city_dir).as_posix()
        known = known_by_type.setdefault(file_type, set(list_raw_files(city, file_type, token=token)))
        if path.name in known:
            skipped += 1
            continue
        upload_raw_file(path, city, file_type, token=token)
        known.add(path.name)
        uploaded += 1
    print(f"uploaded {uploaded} files, skipped {skipped} files (already on HF)")
