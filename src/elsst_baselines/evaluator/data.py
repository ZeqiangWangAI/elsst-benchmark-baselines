import os
from pathlib import Path

from elsst_baselines.common.jsonl import read_jsonl
from elsst_baselines.retrieval.dataset import load_concept_pool


TRACK1_DATASET_REPO = "JohnWang10086/elsst-track1"
TRACK2_DATASET_REPO = "JohnWang10086/elsst-track2"


def repo_root():
    return Path(__file__).resolve().parents[3]


def _token():
    return (
        os.environ.get("ELSST_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _download_dataset_file(repo_id, filename, token=None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required when dataset files are not local") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
    )


def _local_or_public_dataset_file(local_path, repo_id, filename):
    local_path = Path(local_path)
    if local_path.exists():
        return local_path
    return _download_dataset_file(repo_id, filename)


def public_data_root():
    return Path(os.environ.get("ELSST_PUBLIC_DATA_ROOT", repo_root()))


def private_data_root():
    return Path(os.environ.get("ELSST_PRIVATE_DATA_ROOT", repo_root()))


def load_track1_concept_pool(root=None):
    root = Path(root or public_data_root())
    path = _local_or_public_dataset_file(
        root / "track1" / "concept_pool.jsonl",
        os.environ.get("ELSST_TRACK1_REPO", TRACK1_DATASET_REPO),
        "concept_pool.jsonl",
    )
    return load_concept_pool(path)


def load_track1_reference(split, public_root=None, private_root=None):
    if split == "val":
        root = Path(public_root or public_data_root())
        path = _local_or_public_dataset_file(
            root / "track1" / "val.jsonl",
            os.environ.get("ELSST_TRACK1_REPO", TRACK1_DATASET_REPO),
            "val.jsonl",
        )
        return read_jsonl(path)
    if split == "test":
        root = Path(private_root or private_data_root())
        local_path = root / "track1_full" / "test.jsonl"
        if local_path.exists():
            return read_jsonl(local_path)
        private_repo = os.environ.get("ELSST_PRIVATE_TRACK1_REPO")
        if private_repo:
            filename = os.environ.get("ELSST_PRIVATE_TRACK1_TEST_FILE", "test.jsonl")
            return read_jsonl(_download_dataset_file(private_repo, filename, token=_token()))
        raise FileNotFoundError("hidden Track1 test reference data is not configured")
    raise ValueError(f"unsupported split for Track1: {split}")


def load_track2_reference(split, public_root=None, private_root=None):
    if split == "val":
        root = Path(public_root or public_data_root())
        path = _local_or_public_dataset_file(
            root / "track2" / "val.jsonl",
            os.environ.get("ELSST_TRACK2_REPO", TRACK2_DATASET_REPO),
            "val.jsonl",
        )
        return read_jsonl(path)
    if split == "test":
        root = Path(private_root or private_data_root())
        local_path = root / "track2_full" / "test.jsonl"
        if local_path.exists():
            return read_jsonl(local_path)
        private_repo = os.environ.get("ELSST_PRIVATE_TRACK2_REPO")
        if private_repo:
            filename = os.environ.get("ELSST_PRIVATE_TRACK2_TEST_FILE", "test.jsonl")
            return read_jsonl(_download_dataset_file(private_repo, filename, token=_token()))
        raise FileNotFoundError("hidden Track2 test reference data is not configured")
    raise ValueError(f"unsupported split for Track2: {split}")
