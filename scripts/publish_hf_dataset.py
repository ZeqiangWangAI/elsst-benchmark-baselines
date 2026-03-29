#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload a local ELSST track directory to a Hugging Face dataset repo.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Local directory to upload.")
    parser.add_argument("--repo-id", required=True, help="Target dataset repo id, e.g. ZeqiangWangAI/elsst-track1.")
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private.")
    parser.add_argument("--token", help="Optional Hugging Face token. If omitted, the local login is used.")
    parser.add_argument("--commit-message", default="Upload ELSST dataset release")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"source directory not found: {source_dir}")

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(source_dir),
        commit_message=args.commit_message,
    )
    print(f"uploaded {source_dir} -> {args.repo_id}")


if __name__ == "__main__":
    main()
