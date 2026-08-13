"""Creates (or updates) the AutoTuneLab ZeroGPU trainer Space from zerogpu_space/ and
uploads its files. Run once after HF_TOKEN is set, and again any time zerogpu_space/*
changes. Requires HF_TOKEN (write access) and HF_ZEROGPU_SPACE_ID (e.g. "username/space-name")
in backend/.env.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[2]
SPACE_DIR = REPO_ROOT / "zerogpu_space"


def main() -> None:
    load_dotenv(REPO_ROOT / "backend" / ".env")
    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("HF_ZEROGPU_SPACE_ID")
    if not token:
        sys.exit("HF_TOKEN not set in backend/.env")
    if not space_id:
        sys.exit(
            "HF_ZEROGPU_SPACE_ID not set in backend/.env "
            '(e.g. "your-hf-username/autotunelab-zerogpu-trainer")'
        )

    api = HfApi(token=token)
    print(f"Creating/updating Space {space_id} ...")
    # Free accounts can no longer create plain cpu-basic Gradio Spaces (PRO-only now), but
    # ZeroGPU hardware is still free (up to 2 Spaces) — must request it at creation time,
    # not as a follow-up call, or create_repo defaults to the now-paywalled cpu-basic.
    api.create_repo(
        repo_id=space_id, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Deploy AutoTuneLab ZeroGPU trainer",
    )
    print(f"Deployed. Space: https://huggingface.co/spaces/{space_id}")
    print("It may take a minute or two to build before the API is callable.")


if __name__ == "__main__":
    main()
