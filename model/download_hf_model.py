#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model into the local model directory."
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Hugging Face model id, for example: Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision (branch, tag, or commit hash).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face access token for gated/private models.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Optional download directory. Defaults to model/<model_id_with_underscore>.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist in cache.",
    )
    return parser.parse_args()


def build_target_dir(base_dir: Path, model_id: str, local_dir: str | None) -> Path:
    if local_dir:
        return Path(local_dir).expanduser().resolve()
    safe_name = model_id.replace("/", "_")
    return (base_dir / safe_name).resolve()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    target_dir = build_target_dir(script_dir, args.model_id, args.local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[error] missing dependency: huggingface_hub.\n"
            "Install it with: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    try:
        local_path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=args.token,
            force_download=args.force,
        )
    except Exception as exc:
        print(f"[error] failed to download model '{args.model_id}': {exc}", file=sys.stderr)
        return 1

    print(f"[ok] model downloaded: {args.model_id}")
    print(f"[ok] saved to: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
