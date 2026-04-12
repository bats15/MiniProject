"""Main entry point for audio semantic communication pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--config", type=str, default="configs/semantic_mapping.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        cmd = [sys.executable, "training/train_semantic_mapper.py", "--config", args.config]
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for eval mode")
        cmd = [
            sys.executable,
            "training/evaluate_semantic_mapper.py",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
        ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
