#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.dataset_prepare.generate_genesis_franka_dataset import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
