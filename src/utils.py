from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL)
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO, stderr=subprocess.DEVNULL) != 0
        return h.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_run_log(kind: str, payload: dict, log_dir: Path | None = None) -> Path:
    """Save the config/seed/git hash/timings of a run next to the results."""
    log_dir = log_dir or REPO / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = log_dir / f"{kind}-{stamp}.json"
    payload = {"git": git_hash(), "host": platform.node(), "python": platform.python_version(), **payload}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
