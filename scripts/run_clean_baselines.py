"""Clean evaluation of every baseline whose weights are present, then the reproduction table.

    python scripts/run_clean_baselines.py                    # all models, all 4 test sets
    python scripts/run_clean_baselines.py --models ZoomNet PFNet --datasets CHAMELEON
Resumable (finished combinations are skipped). Models with missing weights are skipped with a message.
Order: small specialists first, then the large SAM-based models (SAM-Adapter ViT-H is the slowest).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src import evaluate  # noqa: E402

ORDER = ["SINet-V2", "PFNet", "ZoomNet", "FEDER", "SAM2-UNet", "SAM-Adapter", "Oracle-Box-MobileSAM"]
DATASETS = ["CHAMELEON", "CAMO", "COD10K", "NC4K"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ORDER)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    status = {}
    for m in args.models:
        print(f"\n=== {m} ===")
        try:
            evaluate.main(["--models", m, "--datasets", *args.datasets, "--corruptions", "clean"])
            status[m] = "ok"
        except FileNotFoundError as e:
            status[m] = f"SKIPPED (file missing): {e}"
        except Exception as e:  # noqa: BLE001 -- keep going with the other models, report at the end
            traceback.print_exc()
            status[m] = f"FAILED: {type(e).__name__}: {e}"
    print("\n=== summary ===")
    for m, s in status.items():
        print(f"{m:>22s}: {s}")
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_reproduction.py")], check=False)


if __name__ == "__main__":
    main()
