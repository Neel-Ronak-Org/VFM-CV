"""Download, extract and verify COD datasets and baseline weights.

    python scripts/download_data.py --data            # train + test packages (+ NC4K if missing)
    python scripts/download_data.py --weights sinet_v2 mobile_sam
    python scripts/download_data.py --verify          # counts, pairing, sizes, mask binarity
    python scripts/download_data.py --maps sinet_v2   # officially released prediction maps

If a Google Drive download fails (quota/permission), the script prints the link and
the exact path to save the file to. Re-run it afterwards and it will extract and
organise the manually downloaded archive.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.datasets import MASK_SUBDIR_ALTERNATIVES, CODDataset, load_data_config, verify_dataset  # noqa: E402

SOURCES = yaml.safe_load(open(REPO / "configs" / "sources.yaml"))
TEST_SETS = ("CHAMELEON", "CAMO", "COD10K", "NC4K")


def fetch(entry: dict) -> Path | None:
    dest = REPO / entry["dest"]
    if dest.suffix == "":  # folder destination
        dest.mkdir(parents=True, exist_ok=True)
        if any(p.is_file() for p in dest.rglob("*")):
            print(f"[ok] {dest.relative_to(REPO)} already has files")
            return dest
    elif dest.exists():
        print(f"[ok] {dest.relative_to(REPO)} already present")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "gdrive_folder" in entry:
            import gdown

            out = gdown.download_folder(id=entry["gdrive_folder"], output=str(dest), quiet=False)
            if not out:
                raise RuntimeError("gdown.download_folder returned nothing")
            return dest
        if "gdrive_id" in entry:
            import gdown

            out = gdown.download(id=entry["gdrive_id"], output=str(dest), quiet=False)
            if out is None:
                raise RuntimeError("gdown returned None")
        elif "url" in entry:
            print(f"downloading {entry['url']}")
            urllib.request.urlretrieve(entry["url"], dest)
        else:
            raise RuntimeError("manual download required")
        return dest
    except Exception as e:  # noqa: BLE001
        link = entry.get("manual") or (f"https://drive.google.com/file/d/{entry['gdrive_id']}/view"
                                       if "gdrive_id" in entry else entry.get("url"))
        print(f"\n[MANUAL] Could not download automatically ({e}).\n"
              f"  1. Open: {link}\n  2. Save the file to: {dest}\n  3. Re-run this script.\n")
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        return None


def safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for m in zf.infolist():
            p = (target / m.filename).resolve()
            if not p.is_relative_to(target):
                raise RuntimeError(f"unsafe path in archive: {m.filename}")
        zf.extractall(target)


def find_split_dirs(root: Path) -> dict[str, Path]:
    """Locate folders that contain Imgs/ + a mask folder and guess which dataset they are."""
    found = {}
    for imgs in sorted(root.rglob("Imgs")):
        d = imgs.parent
        if not any((d / m).is_dir() for m in MASK_SUBDIR_ALTERNATIVES) or "__MACOSX" in d.parts:
            continue
        parts = [p.upper() for p in d.relative_to(root).parts] or [root.name.upper()]
        for t in TEST_SETS:
            if t in parts[-1] and "TRAIN" not in "".join(parts):
                found.setdefault(t, d)
        if any("TRAIN" in p for p in parts):
            found.setdefault("train", d)
    return found


def organise(extract_root: Path, raw: Path | None = None) -> None:
    cfg = load_data_config()
    raw = raw or REPO / cfg["root"]
    for key, src in find_split_dirs(extract_root).items():
        dst = raw / (cfg["train"]["dir"] if key == "train" else cfg["test"][key]["dir"])
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"[ok] {key} -> {dst}")


def do_data() -> None:
    staging = REPO / "data" / "raw" / "extracted"
    for key in ("cod_train", "cod_test"):
        arc = fetch(SOURCES["datasets"][key])
        if arc and zipfile.is_zipfile(arc):
            safe_extract(arc, staging / key)
            organise(staging / key)
    cfg = load_data_config()
    if not (REPO / cfg["root"] / cfg["test"]["NC4K"]["dir"]).exists():
        arc = fetch(SOURCES["datasets"]["nc4k"])
        if arc and zipfile.is_zipfile(arc):
            safe_extract(arc, staging / "nc4k")
            organise(staging / "nc4k")


def do_verify() -> bool:
    cfg = load_data_config()
    rows, all_ok = [], True
    specs = [("train", cfg["train"]["expected"])] + [(k, v["expected"]) for k, v in cfg["test"].items()]
    for name, expected in specs:
        try:
            ds = CODDataset.from_config(name, cfg)
        except FileNotFoundError as e:
            print(f"[missing] {name}: {e}")
            rows.append({"dataset": name, "ok": False, "expected": expected, "n_pairs": 0})
            all_ok = False
            continue
        r = verify_dataset(ds, expected)
        rows.append(r)
        all_ok &= r["ok"]
        print(("[ok]  " if r["ok"] else "[FAIL]") + f" {name}: {r['n_pairs']} pairs (expected {expected}); "
              f"missing masks {r['missing_masks']}, missing images {r['missing_images']}, "
              f"size mismatch {r['size_mismatch']}, non-binary {r['nonbinary_masks']}, empty {r['empty_masks']}")
    out = REPO / "results" / "data_check.csv"
    fields = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(REPO)}")
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--weights", nargs="*", help=f"any of: {', '.join(SOURCES['weights'])}")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--maps", nargs="*", help=f"released prediction maps: {', '.join(SOURCES['prediction_maps'])}")
    args = ap.parse_args()
    if not (args.data or args.weights is not None or args.verify or args.maps is not None):
        ap.print_help()
        return
    if args.data:
        do_data()
    if args.weights is not None:
        for w in args.weights or SOURCES["weights"]:
            fetch(SOURCES["weights"][w])
    if args.maps is not None:
        for m in args.maps or SOURCES["prediction_maps"]:
            entry = SOURCES["prediction_maps"][m]
            arc = fetch(entry)
            archives = ([arc] if arc and arc.is_file() else sorted(arc.rglob("*.zip")) if arc else [])
            for a in archives:
                if zipfile.is_zipfile(a):
                    safe_extract(a, REPO / entry["extract_to"])
                    print(f"[ok] {m}: {a.name} -> {entry['extract_to']}")
    if args.verify:
        sys.exit(0 if do_verify() else 1)


if __name__ == "__main__":
    main()
