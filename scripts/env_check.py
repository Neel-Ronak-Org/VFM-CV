"""Print the environment facts that PROGRESS.md needs (OS, Python, CUDA, real VRAM).

    python scripts/env_check.py
Paste the output into PROGRESS.md (or send it back to Claude).
"""
import platform
import shutil
import subprocess
import sys


def main():
    print(f"OS:      {platform.platform()}")
    wsl = "microsoft" in platform.release().lower()
    print(f"WSL:     {wsl}")
    print(f"Python:  {sys.version.split()[0]}")
    if shutil.which("nvidia-smi"):
        q = "name,memory.total,driver_version"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True)
        print(f"nvidia-smi: {out.stdout.strip()}")
    else:
        print("nvidia-smi: not found")
    try:
        import torch

        print(f"torch:   {torch.__version__} (CUDA build {torch.version.cuda})")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"GPU:     {p.name}, {p.total_memory / 2**30:.2f} GiB VRAM, compute {p.major}.{p.minor}")
            print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    except ImportError:
        print("torch:   not installed")
    for pkg in ("numpy", "cv2", "py_sod_metrics", "imagecorruptions", "skimage"):
        try:
            m = __import__(pkg)
            print(f"{pkg:16s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            print(f"{pkg:16s} MISSING ({e.__class__.__name__})")
    total, used, free = shutil.disk_usage(".")
    print(f"Disk free here: {free / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
