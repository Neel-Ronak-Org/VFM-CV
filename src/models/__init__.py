"""Model registry. Wrappers are imported lazily so a missing optional dependency
(or missing weights) for one baseline doesn't break the others."""
from __future__ import annotations

import importlib

# name -> "module:ClassName"
REGISTRY = {
    "debug-centre": "src.models.base:_Constant",
    "SINet-V2": "src.models.sinet_v2:SINetV2",
    "SAM2-UNet": "src.models.sam2_unet:SAM2UNet",
    "ZoomNet": "src.models.zoomnet:ZoomNet",
    "PFNet": "src.models.pfnet:PFNet",
    "FEDER": "src.models.feder:FEDER",
    "SAM-Adapter": "src.models.sam_adapter:SAMAdapter",
    # Ours (src/train.py -> checkpoints/ours/<run>/final.pt):
    "Ours-Clean": "src.models.robust_adapter:OursClean",
    "Ours-Aug": "src.models.robust_adapter:OursAug",
    "Ours-Full": "src.models.robust_adapter:OursFull",
    "Ours-Full-15ep": "src.models.robust_adapter:OursFullEp15",
    # Analysis only -- uses the ground-truth box, NOT a fair competitor:
    "Oracle-Box-MobileSAM": "src.models.oracle_box_sam:OracleBoxMobileSAM",
}


def build_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(REGISTRY)}")
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**kwargs)
