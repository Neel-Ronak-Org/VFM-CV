"""Model registry. Wrappers are imported lazily so a missing optional dependency
(or missing weights) for one baseline doesn't break the others."""
from __future__ import annotations

import importlib

# name -> "module:ClassName"
REGISTRY = {
    "debug-centre": "src.models.base:_Constant",
    "SINet-V2": "src.models.sinet_v2:SINetV2",
}


def build_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(REGISTRY)}")
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**kwargs)
