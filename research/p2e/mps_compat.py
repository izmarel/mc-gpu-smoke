"""MPS (Apple GPU) compatibility shim for SheepRL — Mac-only, inert everywhere else.

Apple's MPS backend cannot create float64 tensors. SheepRL's replay buffer copies
numpy dtypes straight through its single ``get_tensor`` chokepoint, so any float64
data (e.g. Crafter rewards / reward-as-observation) crashes training on MPS with:

    TypeError: Cannot convert a MPS Tensor to float64 dtype ...

This module monkeypatches that one function to downcast float64 -> float32. It does
NOT touch the SheepRL install on disk (which is gitignored and disposable); it
rebinds the function at runtime from your repo, so it survives reinstalls and lives
in version control.

IMPORTANT — quarantine: only ``run_local_mps.py`` imports this. The CUDA/cloud
launcher never imports it, so it cannot affect or "leak into" the cloud run. The
change is also scientifically a no-op: the model trains in float32 regardless.
"""

from __future__ import annotations

import torch

import sheeprl.data.buffers as _buffers
from sheeprl.utils.utils import NUMPY_TO_TORCH_DTYPE_DICT

_orig_get_tensor = _buffers.get_tensor


def _effective_dtype(array, dtype):
    if dtype is not None:
        return dtype
    arr = array.array if isinstance(array, _buffers.MemmapArray) else array
    return NUMPY_TO_TORCH_DTYPE_DICT[arr.dtype]


def _mps_safe_get_tensor(array, dtype=None, clone=False, device="cpu", from_numpy=False):
    """``get_tensor`` wrapper that forces float64 -> float32 (MPS can't do float64)."""
    eff = _effective_dtype(array, dtype)
    if eff == torch.float64:
        eff = torch.float32
    return _orig_get_tensor(
        array, dtype=eff, clone=clone, device=device, from_numpy=from_numpy
    )


def apply() -> None:
    """Rebind ``sheeprl.data.buffers.get_tensor`` to the MPS-safe wrapper (idempotent)."""
    if getattr(_buffers.get_tensor, "_mps_patched", False):
        return
    _mps_safe_get_tensor._mps_patched = True  # type: ignore[attr-defined]
    _buffers.get_tensor = _mps_safe_get_tensor
    print("[mps_compat] sheeprl get_tensor patched: float64 -> float32 for MPS")


apply()
