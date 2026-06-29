"""Local (Mac GPU / MPS) launcher for SheepRL.

Applies the MPS compatibility shim (research/p2e/mps_compat.py), then hands off to
SheepRL's normal hydra entry point. Command-line arguments are passed straight
through, so this behaves exactly like the ``sheeprl`` CLI:

    .venv-sheeprl/bin/python research/p2e/run_local_mps.py \
        exp=p2e_dv3_exploration env=crafter fabric.accelerator=mps ...

The cloud/CUDA path uses the plain ``sheeprl`` CLI and does NOT import this file or
the shim, so nothing here can affect a cloud run.
"""

from __future__ import annotations

import os

# Let any op MPS doesn't implement fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import mps_compat  # noqa: E402,F401  — applies the float64->float32 patch on import
from sheeprl.cli import run  # noqa: E402

if __name__ == "__main__":
    run()
