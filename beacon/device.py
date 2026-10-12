"""Device, dtype, and random-seed helpers.

BEACON relies on GPyTorch/BoTorch, which require double precision for stable
GP fitting. All tensors are therefore created as ``torch.float64`` on the
selected device. Apple-silicon ``mps`` devices do not support float64 and are
rejected with a clear error.
"""

from __future__ import annotations

import random
from typing import Optional, Union

import numpy as np
import torch

DTYPE = torch.double

DeviceLike = Union[str, torch.device, None]


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    """Turn a user device specification into a ``torch.device``.

    Args:
        device: ``"auto"``/``None`` (use CUDA when available, otherwise CPU),
            ``"cpu"``, ``"cuda"``, ``"cuda:<index>"``, or a ``torch.device``.

    Raises:
        ValueError: if a CUDA device is requested but unavailable, or if an
            unsupported device type (e.g. ``mps``) is requested.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"Device '{device}' was requested but CUDA is not available. "
                "Use --device cpu or --device auto."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(
                f"Device '{device}' was requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    elif resolved.type == "mps":
        raise ValueError(
            "The 'mps' backend does not support float64, which GPyTorch/BoTorch "
            "need for stable GP fitting. Use --device cpu instead."
        )
    elif resolved.type != "cpu":
        raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'cuda[:i]'.")
    return resolved


def tkwargs(device: DeviceLike = "cpu") -> dict:
    """Return ``{"dtype": float64, "device": ...}`` for tensor construction."""
    return {"dtype": DTYPE, "device": resolve_device(device)}


def set_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy (global legacy RNG) and PyTorch (CPU and CUDA)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def available_devices() -> list[str]:
    """List the device strings usable on this machine (``cpu`` first)."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices
