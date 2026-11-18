from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def move_to_device(value: Any, device: Any, *, non_blocking: bool = True) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install it in your training environment") from exc
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device, non_blocking=non_blocking) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking=non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking=non_blocking) for item in value]
    return value
