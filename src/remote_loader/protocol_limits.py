from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolLimits:
    max_metadata_bytes: int = 1024 * 1024
    max_batch_bytes: int = 4 * 1024**3
    max_tensor_bytes: int = 2 * 1024**3
    max_tensors: int = 256
    max_dimensions: int = 16
    max_dimension: int = 1_000_000
