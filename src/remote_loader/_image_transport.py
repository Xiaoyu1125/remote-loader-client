"""Derive the HC1 image contract without changing the public logical request."""

from __future__ import annotations

from typing import Any

from .batch_spec import BatchSpec, CanonicalSampleSpec, TensorSchema, VQABatchSpec
from .canonical_spec import ColumnSource
from .exceptions import ProtocolError


def needs_camera_metadata(spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec) -> bool:
    """Skip discovery when no float tensor can have image axes [..., C, H, W]."""
    if isinstance(spec, CanonicalSampleSpec):
        return spec.image_output.enabled and any(
            isinstance(field.source, ColumnSource)
            and field.tensor is not None
            and field.tensor.dtype == "float32"
            and len(field.tensor.shape) >= 3
            for field in spec.fields
        )
    if isinstance(spec, BatchSpec):
        return any(
            isinstance(schema, TensorSchema)
            and schema.dtype == "float32"
            and len(schema.shape) >= 3
            for schema in spec.output.values()
        )
    return False


def compile_wire_spec(
    spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec,
    metadata: dict[str, Any],
) -> BatchSpec | CanonicalSampleSpec | VQABatchSpec:
    """Only catalog camera columns change dtype; targets are never reconstructed."""
    camera_keys = metadata.get("camera_keys")
    if not isinstance(camera_keys, list) or any(not isinstance(key, str) for key in camera_keys):
        raise ProtocolError("dataset metadata camera_keys must be a list of column names")
    cameras = set(camera_keys)
    if isinstance(spec, CanonicalSampleSpec):
        indices = [
            index for index, field in enumerate(spec.fields)
            if isinstance(field.source, ColumnSource)
            and field.source.column in cameras
            and field.tensor is not None
            and field.tensor.dtype == "float32"
        ]
        if not indices or not spec.image_output.enabled:
            return spec
        wire = spec.model_copy(deep=True)
        for index in indices:
            wire.fields[index].tensor.dtype = "uint8"
        wire.image_output.mean = None
        wire.image_output.std = None
        return wire
    if isinstance(spec, BatchSpec):
        names = [
            name for name, schema in spec.output.items()
            if isinstance(schema, TensorSchema)
            and schema.dtype == "float32"
            and spec.sample[name].field in cameras
        ]
        if names:
            wire = spec.model_copy(deep=True)
            for name in names:
                wire.output[name].dtype = "uint8"
            return wire
    return spec
