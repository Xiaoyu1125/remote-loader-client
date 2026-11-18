"""Declarative canonical-sample v2 configuration; no sample processing lives here."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from .protocol_limits import ProtocolLimits

ALLOWED_DTYPES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
_LIMITS = ProtocolLimits()


def validate_image_normalization(mean: Sequence[float] | None, std: Sequence[float] | None) -> None:
    """Validate the native float32 domain without changing request values."""
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be provided together")
    if mean is None:
        return
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must contain three channels")
    try:
        channels = struct.unpack("<6f", struct.pack("<6f", *mean, *std))
    except (OverflowError, struct.error) as exc:
        raise ValueError("mean and std must be finite float32 values") from exc
    if not all(math.isfinite(value) for value in channels):
        raise ValueError("mean and std must be finite float32 values")
    if any(value == 0 for value in channels[3:]):
        raise ValueError("std must not contain zero after float32 conversion")


def _validate_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("constants must be finite literal JSON")


def _validate_path(path: list[str]) -> list[str]:
    if not 1 <= len(path) <= 16 or any(not part or len(part) > 512 for part in path):
        raise ValueError("paths require 1..16 nonempty segments of at most 512 characters")
    return path


def _tensor_constant_shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        if not isinstance(value, (bool, int, float)):
            raise ValueError("tensor constants must contain numbers or booleans")
        return []
    if not value:
        raise ValueError("tensor constants cannot contain empty dimensions")
    inner_shape = _tensor_constant_shape(value[0])
    if any(_tensor_constant_shape(item) != inner_shape for item in value[1:]):
        raise ValueError("tensor constants must be rectangular")
    return [len(value), *inner_shape]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanonicalFormat(_StrictModel):
    type: Literal["canonical_sample"] = "canonical_sample"
    version: Literal[2] = 2

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("canonical format version must be an integer")
        return value


class MetadataRequirements(_StrictModel):
    fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class TemporalWindow(_StrictModel):
    offsets: list[StrictInt] | None = None
    start: StrictInt | None = None
    step: StrictInt | None = None
    count: StrictInt | None = Field(default=None, ge=1, le=4096)

    @model_validator(mode="after")
    def validate_window(self) -> TemporalWindow:
        if self.offsets is None:
            if self.start is None or self.step is None or self.count is None:
                raise ValueError("a window requires offsets or all of start/step/count")
            if self.step == 0:
                raise ValueError("window step must not be zero")
        else:
            if any(item is not None for item in (self.start, self.step, self.count)):
                raise ValueError("offsets and start/step/count are mutually exclusive")
            if not 1 <= len(self.offsets) <= 4096:
                raise ValueError("offsets must contain 1..4096 entries")
            if len(set(self.offsets)) != len(self.offsets):
                raise ValueError("offsets must not contain duplicates")
        return self

    def values(self) -> list[int]:
        if self.offsets is not None:
            return self.offsets
        assert self.start is not None and self.step is not None and self.count is not None
        return list(range(self.start, self.start + self.step * self.count, self.step))

    @property
    def length(self) -> int:
        if self.offsets is not None:
            return len(self.offsets)
        assert self.count is not None
        return self.count


class CanonicalTensorSchema(_StrictModel):
    # The server reserves one protocol dimension for its internal batching.
    shape: list[StrictInt] = Field(max_length=15)
    dtype: str

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: list[int]) -> list[int]:
        if any(size <= 0 or size > _LIMITS.max_dimension for size in value):
            raise ValueError("sample dimensions must be in 1..1000000; scalar shape is []")
        return value

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in ALLOWED_DTYPES:
            raise ValueError(f"unsupported dtype {value!r}")
        return normalized

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * ALLOWED_DTYPES[self.dtype]


class ColumnSource(_StrictModel):
    column: str = Field(min_length=1, max_length=512)


class TaskLookup(_StrictModel):
    table: Literal["tasks"]
    key_column: Literal["task_index"]
    value_column: Literal["task"]


class LookupSource(_StrictModel):
    index_column: str = Field(min_length=1, max_length=512)
    lookup: TaskLookup


class ConstantSource(_StrictModel):
    constant: Any

    @field_validator("constant")
    @classmethod
    def validate_constant(cls, value: Any) -> Any:
        _validate_json(value)
        return value


class TemporalMaskRule(_StrictModel):
    fields: list[list[str]] = Field(min_length=1, max_length=256)
    reduction: Literal["all", "any", "equal"]

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[list[str]]) -> list[list[str]]:
        for path in value:
            _validate_path(path)
        if len({tuple(path) for path in value}) != len(value):
            raise ValueError("temporal mask fields must not contain duplicates")
        return value


class TemporalMaskSource(_StrictModel):
    temporal_mask: TemporalMaskRule


class PaddingMaskSource(_StrictModel):
    is_pad: TemporalMaskRule


class NonemptySource(_StrictModel):
    nonempty: list[str]

    @field_validator("nonempty")
    @classmethod
    def validate_path(cls, value: list[str]) -> list[str]:
        return _validate_path(value)


class FieldBinding(_StrictModel):
    target: list[str]
    source: (
        ColumnSource | LookupSource | ConstantSource | TemporalMaskSource | PaddingMaskSource | NonemptySource
    )
    window: str | None = None
    add_feature_axis: bool = False
    tensor: CanonicalTensorSchema | None = None
    value_type: Literal["string", "json"] | None = None

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: list[str]) -> list[str]:
        return _validate_path(value)

    @model_validator(mode="after")
    def validate_binding(self) -> FieldBinding:
        if (self.tensor is None) == (self.value_type is None):
            raise ValueError("each field requires exactly one of tensor or value_type")
        if self.window is not None and not isinstance(self.source, ColumnSource):
            raise ValueError("only column sources support a temporal window")
        if self.add_feature_axis:
            if not isinstance(self.source, ColumnSource) or self.tensor is None:
                raise ValueError("add_feature_axis requires a column tensor source")
            if not self.tensor.shape or self.tensor.shape[-1] != 1:
                raise ValueError("add_feature_axis requires a trailing singleton output dimension")
        if isinstance(self.source, LookupSource) and self.value_type != "string":
            raise ValueError("tasks lookups require a string output")
        if isinstance(self.source, ConstantSource):
            if self.value_type == "string" and not isinstance(self.source.constant, str):
                raise ValueError("string constants must be strings")
            if self.tensor is not None and _tensor_constant_shape(self.source.constant) != self.tensor.shape:
                raise ValueError("tensor constant shape does not match its declared shape")
        if isinstance(self.source, (TemporalMaskSource, PaddingMaskSource)) and (
            self.tensor is None or self.tensor.dtype != "bool" or len(self.tensor.shape) != 1
        ):
            raise ValueError("temporal masks require one-dimensional bool tensors")
        if isinstance(self.source, NonemptySource):
            if self.tensor is None:
                if self.value_type != "json":
                    raise ValueError("nonempty masks require scalar bool tensors or JSON booleans")
            elif self.tensor.dtype != "bool" or self.tensor.shape:
                raise ValueError("nonempty masks require scalar bool tensors or JSON booleans")
        return self


class CanonicalContract(_StrictModel):
    batch_format: CanonicalFormat = Field(default_factory=CanonicalFormat)
    require_metadata: MetadataRequirements = Field(default_factory=MetadataRequirements)
    windows: dict[str, TemporalWindow] = Field(default_factory=dict)
    fields: list[FieldBinding] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_contract(self) -> CanonicalContract:
        if any(not name or len(name) > 512 for name in self.windows):
            raise ValueError("window names must contain 1..512 characters")
        by_path = {tuple(field.target): field for field in self.fields}
        if len(by_path) != len(self.fields):
            raise ValueError("canonical targets must be unique")
        wire_names = {".".join(path) for path in by_path}
        if len(wire_names) != len(by_path):
            raise ValueError("canonical target paths must have distinct dotted wire names")
        for path, field in by_path.items():
            for length in range(1, len(path)):
                if path[:length] in by_path:
                    raise ValueError(f"canonical target {list(path)!r} overlaps another target")
                if ".".join(path[:length]) in wire_names:
                    raise ValueError("canonical leaf wire names must not alias structural containers")
            if field.window is not None:
                if field.window not in self.windows:
                    raise ValueError(f"unknown temporal window {field.window!r}")
                if field.tensor is not None and (
                    not field.tensor.shape or field.tensor.shape[0] != self.windows[field.window].length
                ):
                    raise ValueError(f"target {list(path)!r} time dimension does not match its window")
        # Validate references only after every windowed column has been checked;
        # references may precede their declarations in the request.
        for field in self.fields:
            if isinstance(field.source, NonemptySource):
                dependency = by_path.get(tuple(field.source.nonempty))
                if dependency is None or dependency.value_type != "string":
                    raise ValueError("nonempty masks must reference a declared string field")
            elif isinstance(field.source, (TemporalMaskSource, PaddingMaskSource)):
                rule = (
                    field.source.is_pad
                    if isinstance(field.source, PaddingMaskSource)
                    else field.source.temporal_mask
                )
                expected_offsets = None
                for reference in rule.fields:
                    dependency = by_path.get(tuple(reference))
                    if (
                        dependency is None
                        or not isinstance(dependency.source, ColumnSource)
                        or dependency.window is None
                        or (dependency.tensor is None and dependency.value_type != "json")
                    ):
                        raise ValueError(
                            "temporal masks must reference declared windowed tensor or JSON columns"
                        )
                    offsets = self.windows[dependency.window].values()
                    if expected_offsets is not None and offsets != expected_offsets:
                        raise ValueError("temporal mask fields must use identical offset sequences")
                    expected_offsets = offsets
                assert field.tensor is not None and expected_offsets is not None
                if field.tensor.shape != [len(expected_offsets)]:
                    raise ValueError("temporal mask shape does not match its referenced window")
        sizes = [field.tensor.nbytes for field in self.fields if field.tensor is not None]
        if any(size > _LIMITS.max_tensor_bytes for size in sizes):
            raise ValueError("canonical tensor exceeds the protocol per-tensor byte limit")
        if sum(sizes) > _LIMITS.max_batch_bytes:
            raise ValueError("canonical sample exceeds the protocol byte limit")
        return self

    def output_schemas(self) -> dict[str, dict[str, Any]]:
        return {
            ".".join(field.target): (
                field.tensor.model_dump(mode="json")
                if field.tensor is not None
                else {"type": field.value_type}
            )
            for field in self.fields
        }

    def wire_schema_hash(self) -> str:
        schemas = self.output_schemas()
        for field in self.fields:
            schemas[".".join(field.target)]["path"] = field.target
        return self._hash(schemas)

    @staticmethod
    def _hash(value: dict[str, Any]) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_request(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def batch_spec_hash(self) -> str:
        return self._hash(self.to_request())
