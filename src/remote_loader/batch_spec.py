from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .canonical_spec import ALLOWED_DTYPES, CanonicalContract, validate_image_normalization
from .exceptions import ConfigurationError
from .protocol_limits import ProtocolLimits

_PROTOCOL_LIMITS = ProtocolLimits()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplingSpec(StrictModel):
    shuffle_seed: int = 42
    drop_n_first_frames: int = Field(default=0, ge=0, le=1_000_000)
    drop_n_last_frames: int = Field(default=0, ge=0, le=1_000_000)
    min_samples_per_episode: int = Field(default=0, ge=0, le=1_000_000)


class SampleField(StrictModel):
    field: str = Field(min_length=1, max_length=512)
    temporal_offsets: list[int] | None = None
    horizon: int | None = Field(default=None, gt=0, le=4096)

    @field_validator("temporal_offsets")
    @classmethod
    def validate_temporal_offsets(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("must not be empty")
        if len(value) > 4096:
            raise ValueError("must contain at most 4096 offsets")
        if len(value) != len(set(value)):
            raise ValueError("must not contain duplicate offsets")
        return value

    @model_validator(mode="after")
    def validate_window(self) -> SampleField:
        if self.temporal_offsets is not None and self.horizon is not None:
            raise ValueError("temporal_offsets and horizon are mutually exclusive")
        return self


class BatchConfig(StrictModel):
    size: int = Field(gt=0, le=4096)


class ImageResizeSpec(StrictModel):
    interpolation: Literal["nearest", "nearest-exact", "bilinear", "bicubic"]
    antialias: bool

    @model_validator(mode="after")
    def validate_antialias(self) -> ImageResizeSpec:
        if self.antialias and self.interpolation in {"nearest", "nearest-exact"}:
            raise ValueError("antialias is not supported for nearest interpolation")
        return self


class TensorSchema(StrictModel):
    shape: list[int] = Field(max_length=_PROTOCOL_LIMITS.max_dimensions)
    dtype: str
    resize: ImageResizeSpec | None = None

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: list[int]) -> list[int]:
        if any(dim <= 0 for dim in value):
            raise ValueError("all dimensions must be positive")
        if any(dim > _PROTOCOL_LIMITS.max_dimension for dim in value):
            raise ValueError(f"a dimension exceeds the {_PROTOCOL_LIMITS.max_dimension:,} limit")
        return value

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in ALLOWED_DTYPES:
            raise ValueError(f"unsupported dtype {value!r}")
        return normalized

    @model_validator(mode="after")
    def validate_resize(self) -> TensorSchema:
        if self.resize is not None:
            if self.dtype != "uint8":
                raise ValueError("resize requires uint8 image output")
            if len(self.shape) < 3:
                raise ValueError("resize requires an image shape [..., C, H, W]")
        return self

    @property
    def nbytes(self) -> int:
        elements = 1
        for dim in self.shape:
            elements *= dim
        return elements * ALLOWED_DTYPES[self.dtype]


class JsonSchema(StrictModel):
    type: Literal["string", "json"]


class ImageOutputSpec(StrictModel):
    enabled: StrictBool = True
    resize_uint8: StrictBool = False
    antialias: StrictBool = True
    mean: list[float] | None = None
    std: list[float] | None = None
    padding: list[int] = Field(default_factory=lambda: [0, 0, 0, 0], min_length=4, max_length=4)

    @field_validator("padding", mode="before")
    @classmethod
    def validate_padding(cls, value: Any) -> list[int]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 4
            or any(type(item) is not int or item < 0 for item in value)
        ):
            raise ValueError("padding must contain four nonnegative integers: left, right, top, bottom")
        return list(value)

    @field_validator("mean", "std", mode="before")
    @classmethod
    def validate_channels(cls, value: Any) -> list[float] | None:
        if value is None:
            return None
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(type(item) not in (int, float) for item in value)
        ):
            raise ValueError("mean and std must contain three finite numbers")
        try:
            channels = [float(item) for item in value]
        except (OverflowError, ValueError) as exc:
            raise ValueError("mean and std must contain three finite numbers") from exc
        return channels

    @model_validator(mode="after")
    def validate_processing(self) -> ImageOutputSpec:
        validate_image_normalization(self.mean, self.std)
        if self.resize_uint8 and not self.antialias:
            raise ValueError("resize_uint8=true requires antialias=true")
        if not self.enabled and (
            self.resize_uint8 or not self.antialias or self.mean is not None or any(self.padding)
        ):
            raise ValueError(
                "disabled image output cannot configure resize, antialias, padding, or normalization"
            )
        return self


class CanonicalSampleSpec(CanonicalContract):
    """One declaratively bound sample per HC frame, without a batch axis."""

    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    image_output: ImageOutputSpec = Field(default_factory=ImageOutputSpec)

    def to_request(self) -> dict[str, Any]:
        payload = super().to_request()
        if payload["sampling"]["min_samples_per_episode"] == 0:
            payload["sampling"].pop("min_samples_per_episode")
        image_output = payload["image_output"]
        if image_output.get("antialias") is True:
            image_output.pop("antialias")
        return payload

    @property
    def output(self) -> dict[str, TensorSchema | JsonSchema]:
        return {
            name: (JsonSchema if "type" in schema else TensorSchema).model_validate(schema)
            for name, schema in self.output_schemas().items()
        }

    @property
    def is_vqa(self) -> bool:
        return False

    @property
    def schema_hash(self) -> str:
        return self.wire_schema_hash()

    @property
    def estimated_batch_bytes(self) -> int:
        return sum(field.tensor.nbytes for field in self.fields if field.tensor is not None)

    @property
    def estimated_metadata_bytes(self) -> int:
        return (
            _PROTOCOL_LIMITS.max_metadata_bytes if any(field.tensor is None for field in self.fields) else 0
        )

    @property
    def estimated_batch_gb(self) -> float:
        return self.estimated_batch_bytes / 1024**3

    @classmethod
    def from_yaml(cls, path: str | Path) -> CanonicalSampleSpec:
        try:
            with Path(path).expanduser().open("rb") as stream:
                return cls.model_validate(yaml.safe_load(stream))
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ConfigurationError(f"invalid batch spec {path}: {exc}") from exc

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CanonicalSampleSpec:
        try:
            return cls.model_validate(value)
        except ValueError as exc:
            raise ConfigurationError(f"invalid batch spec: {exc}") from exc


class BatchSpec(StrictModel):
    """One canonical raw sample per HC frame; the scheduler selects the datasets."""

    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    sample: dict[str, SampleField]
    output: dict[str, TensorSchema | JsonSchema]

    @model_validator(mode="after")
    def validate_outputs(self) -> BatchSpec:
        if not self.sample:
            raise ValueError("sample must contain at least one field")
        if not self.output:
            raise ValueError("output must contain at least one field")
        aliases = {
            name: sample_field.field
            for name, sample_field in self.sample.items()
            if sample_field.field != name
        }
        if aliases:
            raise ValueError(f"field aliases are not supported: {aliases}")
        unknown = set(self.output) - set(self.sample)
        if unknown:
            raise ValueError(f"output contains fields absent from sample: {sorted(unknown)}")
        tensor_count = sum(isinstance(schema, TensorSchema) for schema in self.output.values())
        if tensor_count > _PROTOCOL_LIMITS.max_tensors:
            raise ValueError(
                f"output contains {tensor_count} tensors, exceeding the client protocol "
                f"limit {_PROTOCOL_LIMITS.max_tensors}"
            )
        oversized_tensors = {
            name: schema.nbytes
            for name, schema in self.output.items()
            if isinstance(schema, TensorSchema) and schema.nbytes > _PROTOCOL_LIMITS.max_tensor_bytes
        }
        if oversized_tensors:
            raise ValueError(
                f"output tensors exceed the client protocol per-tensor byte limit "
                f"{_PROTOCOL_LIMITS.max_tensor_bytes}: {oversized_tensors}"
            )
        estimated_bytes = self.estimated_batch_bytes
        if estimated_bytes > _PROTOCOL_LIMITS.max_batch_bytes:
            raise ValueError(
                f"estimated_batch_bytes={estimated_bytes} exceeds the client "
                f"protocol limit {_PROTOCOL_LIMITS.max_batch_bytes}"
            )
        return self

    @property
    def is_vqa(self) -> bool:
        return False

    @property
    def estimated_batch_bytes(self) -> int:
        return sum(schema.nbytes for schema in self.output.values() if isinstance(schema, TensorSchema))

    @property
    def estimated_metadata_bytes(self) -> int:
        return (
            _PROTOCOL_LIMITS.max_metadata_bytes
            if any(isinstance(schema, JsonSchema) for schema in self.output.values())
            else 0
        )

    @property
    def estimated_batch_gb(self) -> float:
        return self.estimated_batch_bytes / 1024**3

    @property
    def schema_hash(self) -> str:
        canonical = json.dumps(
            {
                key: value.model_dump(mode="json", exclude_none=True)
                for key, value in sorted(self.output.items())
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    @property
    def batch_spec_hash(self) -> str:
        canonical = json.dumps(
            self.to_request(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def to_request(self) -> dict[str, Any]:
        request = self.model_dump(mode="json")
        for schema in request["output"].values():
            if schema.get("resize") is None:
                schema.pop("resize", None)
        sampling = request["sampling"]
        if sampling.get("drop_n_first_frames") == 0:
            sampling.pop("drop_n_first_frames", None)
        if sampling.get("drop_n_last_frames") == 0:
            sampling.pop("drop_n_last_frames", None)
        if sampling.get("min_samples_per_episode") == 0:
            sampling.pop("min_samples_per_episode")
        return request

    @classmethod
    def from_yaml(cls, path: str | Path) -> BatchSpec | CanonicalSampleSpec | VQABatchSpec:
        try:
            with Path(path).expanduser().open("rb") as stream:
                payload = yaml.safe_load(stream)
            return cls.from_dict(payload)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ConfigurationError(f"invalid batch spec {path}: {exc}") from exc

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BatchSpec | CanonicalSampleSpec | VQABatchSpec:
        try:
            batch_format = value.get("batch_format") if isinstance(value, dict) else None
            if isinstance(batch_format, dict) and batch_format.get("type") == "canonical_sample":
                return CanonicalSampleSpec.model_validate(value)
            if isinstance(batch_format, dict) and batch_format.get("type") == "vlm_vqa":
                return VQABatchSpec.model_validate(value)
            return cls.model_validate(value)
        except ValueError as exc:
            raise ConfigurationError(f"invalid batch spec: {exc}") from exc


class VQABatchConfig(BatchConfig):
    model_config = ConfigDict(extra="forbid", strict=True)


class VQABatchSpec(StrictModel):
    """Scheduler-selected VLM batches, separate from the fixed-shape raw contract."""

    batch: VQABatchConfig
    batch_format: dict[str, Any] = Field(default_factory=lambda: {"type": "vlm_vqa", "version": 2})
    output: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("batch_format", mode="before")
    @classmethod
    def validate_format(cls, value: Any) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or set(value) != {"type", "version"}
            or value["type"] != "vlm_vqa"
            or type(value["version"]) is not int
            or value["version"] != 2
        ):
            raise ValueError("batch_format must be {type: vlm_vqa, version: 2}")
        return dict(value)

    @model_validator(mode="after")
    def validate_output(self) -> VQABatchSpec:
        minimum_bytes = self.batch.size * 3 * 256 * 256 * ALLOWED_DTYPES["float32"]
        if minimum_bytes > _PROTOCOL_LIMITS.max_tensor_bytes:
            raise ValueError("VQA batch exceeds the protocol per-tensor byte limit even with one image")
        resolved = {
            "pixel_values": {"shape": [self.batch.size, None, 3, 256, 256], "dtype": "float32"},
            "samples": {"type": "list", "length": self.batch.size, "items": "vlm_vqa_sample"},
            "action": {"type": "null"},
            "action_is_pad": {"type": "null"},
            "action_dim_is_pad": {"type": "null"},
            "is_vlm_batch": {"type": "bool", "value": True},
        }
        if self.output:
            # JSON comparison distinguishes bools, floats and integer dimensions.
            try:
                actual = json.dumps(self.output, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("VQA output must be literal JSON") from exc
            if actual != json.dumps(resolved, sort_keys=True):
                raise ValueError("VQA output must match the six-field vlm_vqa schema")
        self.output = resolved
        return self

    @property
    def is_vqa(self) -> bool:
        return True

    @property
    def estimated_batch_bytes(self) -> int:
        # N is dynamic: reserve a ceiling for accounting, never allocate it.
        return _PROTOCOL_LIMITS.max_tensor_bytes

    @property
    def estimated_metadata_bytes(self) -> int:
        return _PROTOCOL_LIMITS.max_metadata_bytes

    @property
    def estimated_batch_gb(self) -> float:
        return self.estimated_batch_bytes / 1024**3

    @property
    def schema_hash(self) -> str:
        canonical = json.dumps(self.output, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    @property
    def batch_spec_hash(self) -> str:
        canonical = json.dumps(self.to_request(), sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def to_request(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> VQABatchSpec:
        try:
            with Path(path).expanduser().open("rb") as stream:
                return cls.model_validate(yaml.safe_load(stream))
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ConfigurationError(f"invalid batch spec {path}: {exc}") from exc

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VQABatchSpec:
        try:
            return cls.model_validate(value)
        except ValueError as exc:
            raise ConfigurationError(f"invalid batch spec: {exc}") from exc
