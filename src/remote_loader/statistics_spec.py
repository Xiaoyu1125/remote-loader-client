"""Strict, offline declarative tensor statistics; no tensor runtime is imported."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from .batch_spec import ImageOutputSpec
from .canonical_spec import CanonicalContract, ConstantSource
from .exceptions import ConfigurationError


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatisticsFormat(_StrictModel):
    type: Literal["tensor_statistics"] = "tensor_statistics"
    version: Literal[1] = 1

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("statistics format version must be an integer")
        return value


class StatisticsScope(_StrictModel):
    dataset_indices: list[StrictInt] = Field(min_length=1)
    grouping: Literal["pooled", "per_dataset"]

    @field_validator("dataset_indices")
    @classmethod
    def validate_indices(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value) or len(set(value)) != len(value):
            raise ValueError("dataset indices must be unique nonnegative integers")
        return sorted(value)


class StatisticsScan(_StrictModel):
    drop_n_first_frames: StrictInt = Field(default=0, ge=0)
    drop_n_last_frames: StrictInt = Field(default=0, ge=0)
    stride: StrictInt = Field(default=1, gt=0)


class StatisticsView(CanonicalContract):
    image_output: ImageOutputSpec = Field(default_factory=ImageOutputSpec)


def _statistics_view_request(view: StatisticsView) -> dict[str, Any]:
    """Serialize a statistics view without dropping required JSON null constants."""
    payload = view.model_dump(mode="json", exclude_none=True)
    for binding, field in zip(view.fields, payload["fields"], strict=True):
        if isinstance(binding.source, ConstantSource) and binding.source.constant is None:
            field["source"]["constant"] = None
    if payload["image_output"].get("antialias") is True:
        payload["image_output"].pop("antialias")
    return payload


class StatisticsMask(_StrictModel):
    field: list[str] = Field(min_length=1)
    axes: list[StrictInt]


class PostReduce(_StrictModel):
    op: Literal["min", "max"]
    axes: list[StrictInt] = Field(min_length=1)


class SimpleMetric(_StrictModel):
    op: Literal["count", "min", "max", "mean"]
    post_reduce: PostReduce | None = None


class StdMetric(_StrictModel):
    op: Literal["std"]
    ddof: StrictInt = Field(ge=0, le=1)
    post_reduce: PostReduce | None = None


class QuantileMetric(_StrictModel):
    op: Literal["quantile"]
    q: float = Field(ge=0, le=1, allow_inf_nan=False)
    method: Literal["lower", "linear"]
    post_reduce: PostReduce | None = None

    @field_validator("q", mode="before")
    @classmethod
    def validate_probability(cls, value: Any) -> Any:
        if type(value) not in (int, float):
            raise ValueError("quantile q must be a finite number")
        return value


Metric = Annotated[SimpleMetric | StdMetric | QuantileMetric, Field(discriminator="op")]


class StatisticsQuery(_StrictModel):
    field: list[str] = Field(min_length=1)
    mask: StatisticsMask | None = None
    reduce_axes: list[Literal["sample"] | StrictInt] = Field(min_length=1)
    non_finite: Literal["error", "exclude"] = "error"
    metrics: dict[str, Metric] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query(self) -> StatisticsQuery:
        if self.reduce_axes.count("sample") != 1 or len(set(self.reduce_axes)) != len(self.reduce_axes):
            raise ValueError("reduce_axes requires sample exactly once and no duplicate axes")
        if any(not name for name in self.metrics):
            raise ValueError("metric names must not be empty")
        return self


class StatisticsExecution(_StrictModel):
    max_memory_bytes: StrictInt = Field(gt=0)


def _validate_axes(axes: list[int], rank: int, label: str) -> None:
    if len(set(axes)) != len(axes) or any(axis < 0 or axis >= rank for axis in axes):
        raise ValueError(f"{label} axes must be unique and within tensor rank {rank}")


class StatisticsSpec(_StrictModel):
    statistics_format: StatisticsFormat = Field(default_factory=StatisticsFormat)
    scope: StatisticsScope
    scan: StatisticsScan = Field(default_factory=StatisticsScan)
    view: StatisticsView
    statistics: dict[str, StatisticsQuery] = Field(min_length=1)
    execution: StatisticsExecution

    @model_validator(mode="after")
    def validate_statistics(self) -> StatisticsSpec:
        fields = {tuple(field.target): field for field in self.view.fields}
        for name, query in self.statistics.items():
            if not name:
                raise ValueError("query names must not be empty")
            target = fields.get(tuple(query.field))
            if target is None or target.tensor is None or target.tensor.dtype == "bool":
                raise ValueError("statistics field must reference a numeric tensor")
            shape = target.tensor.shape
            axes = [axis for axis in query.reduce_axes if isinstance(axis, int)]
            _validate_axes(axes, len(shape), "reduce")
            if query.mask is not None:
                mask = fields.get(tuple(query.mask.field))
                if mask is None or mask.tensor is None or mask.tensor.dtype != "bool":
                    raise ValueError("mask field must reference a bool tensor")
                _validate_axes(query.mask.axes, len(shape), "mask")
                if len(query.mask.axes) != len(mask.tensor.shape):
                    raise ValueError("mask axes must map every mask dimension")
                if any(
                    size not in (1, shape[axis])
                    for size, axis in zip(mask.tensor.shape, query.mask.axes, strict=True)
                ):
                    raise ValueError("mask dimensions must broadcast to their declared axes")
            for metric in query.metrics.values():
                if metric.post_reduce is not None:
                    _validate_axes(metric.post_reduce.axes, len(shape) - len(axes), "post_reduce")
        return self

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StatisticsSpec:
        try:
            return cls.model_validate(value)
        except ValueError as exc:
            raise ConfigurationError(f"invalid statistics spec: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> StatisticsSpec:
        try:
            with Path(path).expanduser().open("rb") as stream:
                return cls.from_dict(yaml.safe_load(stream))
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ConfigurationError(f"invalid statistics spec {path}: {exc}") from exc

    def to_request(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["view"] = _statistics_view_request(self.view)
        return payload

    @property
    def request_hash(self) -> str:
        encoded = json.dumps(
            self.to_request(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
