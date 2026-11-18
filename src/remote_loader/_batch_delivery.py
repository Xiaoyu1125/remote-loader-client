from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .batch_spec import BatchSpec, CanonicalSampleSpec, VQABatchSpec
from .canonical_spec import ColumnSource


def prepare_vqa_samples(batch: dict[str, Any]) -> dict[str, Any]:
    """Restore validated image-slot tuples without changing queued JSON or pixels."""
    samples = []
    for sample in batch["samples"]:
        prepared_sample = sample.copy()
        index = 0
        while (key := f"image{index}") in sample:
            prepared_sample[key] = tuple(sample[key])
            index += 1
        samples.append(prepared_sample)
    prepared = batch.copy()
    prepared["samples"] = samples
    return prepared


@dataclass(frozen=True)
class CanonicalImageDelivery:
    """Restore logical images only after validated uint8 wire storage is consumed."""

    paths: tuple[tuple[str, ...], ...] = ()
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    @classmethod
    def from_spec(cls, spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec) -> CanonicalImageDelivery:
        if spec.is_vqa:
            return cls()
        if isinstance(spec, CanonicalSampleSpec):
            paths = tuple(
                tuple(field.target) for field in spec.fields
                if isinstance(field.source, ColumnSource)
                and field.tensor is not None and field.tensor.dtype == "float32"
            )
            image = spec.image_output
            return cls(
                paths,
                None if image.mean is None else tuple(image.mean),
                None if image.std is None else tuple(image.std),
            )
        return cls(tuple(
            (name,) for name, schema in spec.output.items()
            if getattr(schema, "dtype", None) == "float32"
        ))

    @cached_property
    def _normalization(self) -> tuple[Any, Any]:
        import torch

        return (
            torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1),
            torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1),
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # Spawn receives configuration only, not a parent's cached tensors.
        return type(self), (self.paths, self.mean, self.std)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not self.paths:
            return sample
        import torch

        prepared = sample
        for path in self.paths:
            original = sample
            for segment in path[:-1]:
                original = original[segment]
            value = original[path[-1]]
            # Metadata classified cameras at session startup. Strict wire schema
            # validation leaves all other float32 columns in their logical dtype.
            if value.dtype != torch.uint8:
                continue
            restored = value.to(dtype=torch.float32).div_(255.0)
            if self.mean is not None:
                mean, std = self._normalization
                restored.sub_(mean).div_(std)
            if prepared is sample:
                prepared = sample.copy()
            original, target = sample, prepared
            for segment in path[:-1]:
                if target[segment] is original[segment]:
                    target[segment] = original[segment].copy()
                original, target = original[segment], target[segment]
            target[path[-1]] = restored
        return prepared
