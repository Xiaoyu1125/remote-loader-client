from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install it in your training environment") from exc
    return torch


def _identity(value: Any) -> Any:
    return value


def _validate_preprocess(remote_loader: Any, preprocess: Callable[[Any], Any] | None) -> None:
    if preprocess is not None and not callable(preprocess):
        raise TypeError("preprocess must be callable")
    if remote_loader.spec.is_vqa and preprocess is not None:
        raise ValueError("VQA batches must not pass through a VLA sample preprocessor")


@cache
def _direct_dataset_type() -> type:
    torch = _torch()

    class _DirectDataset(torch.utils.data.IterableDataset):
        handles_distributed_sharding = True

        def __init__(self, source: Any, preprocess: Callable[[Any], Any] | None) -> None:
            super().__init__()
            _validate_preprocess(source, preprocess)
            self.remote_loader = source
            self.preprocess = preprocess
            self.already_batched = source.spec.is_vqa

        def __iter__(self):
            source = iter(self.remote_loader)
            if self.preprocess is None:
                return source
            return map(self.preprocess, source)

        def close(self, *, close_source: bool = True) -> None:
            if not isinstance(close_source, bool):
                raise TypeError("close_source must be a bool")
            if not close_source:
                return
            close = getattr(self.remote_loader, "close", None)
            if not callable(close):
                raise TypeError("remote sample source must expose close()")
            close()

    return _DirectDataset


class RemoteIterableDataset:
    """Expose one rank-local stream to native DataLoader workers without copying its receiver."""

    def __new__(cls, remote_loader: Any, *, preprocess: Callable[[Any], Any] | None = None) -> Any:
        _torch()
        _validate_preprocess(remote_loader, preprocess)
        from ._worker_dataset import _WorkerDataset

        return _WorkerDataset(remote_loader, preprocess)


def make_torch_dataloader(
    remote_loader: Any,
    *,
    batch_size: int | None = None,
    preprocess: Callable[[Any], Any] | None = None,
    collate_fn: Callable[[Any], Any] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    pin_memory_device: str = "",
    **kwargs: Any,
) -> Any:
    """Use native workers for per-sample CPU preprocessing and local VLA batching.

    Without batch_size, yield individual raw/processed VLA samples. Each VLA
    HC frame is one sample; batch_size here is local. VQA bypasses local batching.
    Additional options are passed to torch.utils.data.DataLoader unchanged.
    """
    if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
        raise ValueError("batch_size must be a positive integer or None")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if collate_fn is not None and not callable(collate_fn):
        raise TypeError("collate_fn must be callable")
    if remote_loader.spec.is_vqa and (batch_size is not None or collate_fn is not None):
        raise ValueError("VQA already supplies complete batches; do not batch or collate them again")
    if batch_size is None and collate_fn is not None:
        raise ValueError("a training collate_fn requires an explicit batch_size")
    _torch()
    from ._managed_dataloader import _ManagedDataLoader

    dataset = (
        _direct_dataset_type()(remote_loader, preprocess)
        if num_workers == 0
        else RemoteIterableDataset(remote_loader, preprocess=preprocess)
    )
    if batch_size is None:
        collate_fn = _identity
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    try:
        return _ManagedDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs,
        )
    except BaseException:
        if num_workers > 0:
            dataset.close(close_source=False)
        raise


def close_torch_dataloader(
    dataloader: Any, *, iterator: Any | None = None, close_source: bool = True
) -> None:
    """Cancel source reads and reap native workers.

    SDK-created loaders track all their iterators. For a user-created native
    DataLoader, pass its iterator explicitly when stopping early.
    """
    if not isinstance(close_source, bool):
        raise TypeError("close_source must be a bool")
    dataset = getattr(dataloader, "dataset", None)
    close = getattr(dataset, "close", None)
    if not callable(close):
        raise TypeError("dataloader.dataset must expose close()")
    if iterator is not None and getattr(iterator, "_dataset", None) is not dataset:
        raise ValueError("iterator must belong to dataloader.dataset")

    from ._managed_dataloader import _ManagedDataLoader, _shutdown_iterators

    managed = isinstance(dataloader, _ManagedDataLoader)
    if managed:
        with dataloader._modalink_condition:
            dataloader._modalink_closed = True
    cancel_reads = getattr(dataset, "_cancel_reads", None)
    try:
        if callable(cancel_reads):
            cancel_reads(close_source=close_source)
        else:
            close(close_source=close_source)
    finally:
        try:
            if managed:
                dataloader._modalink_shutdown(iterator)
            else:
                iterators = []
                for candidate in (iterator, getattr(dataloader, "_iterator", None)):
                    if candidate is not None and all(candidate is not known for known in iterators):
                        iterators.append(candidate)
                _shutdown_iterators(iterators)
                dataloader._iterator = None
        finally:
            if callable(cancel_reads):
                close(close_source=False)
