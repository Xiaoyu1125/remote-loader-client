"""Bounded shared storage for one local sample connection.

Tensor reconstruction metadata keeps PyTorch's pickle semantics. Storage bytes
travel through a reusable mapping, and readers always return private storage.
"""

from __future__ import annotations

import ctypes
import io
import pickle
import weakref
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import torch

from ..protocol_limits import ProtocolLimits


@dataclass(frozen=True, slots=True)
class SharedSample:
    name: str | None
    capacity: int
    generation: int
    storages: tuple[tuple[int, int], ...]
    metadata: bytes


@dataclass(frozen=True, slots=True)
class _PreparedSample:
    storages: tuple[torch.UntypedStorage, ...]
    ranges: tuple[tuple[int, int], ...]
    nbytes: int
    metadata: bytes


def _local_metadata_limit(limits: ProtocolLimits) -> int:
    """Bound local pickle bytes independently of the wire JSON representation."""
    # A three-byte JSON float can require nine pickle bytes. Fourfold headroom
    # also covers container/memo opcodes; tensor reducers add dtype globals,
    # offsets and shape/stride pairs rather than copying payload into metadata.
    tensor_metadata = limits.max_tensors * (512 + 32 * limits.max_dimensions)
    return 4 * limits.max_metadata_bytes + tensor_metadata + 4096


class _MetadataBuffer(io.BytesIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def write(self, data: Any) -> int:
        if self.tell() + len(data) > self.limit:
            raise ValueError("sample metadata exceeds the transfer limit")
        return super().write(data)


class _MetadataPickler(pickle.Pickler):
    def __init__(self, stream: io.BytesIO, limits: ProtocolLimits) -> None:
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.limits = limits
        self.storages: list[torch.UntypedStorage] = []
        self.ranges: list[tuple[int, int]] = []
        self.storage_indices: dict[int, int] = {}
        self.tokens: dict[int, tuple[Any, ...]] = {}
        self.references: list[Any] = []
        self.nbytes = 0
        self.owned_bytes = 0

    def persistent_id(self, value: Any) -> tuple[Any, ...] | None:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("local sample transfer requires CPU tensors")
            if value.dim() > self.limits.max_dimensions or any(
                dimension > self.limits.max_dimension for dimension in value.shape
            ):
                raise ValueError("sample tensor shape exceeds the transfer limit")
        if isinstance(value, torch.storage.TypedStorage):
            storage, dtype = value._untyped_storage, value.dtype
        elif isinstance(value, torch.UntypedStorage):
            storage, dtype = value, None
        else:
            return None
        if storage.device.type != "cpu":
            raise ValueError("local sample transfer requires CPU storage")
        identity = id(value)
        if identity in self.tokens:
            return self.tokens[identity]
        if len(self.tokens) >= self.limits.max_tensors:
            raise ValueError("sample has too many storage bindings")
        nbytes = storage.nbytes()
        if nbytes > self.limits.max_tensor_bytes:
            raise ValueError("sample storage exceeds the tensor transfer limit")
        self.owned_bytes += nbytes
        if self.owned_bytes > self.limits.max_batch_bytes:
            raise ValueError("sample reconstructed storage exceeds the transfer limit")
        key = storage._cdata
        index = self.storage_indices.get(key)
        if index is None:
            index = len(self.storages)
            self.storage_indices[key] = index
            self.storages.append(storage)
            self.ranges.append((self.nbytes, nbytes))
            self.nbytes += nbytes
        # Retain wrapper identities until pickling finishes. Distinct wrappers
        # remain distinct on the receiving side, as in ordinary pickle; only
        # their identical input bytes are deduplicated in the transfer mapping.
        token = ("storage", len(self.tokens), index, dtype)
        self.references.append(value)
        self.tokens[identity] = token
        return token


def _prepare_sample(sample: Any, limits: ProtocolLimits, metadata_limit: int) -> _PreparedSample:
    stream = _MetadataBuffer(metadata_limit)
    pickler = _MetadataPickler(stream, limits)
    pickler.dump(sample)
    return _PreparedSample(tuple(pickler.storages), tuple(pickler.ranges), pickler.nbytes, stream.getvalue())


def _destroy_mapping(mapping: SharedMemory) -> None:
    try:
        mapping.close()
    finally:
        with suppress(FileNotFoundError):
            mapping.unlink()


class _OwnedRegion:
    def __init__(self, size: int) -> None:
        self.mapping = SharedMemory(create=True, size=size)
        self.finalizer = weakref.finalize(self, _destroy_mapping, self.mapping)


def _copy_storage(destination: memoryview, storage: torch.UntypedStorage) -> None:
    size = storage.nbytes()
    if size:
        source = (ctypes.c_ubyte * size).from_address(storage.data_ptr())
        with memoryview(source).cast("B") as view:
            destination[:] = view


class SharedSampleWriter:
    """One handler writes; the rank's shutdown thread may close concurrently."""

    def __init__(self, limits: ProtocolLimits | None = None) -> None:
        self.limits = ProtocolLimits() if limits is None else limits
        self._metadata_limit = _local_metadata_limit(self.limits)
        self._region: _OwnedRegion | None = None
        self._generation = 0
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("shared sample writer is closed")

    def write(
        self, sample: Any, *, before_resize: Callable[[int], None] | None = None
    ) -> SharedSample:
        self._check_open()
        prepared = _prepare_sample(sample, self.limits, self._metadata_limit)
        # This local owner keeps mmap alive if close() detaches the cached owner
        # during a copy. No cross-process or per-sample lifetime lock is needed.
        region = self._region
        try:
            self._check_open()
            if prepared.nbytes and (region is None or region.mapping.size < prepared.nbytes):
                if region is not None:
                    if before_resize is None:
                        raise RuntimeError("release the reader mapping before growing a shared sample slot")
                    before_resize(self._generation)
                    self._check_open()
                    self._region = None
                    region = None
                region = _OwnedRegion(prepared.nbytes)
                self._check_open()
                self._generation += 1
                self._region = region
            if prepared.nbytes:
                assert region is not None
                for storage, (offset, size) in zip(prepared.storages, prepared.ranges, strict=True):
                    with region.mapping.buf[offset : offset + size] as destination:
                        _copy_storage(destination, storage)
            self._check_open()
            return SharedSample(
                None if region is None else region.mapping.name,
                0 if region is None else region.mapping.size,
                self._generation,
                prepared.ranges,
                prepared.metadata,
            )
        finally:
            if self._closed:
                self._region = None
            # A retained exception traceback must not retain the transfer mmap.
            region = None

    def close(self) -> None:
        self._closed = True
        self._region = None


class _SampleUnpickler(pickle.Unpickler):
    def __init__(self, packet: SharedSample, mapping: SharedMemory | None, limits: ProtocolLimits) -> None:
        self.stream = io.BytesIO(packet.metadata)
        super().__init__(self.stream)
        self.packet = packet
        self.mapping = mapping
        self.limits = limits
        self.bindings: dict[int, tuple[tuple[Any, ...], Any]] = {}
        self.owned_bytes = 0

    def persistent_load(self, token: Any) -> Any:
        if not isinstance(token, tuple) or len(token) != 4 or token[0] != "storage":
            raise ValueError("invalid shared storage reference")
        _, identity, index, dtype = token
        if (
            type(identity) is not int
            or not 0 <= identity < self.limits.max_tensors
            or type(index) is not int
            or not 0 <= index < len(self.packet.storages)
            or (dtype is not None and not isinstance(dtype, torch.dtype))
        ):
            raise ValueError("invalid shared storage reference")
        cached = self.bindings.get(identity)
        if cached is not None:
            if cached[0] != token:
                raise ValueError("shared storage reference changed within a sample")
            return cached[1]
        offset, size = self.packet.storages[index]
        if dtype is not None and size % torch._utils._element_size(dtype):
            raise ValueError("shared storage size does not match its dtype")
        self.owned_bytes += size
        if self.owned_bytes > self.limits.max_batch_bytes:
            raise ValueError("sample reconstructed storage exceeds the transfer limit")
        owned = torch.UntypedStorage(size, device="cpu")
        if size:
            assert self.mapping is not None
            destination = (ctypes.c_ubyte * size).from_address(owned.data_ptr())
            with (
                self.mapping.buf[offset : offset + size] as source,
                memoryview(destination).cast("B") as target,
            ):
                target[:] = source
        result = (
            owned
            if dtype is None
            else torch.storage.TypedStorage(wrap_storage=owned, dtype=dtype, _internal=True)
        )
        self.bindings[identity] = (token, result)
        return result


class SharedSampleReader:
    """A single worker keeps a mapping; no borrowed view escapes read()."""

    def __init__(self, limits: ProtocolLimits | None = None) -> None:
        self.limits = ProtocolLimits() if limits is None else limits
        self._metadata_limit = _local_metadata_limit(self.limits)
        self._mapping: SharedMemory | None = None
        self._generation: int | None = None

    def _validate(self, packet: SharedSample) -> None:
        if not isinstance(packet, SharedSample):
            raise ValueError("invalid shared sample descriptor")
        if (
            type(packet.capacity) is not int
            or not 0 <= packet.capacity <= self.limits.max_batch_bytes
            or type(packet.generation) is not int
            or packet.generation < 0
            or not isinstance(packet.metadata, bytes)
            or len(packet.metadata) > self._metadata_limit
            or not isinstance(packet.storages, tuple)
            or len(packet.storages) > self.limits.max_tensors
        ):
            raise ValueError("shared sample descriptor exceeds transfer limits")
        if (packet.capacity == 0 and packet.name is not None) or (
            packet.capacity > 0 and (not isinstance(packet.name, str) or not packet.name)
        ):
            raise ValueError("invalid shared sample mapping")
        end = 0
        for entry in packet.storages:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("invalid shared storage range")
            offset, size = entry
            if (
                type(offset) is not int
                or type(size) is not int
                or offset < end
                or not 0 <= size <= self.limits.max_tensor_bytes
                or offset + size > packet.capacity
            ):
                raise ValueError("shared storage points outside the transfer mapping")
            end = offset + size

    def read(self, packet: SharedSample) -> Any:
        self._validate(packet)
        if packet.name is not None:
            if self._mapping is None or self._mapping.name != packet.name:
                self.close()
                mapping = SharedMemory(name=packet.name)
                if mapping.size != packet.capacity:
                    mapping.close()
                    raise ValueError("shared sample mapping size changed")
                self._mapping = mapping
                self._generation = packet.generation
            elif self._generation != packet.generation or self._mapping.size != packet.capacity:
                raise ValueError("shared sample mapping generation changed without replacement")
        elif self._mapping is not None:
            raise ValueError("shared sample mapping disappeared without release")
        reader = _SampleUnpickler(packet, self._mapping, self.limits)
        try:
            sample = reader.load()
            if reader.stream.tell() != len(packet.metadata):
                raise ValueError("unexpected trailing shared sample metadata")
            torch._utils._validate_loaded_sparse_tensors()
            return sample
        except BaseException:
            # Native sparse reconstruction retains tensors until validation.
            # Also release partial results without masking an unpickling error.
            torch._utils._sparse_tensors_to_validate.clear()
            raise

    def close(self) -> None:
        mapping, self._mapping = self._mapping, None
        self._generation = None
        if mapping is not None:
            mapping.close()
