from __future__ import annotations

import ctypes
import json
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from remote_loader.exceptions import ChecksumError, EndOfStream, ProtocolError, SchemaMismatchError
from remote_loader.protocol_limits import ProtocolLimits
from remote_loader.transport._checksum import crc32 as _crc32

MAGIC = b"HC"
PROTOCOL_VERSION = 1
FLAG_END_OF_STREAM = 1
HEADER = struct.Struct("!2sBBHIQ")


@dataclass(frozen=True)
class BatchEnvelope:
    batch: dict[str, Any]
    batch_id: str
    sequence: int
    payload_bytes: int
    metadata_bytes: int = 0

    @property
    def queued_bytes(self) -> int:
        """Tensor payload plus bounded JSON metadata retained by this HC frame."""
        return self.payload_bytes + self.metadata_bytes


def parse_header(raw: bytes, limits: ProtocolLimits) -> tuple[int, int, int]:
    if len(raw) != HEADER.size:
        raise ProtocolError(f"invalid frame header length {len(raw)}")
    magic, version, flags, reserved, metadata_length, payload_length = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ProtocolError("invalid frame magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if reserved != 0:
        raise ProtocolError("reserved frame bits must be zero")
    if flags & ~FLAG_END_OF_STREAM:
        raise ProtocolError(f"unknown frame flags {flags}")
    if flags & FLAG_END_OF_STREAM:
        raise EndOfStream("remote stream completed")
    if metadata_length <= 0 or metadata_length > limits.max_metadata_bytes:
        raise ProtocolError(f"metadata length {metadata_length} is outside the allowed range")
    if payload_length > limits.max_batch_bytes:
        raise ProtocolError(f"payload length {payload_length} exceeds the client limit")
    return metadata_length, payload_length, flags


def decode_batch(
    metadata_raw: bytes,
    payload: bytes,
    *,
    expected_session_id: str,
    expected_schema_hash: str,
    expected_output: Mapping[str, Any] | None = None,
    limits: ProtocolLimits,
) -> BatchEnvelope:
    if not 0 < len(metadata_raw) <= limits.max_metadata_bytes:
        raise ProtocolError("batch metadata exceeds the client limit")
    if len(payload) > limits.max_batch_bytes:
        raise ProtocolError("batch payload exceeds the client limit")
    try:
        metadata = json.loads(metadata_raw, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("batch metadata is not valid UTF-8 JSON") from exc
    if not isinstance(metadata, dict):
        raise ProtocolError("batch metadata must be an object")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported metadata protocol version {metadata.get('protocol_version')!r}")
    if metadata.get("session_id") != expected_session_id:
        raise ProtocolError("batch session_id does not match the negotiated session")
    if metadata.get("schema_hash") != expected_schema_hash:
        raise SchemaMismatchError("batch schema_hash does not match the negotiated schema")
    if metadata.get("payload_length") != len(payload):
        raise ProtocolError("metadata payload_length does not match the received payload")
    _verify_checksum(metadata.get("checksum"), payload)

    if metadata.get("batch_tree_version", 1) != 1:
        raise ProtocolError("unsupported batch tree version")
    expected_tensors = None
    expected_json = None
    is_vqa = False
    if expected_output is not None:
        sample_schema = expected_output.get("samples")
        is_vqa = isinstance(sample_schema, Mapping) and sample_schema.get("items") == "vlm_vqa_sample"
        expected_tensors = {}
        expected_json = {}
        for name, schema in expected_output.items():
            kind = getattr(schema, "type", None) if not isinstance(schema, Mapping) else schema.get("type")
            json_kind = kind in ({"list", "null", "bool"} if is_vqa else {"string", "json"})
            (expected_json if json_kind else expected_tensors)[name] = schema

    tensors = metadata.get("tensors")
    if not isinstance(tensors, list) or len(tensors) > limits.max_tensors:
        raise ProtocolError("invalid tensor descriptor list")
    json_values = metadata.get("json_values", {})
    if not isinstance(json_values, dict):
        raise ProtocolError("json_values must be an object")
    _validate_json_value(json_values)
    retained_metadata = len(metadata_raw) if json_values else 0
    tensor_paths = [_descriptor_path(descriptor) for descriptor in tensors]
    json_paths = (
        _validate_raw_json(json_values, expected_json, tensor_paths)
        if expected_json is not None and not is_vqa
        else set()
    )
    if expected_json is not None and not is_vqa:
        retained_metadata = len(metadata_raw) if json_paths else 0
    batch = json_values
    tensor_names: set[str] = set()
    for descriptor, path in zip(tensors, tensor_paths, strict=True):
        if expected_tensors is not None:
            _validate_expected_descriptor(descriptor, expected_tensors, is_vqa=is_vqa)
        name, tensor = _decode_tensor(descriptor, payload, limits)
        if name in tensor_names:
            raise ProtocolError(f"duplicate tensor field {name!r}")
        tensor_names.add(name)
        if any(path[: len(json_path)] == json_path for json_path in json_paths):
            raise ProtocolError(f"tensor {name!r} collides with a JSON field")
        _insert_tensor(batch, path, tensor)
    if expected_tensors is not None and tensor_names != set(expected_tensors):
        raise SchemaMismatchError(
            f"tensor fields {sorted(tensor_names)} do not match negotiated fields {sorted(expected_tensors)}"
        )

    if is_vqa and expected_json is not None:
        actual_json = set(batch) - tensor_names
        if actual_json != set(expected_json):
            raise SchemaMismatchError(
                f"JSON fields {sorted(actual_json)} do not match negotiated fields {sorted(expected_json)}"
            )
        for name, schema in expected_json.items():
            kind = schema["type"]
            if kind == "null":
                if batch[name] is not None:
                    raise SchemaMismatchError(f"JSON field {name!r} must be null")
                continue
            if kind == "bool":
                if batch[name] is not True:
                    raise SchemaMismatchError(f"JSON field {name!r} must be literal true")
                continue
            length = schema["length"]
            rows = batch[name]
            if not isinstance(rows, list) or len(rows) != length:
                raise SchemaMismatchError(f"JSON field {name!r} must contain {length} rows")
            for row in rows:
                _validate_vqa_sample(row, batch["pixel_values"].shape[1])

    batch_id = metadata.get("batch_id")
    sequence = metadata.get("sequence")
    if not isinstance(batch_id, str) or not batch_id:
        raise ProtocolError("batch_id must be a non-empty string")
    if not isinstance(sequence, int) or sequence < 0:
        raise ProtocolError("sequence must be a non-negative integer")
    return BatchEnvelope(
        batch=batch,
        batch_id=batch_id,
        sequence=sequence,
        payload_bytes=len(payload),
        metadata_bytes=retained_metadata,
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _descriptor_path(descriptor: Any) -> tuple[str, ...]:
    if not isinstance(descriptor, dict):
        raise ProtocolError("tensor descriptor must be an object")
    name = descriptor.get("name")
    if not isinstance(name, str) or not name:
        raise ProtocolError("tensor name must be a non-empty string")
    path = descriptor.get("path", [name])
    if not isinstance(path, list) or not path or any(not isinstance(key, str) or not key for key in path):
        raise ProtocolError(f"tensor {name!r} path must be a nonempty list of nonempty string keys")
    if path != [name] and ".".join(path) != name:
        raise SchemaMismatchError(f"tensor {name!r} path does not identify its negotiated field")
    return tuple(path)


def _insert_tensor(batch: dict[str, Any], path: tuple[str, ...], tensor: Any) -> None:
    target = batch
    for key in path[:-1]:
        if key not in target:
            target[key] = {}
        if not isinstance(target[key], dict):
            raise ProtocolError(f"conflicting tensor/JSON path {path!r}")
        target = target[key]
    if path[-1] in target:
        raise ProtocolError(f"duplicate or conflicting tensor/JSON path {path!r}")
    target[path[-1]] = tensor


def _validate_raw_json(
    values: dict[str, Any], expected: Mapping[str, Any], tensor_paths: list[tuple[str, ...]]
) -> set[tuple[str, ...]]:
    # Traverse sender structure, never split negotiated names into invented paths.
    # A declared JSON object is atomic; only undeclared dictionaries are containers.
    matched: set[str] = set()
    paths: set[tuple[str, ...]] = set()

    def visit(container: dict[str, Any], prefix: tuple[str, ...]) -> None:
        for key, value in container.items():
            path = (*prefix, key)
            name = ".".join(path)
            if name in expected:
                if name in matched:
                    raise SchemaMismatchError(f"duplicate JSON field {name!r}")
                schema = expected[name]
                kind = schema.get("type") if isinstance(schema, Mapping) else schema.type
                if kind == "string" and not isinstance(value, str):
                    raise SchemaMismatchError(f"JSON field {name!r} must be a literal string")
                matched.add(name)
                paths.add(path)
            elif isinstance(value, dict):
                if not value and not any(
                    len(tensor_path) > len(path) and tensor_path[: len(path)] == path
                    for tensor_path in tensor_paths
                ):
                    raise SchemaMismatchError(f"unexpected JSON field {name!r}")
                visit(value, path)
            else:
                raise SchemaMismatchError(f"unexpected JSON field {name!r}")

    visit(values, ())
    if matched != set(expected):
        raise SchemaMismatchError(
            f"JSON fields {sorted(matched)} do not match negotiated fields {sorted(expected)}"
        )
    return paths


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json_value(item)
        return
    raise ProtocolError("JSON values must contain only finite literal JSON values")


def _validate_vqa_sample(row: Any, image_count: int) -> None:
    if not isinstance(row, dict) or any(
        not isinstance(row.get(field), str) for field in ("template", "question", "answer")
    ):
        raise SchemaMismatchError("VQA samples require literal template, question and answer strings")
    if "thinking" in row and not isinstance(row["thinking"], str):
        raise SchemaMismatchError("VQA thinking must be a literal string")
    slots = []
    for name, size in row.items():
        if re.fullmatch(r"image(?:0|[1-9][0-9]*)", name) is None:
            continue
        slots.append(name)
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(dim) is not int or dim != 256 for dim in size)
        ):
            raise SchemaMismatchError("VQA image slots must contain [256, 256] JSON dimensions")
    if not slots or len(slots) > image_count or set(slots) != {f"image{i}" for i in range(len(slots))}:
        raise SchemaMismatchError("VQA image slots must be contiguous from image0 and fit pixel_values")


def _validate_expected_descriptor(
    descriptor: Any, expected_output: Mapping[str, Any], *, is_vqa: bool = False
) -> None:
    if not isinstance(descriptor, dict):
        raise ProtocolError("tensor descriptor must be an object")
    name = descriptor.get("name")
    if not isinstance(name, str) or name not in expected_output:
        raise SchemaMismatchError(f"unexpected tensor field {name!r}")
    if is_vqa and descriptor.get("path", [name]) != [name]:
        raise SchemaMismatchError(f"tensor {name!r} must be a top-level field")
    schema = expected_output[name]
    expected_shape = list(schema.shape if hasattr(schema, "shape") else schema["shape"])
    expected_dtype = str(schema.dtype if hasattr(schema, "dtype") else schema["dtype"])
    shape = descriptor.get("shape")
    if is_vqa and name == "pixel_values" and expected_shape[1:] == [None, 3, 256, 256]:
        valid_shape = (
            isinstance(shape, list)
            and len(shape) == 5
            and all(type(dim) is int and dim > 0 for dim in shape)
            and shape[0] == expected_shape[0]
            and shape[2:] == expected_shape[2:]
        )
    else:
        valid_shape = shape == expected_shape
    if not valid_shape:
        raise SchemaMismatchError(
            f"tensor {name!r} shape {descriptor.get('shape')} does not match {expected_shape}"
        )
    if descriptor.get("dtype") != expected_dtype:
        raise SchemaMismatchError(
            f"tensor {name!r} dtype {descriptor.get('dtype')!r} does not match {expected_dtype!r}"
        )


def _decode_tensor(descriptor: Any, payload: bytes, limits: ProtocolLimits) -> tuple[str, Any]:
    if not isinstance(descriptor, dict):
        raise ProtocolError("tensor descriptor must be an object")
    name = descriptor.get("name")
    dtype_name = descriptor.get("dtype")
    shape = descriptor.get("shape")
    offset = descriptor.get("offset")
    length = descriptor.get("length")
    if not isinstance(name, str) or not name:
        raise ProtocolError("tensor name must be a non-empty string")
    if not isinstance(shape, list) or len(shape) > limits.max_dimensions:
        raise ProtocolError(f"invalid shape for tensor {name!r}")
    if any(type(dim) is not int or dim <= 0 or dim > limits.max_dimension for dim in shape):
        raise ProtocolError(f"shape for tensor {name!r} exceeds client limits")
    if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length < 0:
        raise ProtocolError(f"invalid offset/length for tensor {name!r}")
    if length > limits.max_tensor_bytes or offset + length > len(payload):
        raise ProtocolError(f"tensor {name!r} points outside the received payload")
    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("PyTorch is required; install it in your training environment") from exc
    dtype, item_size = _tensor_dtype_info(dtype_name)
    elements = 1
    for dim in shape:
        elements *= dim
    expected_length = elements * item_size
    if expected_length != length:
        raise ProtocolError(
            f"tensor {name!r} length {length} does not match dtype/shape size {expected_length}"
        )
    try:
        source = memoryview(payload)[offset : offset + length]
        if dtype is torch.bool:
            # Preserve PyTorch's normalization of noncanonical bool storage.
            tensor = torch.frombuffer(bytearray(source), dtype=dtype, count=elements).reshape(shape).clone()
        else:
            tensor = torch.empty(tuple(shape), dtype=dtype, device="cpu")
            target = (ctypes.c_ubyte * length).from_address(tensor.data_ptr())
            memoryview(target).cast("B")[:] = source
    except (RuntimeError, ValueError) as exc:
        raise ProtocolError(f"could not decode tensor {name!r}: {exc}") from exc
    return name, tensor


@lru_cache(maxsize=1)
def _tensor_dtype_metadata() -> dict[str, tuple[Any, int]]:
    # Fixed-size, lazy cache: control-only use does not require importing PyTorch.
    import torch

    names = (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "uint16",
        "uint32",
        "uint64",
    )
    return {
        name: (dtype, torch.empty((), dtype=dtype, device="cpu").element_size())
        for name in names
        if (dtype := getattr(torch, name, None)) is not None
    }


def _tensor_dtype_info(name: Any) -> tuple[Any, int]:
    try:
        return _tensor_dtype_metadata()[name]
    except (KeyError, TypeError) as exc:
        raise ProtocolError(f"unsupported tensor dtype {name!r}") from exc


def _verify_checksum(checksum: Any, payload: bytes) -> None:
    if not isinstance(checksum, dict) or checksum.get("algorithm") != "crc32":
        raise ProtocolError("batch checksum must use crc32")
    expected = checksum.get("value")
    actual = f"{_crc32(payload) & 0xFFFFFFFF:08x}"
    if expected != actual:
        raise ChecksumError("batch payload checksum mismatch")


def encode_batch_frame(
    batch: dict[str, Any],
    *,
    session_id: str,
    batch_id: str,
    sequence: int,
    schema_hash: str,
) -> bytes:
    """Reference encoder used by server implementers and protocol tests."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to encode tensor batches") from exc
    payload_parts: list[bytes] = []
    descriptors: list[dict[str, Any]] = []
    offset = 0
    names: set[str] = set()

    def encode_tree(values: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
        nonlocal offset
        literals: dict[str, Any] = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise ProtocolError("JSON object keys must be strings")
            child_path = (*path, key)
            if isinstance(value, torch.Tensor):
                name = ".".join(child_path)
                descriptor = {"name": name, "path": list(child_path)}
                _descriptor_path(descriptor)
                if name in names:
                    raise ProtocolError(f"duplicate tensor field {name!r}")
                names.add(name)
                tensor = value.detach().cpu().contiguous().resolve_conj().resolve_neg()
                # Materialize lazy view bits before copying logical values, including scalars.
                raw = ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())
                descriptor.update(
                    dtype=str(tensor.dtype).removeprefix("torch."),
                    shape=list(tensor.shape),
                    offset=offset,
                    length=len(raw),
                )
                descriptors.append(descriptor)
                payload_parts.append(raw)
                offset += len(raw)
            elif isinstance(value, dict):
                child_literals = encode_tree(value, child_path)
                if child_literals or not value:
                    literals[key] = child_literals
            else:
                literals[key] = value
        return literals

    json_values = encode_tree(batch, ())
    payload = b"".join(payload_parts)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "batch_tree_version": 1,
        "session_id": session_id,
        "batch_id": batch_id,
        "sequence": sequence,
        "schema_hash": schema_hash,
        "tensors": descriptors,
        "json_values": json_values,
        "payload_length": len(payload),
        "checksum": {"algorithm": "crc32", "value": f"{_crc32(payload) & 0xFFFFFFFF:08x}"},
    }
    _validate_json_value(json_values)
    metadata_raw = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    limits = ProtocolLimits()
    if len(metadata_raw) > limits.max_metadata_bytes:
        raise ProtocolError("batch metadata exceeds the client limit")
    if len(payload) > limits.max_batch_bytes:
        raise ProtocolError("batch payload exceeds the client limit")
    header = HEADER.pack(MAGIC, PROTOCOL_VERSION, 0, 0, len(metadata_raw), len(payload))
    return header + metadata_raw + payload


def encode_end_of_stream() -> bytes:
    return HEADER.pack(MAGIC, PROTOCOL_VERSION, FLAG_END_OF_STREAM, 0, 0, 0)
