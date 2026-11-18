#!/usr/bin/env python3
"""Inspect request schemas offline, or summarize VLA samples/VQA batches with --receive.

Received summaries flatten dictionary paths, omit empty dictionaries, and report
non-tensor types rather than literal JSON values; they are not exact sample dumps.
This separately delivered helper imports the SDK installed in the selected Python.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import remote_loader
from remote_loader import BatchSpec, RemoteDataLoader
from remote_loader.exceptions import RemoteLoaderError


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def emit(value: dict[str, Any], *, error: bool = False) -> None:
    print(
        json.dumps(value, ensure_ascii=False, allow_nan=False),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def describe_fields(item: dict[str, Any], value_count: int) -> dict[str, Any]:
    fields: dict[str, Any] = {}

    def visit(values: dict[str, Any], prefix: str = "") -> None:
        for name, value in values.items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(value, dict):
                visit(value, path)
                continue
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                description: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if hasattr(value, "device"):
                    description["device"] = str(value.device)
                if value_count:
                    preview = value.reshape(-1)[:value_count].tolist()
                    description["values"] = [
                        str(v) if isinstance(v, float) and not math.isfinite(v) else v for v in preview
                    ]
                fields[path] = description
            else:
                fields[path] = {"type": type(value).__name__}

    visit(item)
    return fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("batch.yaml"),
        help="delivered BatchSpec YAML (default: batch.yaml in the current directory)",
    )
    parser.add_argument(
        "--receive",
        action="store_true",
        help="create a session and receive items; otherwise inspect only the YAML",
    )
    parser.add_argument(
        "--items", type=positive_int, help="VLA samples or VQA batches to consume (default: 1)"
    )
    parser.add_argument(
        "--values",
        type=positive_int,
        help="preview tensor elements (max: 32; default: no previews; requires --receive)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=positive_seconds,
        default=60.0,
        help="steady-state item wait in seconds (default: 60); first-frame warmup is 300 seconds",
    )
    args = parser.parse_args(argv)
    if not args.receive and (args.items is not None or args.values is not None):
        parser.error("--items and --values require --receive")
    if args.values is not None and args.values > 32:
        parser.error("--values must be at most 32")

    received = 0
    try:
        spec = BatchSpec.from_yaml(str(args.spec))
        if not args.receive:
            output = spec.output
            fields = (
                output
                if spec.is_vqa
                else {
                    name: schema.model_dump(mode="json", exclude_none=True) for name, schema in output.items()
                }
            )
            tensor_names = {name for name, field in fields.items() if "shape" in field}
            emit(
                {
                    "mode": "schema",
                    "spec_path": str(args.spec.expanduser().resolve()),
                    "client_module": str(Path(remote_loader.__file__).resolve()),
                    "delivery_unit": "vqa_batch" if spec.is_vqa else "canonical_sample",
                    **({"batch_size": spec.batch.size} if spec.is_vqa else {}),
                    "field_count": len(fields),
                    "tensor_count": len(tensor_names),
                    "estimated_batch_bytes": spec.estimated_batch_bytes,
                    "schema_hash": spec.schema_hash,
                    "batch_spec_hash": spec.batch_spec_hash,
                    "fields": {
                        name: (
                            {**field, "bytes": output[name].nbytes}
                            if name in tensor_names and not spec.is_vqa
                            else field
                        )
                        for name, field in fields.items()
                    },
                }
            )
            return 0

        requested = args.items or 1
        with RemoteDataLoader(spec, batch_timeout=args.batch_timeout) as loader:
            for batch_index in range(1, requested + 1):
                try:
                    batch = next(loader)
                except StopIteration:
                    break
                received += 1
                emit(
                    {
                        "mode": "received",
                        "batch_index" if spec.is_vqa else "sample_index": batch_index,
                        "fields": describe_fields(batch, args.values or 0),
                    }
                )
        complete = received == requested
        emit(
            {
                "mode": "result",
                "requested": requested,
                "received": received,
                "complete": complete,
            }
        )
        return 0 if complete else 1
    except KeyboardInterrupt:
        return 130
    except (RemoteLoaderError, RuntimeError, ValueError, TimeoutError, OSError) as exc:
        emit({"mode": "error", "error": type(exc).__name__, "received": received}, error=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
