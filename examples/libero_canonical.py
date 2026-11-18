"""Build native LIBERO canonical samples from the existing server catalog.

This example never registers, reconfigures or releases a training allocation.
--print-spec only queries allocation/metadata and emits the request as JSON.
The stream returns one raw sample per next(loader), not a training batch.
Images remain native uint8 RGB; no resize, normalization or model processing.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from remote_loader import CanonicalSampleSpec, RemoteDataLoader
from remote_loader.auth.token_store import TokenStore
from remote_loader.canonical_spec import ALLOWED_DTYPES
from remote_loader.client_config import ClientConfig, normalize_control_endpoint, validate_control_security
from remote_loader.control.api_client import ControlClient
from remote_loader.exceptions import ConfigurationError, RemoteLoaderError
from remote_loader.scheduler import resolve_control_endpoints

_VECTOR_COLUMNS = ("action", "observation.state")
_CAMERA_COLUMNS = ("observation.images.image", "observation.images.wrist_image")


def _dataset_geometry(dataset: dict) -> dict:
    """Extract only metadata that affects this mapping; never merge task tables."""
    if not isinstance(dataset, dict):
        raise ConfigurationError("each scheduler_datasets entry must be an object")
    fps = dataset.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ConfigurationError("dataset fps must be a finite positive number")
    features = dataset.get("features")
    if not isinstance(features, dict):
        raise ConfigurationError("dataset features must be an object")
    camera_keys = dataset.get("camera_keys")
    if not isinstance(camera_keys, list) or any(column not in camera_keys for column in _CAMERA_COLUMNS):
        raise ConfigurationError("LIBERO image and wrist_image must be declared camera_keys")
    geometry: dict[str, Any] = {"fps": int(fps) if float(fps).is_integer() else fps}
    for column in (*_VECTOR_COLUMNS, *_CAMERA_COLUMNS):
        feature = features.get(column)
        if not isinstance(feature, dict):
            raise ConfigurationError(f"missing LIBERO feature {column!r}")
        shape = feature.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
        ):
            raise ConfigurationError(f"{column}: metadata shape must have positive integer dimensions")
        if "fps" in feature and feature["fps"] != fps:
            raise ConfigurationError(f"{column}: feature fps disagrees with dataset fps")
        names = feature.get("names")
        dtype = feature.get("dtype")
        if column in _VECTOR_COLUMNS:
            if (
                len(shape) != 1
                or not isinstance(dtype, str)
                or dtype not in ALLOWED_DTYPES
                or dtype == "bool"
            ):
                raise ConfigurationError(f"{column}: expected a fixed-width numeric vector")
            geometry[column] = {"shape": list(shape), "dtype": dtype}
        else:
            if len(shape) != 3 or names != ["height", "width", "rgb"] or shape[2] != 3:
                raise ConfigurationError(f"{column}: expected native [height,width,rgb] camera metadata")
            if dtype not in {"video", "image", "uint8"}:
                raise ConfigurationError(f"{column}: unsupported native camera dtype {dtype!r}")
            info = feature.get("info", {})
            if not isinstance(info, dict):
                raise ConfigurationError(f"{column}: camera info must be an object")
            for key, expected in zip(("video.height", "video.width", "video.channels"), shape, strict=True):
                if key in info and info[key] != expected:
                    raise ConfigurationError(f"{column}: {key} disagrees with native shape")
            if "video.fps" in info and info["video.fps"] != fps:
                raise ConfigurationError(f"{column}: video fps disagrees with dataset fps")
            # The service's native camera decoder emits RGB CHW uint8, not video bytes.
            geometry[column] = {"shape": [3, shape[0], shape[1]], "dtype": "uint8"}
    return geometry


def build_libero_spec(metadata: dict) -> CanonicalSampleSpec:
    """Require compatible ordered scheduler catalogs and bind their native values."""
    datasets = metadata.get("scheduler_datasets") if isinstance(metadata, dict) else None
    if not isinstance(datasets, list) or not datasets:
        raise ConfigurationError("metadata must contain nonempty scheduler_datasets")
    geometry = _dataset_geometry(datasets[0])
    for index, dataset in enumerate(datasets[1:], start=1):
        if _dataset_geometry(dataset) != geometry:
            raise ConfigurationError(
                f"scheduler_datasets[{index}] has incompatible LIBERO geometry, dtypes or fps"
            )
    fields: list[dict[str, Any]] = []
    for column in (*_VECTOR_COLUMNS, *_CAMERA_COLUMNS):
        count = 32 if column == "action" else 1
        fields.append(
            {
                "target": [column],
                "source": {"column": column},
                "window": "action" if column == "action" else "observation",
                "tensor": {"shape": [count, *geometry[column]["shape"]], "dtype": geometry[column]["dtype"]},
            }
        )
    fields.extend(
        [
            {"target": ["task"], "source": {"column": "task"}, "value_type": "string"},
            {"target": ["frequency"], "source": {"constant": geometry["fps"]}, "value_type": "json"},
        ]
    )
    for target, columns, count in (
        ("action_is_pad", ("action",), 32),
        ("state_is_pad", ("observation.state",), 1),
        ("image_is_pad", _CAMERA_COLUMNS, 1),
    ):
        fields.append(
            {
                "target": [target],
                "source": {"is_pad": {"fields": [[column] for column in columns], "reduction": "equal"}},
                "tensor": {"shape": [count], "dtype": "bool"},
            }
        )
    return CanonicalSampleSpec.from_dict(
        {
            "batch_format": {"type": "canonical_sample", "version": 2},
            "image_output": {"enabled": False},
            "sampling": {"shuffle_seed": 42, "drop_n_first_frames": 0, "drop_n_last_frames": 7},
            "require_metadata": {"fps": geometry["fps"]},
            "windows": {"observation": {"offsets": [0]}, "action": {"start": 0, "step": 1, "count": 32}},
            "fields": fields,
        }
    )


def _verify_saved(expected: Any, actual: Any, path: str = "sample") -> Any:
    """Compare every tensor byte and every direct value, not just schema."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise ValueError(f"saved {path} is not a tensor")
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise ValueError(f"saved {path} dtype or shape changed")
        before = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        after = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(before, after):
            raise ValueError(f"saved {path} tensor content changed")
        return {"shape": list(actual.shape), "dtype": str(actual.dtype), "content_equal": True}
    if type(expected) is not type(actual):
        raise ValueError(f"saved {path} value type changed")
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"saved {path} keys changed")
        return {key: _verify_saved(value, actual[key], f"{path}.{key}") for key, value in expected.items()}
    if isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f"saved {path} list length changed")
        return [
            _verify_saved(value, actual[index], f"{path}[{index}]") for index, value in enumerate(expected)
        ]
    if expected != actual:
        raise ValueError(f"saved {path} direct value changed")
    return {"value": actual, "value_equal": True}


def benchmark(loader, *, steps: int, save_sample: int, output: Path) -> dict:
    """Start an unstarted loader, consume N samples, close, save one and verify it.

    Numbering is one-based. Startup times loader.start(), not configuration or
    metadata queries. Wait times measure next(loader); loop time also includes
    discarding samples. Prefetch can receive more frames than delivered_samples.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(save_sample) is not int or not 1 <= save_sample <= steps:
        raise ValueError("save_sample must be between 1 and steps")
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    selected = None
    delivered = 0
    next_wait = 0.0
    first_wait = 0.0
    steady_wait = 0.0
    started = time.perf_counter()
    try:
        loader.start()
        startup_seconds = time.perf_counter() - started
        loop_started = time.perf_counter()
        first_delivered = loop_started
        for number in range(1, steps + 1):
            wait_started = time.perf_counter()
            try:
                sample = next(loader)
            except StopIteration as exc:
                raise RuntimeError(f"stream ended after {delivered} of {steps} requested samples") from exc
            received = time.perf_counter()
            wait_seconds = received - wait_started
            next_wait += wait_seconds
            delivered += 1
            if number == 1:
                first_wait = wait_seconds
                first_delivered = received
            else:
                steady_wait += wait_seconds
            if not isinstance(sample, dict):
                raise ValueError("canonical loader must deliver a sample dictionary")
            if number == save_sample:
                selected = sample
            del sample
        loop_finished = time.perf_counter()
    finally:
        teardown_started = time.perf_counter()
        loader.close()
        teardown_seconds = time.perf_counter() - teardown_started
    # No live stream or unselected samples remain while saving/verifying.
    save_started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(selected, stream)
    save_seconds = time.perf_counter() - save_started
    verify_started = time.perf_counter()
    restored = torch.load(output, weights_only=True)
    verification = _verify_saved(selected, restored)
    verification_seconds = time.perf_counter() - verify_started
    loop_seconds = loop_finished - loop_started
    steady_seconds = loop_finished - first_delivered if steps > 1 else 0.0
    return {
        "delivered_samples": delivered,
        "saved_sample_number": save_sample,
        "saved_path": str(output),
        "startup_seconds": startup_seconds,
        "loop_seconds": loop_seconds,
        "summed_next_wait_seconds": next_wait,
        "first_sample_next_wait_seconds": first_wait,
        "startup_to_first_sample_seconds": first_delivered - started,
        "steady_state_samples": max(0, steps - 1),
        "steady_state_loop_seconds": steady_seconds,
        "steady_state_summed_next_wait_seconds": steady_wait,
        "steady_state_samples_per_second": (steps - 1) / steady_seconds if steady_seconds else None,
        "teardown_seconds": teardown_seconds,
        "save_seconds": save_seconds,
        "verification_seconds": verification_seconds,
        "benchmark_total_seconds": time.perf_counter() - started,
        "saved_sample_verified": True,
        "verification": verification,
        "count_scope": "delivered samples; SDK prefetch may receive additional network frames",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", help="explicit existing control endpoint; reuse configured TLS/data security"
    )
    parser.add_argument("--steps", type=int, default=500, help="delivered samples, not network frames")
    parser.add_argument("--save-sample", type=int, default=100, help="one-based sample number to save")
    parser.add_argument("--output", type=Path, default=Path("/output/libero_sample_100.pt"))
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="GET metadata and print request JSON without creating a stream",
    )
    args = parser.parse_args()
    if not args.print_spec and (args.steps <= 0 or not 1 <= args.save_sample <= args.steps):
        raise ValueError("--steps must be positive and 1 <= --save-sample <= --steps")
    config = ClientConfig.load()
    token = TokenStore().load()
    endpoints = (
        (normalize_control_endpoint(args.endpoint),)
        if args.endpoint
        else resolve_control_endpoints(config, token=token)
    )
    ssl_context = config.create_tls_context()
    datasets: list[dict] = []
    for endpoint in endpoints:
        validate_control_security(endpoint)
        control = ControlClient(endpoint, token, ssl_context=ssl_context)
        try:
            metadata = control.get_metadata()
        finally:
            control.close()
        catalog = metadata.get("scheduler_datasets")
        if not isinstance(catalog, list) or not catalog:
            raise ConfigurationError(f"{endpoint}: metadata must contain nonempty scheduler_datasets")
        datasets.extend(catalog)
    # Compare native output shapes, dtypes and fps across every queried node.
    spec = build_libero_spec({"scheduler_datasets": datasets})
    if args.print_spec:
        print(json.dumps(spec.to_request(), indent=2, ensure_ascii=False))
        return
    # Freeze exactly the queried allocation; never re-resolve to unchecked nodes.
    settings: dict[str, Any] = {
        "token": token,
        "prefetch_batches": config.runtime.prefetch_batches,
        "_client_config": config,
        "_track_transfer_state": False,
    }
    if args.endpoint or config.scheduler is None:
        settings["control_endpoint"] = endpoints[0]
    else:
        settings["_control_endpoints"] = endpoints
    loader = RemoteDataLoader(spec, **settings)
    result = benchmark(loader, steps=args.steps, save_sample=args.save_sample, output=args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (RemoteLoaderError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"event": "error", "error": str(exc)}), flush=True)
        raise SystemExit(1) from None
