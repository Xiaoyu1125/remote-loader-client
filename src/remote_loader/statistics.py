"""One-endpoint finite statistics jobs, independent of training loader resources."""

from __future__ import annotations

import copy
import math
import re
import time
from ipaddress import ip_address
from typing import Any

import httpx

from .auth import TokenStore
from .canonical_spec import ALLOWED_DTYPES
from .client_config import ClientConfig, normalize_control_endpoint, validate_control_security
from .control.api_client import ControlClient
from .exceptions import (
    ConfigurationError,
    ProtocolError,
    RemoteLoaderError,
    StatisticsCancelledError,
    StatisticsFailedError,
    StatisticsTimeoutError,
)
from .scheduler import SchedulerClient
from .statistics_spec import StatisticsSpec


def _dataset_indices(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(index) is not int or index < 0 for index in value)
        or value != sorted(set(value))
    ):
        raise ProtocolError(f"{label} must contain ordered unique nonnegative dataset indices")
    return value


def _validate_tensor_values(value: Any, shape: list[int], dtype: str, axis: int = 0) -> None:
    if axis < len(shape):
        if not isinstance(value, list) or len(value) != shape[axis]:
            raise ProtocolError("statistics values do not match their declared shape")
        for item in value:
            _validate_tensor_values(item, shape, dtype, axis + 1)
        return
    if dtype.startswith(("int", "uint")):
        bits = int(dtype.removeprefix("u").removeprefix("int"))
        lower = 0 if dtype.startswith("uint") else -(1 << (bits - 1))
        upper = (1 << bits) - 1 if lower == 0 else (1 << (bits - 1)) - 1
        if type(value) is not int or not lower <= value <= upper:
            raise ProtocolError("statistics integer values are invalid for their declared dtype")
    elif type(value) not in (int, float):
        raise ProtocolError("statistics values must be finite numbers, not booleans")
    else:
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ProtocolError("statistics values must be finite numbers")


def _validate_metric(value: Any, expected: tuple[list[int], str] | None) -> None:
    if not isinstance(value, dict) or not {"shape", "dtype", "values"} <= value.keys():
        raise ProtocolError("statistics metric requires shape, dtype and values")
    shape, dtype = value["shape"], value["dtype"]
    if (
        not isinstance(shape, list)
        or len(shape) > 15
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ProtocolError("statistics metric shape must contain positive integer dimensions")
    if not isinstance(dtype, str) or dtype not in ALLOWED_DTYPES or dtype == "bool":
        raise ProtocolError("statistics metric dtype must be a supported numeric dtype")
    if expected is not None and (shape, dtype) != expected:
        raise ProtocolError("statistics metric shape or dtype differs from the submitted spec")
    _validate_tensor_values(value["values"], shape, dtype)


def _expected_metrics(spec: StatisticsSpec) -> dict[str, dict[str, tuple[list[int], str]]]:
    fields = {tuple(field.target): field for field in spec.view.fields}
    expected = {}
    for name, query in spec.statistics.items():
        tensor = fields[tuple(query.field)].tensor
        assert tensor is not None
        retained = [size for axis, size in enumerate(tensor.shape) if axis not in query.reduce_axes]
        metrics = {}
        for metric_name, metric in query.metrics.items():
            shape = retained
            if metric.post_reduce is not None:
                shape = [size for axis, size in enumerate(shape) if axis not in metric.post_reduce.axes]
            if metric.op == "count":
                dtype = "int64"
            elif metric.op in {"mean", "std"} or (metric.op == "quantile" and metric.method == "linear"):
                dtype = "float64"
            else:
                dtype = tensor.dtype
            metrics[metric_name] = (shape, dtype)
        expected[name] = metrics
    return expected


def _validate_result(value: dict[str, Any], spec: StatisticsSpec | None) -> None:
    result_format = value.get("statistics_format")
    if (
        result_format != {"type": "tensor_statistics_result", "version": 1}
        or type(result_format.get("version")) is not int
    ):
        raise ProtocolError("unsupported statistics result format")
    if value.get("algorithm") != "tensor-statistics-v1":
        raise ProtocolError("unsupported statistics algorithm")
    descriptors = value.get("datasets")
    if not isinstance(descriptors, list) or not descriptors:
        raise ProtocolError("statistics result requires dataset descriptors")
    indices = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            raise ProtocolError("statistics dataset descriptor must be an object")
        index, key, version = (
            descriptor.get("dataset_index"),
            descriptor.get("dataset_key"),
            descriptor.get("version"),
        )
        if not isinstance(key, str) or not key:
            raise ProtocolError("statistics dataset descriptor requires an opaque key")
        if not ((type(version) is int and version > 0) or (isinstance(version, str) and bool(version))):
            raise ProtocolError("statistics manifest version must be a positive integer or nonempty string")
        indices.append(index)
    _dataset_indices(indices, "statistics datasets")
    if spec is not None and indices != spec.scope.dataset_indices:
        raise ProtocolError("statistics datasets differ from the submitted scope")
    coverage = value.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("complete") is not True:
        raise ProtocolError("statistics result does not have complete coverage")
    rows = coverage.get("datasets")
    if not isinstance(rows, list) or len(rows) != len(indices):
        raise ProtocolError("statistics coverage requires one row for every selected dataset")
    covered = []
    for row in rows:
        if not isinstance(row, dict):
            raise ProtocolError("statistics coverage row must be an object")
        covered.append(row.get("dataset_index"))
        eligible, scanned, selected = (
            row.get("eligible_anchors"),
            row.get("scanned_anchors"),
            row.get("selected_anchors"),
        )
        if any(type(count) is not int or count < 0 for count in (eligible, scanned, selected)):
            raise ProtocolError("statistics coverage counts must be nonnegative integers")
        if scanned != eligible or selected > eligible or (eligible > 0 and selected == 0):
            raise ProtocolError("statistics coverage counts do not describe a complete scan")
        if spec is not None and selected != (eligible + spec.scan.stride - 1) // spec.scan.stride:
            raise ProtocolError("statistics selected coverage differs from the submitted stride")
    if _dataset_indices(covered, "statistics coverage") != indices:
        raise ProtocolError("statistics coverage does not match dataset descriptors")
    groups = value.get("results")
    if not isinstance(groups, list) or not groups:
        raise ProtocolError("statistics result requires nonempty result groups")
    expected = _expected_metrics(spec) if spec is not None else None
    partitions = []
    for group in groups:
        if not isinstance(group, dict):
            raise ProtocolError("statistics result group must be an object")
        partitions.append(_dataset_indices(group.get("dataset_indices"), "statistics group"))
        queries = group.get("statistics")
        if not isinstance(queries, dict) or not queries or any(not name for name in queries):
            raise ProtocolError("statistics result requires nonempty named queries")
        if expected is not None and queries.keys() != expected.keys():
            raise ProtocolError("statistics query names differ from the submitted spec")
        for name, metrics in queries.items():
            if not isinstance(metrics, dict) or not metrics or any(not key for key in metrics):
                raise ProtocolError("statistics query requires nonempty named metrics")
            if expected is not None and metrics.keys() != expected[name].keys():
                raise ProtocolError("statistics metric names differ from the submitted spec")
            for metric_name, metric in metrics.items():
                _validate_metric(metric, None if expected is None else expected[name][metric_name])
    pooled, per_dataset = [indices], [[index] for index in indices]
    if spec is None:
        if partitions not in (pooled, per_dataset):
            raise ProtocolError("statistics groups must completely partition their selected datasets")
    elif partitions != (pooled if spec.scope.grouping == "pooled" else per_dataset):
        raise ProtocolError("statistics result groups differ from the submitted grouping")


class StatisticsClient:
    """Submit and retrieve finite jobs without creating or releasing allocations.

    The endpoint is fixed for this client's lifetime. Closing the client only closes
    HTTP resources; neither close nor a wait timeout cancels a running remote job.
    """

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        *,
        client_config: ClientConfig | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        endpoint = normalize_control_endpoint(endpoint)
        validate_control_security(endpoint)
        if type(max_retries) is not int or max_retries < 0:
            raise ConfigurationError("max_retries must be a nonnegative integer")
        for timeout in (connect_timeout, read_timeout):
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ConfigurationError("HTTP timeouts must be finite positive numbers")
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ConfigurationError("token must be a nonempty string")
        config = ClientConfig.load() if client_config is None else client_config
        if not isinstance(config, ClientConfig):
            raise ConfigurationError("client_config must be a ClientConfig")
        ssl_context = config.create_tls_context()
        self._endpoint = endpoint
        self._control = ControlClient(
            endpoint,
            TokenStore().load() if token is None else token,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_retries=max_retries,
            ssl_context=ssl_context,
            transport=transport,
        )
        self._identities: dict[str, tuple[str, str]] = {}
        self._specs: dict[str, StatisticsSpec] = {}
        self._closed = False

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def catalog(self) -> dict[str, Any]:
        """Return this endpoint's ordered catalog with explicit zero-based indices."""
        self._check_open()
        value = copy.deepcopy(self._control.get_metadata())
        catalog_id = value.get("catalog_id")
        datasets = value.get("scheduler_datasets")
        if not isinstance(catalog_id, str) or not catalog_id:
            raise ProtocolError("metadata response does not provide a statistics catalog_id")
        if not isinstance(datasets, list) or any(not isinstance(item, dict) for item in datasets):
            raise ProtocolError("metadata response does not provide an ordered scheduler catalog")
        value["scheduler_datasets"] = [
            {**dataset, "dataset_index": index} for index, dataset in enumerate(datasets)
        ]
        return value

    @classmethod
    def from_config(
        cls,
        config: ClientConfig | None = None,
        *,
        local_rank: int = 0,
        token: str | None = None,
    ) -> StatisticsClient:
        """Select the first ready VLA node from the training's read-only status.

        local_rank is accepted for compatibility; discovery covers the whole run.
        """
        if type(local_rank) is not int or local_rank < 0:
            raise ConfigurationError("local_rank must be a nonnegative integer")
        config = ClientConfig.load() if config is None else config
        token = TokenStore().load() if token is None else token
        settings = config.scheduler
        if settings is None:
            return cls(config.control_endpoint, token, client_config=config)

        scheduler = SchedulerClient(settings.endpoint, token=token, verify=settings.verify_tls)
        try:
            resources = scheduler.training_status(config.training.training_id)
        finally:
            scheduler.close()
        training_id = config.training.training_id.strip()
        gpus = resources.get("gpus")
        if not isinstance(gpus, list):
            raise ProtocolError("scheduler status must contain a GPU list")
        for gpu in gpus:
            if not isinstance(gpu, dict) or not isinstance(gpu.get("training_id"), str):
                raise ProtocolError("scheduler status contains an invalid GPU record")
            if gpu["training_id"] != training_id:
                continue
            nodes = gpu.get("cpu_nodes")
            if not isinstance(nodes, list):
                raise ProtocolError("scheduler status must contain a CPU node list")
            for node in nodes:
                if not isinstance(node, dict):
                    raise ProtocolError("scheduler status contains an invalid CPU node")
                if node.get("state") != "ready" or node.get("error"):
                    continue
                address = node.get("ip")
                if not isinstance(address, str) or "%" in address:
                    raise ProtocolError("scheduler status contains an invalid node IP")
                try:
                    address = ip_address(address)
                except ValueError as exc:
                    raise ProtocolError("scheduler status contains an invalid node IP") from exc
                host = f"[{address}]" if address.version == 6 else str(address)
                endpoint = f"{settings.control_scheme}://{host}:{settings.control_port}"
                return cls(endpoint, token, client_config=config)
        raise ConfigurationError(f"training {training_id!r} has no ready VLA node for statistics")

    def _check_open(self) -> None:
        if self._closed:
            raise RemoteLoaderError("statistics client is closed")

    def _check_job_id(self, job_id: str) -> None:
        self._check_open()
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", job_id):
            raise ConfigurationError("job_id must contain only letters, digits, underscores or hyphens")

    def _bind_identity(self, value: dict[str, Any], job_id: str | None = None) -> str:
        actual_id = value.get("job_id")
        if not isinstance(actual_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", actual_id):
            raise ProtocolError("statistics response has an invalid job_id")
        if job_id is not None and actual_id != job_id:
            raise ProtocolError("statistics response belongs to a different job")
        identity = (value.get("catalog_id"), value.get("request_hash"))
        if not all(isinstance(item, str) and item for item in identity):
            raise ProtocolError("statistics response is missing catalog/spec identity")
        previous = self._identities.get(actual_id)
        if previous is not None and previous != identity:
            raise ProtocolError("statistics response changed its bound catalog/spec identity")
        self._identities[actual_id] = identity
        return actual_id

    def _job(self, value: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
        if value.get("status") not in {"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}:
            raise ProtocolError("statistics response has an invalid job status")
        if "error" not in value:
            raise ProtocolError("statistics response is missing error state")
        self._bind_identity(value, job_id)
        return value

    def submit(self, spec: StatisticsSpec, *, idempotency_key: str | None = None) -> dict[str, Any]:
        self._check_open()
        if not isinstance(spec, StatisticsSpec):
            raise ConfigurationError("submit requires a StatisticsSpec")
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not idempotency_key):
            raise ConfigurationError("idempotency_key must be a nonempty string")
        # Revalidate and detach mutable caller-owned models before any HTTP request.
        frozen = StatisticsSpec.from_dict(spec.to_request())
        catalog_id = self.catalog()["catalog_id"]
        if not isinstance(catalog_id, str) or not catalog_id:
            raise ProtocolError("metadata response does not provide a statistics catalog_id")
        job = self._control.submit_statistics(
            catalog_id, frozen.to_request(), idempotency_key=idempotency_key
        )
        if job.get("catalog_id") != catalog_id or job.get("request_hash") != frozen.request_hash:
            raise ProtocolError("submitted job does not match the requested catalog/spec identity")
        job = self._job(job)
        self._specs[job["job_id"]] = frozen
        return job

    def status(self, job_id: str) -> dict[str, Any]:
        self._check_job_id(job_id)
        return self._job(self._control.get_statistics(job_id), job_id)

    def _result(self, job_id: str, *, deadline: float | None = None) -> dict[str, Any]:
        value = self._control.get_statistics_result(job_id, deadline=deadline)
        if deadline is not None and time.monotonic() >= deadline:
            raise StatisticsTimeoutError("statistics wait timed out; remote job was not cancelled")
        _validate_result(value, self._specs.get(job_id))
        self._bind_identity(value, job_id)
        return value

    def result(self, job_id: str) -> dict[str, Any]:
        self._check_job_id(job_id)
        return self._result(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        self._check_job_id(job_id)
        return self._job(self._control.cancel_statistics(job_id), job_id)

    def wait(self, job_id: str, *, timeout: float = 300.0, poll_interval: float = 0.25) -> dict[str, Any]:
        self._check_job_id(job_id)
        for name, value in (("timeout", timeout), ("poll_interval", poll_interval)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigurationError(f"{name} must be a finite positive number")
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise StatisticsTimeoutError("statistics wait timed out; remote job was not cancelled")
            job = self._job(self._control.get_statistics(job_id, deadline=deadline), job_id)
            if time.monotonic() >= deadline:
                raise StatisticsTimeoutError("statistics wait timed out; remote job was not cancelled")
            if job["status"] == "FAILED":
                raise StatisticsFailedError(f"statistics job {job_id} failed: {job['error']}")
            if job["status"] == "CANCELLED":
                raise StatisticsCancelledError(f"statistics job {job_id} was cancelled")
            if job["status"] == "SUCCEEDED":
                return self._result(job_id, deadline=deadline)
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._control.close()

    def __enter__(self) -> StatisticsClient:
        self._check_open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
