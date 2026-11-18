from __future__ import annotations

import math
import os
import ssl
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import replace
from enum import Enum
from typing import Any

from ._batch_delivery import CanonicalImageDelivery
from .auth.token_store import TokenStore
from .batch_spec import BatchSpec, CanonicalSampleSpec, VQABatchSpec
from .checkpoint import LoaderCheckpoint
from .client_config import (
    ClientConfig,
    TrainingConfig,
    normalize_control_endpoint,
)
from .control.models import SessionInfo
from .exceptions import (
    BatchTimeoutError,
    ConfigurationError,
    EndOfStream,
    SchemaMismatchError,
    SessionStateError,
)
from .multi_node import LoaderPausedError, MultiNodeCheckpoint, MultiNodeLoader, validate_checkpoint
from .scheduler import resolve_control_endpoints
from .telemetry.metrics import LoaderMetrics
from .transfer_state import TransferSnapshot, TransferStateStore, TransferStatus, utc_now
from .transport.connection import BatchConnection

_DEFAULT_PREFETCH_BATCHES = 8
_DEFAULT_BATCH_TIMEOUT_SECONDS = 10.0
_TRANSFER_STATE_WRITE_INTERVAL_SECONDS = 1.0


class _UnsetPrefetchBytes:
    pass


_UNSET_PREFETCH_BYTES = _UnsetPrefetchBytes()


class LoaderState(str, Enum):
    NEW = "NEW"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


class _WireSampleIterator(Iterator[dict[str, Any]]):
    def __init__(self, loader: RemoteDataLoader) -> None:
        self.loader = loader

    def __next__(self) -> dict[str, Any]:
        if self.loader.state == "NEW":
            self.loader.start()
        return self.loader._next_wire_sample()


class RemoteDataLoader(Iterator[dict[str, Any]]):
    """Yield logical VLA canonical samples or complete VQA v2 training batches."""

    def __init__(
        self,
        spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec,
        *,
        control_endpoint: str | None = None,
        training_id: str | None = None,
        scheduler_endpoint: str | None = None,
        node_control_port: int | None = None,
        token: str | None = None,
        token_store: TokenStore | None = None,
        prefetch_batches: int | None = None,
        prefetch_bytes: int | _UnsetPrefetchBytes | None = _UNSET_PREFETCH_BYTES,
        batch_timeout: float | None = None,
        session_timeout: float = 300.0,
        poll_interval: float = 0.5,
        rank: int | None = None,
        world_size: int | None = None,
        local_rank: int | None = None,
        max_reconnects: int = 8,
        _control_client: Any | None = None,
        _connection_factory: Callable[[SessionInfo, int], BatchConnection] | None = None,
        _transfer_state_store: TransferStateStore | None = None,
        _track_transfer_state: bool = True,
        _control_endpoints: tuple[str, ...] | None = None,
        _client_config: ClientConfig | None = None,
        _ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.spec = spec
        # A loader session and every checkpoint derived from it must describe the
        # same request. Keep a private execution snapshot while retaining the
        # caller-owned public spec for inspection and backward compatibility.
        self._session_spec = spec.model_copy(deep=True)
        self._delivery = CanonicalImageDelivery.from_spec(self._session_spec)
        self._batch_spec_hash = self._session_spec.batch_spec_hash
        explicit_scheduler = any(
            value is not None for value in (training_id, scheduler_endpoint, node_control_port)
        )
        if control_endpoint is not None and explicit_scheduler:
            raise ConfigurationError("explicit control_endpoint cannot be combined with scheduler settings")
        injected_network = _control_client is not None and _connection_factory is not None
        client_config = _client_config
        if client_config is None and not injected_network:
            client_config = ClientConfig.load()
        if client_config is None:
            client_config = ClientConfig.default()
        ssl_context = _ssl_context or (
            ssl.create_default_context() if injected_network else client_config.create_tls_context()
        )
        scheduled_config = client_config if control_endpoint is None else None
        if explicit_scheduler:
            assert scheduled_config is not None
            settings = scheduled_config.resolve_scheduler(scheduler_endpoint)
            training = scheduled_config.training
            if training_id is not None:
                selected = TrainingConfig(training_id=training_id)
                if selected.training_id != training.training_id:
                    # Another run must not inherit this run's recorded allocation.
                    training = selected
            settings = replace(
                settings,
                control_port=settings.control_port if node_control_port is None else node_control_port,
            )
            scheduled_config = replace(scheduled_config, scheduler=settings, training=training)
        if scheduled_config is not None and scheduled_config.scheduler is None:
            scheduled_config = None
        if _control_endpoints is not None and scheduled_config is None:
            raise ConfigurationError("a frozen node assignment requires scheduler mode")
        if prefetch_batches is None:
            prefetch_batches = (
                _DEFAULT_PREFETCH_BATCHES if client_config is None else client_config.runtime.prefetch_batches
            )
        if isinstance(prefetch_bytes, _UnsetPrefetchBytes):
            prefetch_bytes = None
        self._prefetch_bytes_is_auto = prefetch_bytes is None
        if batch_timeout is None:
            batch_timeout = _DEFAULT_BATCH_TIMEOUT_SECONDS
        if type(prefetch_batches) is not int or not 1 <= prefetch_batches <= 1024:
            raise ValueError("prefetch_batches must be an integer between 1 and 1024")
        if prefetch_bytes is not None and (type(prefetch_bytes) is not int or prefetch_bytes <= 0):
            raise ValueError("prefetch_bytes must be a positive integer or None")
        if not math.isfinite(batch_timeout) or batch_timeout <= 0:
            raise ValueError("batch_timeout must be a finite positive number")
        if control_endpoint is None:
            assert client_config is not None
            control_endpoint = client_config.control_endpoint
        control_endpoint = normalize_control_endpoint(control_endpoint)
        if scheduled_config is None and not control_endpoint.startswith("https://"):
            raise ConfigurationError("control_endpoint must use HTTPS")
        self._control_endpoint = control_endpoint
        self._token = token
        self._token_store = token_store or TokenStore()
        self.prefetch_batches = prefetch_batches
        self.prefetch_bytes = (
            (self._session_spec.estimated_batch_bytes + self._session_spec.estimated_metadata_bytes)
            * prefetch_batches
            if prefetch_bytes is None
            else prefetch_bytes
        )
        self.batch_timeout = batch_timeout
        self.session_timeout = session_timeout
        self.poll_interval = poll_interval
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else local_rank
        if type(self.local_rank) is not int or self.local_rank < 0:
            raise ValueError("local_rank must be a non-negative integer")
        configured_world_size = None
        if scheduled_config is not None:
            training = scheduled_config.training
            if training.num_nodes is not None and training.gpus_per_node is not None:
                configured_world_size = training.num_nodes * training.gpus_per_node
        if world_size is None:
            environment_world_size = os.environ.get("WORLD_SIZE")
            self.world_size = (
                int(environment_world_size)
                if environment_world_size is not None
                else (configured_world_size or 1)
            )
        else:
            self.world_size = world_size
        if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
            raise ValueError("rank/world_size must satisfy 0 <= rank < world_size")
        if configured_world_size is not None and self.world_size != configured_world_size:
            raise ConfigurationError(
                f"world_size={self.world_size} does not match "
                f"training.num_nodes * training.gpus_per_node={configured_world_size}"
            )
        self.max_reconnects = max_reconnects
        self.ssl_context = ssl_context
        self._scheduled_config = scheduled_config
        self._control_endpoints = None if _control_endpoints is None else tuple(_control_endpoints)
        self._runtime: MultiNodeLoader | None = None
        self._checkpoint: LoaderCheckpoint | MultiNodeCheckpoint | None = None
        self._control: Any | None = _control_client
        self._custom_connection_factory = _connection_factory
        self._lifecycle_lock = threading.RLock()
        self._close_requested = False
        self._state = LoaderState.NEW
        # Filesystem writes serialize independently of bootstrap/runtime state.
        self._transfer_lock = threading.Lock()
        self._metrics = LoaderMetrics()
        self._transfer_state_store = (
            (_transfer_state_store or TransferStateStore()) if _track_transfer_state else None
        )
        self._transfer_snapshot: TransferSnapshot | None = None
        self._transfer_state_finished = False
        self._last_transfer_state_write = 0.0

    @classmethod
    def from_yaml(cls, path: str, **kwargs: Any) -> RemoteDataLoader:
        return cls(BatchSpec.from_yaml(path), **kwargs)

    @property
    def batch_spec_hash(self) -> str:
        """Return the immutable BatchSpec identity bound to this loader session."""
        return self._batch_spec_hash

    @property
    def state(self) -> str:
        runtime = self._runtime
        if runtime is not None:
            state = runtime.state
            if state != "NEW" or self._state is not LoaderState.STARTING:
                return state
        return self._state.value

    @property
    def session_id(self) -> str | None:
        return None if self._runtime is None else self._runtime.session_id

    @property
    def dataset_metadata(self) -> dict[str, dict[str, Any]]:
        """Return complete metadata per control endpoint as a detached deep snapshot.

        Direct and scheduled loaders share this shape; before start it is empty.
        """
        with self._lifecycle_lock:
            runtime = self._runtime
        return {} if runtime is None else runtime.dataset_metadata

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._close_requested:
                raise SessionStateError("a closed loader cannot be restarted")
            if self.state in {"RUNNING", "PAUSED"}:
                return
            if self._state is LoaderState.STARTING:
                raise SessionStateError("loader is already starting")
            if self._runtime is not None:
                raise SessionStateError(f"loader cannot be restarted (state={self.state})")
            self._state = LoaderState.STARTING
        try:
            self._begin_transfer_state()
            self._start_runtime()
            with self._lifecycle_lock:
                if self._close_requested:
                    raise SessionStateError("loader closed while starting")
                self._state = LoaderState.RUNNING
            self._update_transfer_state(force=True)
        except BaseException:
            if not self._close_requested:
                self._finish_transfer_state("FAILED")
            with suppress(Exception):
                self.close()
            raise

    def _scheduler_assignment_id(self) -> str:
        assert self._scheduled_config is not None and self._scheduled_config.scheduler is not None
        training_id = self._scheduled_config.training.training_id
        if training_id is None:
            raise ConfigurationError(
                "scheduler requires a preassigned training_id in the client configuration"
            )
        # Preserve the existing checkpoint's opaque GPU ID and global process identity.
        return f"{training_id}:{self.rank}"

    def _start_runtime(self) -> None:
        scheduled = self._scheduled_config
        token = self._token
        endpoints = (self._control_endpoint,)
        gpu_id = None
        if scheduled is not None:
            gpu_id = self._scheduler_assignment_id()
            if self._control_endpoints is None:
                token = token or self._token_store.load()
            endpoints = (
                resolve_control_endpoints(
                    scheduled,
                    local_rank=self.local_rank,
                    token=token,
                    **({"vqa": True} if self._session_spec.is_vqa else {}),
                )
                if self._control_endpoints is None
                else self._control_endpoints
            )
        with self._lifecycle_lock:
            if self._close_requested:
                raise SessionStateError("loader closed while starting")
            # Bootstrap owns this client until construction succeeds. Neither
            # discovery nor client construction may block concurrent close().
            control, self._control = self._control, None
        runtime = None
        try:
            if control is None:
                token = token or self._token_store.load()
            runtime = MultiNodeLoader(
                self._session_spec,
                endpoints,
                token,
                gpu_id=gpu_id,
                prefetch_batches=self.prefetch_batches,
                prefetch_bytes=self.prefetch_bytes,
                prefetch_bytes_is_auto=self._prefetch_bytes_is_auto,
                batch_timeout=self.batch_timeout,
                session_timeout=self.session_timeout,
                poll_interval=self.poll_interval,
                rank=self.rank,
                world_size=self.world_size,
                max_reconnects=self.max_reconnects,
                ssl_context=self.ssl_context,
                metrics=self._metrics,
                checkpoint=self._checkpoint,
                control_client=control,
                connection_factory=self._custom_connection_factory,
            )
            control = None
            with self._lifecycle_lock:
                if self._close_requested:
                    raise SessionStateError("loader closed while starting")
                self._runtime = runtime
            runtime.start()
            if self._prefetch_bytes_is_auto:
                self.prefetch_bytes = runtime.byte_capacity
        finally:
            if runtime is not None and self._runtime is not runtime:
                runtime.close()
            if control is not None:
                self._close_control(control)

    def __iter__(self) -> RemoteDataLoader:
        if self.state == "NEW":
            self.start()
        return self

    def __next__(self) -> dict[str, Any]:
        return self._delivery(self._next_wire_sample())

    def _iter_wire_samples(self) -> Iterator[dict[str, Any]]:
        """Provide lazy raw consumption for the parent worker IPC broker."""
        return _WireSampleIterator(self)

    def _next_wire_sample(self) -> dict[str, Any]:
        runtime = self._runtime
        if runtime is None:
            if self.state == "CLOSED":
                raise StopIteration
            raise SessionStateError(f"loader is not running (state={self.state})")
        try:
            item = runtime.next_item()
        except EndOfStream as exc:
            self._finish_transfer_state("COMPLETED")
            self.close()
            raise StopIteration from exc
        except (BatchTimeoutError, LoaderPausedError):
            raise
        except Exception:
            if runtime.state == "FAILED":
                self._finish_transfer_state("FAILED")
                with suppress(Exception):
                    self.close()
            raise
        self._update_transfer_state()
        return item

    def pause(self) -> None:
        runtime = self._runtime
        if runtime is None or runtime.state != "RUNNING":
            raise SessionStateError("only a running loader can be paused")
        try:
            runtime.pause()
        except Exception:
            if runtime.state == "FAILED":
                self._finish_transfer_state("FAILED")
                with suppress(Exception):
                    self.close()
            raise
        self._update_transfer_state(force=True)

    def resume(self) -> None:
        if self.state == "NEW" and self._checkpoint is not None:
            self.start()
            return
        runtime = self._runtime
        if runtime is None or runtime.state != "PAUSED":
            raise SessionStateError("only a paused loader can be resumed")
        try:
            runtime.resume()
        except Exception:
            if runtime.state == "FAILED":
                self._finish_transfer_state("FAILED")
                with suppress(Exception):
                    self.close()
            raise
        self._update_transfer_state(force=True)

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._close_requested:
                return
            self._close_requested = True
            self._state = LoaderState.CLOSED
            runtime = self._runtime
            control, self._control = self._control, None
        try:
            if runtime is not None:
                runtime.close()
            if control is not None:
                self._close_control(control)
        finally:
            self._finish_transfer_state("STOPPED")

    @staticmethod
    def _close_control(control: Any) -> None:
        if hasattr(control, "close"):
            control.close()

    def state_dict(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self._runtime is not None:
                return self._runtime.state_dict()
            if isinstance(self._checkpoint, MultiNodeCheckpoint):
                return self._checkpoint.to_state_dict()
            if self._checkpoint is not None:
                return self._checkpoint.model_copy(update={"session_id": None}).to_state_dict()
            if self._scheduled_config is not None:
                raise SessionStateError("start the scheduled loader before saving its node assignment")
            return LoaderCheckpoint(
                session_id=None,
                last_consumed_sequence=-1,
                last_batch_id=None,
                schema_hash=self._session_spec.schema_hash,
                batch_spec_hash=self._batch_spec_hash,
                rank=self.rank,
                world_size=self.world_size,
            ).to_state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with self._lifecycle_lock:
            if self._state is not LoaderState.NEW:
                raise SessionStateError("loader state must be restored before start()")
            if self._scheduled_config is not None:
                assert self._scheduled_config.scheduler is not None
                self._checkpoint = validate_checkpoint(
                    state,
                    self._session_spec,
                    gpu_id=self._scheduler_assignment_id(),
                    rank=self.rank,
                    world_size=self.world_size,
                )
                return
            checkpoint = LoaderCheckpoint.from_state_dict(state)
            if checkpoint.schema_hash != self._session_spec.schema_hash:
                raise SchemaMismatchError("checkpoint schema_hash does not match the BatchSpec")
            if checkpoint.batch_spec_hash != self._batch_spec_hash:
                raise SchemaMismatchError("checkpoint BatchSpec hash does not match the current BatchSpec")
            if (checkpoint.rank, checkpoint.world_size) != (self.rank, self.world_size):
                raise SchemaMismatchError("checkpoint rank/world_size does not match the current process")
            self._checkpoint = checkpoint

    def metrics(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime.metrics()
        return self._metrics.snapshot(
            queue_size=0,
            queue_capacity=self.prefetch_batches,
            queued_bytes=0,
            state=self._state.value,
        )

    def _begin_transfer_state(self) -> None:
        if self._transfer_state_store is None:
            return
        with self._transfer_lock:
            if self._close_requested:
                return
            try:
                self._transfer_snapshot = self._transfer_state_store.begin(self._session_spec)
                self._last_transfer_state_write = time.monotonic()
            except Exception:
                self._transfer_snapshot = None

    def _update_transfer_state(self, *, force: bool = False) -> None:
        if self._transfer_state_store is None or self._transfer_state_finished:
            return
        now = time.monotonic()
        if not force and now - self._last_transfer_state_write < _TRANSFER_STATE_WRITE_INTERVAL_SECONDS:
            return
        with self._transfer_lock:
            if self._transfer_snapshot is None or self._transfer_state_finished or self._close_requested:
                return
            now = time.monotonic()
            if not force and now - self._last_transfer_state_write < _TRANSFER_STATE_WRITE_INTERVAL_SECONDS:
                return
            runtime = self._runtime
            metrics, last_batch_at = (
                (self.metrics(), None) if runtime is None else runtime.transfer_snapshot()
            )
            status = metrics["state"]
            if status not in {"STARTING", "RUNNING", "PAUSED"}:
                return
            snapshot = self._transfer_snapshot.model_copy(
                update={
                    "status": status,
                    "session_id": self.session_id,
                    "updated_at": utc_now(),
                    "last_batch_at": last_batch_at,
                    "batches_received": metrics["batches_received"],
                    "batches_consumed": metrics["batches_consumed"],
                    "bytes_received": metrics["bytes_received"],
                }
            )
            self._transfer_snapshot = snapshot
            try:
                self._transfer_state_store.publish(snapshot)
            except Exception:
                pass
            finally:
                self._last_transfer_state_write = now

    def _finish_transfer_state(self, status: TransferStatus) -> None:
        if self._transfer_state_store is None:
            return
        with self._transfer_lock:
            if self._transfer_snapshot is None or self._transfer_state_finished:
                return
            if status == "COMPLETED" and self._close_requested:
                status = "STOPPED"
            runtime = self._runtime
            metrics, last_batch_at = (
                (self.metrics(), None) if runtime is None else runtime.transfer_snapshot()
            )
            now = utc_now()
            snapshot = self._transfer_snapshot.model_copy(
                update={
                    "status": status,
                    "session_id": self.session_id,
                    "updated_at": now,
                    "finished_at": now,
                    "last_batch_at": last_batch_at,
                    "batches_received": metrics["batches_received"],
                    "batches_consumed": metrics["batches_consumed"],
                    "bytes_received": metrics["bytes_received"],
                }
            )
            self._transfer_snapshot = snapshot
            self._transfer_state_finished = True
            with suppress(Exception):
                self._transfer_state_store.finish(snapshot)

    @staticmethod
    def to_device(value: Any, device: Any, *, non_blocking: bool = True) -> Any:
        from .pytorch.device_transfer import move_to_device

        return move_to_device(value, device, non_blocking=non_blocking)

    def to_torch_dataloader(self, **kwargs: Any) -> Any:
        from .pytorch.iterable_dataset import make_torch_dataloader

        return make_torch_dataloader(self, **kwargs)

    def __enter__(self) -> RemoteDataLoader:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
