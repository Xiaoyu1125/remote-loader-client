"""Concurrent node streams with whole-frame delivery, replay, and shared credit."""

from __future__ import annotations

import ssl
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ._batch_delivery import prepare_vqa_samples
from ._image_transport import compile_wire_spec, needs_camera_metadata
from .batch_spec import BatchSpec, CanonicalSampleSpec, VQABatchSpec
from .checkpoint import LoaderCheckpoint
from .client_config import normalize_control_endpoint
from .control.api_client import ControlClient
from .control.models import SessionInfo, SessionStatus
from .exceptions import (
    BatchTimeoutError,
    ChecksumError,
    EndOfStream,
    ProtocolError,
    SchemaMismatchError,
    ServiceUnavailableError,
    SessionExpiredError,
    SessionStateError,
)
from .telemetry.metrics import LoaderMetrics
from .transport.connection import BatchConnection, CreditUpdate, TcpBatchConnection
from .transport.protocol import BatchEnvelope, ProtocolLimits

_WARMUP_TIMEOUT_SECONDS = 300.0


class LoaderPausedError(SessionStateError):
    """A waiting consumer was interrupted by pause, not a stream failure."""


class MultiNodeCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["multi_node"] = "multi_node"
    format_version: Literal[1] = 1
    schema_hash: str
    batch_spec_hash: str
    gpu_id: str
    rank: int = Field(ge=0)
    world_size: int = Field(ge=1)
    consumed_batches: int = Field(default=0, ge=0)
    nodes: dict[str, LoaderCheckpoint]

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def validate_checkpoint(
    state: dict[str, Any],
    spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec,
    *,
    gpu_id: str,
    rank: int,
    world_size: int,
) -> MultiNodeCheckpoint:
    checkpoint = MultiNodeCheckpoint.model_validate(state)
    expected = (
        spec.schema_hash,
        spec.batch_spec_hash,
        rank,
        world_size,
    )
    actual = (
        checkpoint.schema_hash,
        checkpoint.batch_spec_hash,
        checkpoint.rank,
        checkpoint.world_size,
    )
    if actual != expected or checkpoint.gpu_id != gpu_id:
        raise SchemaMismatchError("checkpoint BatchSpec, GPU ID or rank/world_size does not match this run")
    if not checkpoint.nodes:
        raise SchemaMismatchError("multi-node checkpoint has no node assignment")
    for endpoint, child in checkpoint.nodes.items():
        if normalize_control_endpoint(endpoint) != endpoint:
            raise SchemaMismatchError("checkpoint node endpoint is not canonical")
        if (
            child.schema_hash,
            child.batch_spec_hash,
            child.rank,
            child.world_size,
        ) != expected:
            raise SchemaMismatchError("checkpoint node BatchSpec or training topology does not match")
    return checkpoint


@dataclass
class _Node:
    endpoint: str
    control: Any
    session: SessionInfo | None = None
    connection: BatchConnection | None = None
    connecting: BatchConnection | None = None
    starting: bool = False
    control_busy: int = 0
    control_closed: bool = False
    checkpoint: LoaderCheckpoint | None = None
    consumed: int = -1
    highest_enqueued: int = -1
    last_batch_id: str | None = None
    # All three kinds of reservation (permission, claimed credit, queued/pending
    # envelopes) count in owned, and hence in the global reservation count.
    permission: int = 0
    grants: int = 0
    owned: int = 0
    credit_limit: int = 0
    warmup_deadline: float | None = None
    data_started: bool = False
    done: bool = False
    seen: set[str] = field(default_factory=set)
    seen_order: deque[str] = field(default_factory=deque)
    received_batches: int = 0
    consumed_batches: int = 0
    received_bytes: int = 0
    reconnects: int = 0


class MultiNodeLoader:
    """One condition-coordinated runtime for direct and scheduled batch streams."""

    def __init__(
        self,
        spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec,
        endpoints: tuple[str, ...],
        token: str | None,
        *,
        gpu_id: str | None,
        prefetch_batches: int,
        prefetch_bytes: int,
        batch_timeout: float,
        session_timeout: float,
        poll_interval: float,
        rank: int,
        world_size: int,
        max_reconnects: int,
        metrics: LoaderMetrics,
        prefetch_bytes_is_auto: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        checkpoint: LoaderCheckpoint | MultiNodeCheckpoint | None = None,
        control_client: Any | None = None,
        connection_factory: Callable[[SessionInfo, int], BatchConnection] | None = None,
    ) -> None:
        normalized = tuple(normalize_control_endpoint(endpoint) for endpoint in endpoints)
        if not normalized or len(set(normalized)) != len(normalized):
            raise SessionStateError("node assignment must contain distinct control endpoints")
        if any(not endpoint.startswith("https://") for endpoint in normalized):
            raise SessionStateError("all loader control endpoints must use HTTPS")
        if gpu_id is None and len(normalized) != 1:
            raise SessionStateError("direct loading requires exactly one control endpoint")
        if (control_client is not None or connection_factory is not None) and len(normalized) != 1:
            raise SessionStateError("dependency injection requires exactly one control endpoint")
        if isinstance(checkpoint, MultiNodeCheckpoint):
            if gpu_id is None or set(checkpoint.nodes) != set(normalized):
                raise SchemaMismatchError("checkpoint node assignment does not match scheduler nodes")
        elif checkpoint is not None and gpu_id is not None:
            raise SchemaMismatchError("scheduled loading requires a multi-node checkpoint")
        self.spec = spec
        self._schema_hash = spec.schema_hash
        self._batch_spec_hash = spec.batch_spec_hash
        self._wire_spec = spec
        self._wire_schema_hash = self._schema_hash
        self._wire_batch_spec_hash = self._batch_spec_hash
        self._prefetch_bytes_is_auto = prefetch_bytes_is_auto
        self.gpu_id = gpu_id
        self.capacity = prefetch_batches
        self.byte_capacity = prefetch_bytes
        self.batch_timeout = batch_timeout
        self.session_timeout = session_timeout
        self.poll_interval = poll_interval
        self.rank = rank
        self.world_size = world_size
        self.max_reconnects = max_reconnects
        self.ssl_context = ssl_context
        self._metrics = metrics
        self._connection_factory = connection_factory
        self._condition = threading.Condition()
        self._control_operations = threading.Lock()
        self._delivery_active = False
        self._first_batch_at: float | None = None
        self._last_batch_at: datetime | None = None
        self._stop = threading.Event()
        self._paused = False
        self._generation = 0
        self._closed = False
        self._state = "NEW"
        self._error: BaseException | None = None
        self._ready: deque[tuple[_Node, BatchEnvelope]] = deque()
        self._pending: tuple[_Node, BatchEnvelope] | None = None
        self._queued_bytes = 0
        self._reserved = 0
        self._unfinished = len(normalized)
        self._threads: list[threading.Thread] = []
        self._next_credit_node = 0
        self._consumed_batches = (
            checkpoint.consumed_batches if isinstance(checkpoint, MultiNodeCheckpoint) else 0
        )
        self._nodes: list[_Node] = []
        try:
            for endpoint in normalized:
                control = (
                    control_client
                    if control_client is not None
                    else ControlClient(endpoint, token, ssl_context=ssl_context)
                )
                node = _Node(endpoint, control)
                self._nodes.append(node)
                node.checkpoint = (
                    checkpoint.nodes[endpoint] if isinstance(checkpoint, MultiNodeCheckpoint) else checkpoint
                )
                if node.checkpoint is not None:
                    node.consumed = node.checkpoint.last_consumed_sequence
                    node.highest_enqueued = node.consumed
                    node.last_batch_id = node.checkpoint.last_batch_id
        except BaseException:
            for node in self._nodes:
                if node.control is not control_client:
                    with suppress(Exception):
                        node.control.close()
            raise

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def session_id(self) -> str | None:
        with self._condition:
            session = self._nodes[0].session
            return session.session_id if self.gpu_id is None and session is not None else None

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise SessionStateError("a closed loader cannot be restarted")
            if self._state in {"RUNNING", "PAUSED"}:
                return
            if self._state != "NEW":
                raise SessionStateError(f"loader cannot start in state {self._state}")
            self._state = "STARTING"
        try:
            with ThreadPoolExecutor(
                max_workers=len(self._nodes), thread_name_prefix="remote-loader-start"
            ) as pool:
                futures = []
                try:
                    if needs_camera_metadata(self.spec):
                        futures = [pool.submit(self._node_wire_spec, node) for node in self._nodes]
                        wire_specs = [future.result() for future in futures]
                        wire = wire_specs[0]
                        if any(candidate.batch_spec_hash != wire.batch_spec_hash for candidate in wire_specs[1:]):
                            raise SchemaMismatchError("nodes disagree on the derived image wire contract")
                        self._wire_spec = wire
                        self._wire_schema_hash = wire.schema_hash
                        self._wire_batch_spec_hash = wire.batch_spec_hash
                    if self._prefetch_bytes_is_auto:
                        self.byte_capacity = (
                            self._wire_spec.estimated_batch_bytes + self._wire_spec.estimated_metadata_bytes
                        ) * self.capacity
                    futures = [pool.submit(self._start_node, node) for node in self._nodes]
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    self._stop.set()
                    for future in futures:
                        future.cancel()
                    raise
            with self._condition:
                if self._stop.is_set():
                    raise SessionStateError("loader closed while starting node sessions")
                self._state = "RUNNING"
                self._metrics.set_running(True)
                self._threads = [
                    threading.Thread(
                        target=self._receive, args=(node,), name=f"remote-loader-{index}", daemon=True
                    )
                    for index, node in enumerate(self._nodes)
                ]
                for thread in self._threads:
                    thread.start()
                wake = self._allocate_credits()
            self._notify_credit(wake)
        except BaseException:
            self.close()
            raise

    def _node_wire_spec(self, node: _Node) -> BatchSpec | CanonicalSampleSpec | VQABatchSpec:
        with self._condition:
            if self._stop.is_set():
                raise SessionStateError("loader closed while discovering camera metadata")
            node.control_busy += 1
        try:
            return compile_wire_spec(self.spec, node.control.get_metadata())
        finally:
            with self._condition:
                node.control_busy -= 1
                closed = self._closed
            if closed:
                self._close_node(node)

    def _start_node(self, node: _Node) -> None:
        with self._condition:
            if self._stop.is_set():
                return
            node.starting = True
        try:
            session = node.control.create_session(
                self._wire_spec,
                prefetch_batches=self.capacity,
                rank=self.rank,
                world_size=self.world_size,
                checkpoint=(
                    node.checkpoint.model_copy(update={
                        "schema_hash": self._wire_schema_hash,
                        "batch_spec_hash": self._wire_batch_spec_hash,
                    }) if node.checkpoint is not None else None
                ),
            )
            with self._condition:
                node.session = session
            deadline = time.monotonic() + self.session_timeout
            while session.status not in {SessionStatus.READY, SessionStatus.RUNNING}:
                self._check_session(session)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"node {node.endpoint} did not become READY before session_timeout")
                if self._stop.wait(self.poll_interval):
                    return
                refreshed = node.control.get_session(session.session_id)
                if refreshed.session_id != session.session_id:
                    raise ProtocolError("node changed session identity while starting")
                session = refreshed
                with self._condition:
                    node.session = session
            self._check_session(session)
            if not session.data_endpoint or not session.stream_token:
                raise SessionStateError(f"node {node.endpoint} is missing data connection details")
            if self._stop.is_set():
                return
            limits = session.frame_limits
            frame_bytes = limits.get("max_frame_bytes") if isinstance(limits, dict) else None
            if type(frame_bytes) is not int or frame_bytes <= 0:
                raise ProtocolError(f"node {node.endpoint} is missing a positive max_frame_bytes limit")
            metrics = node.control.get_metrics(session.session_id)
            replay_bytes = metrics.get("replay_max_bytes") if isinstance(metrics, dict) else None
            if type(replay_bytes) is not int or replay_bytes < frame_bytes:
                raise ProtocolError(f"node {node.endpoint} has no valid replay window for one frame")
            with self._condition:
                node.credit_limit = min(self.capacity, replay_bytes // frame_bytes)

        finally:
            with self._condition:
                node.starting = False
                closed = self._closed
            if closed:
                self._close_node(node)

    def _check_session(self, session: SessionInfo) -> None:
        if session.status is SessionStatus.EXPIRED:
            raise SessionExpiredError("remote loader node session expired")
        if session.status in {SessionStatus.FAILED, SessionStatus.CLOSED}:
            raise SessionStateError(session.error or f"node session entered {session.status.value}")
        if session.schema_hash != self._wire_schema_hash:
            raise SchemaMismatchError("node session schema_hash does not match the requested BatchSpec")

    def _connect(self, node: _Node, generation: int) -> BatchConnection | None:
        with self._condition:
            if self._stop.is_set() or self._paused or generation != self._generation:
                return None
            session = node.session
            node.control_busy += 1
        assert session is not None
        try:
            refreshed = (
                session
                if self._connection_factory is not None
                else node.control.get_session(session.session_id)
            )
            self._check_session(refreshed)
            if refreshed.session_id != session.session_id:
                raise ProtocolError("node changed session identity while reconnecting")
            with self._condition:
                consumed = node.consumed
            connection = (
                self._connection_factory(refreshed, consumed)
                if self._connection_factory is not None
                else TcpBatchConnection(
                    refreshed,
                    schema_hash=self._wire_schema_hash,
                    expected_output=self._wire_spec.output,
                    last_consumed_sequence=consumed,
                    limits=ProtocolLimits(),
                    ssl_context=self.ssl_context,
                )
            )
        finally:
            with self._condition:
                node.control_busy -= 1
                closed = self._closed
            if closed:
                self._close_node(node)
        with self._condition:
            stale = self._stop.is_set() or self._paused or generation != self._generation
            if not stale:
                node.connecting = connection
        if stale:
            with suppress(Exception):
                connection.close()
            return None
        try:
            connection.connect()
        except BaseException:
            with suppress(Exception):
                connection.close()
            raise
        finally:
            with self._condition:
                if node.connecting is connection:
                    node.connecting = None
        wake = None
        with self._condition:
            stale = self._stop.is_set() or self._paused or generation != self._generation
            if not stale:
                node.session = refreshed
                node.connection = connection
                wake = self._allocate_credits()
                self._condition.notify_all()
        self._notify_credit(wake)
        if stale:
            with suppress(Exception):
                connection.close()
            return None
        return connection

    def _drop(self, node: _Node, expected: BatchConnection | None = None) -> None:
        wake = None
        with self._condition:
            connecting = None
            if expected is not None and node.connection is not expected:
                connection = expected
            else:
                connection, node.connection = node.connection, None
                connecting, node.connecting = node.connecting, None
                released = node.grants + node.permission
                node.owned -= released
                self._reserved -= released
                node.grants = node.permission = 0
                wake = self._allocate_credits()
            self._condition.notify_all()
        try:
            self._notify_credit(wake)
        finally:
            for candidate in (connection, connecting):
                if candidate is not None:
                    with suppress(Exception):
                        candidate.close()

    def _allocate_credits(self) -> list[BatchConnection] | None:
        """Reserve under the condition; return local wakes for emission outside it."""
        if self._paused or self._stop.is_set() or not self._unfinished:
            return
        available = self.capacity - self._reserved
        if available <= 0:
            return
        wake = None
        share = (self.capacity + self._unfinished - 1) // self._unfinished
        for offset in range(len(self._nodes)):
            index = (self._next_credit_node + offset) % len(self._nodes)
            node = self._nodes[index]
            if node.connection is None or node.done:
                continue
            grant = min(
                available,
                share - node.owned,
                node.credit_limit - node.owned,
            )
            if grant <= 0:
                continue
            if wake is None:
                wake = []
            wake.append(node.connection)
            node.permission += grant
            node.owned += grant
            self._reserved += grant
            available -= grant
            next_node = (index + 1) % len(self._nodes)
            if not available:
                self._next_credit_node = next_node
                break
        return wake

    @staticmethod
    def _notify_credit(connections: list[BatchConnection] | None) -> None:
        if connections is not None:
            for connection in connections:
                connection.notify_credit()

    def _claim_credit(self, node: _Node, connection: BatchConnection, generation: int) -> CreditUpdate | None:
        with self._condition:
            if (
                node.connection is not connection
                or generation != self._generation
                or self._paused
                or self._closed
                or self._error is not None
                or node.done
                or not node.permission
            ):
                return None
            grant = node.permission
            node.permission = 0
            node.grants += grant
            if not node.data_started and node.warmup_deadline is None:
                node.warmup_deadline = time.monotonic() + _WARMUP_TIMEOUT_SECONDS
            size = len(self._ready) + (self._pending is not None)
            return CreditUpdate(
                capacity=self.capacity,
                size=size,
                available=self.capacity - size,
                queued_bytes=self._queued_bytes,
                grant_batches=grant,
                last_consumed_sequence=node.consumed,
                warmup_deadline=node.warmup_deadline,
            )

    def _mark_data_received(self, node: _Node, connection: BatchConnection, generation: int) -> None:
        with self._condition:
            if node.connection is connection and generation == self._generation and not self._stop.is_set():
                node.data_started = True
                node.warmup_deadline = None

    def _receive(self, node: _Node) -> None:
        failures = 0
        credit_supplier = None
        on_data_received = None
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: not self._paused or self._stop.is_set())
                if self._stop.is_set():
                    return
                connection = node.connection
                generation = self._generation
            try:
                if connection is None:
                    connection = self._connect(node, generation)
                    if connection is None:
                        continue
                    credit_supplier = partial(self._claim_credit, node, connection, generation)
                    on_data_received = partial(self._mark_data_received, node, connection, generation)
                assert credit_supplier is not None
                envelope = connection.receive_batch(
                    credit_supplier=credit_supplier, on_data_received=on_data_received
                )
                failures = 0
                self._enqueue(node, connection, envelope)
            except EndOfStream:
                with self._condition:
                    current = (
                        node.connection is connection
                        and generation == self._generation
                        and not self._stop.is_set()
                    )
                    if current:
                        node.done = True
                        self._unfinished -= 1
                self._drop(node, connection)
                if current:
                    return
            except (ServiceUnavailableError, TimeoutError, OSError):
                self._drop(node, connection)
                with self._condition:
                    if self._stop.is_set():
                        return
                    if self._paused or generation != self._generation:
                        continue
                    failures += 1
                    exhausted = failures > self.max_reconnects
                    if not exhausted:
                        node.reconnects += 1
                        self._metrics.add("reconnect_count")
                        self._condition.notify_all()
                if exhausted:
                    self._fail(
                        ServiceUnavailableError(
                            f"node {node.endpoint} failed after {self.max_reconnects} reconnects"
                        ),
                        generation=generation,
                    )
                else:
                    self._stop.wait(min(0.5 * 2 ** (failures - 1), 8.0))
            except BaseException as exc:
                self._drop(node, connection)
                with self._condition:
                    if generation != self._generation or self._stop.is_set():
                        continue
                    if isinstance(exc, ChecksumError):
                        self._metrics.add("checksum_failure_count")
                self._fail(exc, generation=generation)

    def _enqueue(self, node: _Node, connection: BatchConnection, envelope: BatchEnvelope) -> None:
        wake = None
        try:
            with self._condition:
                if node.connection is not connection or self._paused or self._stop.is_set():
                    return
                if node.grants <= 0:
                    raise ProtocolError(f"node {node.endpoint} sent a batch without credit")
                node.grants -= 1
                queued = False
                try:
                    if envelope.sequence <= node.highest_enqueued or envelope.batch_id in node.seen:
                        self._metrics.add("duplicate_count")
                        return
                    if envelope.queued_bytes > self.byte_capacity:
                        raise ProtocolError("batch payload exceeds global prefetch byte capacity")
                    while self._queued_bytes + envelope.queued_bytes > self.byte_capacity:
                        if self._stop.is_set() or self._paused or node.connection is not connection:
                            return
                        self._condition.wait()
                    if self._stop.is_set() or self._paused or node.connection is not connection:
                        return
                    if self._first_batch_at is None:
                        self._first_batch_at = time.monotonic()
                    self._ready.append((node, envelope))
                    queued = True
                    self._queued_bytes += envelope.queued_bytes
                    node.highest_enqueued = envelope.sequence
                    node.seen.add(envelope.batch_id)
                    node.seen_order.append(envelope.batch_id)
                    if len(node.seen_order) > 8192:
                        node.seen.discard(node.seen_order.popleft())
                    node.received_batches += 1
                    node.received_bytes += envelope.payload_bytes
                    self._metrics.add("batches_received")
                    self._metrics.add("bytes_received", envelope.payload_bytes)
                    self._metrics.observe_received(envelope.sequence)
                finally:
                    if not queued:
                        node.owned -= 1
                        self._reserved -= 1
                    wake = self._allocate_credits()
                    self._condition.notify_all()
        finally:
            self._notify_credit(wake)

    def _check_delivery_state(self) -> None:
        if self._error is not None:
            raise self._error
        if self._closed:
            raise EndOfStream("loader is closed")
        if self._paused:
            raise LoaderPausedError("loader is paused")
        if self._state not in {"RUNNING", "COMPLETED"}:
            raise SessionStateError(f"loader is not running (state={self._state})")

    def next_item(self) -> dict[str, Any]:
        started = time.monotonic()
        with self._condition:
            deadline = started + (
                _WARMUP_TIMEOUT_SECONDS if self._first_batch_at is None else self.batch_timeout
            )
            if self._pending is None and not self._ready:
                self._metrics.add("queue_empty_count")
            while True:
                self._check_delivery_state()
                # EOF must not overtake an admitted frame or its atomic commit.
                if (
                    not self._delivery_active
                    and self._pending is None
                    and not self._ready
                    and not self._unfinished
                ):
                    self._state = "COMPLETED"
                    self._metrics.set_running(False)
                    raise EndOfStream("all remote node streams completed")
                if self._first_batch_at is not None:
                    deadline = min(deadline, max(started, self._first_batch_at) + self.batch_timeout)
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    raise BatchTimeoutError("timed out waiting for a remote item")
                if not self._delivery_active and (self._pending is not None or self._ready):
                    self._delivery_active = True
                    if self._pending is None:
                        self._pending = self._ready.popleft()
                    node, envelope = self._pending
                    wait_seconds = now - started
                    break
                self._condition.wait(remaining)
        try:
            try:
                item = prepare_vqa_samples(envelope.batch) if self.spec.is_vqa else envelope.batch
            except BaseException as exc:
                self._fail(exc)
                with self._condition:
                    if self._error is None:
                        raise EndOfStream("loader closed during item delivery") from exc
            with self._condition:
                if self._error is not None:
                    raise self._error
                if self._closed:
                    raise EndOfStream("loader closed during item delivery")
                # Pause blocks admission, not the commit of an admitted result.
                # Retain the admitted frame's replay reservation until commit.
                node.consumed = envelope.sequence
                node.last_batch_id = envelope.batch_id
                self._pending = None
                node.owned -= 1
                self._reserved -= 1
                self._queued_bytes -= envelope.queued_bytes
                node.consumed_batches += 1
                self._consumed_batches += 1
                self._metrics.add("batches_consumed")
                self._metrics.observe_consumed(envelope.sequence)
                self._last_batch_at = datetime.now(timezone.utc)
                self._metrics.add("batch_wait_seconds", wait_seconds)
                return item
        finally:
            with self._condition:
                self._delivery_active = False
                wake = self._allocate_credits()
                self._condition.notify_all()
            self._notify_credit(wake)

    def _fail(self, error: BaseException, *, generation: int | None = None) -> None:
        with self._condition:
            if (
                self._closed
                or self._error is not None
                or (generation is not None and generation != self._generation)
            ):
                return
            self._error = error
            self._state = "FAILED"
            self._generation += 1
            self._stop.set()
            self._metrics.set_running(False)
            self._condition.notify_all()
        for node in self._nodes:
            self._drop(node)

    def _change_sessions(self, *, resume: bool, generation: int) -> None:
        for node in self._nodes:
            with self._condition:
                if self._stop.is_set() or generation != self._generation:
                    return
                if node.session is None or node.done:
                    continue
                session, consumed = node.session, node.consumed
                node.control_busy += 1
            try:
                refreshed = (
                    node.control.resume_session(
                        session.session_id, last_consumed_sequence=consumed, schema_hash=self._wire_schema_hash
                    )
                    if resume
                    else node.control.pause_session(session.session_id, consumed)
                )
                self._check_session(refreshed)
                if refreshed.session_id != session.session_id:
                    raise ProtocolError("node changed session identity during lifecycle operation")
                with self._condition:
                    if not self._stop.is_set() and generation == self._generation:
                        node.session = refreshed
            finally:
                with self._condition:
                    node.control_busy -= 1
                    closed = self._closed
                if closed:
                    self._close_node(node)

    def pause(self) -> None:
        with self._control_operations:
            with self._condition:
                if self._state != "RUNNING":
                    raise SessionStateError("only a running loader can be paused")
                self._paused = True
                self._state = "PAUSED"
                self._generation += 1
                generation = self._generation
                self._metrics.set_running(False)
                self._condition.notify_all()
            for node in self._nodes:
                self._drop(node)
            try:
                self._change_sessions(resume=False, generation=generation)
            except BaseException as exc:
                self._fail(exc, generation=generation)
                raise

    def resume(self) -> None:
        with self._control_operations:
            with self._condition:
                if self._state != "PAUSED":
                    raise SessionStateError("only a paused loader can be resumed")
                generation = self._generation
            try:
                self._change_sessions(resume=True, generation=generation)
            except BaseException as exc:
                self._fail(exc, generation=generation)
                raise
            with self._condition:
                if self._stop.is_set() or generation != self._generation:
                    return
                self._paused = False
                self._state = "RUNNING"
                self._metrics.set_running(True)
                wake = self._allocate_credits()
                self._condition.notify_all()
            self._notify_credit(wake)

    def _close_node(self, node: _Node) -> None:
        # Late create/refresh/control completion retains ownership until it can
        # close the actual session. close() never waits for those HTTP calls.
        with self._condition:
            if node.starting or node.control_busy or node.control_closed:
                return
            node.control_closed = True
            session = node.session
        if session is not None:
            with suppress(Exception):
                node.control.close_session(session.session_id)
        with suppress(Exception):
            node.control.close()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._state = "CLOSED"
            self._generation += 1
            self._stop.set()
            self._metrics.set_running(False)
            while self._ready:
                node, _ = self._ready.popleft()
                node.owned -= 1
                self._reserved -= 1
            if self._pending is not None:
                node, _ = self._pending
                node.owned -= 1
                self._reserved -= 1
                self._pending = None
            self._queued_bytes = 0
            self._condition.notify_all()
        for node in self._nodes:
            self._drop(node)
        for node in self._nodes:
            self._close_node(node)

    @property
    def dataset_metadata(self) -> dict[str, dict[str, Any]]:
        """Return a detached metadata snapshot keyed by initialized control endpoint."""
        with self._condition:
            metadata = {
                node.endpoint: node.session.dataset_metadata
                for node in self._nodes
                if node.session is not None
            }
        # Session updates replace objects; copy their metadata outside the stream lock.
        return deepcopy(metadata)

    def _node_checkpoint(self, node: _Node) -> LoaderCheckpoint:
        return LoaderCheckpoint(
            session_id=None if node.session is None else node.session.session_id,
            last_consumed_sequence=node.consumed,
            last_batch_id=node.last_batch_id,
            schema_hash=self._schema_hash,
            batch_spec_hash=self._batch_spec_hash,
            rank=self.rank,
            world_size=self.world_size,
        )

    def state_dict(self) -> dict[str, Any]:
        with self._condition:
            states = {node.endpoint: self._node_checkpoint(node) for node in self._nodes}
            if self.gpu_id is None:
                return states[self._nodes[0].endpoint].to_state_dict()
            return MultiNodeCheckpoint(
                schema_hash=self._schema_hash,
                batch_spec_hash=self._batch_spec_hash,
                gpu_id=self.gpu_id,
                rank=self.rank,
                world_size=self.world_size,
                consumed_batches=self._consumed_batches,
                nodes=states,
            ).to_state_dict()

    def metrics(self) -> dict[str, Any]:
        with self._condition:
            result = self._metrics.snapshot(
                queue_size=len(self._ready) + (self._pending is not None),
                queue_capacity=self.capacity,
                queued_bytes=self._queued_bytes,
                state=self._state,
            )
            if self.gpu_id is not None:
                result["nodes"] = {
                    node.endpoint: {
                        "session_id": None if node.session is None else node.session.session_id,
                        "completed": node.done,
                        "connected": node.connection is not None,
                        "batches_received": node.received_batches,
                        "batches_consumed": node.consumed_batches,
                        "bytes_received": node.received_bytes,
                        "last_consumed_sequence": node.consumed,
                        "reconnect_count": node.reconnects,
                    }
                    for node in self._nodes
                }
            return result

    def transfer_snapshot(self) -> tuple[dict[str, Any], datetime | None]:
        """Capture reporting metrics and their last committed delivery together."""
        with self._condition:
            return self.metrics(), self._last_batch_at
