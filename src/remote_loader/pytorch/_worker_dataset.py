"""Worker-readable views of the parent's sample buffer, without shared receive locks."""

from __future__ import annotations

import os
import pickle
import socket
import socketserver
import struct
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import suppress
from multiprocessing import resource_tracker
from multiprocessing.connection import Client, Connection, answer_challenge, deliver_challenge
from typing import Any

import torch
from torch._utils import ExceptionWrapper

from .._batch_delivery import CanonicalImageDelivery
from ..protocol_limits import ProtocolLimits
from ._shared_sample import SharedSample, SharedSampleReader, SharedSampleWriter, _local_metadata_limit

# The parent handler thread holding this process-local lock vanishes at fork.
# Reset the child copy without adding a fork barrier or a per-sample lock.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=resource_tracker._resource_tracker._lock._at_fork_reinit,
    )

_SAMPLE = 0
_END = 1
_ERROR = 2
_RELEASE = 3
_REQUEST = struct.Struct("!cQQ")
_RELEASE_ACK = struct.Struct("!cQ")
_MAX_CONTROL_BYTES = 2 * _local_metadata_limit(ProtocolLimits())


def _shutdown_connection(connection: Connection) -> None:
    with suppress(OSError):
        sock = socket.socket(fileno=connection.fileno())
        try:
            sock.shutdown(socket.SHUT_RDWR)
        finally:
            sock.detach()


def _send_message(connection: Connection, kind: int, value: Any) -> None:
    payload = pickle.dumps((kind, value), protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_CONTROL_BYTES:
        raise ValueError("local sample control message exceeds the transfer limit")
    connection.send_bytes(payload)


def _close_sample_connection(reader: SharedSampleReader, connection: Connection) -> None:
    try:
        reader.close()
    finally:
        connection.close()


class _SampleHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection = Connection(self.request.detach())
        writer = SharedSampleWriter()
        authenticated = False
        with server.connections_lock:
            if server.stopped.is_set():
                connection.close()
                return
            server.connections.add(connection)
            server._writers.add(writer)

        def release_reader(generation: int) -> None:
            _send_message(connection, _RELEASE, generation)
            response = connection.recv_bytes(maxlength=_RELEASE_ACK.size)
            if response != _RELEASE_ACK.pack(b"R", generation):
                raise ValueError("worker did not release the previous shared sample mapping")

        last_delivery = (0, 0)
        try:
            deliver_challenge(connection, server.authkey)
            answer_challenge(connection, server.authkey)
            authenticated = True
            while not server.stopped.is_set():
                request = connection.recv_bytes(maxlength=_REQUEST.size)
                if len(request) != _REQUEST.size:
                    raise ValueError("invalid local sample request")
                operation, sequence, generation = _REQUEST.unpack(request)
                if operation != b"N" or (sequence, generation) != last_delivery:
                    raise ValueError("worker acknowledged a different shared sample delivery")
                try:
                    sample = server.next_sample()
                except StopIteration:
                    _send_message(connection, _END, None)
                    return
                except Exception:
                    _send_message(connection, _ERROR, ExceptionWrapper(where="in the remote sample receiver"))
                    continue
                try:
                    descriptor = writer.write(sample, before_resize=release_reader)
                except Exception:
                    _send_message(
                        connection,
                        _ERROR,
                        ExceptionWrapper(where="while transferring a remote sample to a worker"),
                    )
                    continue
                finally:
                    # The shared slot, not the source tensor, survives delivery.
                    sample = None
                sequence = last_delivery[0] + 1
                _send_message(connection, _SAMPLE, (sequence, descriptor))
                last_delivery = (sequence, descriptor.generation)
        except (EOFError, OSError):
            pass
        except Exception:
            if not authenticated:
                raise
            with suppress(OSError):
                _send_message(connection, _ERROR, ExceptionWrapper(where="in local sample transport"))
        finally:
            writer.close()
            with server.connections_lock:
                server._writers.discard(writer)
                server.connections.discard(connection)
                connection.close()


class _SampleBuffer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, source: Any) -> None:
        # Fork workers must inherit the parent tracker rather than launch their own.
        if os.name == "posix":
            resource_tracker.ensure_running()
        self.owner_pid = os.getpid()
        self.authkey = os.urandom(32)
        self.stopped = threading.Event()
        self._cancelled = threading.Event()
        self._source_close_requested = False
        self._source_closed = False
        self.connections_lock = threading.Lock()
        self.connections: set[Connection] = set()
        self._writers: set[SharedSampleWriter] = set()
        self._handler_threads_lock = threading.Lock()
        self._handler_threads: set[threading.Thread] = set()
        self._source_lock = threading.Lock()
        self._source = source
        self._iterator: Iterator[Any] | None = None
        super().__init__(("127.0.0.1", 0), _SampleHandler)
        try:
            self._thread = threading.Thread(
                target=self.serve_forever,
                kwargs={"poll_interval": 0.05},
                name="remote-loader-samples",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            self.server_close()
            raise

    def process_request(self, request: Any, client_address: Any) -> None:
        thread = threading.Thread(
            target=self.process_request_thread,
            args=(request, client_address),
            daemon=self.daemon_threads,
        )
        with self._handler_threads_lock:
            self._handler_threads.add(thread)
        try:
            thread.start()
        except BaseException:
            with self._handler_threads_lock:
                self._handler_threads.discard(thread)
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        thread = threading.current_thread()
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._handler_threads_lock:
                self._handler_threads.discard(thread)

    def next_sample(self) -> Any:
        # The only shared consumer lock stays in the receiver process. Killing a
        # worker closes its private connection, never a process-shared read lock.
        if self._cancelled.is_set():
            raise StopIteration
        with self._source_lock:
            if self._cancelled.is_set():
                raise StopIteration
            if self._iterator is None:
                self._iterator = self._source._iter_wire_samples()
            return next(self._iterator)

    def cancel_reads(self, *, close_source: bool) -> None:
        """End reads while keeping IPC available to workers resetting their iterators."""
        if os.getpid() != self.owner_pid:
            return
        if not isinstance(close_source, bool):
            raise TypeError("close_source must be a bool")
        self._cancelled.set()
        if close_source and not self._source_close_requested:
            self._source_close_requested = True
            close = getattr(self._source, "close", None)
            if not callable(close):
                raise TypeError("remote sample source must expose close()")
            close()
            self._source_closed = True

    def close(self, *, close_source: bool = False) -> None:
        if os.getpid() != self.owner_pid:
            return
        try:
            self.cancel_reads(close_source=close_source)
        finally:
            if not self.stopped.is_set():
                self.stopped.set()
                with self.connections_lock:
                    for connection in self.connections:
                        _shutdown_connection(connection)
                    writers = tuple(self._writers)
                # A blocked source may deliberately outlive this dataset when
                # close_source=False. Its handler must not retain transfer slots.
                for writer in writers:
                    writer.close()
                self.shutdown()
                self.server_close()
                self._thread.join()
        if self._source_closed:
            while True:
                with self._handler_threads_lock:
                    handlers = list(self._handler_threads)
                if not handlers:
                    break
                for handler in handlers:
                    handler.join()


class _SampleIterator:
    def __init__(
        self,
        preprocess: Callable[[Any], Any] | None,
        *,
        delivery: CanonicalImageDelivery,
        source: Iterator[Any] | None = None,
        address: tuple[str, int] | None = None,
        authkey: bytes | None = None,
    ) -> None:
        self.preprocess = preprocess
        self.delivery = delivery
        self.source = source
        self.connection = None
        self.finished = False
        self._acknowledged = (0, 0)
        self._reader = None
        if source is None:
            self.connection = Client(address, authkey=authkey)
            try:
                sock = socket.socket(fileno=self.connection.fileno())
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                finally:
                    sock.detach()
                self._reader = SharedSampleReader()
                self._finalizer = weakref.finalize(
                    self, _close_sample_connection, self._reader, self.connection
                )
            except BaseException:
                self.connection.close()
                raise

    def __iter__(self) -> _SampleIterator:
        return self

    def _receive(self) -> tuple[int, Any]:
        assert self.connection is not None and self._reader is not None
        self.connection.send_bytes(_REQUEST.pack(b"N", *self._acknowledged))
        while True:
            packet = pickle.loads(self.connection.recv_bytes(maxlength=_MAX_CONTROL_BYTES))
            if not isinstance(packet, tuple) or len(packet) != 2:
                raise ValueError("invalid local sample response")
            kind, value = packet
            if kind != _RELEASE:
                return kind, value
            if type(value) is not int or value < 0:
                raise ValueError("invalid shared sample mapping generation")
            self._reader.close()
            self.connection.send_bytes(_RELEASE_ACK.pack(b"R", value))

    def __next__(self) -> Any:
        if self.finished:
            raise StopIteration
        if self.source is not None:
            sample = next(self.source)
        else:
            try:
                kind, payload = self._receive()
            except (EOFError, OSError) as exc:
                self.finished = True
                self._finalizer()
                raise RuntimeError("remote sample receiver connection closed before end of stream") from exc
            except Exception:
                self.finished = True
                self._finalizer()
                raise
            if kind == _END:
                self.finished = True
                self._finalizer()
                raise StopIteration
            if kind == _ERROR:
                # Source pause/timeout and sample-encoding failures remain
                # recoverable through the same native worker iterator.
                payload.reraise()
            try:
                if kind != _SAMPLE or not isinstance(payload, tuple) or len(payload) != 2:
                    raise ValueError("invalid shared sample delivery")
                sequence, descriptor = payload
                if (
                    type(sequence) is not int
                    or sequence != self._acknowledged[0] + 1
                    or not isinstance(descriptor, SharedSample)
                ):
                    raise ValueError("unexpected shared sample delivery sequence")
                assert self._reader is not None
                sample = self._reader.read(descriptor)
                # Only private storage reaches preprocess/collate. This ACK is
                # carried by NEXT, so ordinary deliveries add no extra round trip.
                self._acknowledged = (sequence, descriptor.generation)
            except Exception:
                self.finished = True
                self._finalizer()
                raise
        sample = self.delivery(sample)
        return sample if self.preprocess is None else self.preprocess(sample)


class _WorkerDataset(torch.utils.data.IterableDataset):
    handles_distributed_sharding = True

    def __init__(self, source: Any, preprocess: Callable[[Any], Any] | None) -> None:
        super().__init__()
        self.remote_loader = source
        self.preprocess = preprocess
        self._delivery = source._delivery
        self.already_batched = source.spec.is_vqa
        self._buffer = _SampleBuffer(source)
        self._owner_pid = self._buffer.owner_pid
        self._address = self._buffer.server_address
        self._authkey = self._buffer.authkey
        self._finalizer = weakref.finalize(self, self._buffer.close)

    def close(self, *, close_source: bool = True) -> None:
        """Stop the local sample buffer and, by default, interrupt its remote source."""
        self._buffer.close(close_source=close_source)
        self._finalizer.detach()

    def _cancel_reads(self, *, close_source: bool) -> None:
        self._buffer.cancel_reads(close_source=close_source)

    def __getstate__(self) -> dict[str, Any]:
        # Only delivery configuration, CPU transform and local reader address cross
        # process boundaries; never serialize the live network loader.
        return {
            "preprocess": self.preprocess,
            "_delivery": self._delivery,
            "already_batched": self.already_batched,
            "_owner_pid": self._owner_pid,
            "_address": self._address,
            "_authkey": self._authkey,
        }

    def __iter__(self) -> Iterator[Any]:
        if os.getpid() == self._owner_pid:
            return _SampleIterator(
                self.preprocess, delivery=self._delivery,
                source=self.remote_loader._iter_wire_samples(),
            )
        return _SampleIterator(
            self.preprocess, delivery=self._delivery, address=self._address, authkey=self._authkey,
        )
