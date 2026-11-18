from __future__ import annotations

import json
import selectors
import socket
import ssl
import struct
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from remote_loader.control.models import SessionInfo
from remote_loader.exceptions import ProtocolError, ServiceUnavailableError

from .protocol import HEADER, BatchEnvelope, ProtocolLimits, decode_batch, parse_header


@dataclass(frozen=True, slots=True)
class CreditUpdate:
    capacity: int
    size: int
    available: int
    queued_bytes: int
    grant_batches: int
    last_consumed_sequence: int = -1
    warmup_deadline: float | None = None


class BatchConnection(Protocol):
    def connect(self) -> None: ...

    def receive_batch(
        self,
        *,
        credit_supplier: Callable[[], CreditUpdate | None],
        on_data_received: Callable[[], None] | None = None,
    ) -> BatchEnvelope: ...

    def notify_credit(self) -> None: ...

    def close(self) -> None: ...


class TcpBatchConnection:
    """TLS-capable persistent TCP data plane using the HC framed protocol."""

    def __init__(
        self,
        session: SessionInfo,
        *,
        schema_hash: str,
        expected_output: dict[str, Any] | None = None,
        last_consumed_sequence: int,
        limits: ProtocolLimits,
        connect_timeout: float = 10.0,
        read_timeout: float = 10.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not session.data_endpoint or not session.stream_token:
            raise ProtocolError("READY session is missing data_endpoint or stream_token")
        self.session = session
        self.schema_hash = schema_hash
        self.expected_output = expected_output
        self.last_consumed_sequence = last_consumed_sequence
        self.limits = limits
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.ssl_context = ssl_context
        self._socket: socket.socket | None = None
        self._closed = False
        self._state_lock = threading.Lock()
        self._active = False
        self._selector: selectors.BaseSelector | None = None
        self._wake_reader: socket.socket | None = None
        self._wake_writer: socket.socket | None = None
        self._pending: memoryview | None = None
        self._pending_grants = 0
        self._write_deadline: float | None = None
        self._write_remaining: float | None = None
        self._sslobj: ssl.SSLObject | None = None
        self._incoming: ssl.MemoryBIO | None = None
        self._outgoing: ssl.MemoryBIO | None = None
        self._ciphertext: memoryview | None = None
        self._tls_write_chunk: memoryview | None = None
        self._tls_write_need = 0
        self._wire_sent = 0
        self._credit_end: int | None = None
        self._wire_eof = False
        self._write_error: BaseException | None = None
        self._wire_grants = 0
        self._sent_credit = False
        self._warmup_deadline: float | None = None
        self._data_started = False

    # Backpressure bounds both BIOs; a single TLS operation can additionally
    # produce one bounded application record or its context's handshake flight.
    _BIO_HIGH_WATER = 64 * 1024
    _TCP_READ_CHUNK_SIZE = 256 * 1024
    _TLS_RECORD_BYTES = 16 * 1024

    def connect(self) -> None:
        with self._state_lock:
            if self._closed:
                raise ServiceUnavailableError("batch connection is closed")
            if self._active or self._socket is not None:
                raise RuntimeError("batch connection already has an I/O owner")
            self._active = True
        sock: socket.socket | None = None
        try:
            host, port = self._parse_endpoint(self.session.data_endpoint or "")
            sock = socket.create_connection((host, port), timeout=self.connect_timeout)
            self._publish_socket(sock)
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._ensure_reactor(sock)
            context = self.ssl_context or ssl.create_default_context()
            self._incoming, self._outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
            # No fd transfer: cancellation always interrupts the same raw
            # socket, and only this owner ever touches the SSLObject.
            self._sslobj = context.wrap_bio(self._incoming, self._outgoing, server_hostname=host)
            self._get_socket()
            self._handshake_operation(
                sock, self._sslobj.do_handshake, time.monotonic() + self.read_timeout
            )
            self._handshake(sock)
            self._get_socket()
        except BaseException as exc:
            self.close()
            with self._state_lock:
                if sock is not None and sock is not self._socket:
                    with suppress(OSError):
                        sock.close()
            if isinstance(exc, OSError):
                raise ServiceUnavailableError(f"could not connect to the batch service: {exc}") from exc
            raise
        finally:
            self._finish_io()

    def _publish_socket(self, sock: socket.socket) -> None:
        with self._state_lock:
            if self._closed:
                raise ServiceUnavailableError("batch connection is closed")
            self._socket = sock

    def _get_socket(self) -> socket.socket:
        with self._state_lock:
            sock = None if self._closed else self._socket
        if sock is None:
            raise ServiceUnavailableError("batch connection is not open")
        return sock

    def _handshake(self, sock: socket.socket) -> None:
        request = {
            "type": "handshake",
            "protocol_version": self.session.protocol_version,
            "stream_token": self.session.stream_token,
            "session_id": self.session.session_id,
            "schema_hash": self.schema_hash,
            "max_batch_bytes": self.limits.max_batch_bytes,
            "last_consumed_sequence": self.last_consumed_sequence,
        }
        deadline = time.monotonic() + self.read_timeout
        pending = memoryview(self._encode_json(request))
        while pending:
            chunk = pending[: self._TLS_RECORD_BYTES]
            sent = self._handshake_operation(
                sock,
                lambda chunk=chunk: self._sslobj.write(chunk) if self._sslobj else sock.send(chunk),
                deadline,
                writing=True,
            )
            if not sent:
                raise ServiceUnavailableError("batch connection closed during handshake")
            pending = pending[sent:]
        length = struct.unpack("!I", self._handshake_exact(sock, 4, deadline))[0]
        if length <= 0 or length > 64 * 1024:
            raise ProtocolError("invalid handshake response length")
        try:
            response = json.loads(self._handshake_exact(sock, length, deadline))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("handshake response is not valid JSON") from exc
        if not isinstance(response, dict):
            raise ProtocolError("handshake response must be an object")
        if response.get("status") != "OK":
            raise ProtocolError(str(response.get("error") or "data handshake rejected"))
        if response.get("protocol_version") != self.session.protocol_version:
            raise ProtocolError("server selected an unexpected protocol version")
        if response.get("schema_hash") != self.schema_hash:
            raise ProtocolError("server handshake schema_hash mismatch")

    def receive_batch(
        self,
        *,
        credit_supplier: Callable[[], CreditUpdate | None],
        on_data_received: Callable[[], None] | None = None,
    ) -> BatchEnvelope:
        with self._state_lock:
            if self._closed or self._socket is None:
                raise ServiceUnavailableError("batch connection is not open")
            if self._active:
                raise RuntimeError("batch connection already has an I/O owner")
            self._active = True
            sock = self._socket
        try:
            self._ensure_reactor(sock)
            if self._write_remaining is not None:
                self._write_deadline = time.monotonic() + self._write_remaining
            return self._receive_frame(sock, credit_supplier, on_data_received)
        except BaseException as exc:
            self.close()
            if isinstance(exc, TimeoutError):
                raise TimeoutError("batch socket read/write timed out") from exc
            if isinstance(exc, OSError):
                raise ServiceUnavailableError(f"batch connection failed: {exc}") from exc
            raise
        finally:
            self._pause_write_clock()
            self._finish_io()

    def _ensure_reactor(self, sock: socket.socket) -> None:
        if self._selector is not None:
            return
        selector = selectors.DefaultSelector()
        reader = writer = None
        try:
            reader, writer = socket.socketpair()
            reader.setblocking(False)
            writer.setblocking(False)
            sock.setblocking(False)
            selector.register(reader, selectors.EVENT_READ, "wake")
            selector.register(sock, selectors.EVENT_READ, "data")
            with self._state_lock:
                if self._closed:
                    raise ServiceUnavailableError("batch connection is closed")
                self._selector = selector
                self._wake_reader, self._wake_writer = reader, writer
        except BaseException:
            with suppress(OSError):
                selector.close()
            for resource in (reader, writer):
                if resource is not None:
                    with suppress(OSError):
                        resource.close()
            raise
        # The first supplier pull follows publication. Notifications before that
        # point may coalesce away; their runtime permission cannot be lost.

    def _output_size(self) -> int:
        return (len(self._ciphertext) if self._ciphertext is not None else 0) + (
            self._outgoing.pending if self._outgoing is not None else 0
        )

    def _flush_tls(self, sock: socket.socket) -> bool:
        outgoing = self._outgoing
        assert outgoing is not None
        if self._ciphertext is None:
            if not outgoing.pending:
                return False
            self._ciphertext = memoryview(outgoing.read(self._BIO_HIGH_WATER))
        sent = sock.send(self._ciphertext)
        if not sent:
            raise ServiceUnavailableError("batch connection closed while sending TLS records")
        self._wire_sent += sent
        self._ciphertext = self._ciphertext[sent:]
        if not self._ciphertext:
            self._ciphertext = None
        return True

    def _feed_tls(self, sock: socket.socket) -> bool:
        incoming = self._incoming
        assert incoming is not None
        if self._wire_eof or incoming.pending >= self._BIO_HIGH_WATER:
            return False
        chunk = sock.recv(self._BIO_HIGH_WATER - incoming.pending)
        if chunk:
            incoming.write(chunk)
        else:
            self._wire_eof = True
            incoming.write_eof()
        return True

    def _wait(self, sock: socket.socket, interest: int, timeout: float | None) -> int:
        selector = self._selector
        assert selector is not None
        if selector.get_key(sock).events != interest:
            selector.modify(sock, interest, "data")
        ready = 0
        for key, mask in selector.select(timeout):
            if key.data == "wake":
                # Bounded coalescing leaves concurrent notifications readable.
                with suppress(BlockingIOError):
                    key.fileobj.recv(self._BIO_HIGH_WATER)
            else:
                ready |= mask
        self._get_socket()
        return ready

    def _handshake_operation(
        self,
        sock: socket.socket,
        operation: Callable[[], Any],
        deadline: float,
        *,
        writing: bool = False,
    ) -> Any:
        done = False
        result = None
        need = selectors.EVENT_WRITE if writing else selectors.EVENT_READ
        operation_ready = read_ready = write_ready = True
        while True:
            self._get_socket()
            progressed = False
            if not done and operation_ready and self._output_size() < self._BIO_HIGH_WATER:
                try:
                    result = operation()
                except ssl.SSLWantReadError:
                    need = selectors.EVENT_READ
                except ssl.SSLWantWriteError:
                    need = selectors.EVENT_WRITE
                except BlockingIOError:
                    need = selectors.EVENT_WRITE if writing else selectors.EVENT_READ
                else:
                    done = True
                operation_ready = False
            if self._sslobj is not None:
                if self._output_size() and write_ready:
                    try:
                        progressed = self._flush_tls(sock)
                    except BlockingIOError:
                        write_ready = False
                    if progressed:
                        # Draining output can re-enable an operation that still
                        # has buffered TLS input, without a new raw read event.
                        operation_ready = True
                if not done and need == selectors.EVENT_READ and read_ready:
                    try:
                        fed = self._feed_tls(sock)
                    except BlockingIOError:
                        read_ready = False
                    else:
                        progressed |= fed
                        operation_ready |= fed
            if done and not self._output_size():
                return result
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("batch handshake timed out")
            if progressed:
                continue
            interest = need if not done else 0
            if self._sslobj is not None:
                assert self._incoming is not None
                if (
                    self._wire_eof
                    or self._incoming.pending >= self._BIO_HIGH_WATER
                    or self._output_size() >= self._BIO_HIGH_WATER
                ):
                    interest &= ~selectors.EVENT_READ
                if self._output_size():
                    interest |= selectors.EVENT_WRITE
            if not interest:
                raise ServiceUnavailableError("TLS handshake made no I/O progress")
            ready = self._wait(sock, interest, deadline - now)
            read_ready = bool(ready & selectors.EVENT_READ)
            write_ready = bool(ready & selectors.EVENT_WRITE)
            if self._sslobj is None:
                operation_ready = bool(ready & need)
            elif need == selectors.EVENT_WRITE and write_ready:
                operation_ready = True

    def _handshake_exact(self, sock: socket.socket, size: int, deadline: float) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._handshake_operation(
                sock,
                lambda remaining=remaining: (
                    self._sslobj.read(remaining) if self._sslobj else sock.recv(remaining)
                ),
                deadline,
            )
            if not chunk:
                raise ServiceUnavailableError("batch connection closed during handshake")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _arm_write_clock(self) -> None:
        if self._write_remaining is None:
            self._write_remaining = self.read_timeout
            self._write_deadline = time.monotonic() + self.read_timeout

    def _pause_write_clock(self) -> None:
        if self._write_deadline is not None:
            self._write_remaining = max(0.0, self._write_deadline - time.monotonic())
            self._write_deadline = None

    def _complete_credit(self) -> None:
        self._pending = None
        grants = self._pending_grants
        self._pending_grants = 0
        self._credit_end = None
        self._wire_grants += grants
        self._sent_credit = True
        self._settle_write_clock()

    def _read_deadline(self) -> float:
        if self._warmup_deadline is not None and self._write_error is None:
            return self._warmup_deadline
        return time.monotonic() + self.read_timeout

    def _receive_frame(
        self,
        sock: socket.socket,
        credit_supplier: Callable[[], CreditUpdate | None],
        on_data_received: Callable[[], None] | None,
    ) -> BatchEnvelope:
        tls = self._sslobj
        read_ready = write_ready = tls_read_ready = True
        tls_write_ready = True
        tls_read_needs_write = False
        read_deadline = self._read_deadline() if self._wire_grants or self._write_error is not None else None
        parts: list[bytes] = []
        stage, remaining, payload_length = 0, HEADER.size, 0
        metadata = b""
        partial = False
        while True:
            self._get_socket()
            progressed = False
            if self._write_deadline is not None and time.monotonic() >= self._write_deadline:
                self._fail_write(TimeoutError("batch credit write timed out"))
                tls_read_ready = True
            if self._pending is None and self._write_error is None:
                update = credit_supplier()
                self._get_socket()
                if update is not None:
                    self._pending = memoryview(
                        self._encode_json(
                            {
                                "type": "credit",
                                "queue_capacity": update.capacity,
                                "queue_size": update.size,
                                "available_batches": update.available,
                                "queued_bytes": update.queued_bytes,
                                "grant_batches": update.grant_batches,
                                "last_consumed_sequence": update.last_consumed_sequence,
                            }
                        )
                    )
                    self._pending_grants = update.grant_batches
                    if not self._data_started:
                        self._warmup_deadline = update.warmup_deadline
                    self._arm_write_clock()
                    tls_write_ready = True
            if (
                tls is not None
                and self._pending
                and tls_write_ready
                and (self._output_size() < self._BIO_HIGH_WATER and not tls_read_needs_write)
            ):
                if self._tls_write_chunk is None:
                    self._tls_write_chunk = self._pending[: self._TLS_RECORD_BYTES]
                try:
                    sent = tls.write(self._tls_write_chunk)
                except ssl.SSLWantReadError:
                    self._tls_write_need = selectors.EVENT_READ
                    tls_write_ready = False
                except ssl.SSLWantWriteError:
                    # MemoryBIO has no socket backpressure. If OpenSSL does
                    # request output progress, retry this exact write before
                    # making any other SSL operation.
                    self._tls_write_need = selectors.EVENT_WRITE
                    tls_write_ready = False
                else:
                    if not sent:
                        raise ServiceUnavailableError("TLS closed while encoding credit")
                    progressed = True
                    self._pending = self._pending[sent:]
                    self._tls_write_chunk = None
                    self._tls_write_need = 0
                    if not self._pending:
                        # Freeze this credit's boundary BEFORE reads can append
                        # post-handshake records to the same outgoing BIO.
                        self._credit_end = self._wire_sent + self._output_size()
            if self._write_error is None and write_ready:
                try:
                    if tls is not None:
                        sent_output = self._flush_tls(sock) if self._output_size() else False
                        progressed |= sent_output
                        if sent_output:
                            if self._tls_write_need == selectors.EVENT_WRITE:
                                tls_write_ready = True
                            if tls_read_needs_write:
                                tls_read_ready = True
                        if self._credit_end is not None and self._wire_sent >= self._credit_end:
                            self._complete_credit()
                    elif self._pending is not None:
                        sent = sock.send(self._pending)
                        if not sent:
                            raise ServiceUnavailableError("batch connection closed while sending credit")
                        progressed = True
                        self._pending = self._pending[sent:]
                        if not self._pending:
                            self._complete_credit()
                except BlockingIOError:
                    write_ready = False
                except (OSError, ServiceUnavailableError) as exc:
                    self._fail_write(exc)
                    tls_read_ready = True
            if self._write_error is not None:
                failure_deadline = self._read_deadline()
                read_deadline = (
                    failure_deadline if read_deadline is None else min(read_deadline, failure_deadline)
                )
            elif read_deadline is None and self._wire_grants:
                read_deadline = self._read_deadline()
            chunk = None
            if tls is None and read_ready:
                try:
                    chunk = sock.recv(min(remaining, self._TCP_READ_CHUNK_SIZE))
                except BlockingIOError:
                    read_ready = False
            elif (
                tls is not None
                and tls_read_ready
                and (
                    self._tls_write_need != selectors.EVENT_WRITE
                    and self._output_size() < self._BIO_HIGH_WATER
                )
            ):
                assert self._incoming is not None
                before_in, before_out = self._incoming.pending, self._output_size()
                try:
                    chunk = tls.read(min(remaining, self._BIO_HIGH_WATER))
                except ssl.SSLWantReadError:
                    tls_read_ready = False
                    tls_read_needs_write = False
                except ssl.SSLWantWriteError:
                    tls_read_ready = False
                    tls_read_needs_write = True
                else:
                    tls_read_needs_write = False
                # CertificateRequest/Finished/key updates can advance TLS with
                # no application bytes. That progress must retry a WANT_READ
                # credit writer, not strand it waiting for an HC frame.
                protocol_progress = self._incoming.pending != before_in or self._output_size() != before_out
                if self._write_error is not None:
                    self._discard_tls_output()
                progressed |= protocol_progress
                if self._tls_write_need == selectors.EVENT_READ:
                    # SSL can consume handshake bytes already in its internal
                    # record buffer, invisible to MemoryBIO.pending. Retry once
                    # after this read attempt; WANT_READ then parks both sides.
                    tls_write_ready = True
                    progressed = True
            if chunk is not None:
                if not chunk:
                    raise ServiceUnavailableError("batch connection closed while receiving a frame")
                if not self._data_started:
                    self._data_started = True
                    self._warmup_deadline = None
                    if on_data_received is not None:
                        on_data_received()
                progressed = partial = True
                read_deadline = time.monotonic() + self.read_timeout
                parts.append(chunk)
                remaining -= len(chunk)
                if remaining == 0:
                    raw = b"".join(parts)
                    parts = []
                    if stage == 0:
                        remaining, payload_length, _ = parse_header(raw, self.limits)
                        stage = 1
                    elif stage == 1:
                        metadata = raw
                        remaining, stage = payload_length, 2
                        if remaining:
                            continue
                        raw = b""
                    if stage == 2 and remaining == 0:
                        self._get_socket()
                        # Decoder/caller/enqueue work is not active socket I/O.
                        self._settle_write_clock()
                        self._pause_write_clock()
                        envelope = decode_batch(
                            metadata,
                            raw,
                            expected_session_id=self.session.session_id,
                            expected_schema_hash=self.schema_hash,
                            expected_output=self.expected_output,
                            limits=self.limits,
                        )
                        self._wire_grants = max(0, self._wire_grants - 1)
                        return envelope
            if (
                tls is not None
                and read_ready
                and (
                    not tls_read_ready or self._tls_write_need or self._output_size() >= self._BIO_HIGH_WATER
                )
            ):
                try:
                    fed = self._feed_tls(sock)
                except BlockingIOError:
                    read_ready = False
                else:
                    progressed |= fed
                    if fed:
                        tls_read_ready = True
                        if self._tls_write_need == selectors.EVENT_READ:
                            tls_write_ready = True
            self._settle_write_clock()
            now = time.monotonic()
            if self._write_deadline is not None and now >= self._write_deadline:
                self._fail_write(TimeoutError("batch credit write timed out"))
                failure_deadline = now + self.read_timeout
                read_deadline = (
                    failure_deadline if read_deadline is None else min(read_deadline, failure_deadline)
                )
                tls_read_ready = True
                # Retry buffered TLS data before waiting for another raw event.
                continue
            if read_deadline is not None and now >= read_deadline:
                raise TimeoutError("batch socket read timed out")
            if progressed:
                continue
            # Permission starvation is local, not a network read timeout.
            if not partial and not self._wire_grants and self._write_error is None:
                read_deadline = None
            deadlines = [value for value in (read_deadline, self._write_deadline) if value is not None]
            timeout = max(0.0, min(deadlines) - now) if deadlines else None
            interest = selectors.EVENT_READ
            if self._write_error is None and (
                self._output_size() if tls is not None else self._pending is not None
            ):
                interest |= selectors.EVENT_WRITE
            # At the input high-water mark, only output can unblock TLS. Avoid
            # repeatedly selecting an already-readable raw fd without reading.
            if (
                tls is not None
                and self._incoming is not None
                and (self._wire_eof or self._incoming.pending >= self._BIO_HIGH_WATER)
            ):
                interest &= ~selectors.EVENT_READ
            if not interest:
                # The damaged output cannot unblock more protocol progress.
                if self._write_error is not None:
                    raise self._write_error
                raise ServiceUnavailableError("TLS protocol made no I/O progress")
            ready = self._wait(sock, interest, timeout)
            read_ready = bool(ready & selectors.EVENT_READ)
            write_ready = bool(ready & selectors.EVENT_WRITE)

    def _settle_write_clock(self) -> None:
        if self._write_error is not None:
            return
        if self._pending is not None or self._output_size():
            self._arm_write_clock()
        else:
            self._write_remaining = self._write_deadline = None

    def _fail_write(self, error: BaseException) -> None:
        self._pending = None
        self._pending_grants = 0
        self._credit_end = None
        self._tls_write_chunk = None
        self._tls_write_need = 0
        self._write_remaining = self._write_deadline = None
        if not self._sent_credit:
            raise error
        # Stop all raw output, but retain TLS state and readable data/EOS. A
        # damaged partial control is never re-encoded or counted as permission.
        self._write_error = error
        self._warmup_deadline = None
        self._discard_tls_output()

    def _discard_tls_output(self) -> None:
        # Output is permanently unusable after a raw write failure. Retaining
        # it would block readable records at the high-water mark; discard only
        # outgoing bytes, never the TLS state or its buffered incoming data.
        self._ciphertext = None
        if self._outgoing is not None:
            while self._outgoing.pending:
                self._outgoing.read(self._BIO_HIGH_WATER)

    def notify_credit(self) -> None:
        with self._state_lock:
            if not self._closed and self._wake_writer is not None:
                with suppress(BlockingIOError, OSError):
                    self._wake_writer.send(b"\0")

    @staticmethod
    def _encode_json(payload: dict[str, Any]) -> bytes:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        if len(raw) > 1024 * 1024:
            raise ProtocolError("outgoing control message is too large")
        return struct.pack("!I", len(raw)) + raw

    @staticmethod
    def _parse_endpoint(endpoint: str) -> tuple[str, int]:
        parsed = urlparse(endpoint)
        if parsed.scheme != "tls" or not parsed.hostname or not parsed.port:
            raise ProtocolError(f"invalid data endpoint {endpoint!r}")
        return parsed.hostname, parsed.port

    def _dispose_locked(self) -> None:
        # Only the inactive owner disposes registered descriptors. Holding the
        # state lock also serializes shutdown/notify against descriptor reuse.
        if self._selector is not None:
            with suppress(OSError):
                self._selector.close()
            self._selector = None
        for resource in (self._socket, self._wake_reader, self._wake_writer):
            if resource is not None:
                with suppress(OSError):
                    resource.close()
        self._socket = self._wake_reader = self._wake_writer = None
        self._pending = self._ciphertext = self._tls_write_chunk = None
        self._sslobj = self._incoming = self._outgoing = None

    def _finish_io(self) -> None:
        with self._state_lock:
            self._active = False
            if self._closed:
                self._dispose_locked()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            if self._wake_writer is not None:
                with suppress(BlockingIOError, OSError):
                    self._wake_writer.send(b"\0")
            if self._socket is not None:
                with suppress(OSError):
                    self._socket.shutdown(socket.SHUT_RDWR)
            if not self._active:
                self._dispose_locked()
