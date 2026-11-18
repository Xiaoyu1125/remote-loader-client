from __future__ import annotations

import threading
import time
from typing import Any


class LoaderMetrics:
    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.batches_received = 0
        self.batches_consumed = 0
        self.bytes_received = 0
        self.queue_empty_count = 0
        self.batch_wait_seconds = 0.0
        self.reconnect_count = 0
        self.duplicate_count = 0
        self.checksum_failure_count = 0
        self.last_received_sequence = -1
        self.last_consumed_sequence = -1
        self._running_since: float | None = None
        self._running_seconds = 0.0
        self._lock = threading.Lock()

    def add(self, name: str, value: int | float = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + value)

    def set_running(self, running: bool) -> None:
        now = time.monotonic()
        with self._lock:
            if running:
                if self._running_since is None:
                    self._running_since = now
                return
            if self._running_since is not None:
                self._running_seconds += now - self._running_since
                self._running_since = None

    def observe_received(self, sequence: int) -> None:
        with self._lock:
            self.last_received_sequence = max(self.last_received_sequence, sequence)

    def observe_consumed(self, sequence: int) -> None:
        with self._lock:
            self.last_consumed_sequence = max(self.last_consumed_sequence, sequence)

    def snapshot(
        self,
        *,
        queue_size: int,
        queue_capacity: int,
        queued_bytes: int,
        state: str,
    ) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            elapsed = max(now - self.started_at, 1e-9)
            active_seconds = self._running_seconds
            if self._running_since is not None:
                active_seconds += now - self._running_since
            active_seconds = max(active_seconds, 1e-9)
            return {
                "state": state,
                "batches_received": self.batches_received,
                "batches_consumed": self.batches_consumed,
                "bytes_received": self.bytes_received,
                "receive_batches_per_s": self.batches_received / elapsed,
                "receive_bytes_per_s": self.bytes_received / elapsed,
                "consume_batches_per_s": self.batches_consumed / active_seconds,
                "queue_size": queue_size,
                "queue_capacity": queue_capacity,
                "available_batches": max(queue_capacity - queue_size, 0),
                "queued_bytes": queued_bytes,
                "last_received_sequence": self.last_received_sequence,
                "last_consumed_sequence": self.last_consumed_sequence,
                "queue_empty_count": self.queue_empty_count,
                "batch_wait_seconds": self.batch_wait_seconds,
                "active_seconds": active_seconds,
                "consumer_data_stall_ratio": min(self.batch_wait_seconds / active_seconds, 1.0),
                "reconnect_count": self.reconnect_count,
                "duplicate_count": self.duplicate_count,
                "checksum_failure_count": self.checksum_failure_count,
            }
