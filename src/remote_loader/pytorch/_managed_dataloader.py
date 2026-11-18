"""Track native PyTorch iterators without extending their lifetime."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from queue import Full
from threading import Condition
from typing import Any
from weakref import WeakSet

from torch.utils.data import DataLoader, _utils


def _shutdown_iterators(iterators: Iterable[Any]) -> None:
    failure: BaseException | None = None
    for iterator in iterators:
        try:
            try:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
                elif getattr(iterator, "_num_workers", 0) > 0:
                    raise RuntimeError("PyTorch DataLoader iterator cannot shut down workers")
            finally:
                # A concurrent persistent reset can still be waiting for resume
                # acknowledgements. Native shutdown leaves this receive queue
                # open without pin_memory, so that wait would otherwise retry forever.
                queue = getattr(iterator, "_worker_result_queue", None)
                if queue is not None:
                    queue.cancel_join_thread()
                    queue.close()
                for worker in getattr(iterator, "_workers", ()):
                    worker.join(timeout=5)
                    if worker.is_alive():
                        raise RuntimeError("PyTorch DataLoader worker did not stop")
        except BaseException as exc:
            # Reap the other iterators even if one shutdown fails or is interrupted.
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


def _resume_pending_resets(iterators: Iterable[Any]) -> None:
    """Wake persistent resets through the queue consumed by _get_data()."""
    for iterator in iterators:
        queue = getattr(iterator, "_data_queue", None)
        if queue is None:
            continue
        acknowledgement = _utils.worker._ResumeIteration(
            getattr(iterator, "_shared_seed", None)
        )
        for _ in getattr(iterator, "_workers", ()):
            with suppress(AttributeError, Full, OSError, ValueError):
                queue.put_nowait((acknowledgement, None))


class _ManagedDataLoader(DataLoader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._modalink_closed = False
        self._modalink_creating = 0
        self._modalink_condition = Condition()
        self._modalink_iterators: WeakSet[Any] = WeakSet()
        self._modalink_pending: list[Any] = []

    def __iter__(self) -> Any:
        with self._modalink_condition:
            if self._modalink_closed:
                raise RuntimeError("PyTorch DataLoader is closed")
            self._modalink_creating += 1
        try:
            # Persistent iterator reset may wait for workers. Closing must remain
            # free to interrupt their source while this call is in progress.
            try:
                iterator = super().__iter__()
            except Exception as exc:
                with self._modalink_condition:
                    if self._modalink_closed:
                        raise RuntimeError(
                            "PyTorch DataLoader was closed while creating an iterator"
                        ) from exc
                raise
            with self._modalink_condition:
                if not self._modalink_closed:
                    self._modalink_iterators.add(iterator)
                    return iterator
                # Hand ownership to close(), including iterators that finish
                # starting after its first snapshot. Do not race two shutdowns.
                self._modalink_pending.append(iterator)
            raise RuntimeError("PyTorch DataLoader was closed while creating an iterator")
        finally:
            with self._modalink_condition:
                self._modalink_creating -= 1
                self._modalink_condition.notify_all()

    def _modalink_shutdown(self, iterator: Any | None) -> None:
        with self._modalink_condition:
            known = set(self._modalink_iterators)
            known.update(self._modalink_pending)
            known.update(candidate for candidate in (iterator, self._iterator) if candidate is not None)
            reset_in_progress = self._modalink_creating > 0
        if reset_in_progress:
            # Supply native reset acknowledgements before terminating a worker
            # that may be blocked indefinitely in preprocess/collate.
            _resume_pending_resets(known)
        try:
            # Stop known workers before waiting for an in-flight persistent
            # reset: its current preprocess/collate may never return.
            _shutdown_iterators(known)
        finally:
            with self._modalink_condition:
                self._modalink_condition.wait_for(lambda: self._modalink_creating == 0)
                late = set(self._modalink_pending) - known
            _shutdown_iterators(late)
        self._modalink_pending.clear()
        self._modalink_iterators.clear()
        self._iterator = None
