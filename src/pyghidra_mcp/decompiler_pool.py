import queue
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ghidra.app.decompiler import DecompInterface


class DecompilerPool:
    def __init__(
        self,
        factory: Callable[[], "DecompInterface"],
        *,
        size: int = 1,
    ) -> None:
        self._factory = factory
        self._size = max(1, size)
        self._queue: queue.LifoQueue[DecompInterface] = queue.LifoQueue(maxsize=self._size)
        self._created: list[DecompInterface] = []
        self._created_lock = threading.Lock()
        self._available = threading.Condition(self._created_lock)
        self._creating = 0

    def _create(self) -> "DecompInterface":
        # _ensure_available reserves capacity before releasing the condition.
        # Factory calls may be slow, so run them without holding the pool lock.
        try:
            decompiler = self._factory()
        except BaseException:
            with self._available:
                self._creating -= 1
                self._available.notify()
            raise
        with self._available:
            self._creating -= 1
            self._created.append(decompiler)
        return decompiler

    def _ensure_available(self) -> "DecompInterface":
        with self._available:
            while True:
                try:
                    return self._queue.get_nowait()
                except queue.Empty:
                    if len(self._created) + self._creating < self._size:
                        self._creating += 1
                        break
                    # Condition.wait releases the pool lock. In particular,
                    # waiters must not prevent cache invalidation or a factory
                    # that is already running from publishing its result.
                    self._available.wait()
        return self._create()

    @contextmanager
    def acquire(self):
        decompiler = self._ensure_available()
        try:
            yield decompiler
        finally:
            with self._available:
                # Atomic reservations keep the return queue within capacity.
                # Fail rather than silently hang if that invariant is broken.
                self._queue.put_nowait(decompiler)
                self._available.notify()

    def invalidate_all(self) -> None:
        with self._created_lock:
            decompilers = list(self._created)

        for decompiler in decompilers:
            for method_name in ("flushCache", "resetDecompiler"):
                method = getattr(decompiler, method_name, None)
                if method is not None:
                    method()
                    break

    def dispose(self) -> None:
        with self._created_lock:
            decompilers = list(self._created)
            self._created.clear()

        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        for decompiler in decompilers:
            for method_name in ("dispose", "closeProgram"):
                method = getattr(decompiler, method_name, None)
                if method is not None:
                    method()
                    break
