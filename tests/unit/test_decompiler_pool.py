import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from pyghidra_mcp.decompiler_pool import DecompilerPool


@pytest.mark.parametrize("size", [1, 2])
def test_concurrent_cold_start_reserves_capacity_before_creating(size):
    factory_entered = threading.Event()
    release_factories = threading.Event()
    counts_lock = threading.Lock()
    factory_calls = 0
    active = 0
    peak_active = 0

    def factory():
        nonlocal factory_calls
        with counts_lock:
            factory_calls += 1
            factory_entered.set()
        assert release_factories.wait(3)
        return Mock()

    pool = DecompilerPool(factory, size=size)
    all_attempted = threading.Event()
    attempted = set()
    original_get = pool._queue.get_nowait

    def observe_attempt():
        try:
            return original_get()
        finally:
            with counts_lock:
                attempted.add(threading.get_ident())
                if len(attempted) == 12:
                    all_attempted.set()

    pool._queue.get_nowait = observe_attempt
    start = threading.Barrier(12)
    results = []
    errors = []

    def worker():
        nonlocal active, peak_active
        try:
            start.wait(3)
            with pool.acquire() as decompiler:
                with counts_lock:
                    active += 1
                    peak_active = max(peak_active, active)
                with counts_lock:
                    active -= 1
            results.append(decompiler)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(12)]
    for thread in threads:
        thread.start()
    try:
        assert factory_entered.wait(3)
        assert all_attempted.wait(3)
        calls_while_factories_blocked = factory_calls
    finally:
        release_factories.set()
        deadline = time.monotonic() + 3
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        blocked_threads = sum(thread.is_alive() for thread in threads)
        # Also let the regression test terminate against the old broken pool.
        while any(thread.is_alive() for thread in threads) and time.monotonic() < deadline + 1:
            try:
                pool._queue.get(timeout=0.01)
            except queue.Empty:
                pass

    assert not errors
    assert blocked_threads == 0
    assert calls_while_factories_blocked <= size
    assert len(results) == 12
    assert factory_calls <= size
    assert len({id(result) for result in results}) <= size
    assert peak_active <= size
    assert pool._creating == 0


def test_factory_failure_releases_reserved_capacity():
    first_factory_entered = threading.Event()
    fail_first_factory = threading.Event()
    calls = 0
    decompiler = Mock()

    def factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            first_factory_entered.set()
            assert fail_first_factory.wait(3)
            raise ValueError("factory failed")
        return decompiler

    pool = DecompilerPool(factory, size=1)

    def acquire_once():
        with pool.acquire() as result:
            return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(acquire_once)
        assert first_factory_entered.wait(3)
        second = executor.submit(acquire_once)
        fail_first_factory.set()
        with pytest.raises(ValueError, match="factory failed"):
            first.result(timeout=3)
        assert second.result(timeout=3) is decompiler
    assert pool._creating == 0


def test_exhausted_pool_does_not_hold_lock_needed_for_invalidation():
    decompiler = Mock()
    pool = DecompilerPool(lambda: decompiler, size=1)
    waiter_started = threading.Event()

    def acquire_once():
        waiter_started.set()
        with pool.acquire() as result:
            return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        with pool.acquire():
            waiting = executor.submit(acquire_once)
            assert waiter_started.wait(3)
            invalidated = executor.submit(pool.invalidate_all)
            invalidated.result(timeout=3)
        assert waiting.result(timeout=3) is decompiler
    decompiler.flushCache.assert_called_once()


def test_body_exception_returns_decompiler_for_reuse():
    decompiler = Mock()
    factory = Mock(return_value=decompiler)
    pool = DecompilerPool(factory, size=1)
    with pytest.raises(ValueError, match="body failed"), pool.acquire():
        raise ValueError("body failed")
    with pool.acquire() as result:
        assert result is decompiler
    factory.assert_called_once()


def test_dispose_closes_each_created_decompiler_once():
    created = [Mock(), Mock()]
    pool = DecompilerPool(Mock(side_effect=created), size=2)
    with pool.acquire():
        with pool.acquire():
            pass
    pool.dispose()
    pool.dispose()
    for decompiler in created:
        decompiler.dispose.assert_called_once()
