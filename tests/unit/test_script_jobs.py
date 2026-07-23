"""Unit tests for ScriptJobRegistry — the in-memory job store used by
``run_script_async`` / ``run_inline_script_async`` / ``poll_script_job``.
"""

import threading
import time

from pyghidra_mcp.script_jobs import MAX_RETAINED_JOBS, ScriptJobRegistry


def _wait_for(predicate, timeout=2.0, interval=0.005):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"Timed out waiting for {predicate!r}")


def test_submit_assigns_unique_ids_and_returns_job():
    reg = ScriptJobRegistry()
    try:
        j1 = reg.submit("p", lambda: {"stdout": "a"})
        j2 = reg.submit("p", lambda: {"stdout": "b"})
        assert j1.job_id != j2.job_id
        assert j1.program_name == "p"
        assert j1.status in ("queued", "running", "completed")
    finally:
        reg.close()


def test_completed_job_has_result_payload_in_to_dict():
    reg = ScriptJobRegistry()
    try:
        job = reg.submit(
            "p",
            lambda: {
                "stdout": "hi",
                "stderr": "",
                "result_repr": "42",
                "committed": True,
                "error": None,
            },
        )
        _wait_for(lambda: job.status == "completed")
        d = job.to_dict()
        assert d["status"] == "completed"
        assert d["stdout"] == "hi"
        assert d["result_repr"] == "42"
        assert d["committed"] is True
        assert d["completed_at"] is not None
        assert d["started_at"] is not None
    finally:
        reg.close()


def test_failing_job_is_marked_failed_with_error_message():
    reg = ScriptJobRegistry()
    try:
        def boom():
            raise RuntimeError("script blew up")

        job = reg.submit("p", boom)
        _wait_for(lambda: job.status == "failed")
        d = job.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "script blew up"
        # Result fields stay None on failure.
        assert d["stdout"] is None
        assert d["committed"] is None
    finally:
        reg.close()


def test_get_returns_none_for_unknown_id():
    reg = ScriptJobRegistry()
    try:
        assert reg.get("does-not-exist") is None
    finally:
        reg.close()


def test_lru_eviction_drops_oldest_jobs_past_capacity():
    reg = ScriptJobRegistry()
    try:
        # Quick-completing payloads so the worker doesn't backlog.
        first_id = reg.submit("p", lambda: {"stdout": "first"}).job_id
        for _ in range(MAX_RETAINED_JOBS):
            reg.submit("p", lambda: {"stdout": "x"})
        # The oldest job should now be evicted.
        assert reg.get(first_id) is None
    finally:
        reg.close()


def test_jobs_serialize_under_single_worker():
    """Two jobs submitted back-to-back should run one after the other."""
    reg = ScriptJobRegistry()
    try:
        order: list[str] = []
        first_started = threading.Event()
        first_can_finish = threading.Event()

        def slow_first():
            order.append("first-start")
            first_started.set()
            first_can_finish.wait(timeout=2.0)
            order.append("first-end")
            return {"stdout": ""}

        def fast_second():
            order.append("second")
            return {"stdout": ""}

        reg.submit("p", slow_first)
        reg.submit("p", fast_second)
        first_started.wait(timeout=2.0)
        # While the first is parked, the second must NOT have run yet.
        assert "second" not in order
        first_can_finish.set()
        _wait_for(lambda: "second" in order)
        assert order == ["first-start", "first-end", "second"]
    finally:
        reg.close()
