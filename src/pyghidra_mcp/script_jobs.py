"""In-memory async job registry for `run_*_script_async` tools.

MCP requests have a 2-minute timeout; long-running Ghidra scripts need to
return immediately with a job handle and let the caller poll for the result.
This module owns that small piece of state.

Single worker thread by design — Ghidra mutations are not safe to run
concurrently across transactions. Submissions queue up behind whatever job
is currently running.
"""

import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

logger = logging.getLogger(__name__)


# Bound on retained jobs (completed + in-flight). Old entries are evicted
# FIFO so polling against an unknown id eventually returns None.
MAX_RETAINED_JOBS = 100

ScriptJobState = Literal["queued", "running", "completed", "failed"]


class ScriptJob:
    """Mutable record of one async script execution.

    Field timestamps are seconds-since-epoch (``time.time()``); ``None``
    until the corresponding lifecycle event happens. Result fields stay
    ``None`` until ``status`` reaches a terminal state.
    """

    __slots__ = (
        "completed_at",
        "error",
        "job_id",
        "program_name",
        "result",
        "started_at",
        "status",
        "submitted_at",
    )

    def __init__(self, job_id: str, program_name: str) -> None:
        self.job_id: str = job_id
        self.program_name: str = program_name
        self.status: ScriptJobState = "queued"
        self.submitted_at: float = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.result: dict[str, Any] | None = None
        self.error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render the job as a flat dict for response models.

        Result fields (stdout/stderr/...) are spread to the top level when
        the job has completed; on failure ``error`` is populated and result
        fields stay ``None``.
        """
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "program_name": self.program_name,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stdout": None,
            "stderr": None,
            "result_repr": None,
            "committed": None,
            "error": self.error,
        }
        if self.result is not None:
            payload["stdout"] = self.result.get("stdout")
            payload["stderr"] = self.result.get("stderr")
            payload["result_repr"] = self.result.get("result_repr")
            payload["committed"] = self.result.get("committed")
            # ``error`` from the script's own try/except (run_script returns it
            # in the result dict) takes precedence over the registry-level None.
            if self.error is None:
                payload["error"] = self.result.get("error")
        return payload


class ScriptJobRegistry:
    """In-memory registry of script jobs with a single-worker executor.

    Jobs run serially so concurrent Ghidra transactions can't corrupt each
    other. The registry is cheap to construct and shuts down its executor
    on ``close()``.
    """

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, ScriptJob] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pyghidra-mcp-script-job"
        )

    def submit(self, program_name: str, fn: Callable[[], dict]) -> ScriptJob:
        """Register a new job and queue it for execution.

        ``fn`` is a zero-arg callable that returns the script's result dict
        (the same shape as the synchronous ``run_script`` / ``run_inline_script``
        return value). Exceptions raised by ``fn`` are caught and reported via
        ``ScriptJob.error``.
        """
        job_id = uuid.uuid4().hex
        job = ScriptJob(job_id, program_name)
        with self._lock:
            self._jobs[job_id] = job
            while len(self._jobs) > MAX_RETAINED_JOBS:
                evicted_id, _ = self._jobs.popitem(last=False)
                logger.debug("Evicted oldest script job: %s", evicted_id)
        self._executor.submit(self._run, job, fn)
        return job

    def get(self, job_id: str) -> ScriptJob | None:
        """Look up a job by id. Returns ``None`` if unknown or evicted."""
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job: ScriptJob, fn: Callable[[], dict]) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = time.time()
        try:
            result = fn()
            with self._lock:
                job.result = result
                job.status = "completed"
                job.completed_at = time.time()
        except Exception as e:
            logger.exception("Async script job %s failed", job.job_id)
            with self._lock:
                job.error = str(e)
                job.status = "failed"
                job.completed_at = time.time()

    def close(self) -> None:
        """Shut down the worker. Pending jobs are abandoned."""
        self._executor.shutdown(wait=False)
