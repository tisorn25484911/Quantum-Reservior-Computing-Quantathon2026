"""jobs.py -- in-process job registry for runs that take longer than a request.

A reservoir sweep is seconds, not milliseconds, and blocking a request for that
long makes the UI feel broken and risks a proxy timeout. Jobs run on a worker
thread; the route returns an id immediately and the page polls for progress.

Deliberately no broker. A single-process registry plus JSON on disk is the
right weight for a demo: nothing to install, nothing to keep running alongside
the app. The trade-off is explicit -- in-flight jobs die with the process,
while *finished* ones survive because they are written to ``results/``.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..config import RESULTS_DIR

MAX_LIVE_JOBS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = "queued"
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None
    traceback: str | None = None

    def public(self, include_result: bool = False) -> dict:
        d = {"id": self.id, "kind": self.kind, "label": self.label,
             "status": self.status, "progress": round(self.progress, 3),
             "message": self.message, "created_at": self.created_at,
             "finished_at": self.finished_at, "error": self.error}
        if include_result:
            d["result"] = self.result
        return d


class JobRegistry:
    """Thread-safe registry. One lock; every mutation goes through it."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------
    def submit(self, kind: str, label: str,
               fn: Callable[..., dict], *args: Any, **kwargs: Any) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        threading.Thread(target=self._run, args=(job, fn, args, kwargs),
                         daemon=True).start()
        return job

    def _run(self, job: Job, fn, args, kwargs) -> None:
        def progress(frac: float, msg: str) -> None:
            with self._lock:
                job.progress = float(frac)
                job.message = str(msg)

        with self._lock:
            job.status, job.message = "running", "starting"
        try:
            result = fn(*args, progress=progress, **kwargs)
            self._persist(job.id, result)
            with self._lock:
                job.result, job.status = result, "done"
                job.progress, job.message = 1.0, "done"
                job.finished_at = _now()
        except Exception as exc:                       # surfaced, not swallowed
            with self._lock:
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
                job.message = "failed"
                job.finished_at = _now()

    # -- access ---------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        # Not in memory: it may be a finished run from an earlier process.
        return self._load(job_id)

    def list(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.public() for j in jobs[:limit]]

    def _evict_locked(self) -> None:
        """Drop the oldest finished jobs once the registry is full.

        Only finished ones are eligible: evicting a running job would orphan a
        thread that is still writing. Their results are already on disk, so
        eviction loses nothing permanent.
        """
        if len(self._jobs) <= MAX_LIVE_JOBS:
            return
        finished = sorted((j for j in self._jobs.values()
                           if j.status in ("done", "error")),
                          key=lambda j: j.created_at)
        for j in finished[:len(self._jobs) - MAX_LIVE_JOBS]:
            self._jobs.pop(j.id, None)

    # -- persistence ----------------------------------------------------
    def _persist(self, job_id: str, result: dict) -> None:
        path = RESULTS_DIR / f"{job_id}.json"
        tmp = path.with_suffix(".json.tmp")
        # Write-then-rename so a reader never sees a half-written file.
        tmp.write_text(json.dumps(result, indent=2, default=str))
        tmp.replace(path)

    def _load(self, job_id: str) -> Job | None:
        path = RESULTS_DIR / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        # Recover the run kind from distinctive keys so a restored result
        # renders with the right template after a process restart.
        if "max_delay" in result:
            kind = "memory"
        elif "n_free" in result:
            kind = "freerun"
        elif "useful_horizons" in result:
            kind = "anomaly"
        elif "nrmse" in result:
            kind = "sweep"
        else:
            kind = "forecast"
        label = str(result.get("config", {}).get("dataset", job_id))
        return Job(id=job_id, kind=kind, label=label, status="done",
                   progress=1.0, message="done (restored from disk)",
                   result=result, finished_at=_now())


registry = JobRegistry()
