"""A small job queue in a Postgres table.

Jobs are claimed with ``SELECT ... FOR UPDATE SKIP LOCKED`` so several workers never take the
same job. Enqueueing happens in the caller's transaction, so a run and its jobs appear together
or not at all. On SQLite (dev/test) the row lock is skipped and the conditional UPDATE alone
prevents double claims.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..util import utcnow
from .models import Job

RETRY_BASE_S = float(os.environ.get("RLENS_RETRY_BASE_SECONDS", "1"))


@dataclass
class ClaimedJob:
    id: int
    queue: str
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


def enqueue(
    session: Session,
    kind: str,
    payload: dict[str, Any],
    queue: str = "default",
    run_after: datetime | None = None,
    max_attempts: int = 3,
) -> Job:
    job = Job(queue=queue, kind=kind, payload=payload, run_after=run_after or utcnow(), max_attempts=max_attempts)
    session.add(job)
    return job


def enqueue_many(session: Session, kind: str, payloads: list[dict[str, Any]], queue: str, max_attempts: int = 3) -> None:
    now = utcnow()
    session.add_all([Job(queue=queue, kind=kind, payload=p, run_after=now, max_attempts=max_attempts) for p in payloads])


def queued_queues(session: Session, prefix: str = "") -> list[str]:
    stmt = select(Job.queue).where(Job.status == "queued", Job.run_after <= utcnow()).distinct()
    if prefix:
        stmt = stmt.where(Job.queue.like(f"{prefix}%"))
    return list(session.scalars(stmt))


def claim(session: Session, queue: str, worker_id: str, limit: int = 1) -> list[ClaimedJob]:
    """Claim up to ``limit`` ready jobs from one queue. Commits."""
    if limit <= 0:
        return []
    now = utcnow()
    stmt = (
        select(Job.id)
        .where(Job.queue == queue, Job.status == "queued", Job.run_after <= now)
        .order_by(Job.id)
        .limit(limit)
    )
    if session.get_bind().dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    ids = list(session.scalars(stmt))
    claimed: list[ClaimedJob] = []
    for job_id in ids:
        result = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "queued")
            .values(status="running", locked_by=worker_id, locked_at=now, attempts=Job.attempts + 1)
        )
        if result.rowcount == 1:
            job = session.get(Job, job_id)
            claimed.append(ClaimedJob(job.id, job.queue, job.kind, dict(job.payload), job.attempts, job.max_attempts))
    session.commit()
    return claimed


def complete(session: Session, job_id: int) -> None:
    session.execute(update(Job).where(Job.id == job_id).values(status="done", finished_at=utcnow(), last_error=None))


def fail(session: Session, job: ClaimedJob, error: str) -> bool:
    """Record a failure. Returns True if the job will be retried."""
    retry = job.attempts < job.max_attempts
    values: dict[str, Any] = {"last_error": error[:4000], "locked_by": None, "locked_at": None}
    if retry:
        values.update(status="queued", run_after=utcnow() + timedelta(seconds=min(60.0, RETRY_BASE_S * 2 ** (job.attempts - 1))))
    else:
        values.update(status="failed", finished_at=utcnow())
    session.execute(update(Job).where(Job.id == job.id).values(**values))
    return retry


def delay(session: Session, job: ClaimedJob, seconds: float) -> None:
    """Put a job back without counting the attempt (e.g. waiting for a baseline run)."""
    session.execute(
        update(Job)
        .where(Job.id == job.id)
        .values(status="queued", run_after=utcnow() + timedelta(seconds=seconds), attempts=Job.attempts - 1, locked_by=None, locked_at=None)
    )


def requeue_stale(session: Session, older_than_s: float) -> int:
    """Jobs whose worker died mid-flight go back to the queue."""
    cutoff = utcnow() - timedelta(seconds=older_than_s)
    result = session.execute(
        update(Job).where(Job.status == "running", Job.locked_at < cutoff).values(status="queued", locked_by=None, locked_at=None)
    )
    return result.rowcount or 0


def cancel_run_jobs(session: Session, run_id: str) -> int:
    result = session.execute(
        update(Job)
        .where(Job.status == "queued", Job.payload["run_id"].as_string() == run_id)
        .values(status="done", finished_at=utcnow(), last_error="cancelled")
    )
    return result.rowcount or 0


def counts(session: Session) -> dict[str, int]:
    rows = session.execute(select(Job.status, func.count()).group_by(Job.status)).all()
    return {status: n for status, n in rows}
