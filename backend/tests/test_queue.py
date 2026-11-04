import threading

from sqlalchemy import select

from regressionlens.db import queue
from regressionlens.db.models import Job


def test_claim_complete_fail_delay(stack):
    with stack.db.session() as s:
        queue.enqueue_many(s, "execute", [{"n": i} for i in range(5)], queue="exec:a", max_attempts=2)
        queue.enqueue(s, "grade", {"n": 99}, queue="grade")
    with stack.db.session() as s:
        assert sorted(queue.queued_queues(s)) == ["exec:a", "grade"]
        jobs = queue.claim(s, "exec:a", "w1", limit=3)
    assert [j.payload["n"] for j in jobs] == [0, 1, 2]
    with stack.db.session() as s:
        queue.complete(s, jobs[0].id)
        assert queue.fail(s, jobs[1], "boom") is True  # attempt 1 of 2: retried
        queue.delay(s, jobs[2], 0)
    with stack.db.session() as s:
        again = queue.claim(s, "exec:a", "w1", limit=10)
    # job 1 is waiting for its backoff (10 ms); job 2 was put back without spending an attempt
    assert {j.payload["n"] for j in again} >= {2, 3, 4}
    delayed = next(j for j in again if j.payload["n"] == 2)
    assert delayed.attempts == 1
    with stack.db.session() as s:
        failed_once = s.get(Job, jobs[1].id)
        assert failed_once.status == "queued" and failed_once.last_error == "boom"


def test_permanent_failure_and_stale_requeue(stack):
    with stack.db.session() as s:
        queue.enqueue(s, "execute", {}, queue="q", max_attempts=1)
    with stack.db.session() as s:
        [job] = queue.claim(s, "q", "w")
        assert queue.fail(s, job, "fatal") is False
    with stack.db.session() as s:
        assert s.get(Job, job.id).status == "failed"
        queue.enqueue(s, "execute", {}, queue="q2")
    with stack.db.session() as s:
        [running] = queue.claim(s, "q2", "dead-worker")
    with stack.db.session() as s:
        assert queue.requeue_stale(s, older_than_s=3600) == 0
        assert queue.requeue_stale(s, older_than_s=-1) == 1
        assert s.get(Job, running.id).status == "queued"


def test_concurrent_claims_never_overlap(stack):
    with stack.db.session() as s:
        queue.enqueue_many(s, "execute", [{"n": i} for i in range(200)], queue="exec:race")
    claimed: list[int] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        while True:
            with stack.db.session() as s:
                jobs = queue.claim(s, "exec:race", name, limit=7)
            if not jobs:
                return
            with lock:
                claimed.extend(j.id for j in jobs)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == 200 and len(set(claimed)) == 200
    with stack.db.session() as s:
        assert set(s.scalars(select(Job.status).where(Job.queue == "exec:race"))) == {"running"}
