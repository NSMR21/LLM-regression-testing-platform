import socket
import threading
import time

import uvicorn
from sqlalchemy import func, select
from typer.testing import CliRunner

from regressionlens.cli.main import app as cli_app
from regressionlens.db.models import Job, Run, Trial
from regressionlens.spec import SuiteSpec

from .helpers import EXAMPLE
from .server_fixtures import ADMIN, client_for, example_body


def start_and_finish(stack, body) -> dict:
    created = stack.http.post("/api/v1/runs", json=body)
    assert created.status_code == 202, created.text
    stack.worker.run_until_idle()
    run = stack.http.get(f"/api/v1/runs/{created.json()['run_id']}").json()
    assert run["status"] == "completed", run
    return run


def test_health_and_auth(stack):
    assert stack.http.get("/api/v1/health").json()["status"] == "ok"
    assert stack.http.get("/api/v1/me").json() == {"project": "default", "scopes": ["admin", "read", "run"]}
    anonymous = stack.http.get("/api/v1/runs", headers={"Authorization": ""})
    assert anonymous.status_code == 401
    assert stack.http.get("/api/v1/runs", headers={"Authorization": "Bearer nope"}).status_code == 401
    token = stack.http.post("/api/v1/tokens", json={"name": "ci-read", "scopes": "read"}).json()["token"]
    read_only = {"Authorization": f"Bearer {token}"}
    assert stack.http.get("/api/v1/runs", headers=read_only).status_code == 200
    assert stack.http.post("/api/v1/runs", json=example_body(), headers=read_only).status_code == 403
    other = stack.http.post("/api/v1/tokens", json={"project": "other-team", "scopes": "read,run"}).json()["token"]
    assert stack.http.get("/api/v1/me", headers={"Authorization": f"Bearer {other}"}).json()["project"] == "other-team"


def test_run_lifecycle(stack):
    run = start_and_finish(stack, example_body(repeats=2))
    assert run["progress"] == {"total": 60, "executed": 60, "graded": 60}
    assert run["kind"] == "run" and run["fingerprint"].startswith("sha256:")
    assert run["summary"]["n_trials"] == 60 and "pass_rate" in run["summary"]["metric_ci"]
    assert run["verdict"] in {"pass", "fail"}
    assert any("no baseline" in n for n in run["verdict_report"]["notes"])
    trials = stack.http.get(f"/api/v1/runs/{run['run_id']}/trials", params={"case_key": "express-cost"}).json()
    assert len(trials) == 2 and trials[0]["messages"][0]["role"] == "system"
    assert "answer" in trials[0]["scores"]
    verdict = stack.http.get(f"/api/v1/runs/{run['run_id']}/verdict").json()
    assert verdict["status"] == "completed" and verdict["url"] == f"http://rlens.test/runs/{run['run_id']}"
    with stack.db.session() as s:
        assert s.scalar(select(func.count()).select_from(Job).where(Job.status != "done")) == 0


def test_bad_specs_are_rejected(stack):
    body = example_body()
    body["suite"]["cases"][0]["graders"] = [{"id": "x", "type": "nope"}]
    response = stack.http.post("/api/v1/runs", json=body)
    assert response.status_code == 400 and "unknown grader type" in response.json()["detail"]
    body = example_body()
    body["prompt"]["template"] = "{{ missing_variable }}"
    assert stack.http.post("/api/v1/runs", json=body).status_code == 400
    body = example_body()
    body["model"] = {"name": "m", "provider": "ollama", "model": "x", "endpoint": "http://127.0.0.1:9"}
    unreachable = stack.http.post("/api/v1/runs", json=body)
    assert unreachable.status_code == 502 and "cannot resolve model" in unreachable.json()["detail"]


def test_baseline_regression_and_trends(stack):
    base = start_and_finish(stack, example_body(repeats=4, run_seed=1))
    promoted = stack.http.put("/api/v1/baselines/main", json={"run_id": base["run_id"]})
    assert promoted.status_code == 200
    worse = start_and_finish(stack, example_body("v2-shorter.yaml", repeats=4, baseline="main", run_seed=2))
    assert worse["baseline_run_id"] == base["run_id"]
    assert worse["verdict"] == "fail"
    assert any(m["name"] == "pass_rate" and m["status"] == "fail" for m in worse["verdict_report"]["metrics"])
    assert "prompt.template" in worse["verdict_report"]["config_diff"]
    trends = stack.http.get("/api/v1/suites/support-qa/trends").json()
    assert [p["run_id"] for p in trends["points"]] == [base["run_id"], worse["run_id"]]
    assert trends["points"][1]["config_changed"] is True
    assert trends["baseline_events"][0]["run_id"] == base["run_id"]
    history = stack.http.get("/api/v1/suites/support-qa/cases/express-cost/history").json()
    assert len(history["points"]) == 2 and history["points"][0]["repeats"] == 4
    assert stack.http.get("/api/v1/baselines").json()[0]["name"] == "main"
    suites = stack.http.get("/api/v1/suites").json()
    assert suites[0]["name"] == "support-qa" and suites[0]["runs"] == 2
    adhoc = stack.http.post("/api/v1/comparisons", json={"baseline_run_id": base["run_id"], "candidate_run_id": worse["run_id"]}).json()
    assert adhoc["verdict"] == "fail"
    missing = start_and_finish(stack, example_body(repeats=1, baseline="does-not-exist"))
    assert any("not found" in n for n in missing["verdict_report"]["notes"])


def test_fresh_baseline_runs_both_configs(stack):
    base = start_and_finish(stack, example_body(repeats=2))
    stack.http.put("/api/v1/baselines/main", json={"run_id": base["run_id"]})
    cand = start_and_finish(stack, example_body("v2-shorter.yaml", repeats=2, baseline="main", fresh_baseline=True))
    fresh = stack.http.get(f"/api/v1/runs/{cand['baseline_run_id']}").json()
    assert fresh["kind"] == "fresh_baseline" and fresh["source_run_id"] == base["run_id"]
    assert fresh["status"] == "completed" and fresh["prompt"] == "support-v1"
    assert cand["verdict_report"]["baseline_run_id"] == fresh["run_id"]


def test_rerun_reproduces_and_refuses_changed_weights(stack):
    original = start_and_finish(stack, example_body(repeats=2))
    response = stack.http.post(f"/api/v1/runs/{original['run_id']}/rerun")
    assert response.status_code == 202
    stack.worker.run_until_idle()
    again = stack.http.get(f"/api/v1/runs/{response.json()['run_id']}").json()
    assert again["kind"] == "rerun" and again["fingerprint"] == original["fingerprint"]
    assert again["run_seed"] == original["run_seed"]
    # Same seeds on a deterministic simulator: identical outputs, so the paired comparison is exactly zero.
    primary = next(m for m in again["verdict_report"]["metrics"] if m["name"] == "pass_rate")
    assert primary["delta"] == 0 and primary["status"] == "pass"
    assert not any("dropped" in r for r in again["verdict_report"]["reasons"])
    stack.providers.digest_override = "sha256:different"
    refused = stack.http.post(f"/api/v1/runs/{original['run_id']}/rerun")
    assert refused.status_code == 409 and "weights changed" in refused.json()["detail"]
    assert stack.http.post(f"/api/v1/runs/{original['run_id']}/rerun", json={"force": True}).status_code == 202


def test_rescore_reuses_outputs(stack):
    original = start_and_finish(stack, example_body(repeats=2))
    response = stack.http.post(f"/api/v1/runs/{original['run_id']}/rescore")
    assert response.status_code == 202, response.text
    stack.worker.run_until_idle()
    rescored = stack.http.get(f"/api/v1/runs/{response.json()['run_id']}").json()
    assert rescored["kind"] == "rescore" and rescored["status"] == "completed"
    assert rescored["summary"]["metrics"] == original["summary"]["metrics"]
    a = stack.http.get(f"/api/v1/runs/{original['run_id']}/trials").json()
    b = stack.http.get(f"/api/v1/runs/{rescored['run_id']}/trials").json()
    assert [t["output"] for t in a] == [t["output"] for t in b]


def test_cancel_stops_work(stack):
    created = stack.http.post("/api/v1/runs", json=example_body()).json()
    cancelled = stack.http.post(f"/api/v1/runs/{created['run_id']}/cancel").json()
    assert cancelled["status"] == "cancelled"
    stack.worker.run_until_idle()
    with stack.db.session() as s:
        assert s.scalar(select(func.count()).select_from(Trial).where(Trial.run_id == created["run_id"], Trial.status != "pending")) == 0


def test_job_failures_mark_run_errored(stack):
    created = stack.http.post("/api/v1/runs", json=example_body(repeats=1)).json()
    stack.providers.broken = True
    stack.worker.run_until_idle()
    run = stack.http.get(f"/api/v1/runs/{created['run_id']}").json()
    assert run["status"] == "errored" and "provider construction failed" in run["error"]


def _judge_suite(n: int = 40) -> dict:
    return SuiteSpec.model_validate(
        {
            "name": "judged",
            "graders": [
                {
                    "id": "judge",
                    "type": "llm_judge",
                    "rubric": "Does the response give the reference answer?",
                    "judge_model": {"provider": "simulated", "model": "sim-judge", "params": {"temperature": 0}, "simulation": {"judge_accuracy": 0.9}},
                }
            ],
            "cases": [{"key": f"k{i:02d}", "input": {"question": f"q{i}", "format": "free"}, "expected": f"a{i}"} for i in range(n)],
        }
    ).model_dump(mode="json")


def test_labels_and_calibration(stack):
    body = example_body(repeats=2)
    body["suite"] = _judge_suite()
    run = start_and_finish(stack, body)
    assert run["verdict_report"]["graders"]["judge"]["trust"] == "uncalibrated"
    queue_items = stack.http.get("/api/v1/suites/judged/graders/judge/label-queue", params={"labeler": "alice", "n": 60}).json()
    assert len(queue_items) == 60
    assert "passed" not in queue_items[0] and queue_items[0]["rubric"].startswith("Does the response")
    trials = {t["trial_id"]: t for t in stack.http.get(f"/api/v1/runs/{run['run_id']}/trials").json()}
    for item in queue_items:
        truth = trials[item["trial_id"]]["output"] == item["expected"]
        r = stack.http.post("/api/v1/labels", json={"trial_id": item["trial_id"], "grader_id": "judge", "label": 1.0 if truth else 0.0, "labeler": "alice"})
        assert r.status_code == 201, r.text
    report = stack.http.get("/api/v1/suites/judged/graders/judge/calibration").json()
    current = report["current"]
    assert report["current_version"] == "llm_judge@1.0" and current["n"] == 60
    assert 0.6 <= current["accuracy"] <= 1.0
    assert current["trust"] in {"calibrated", "weak"}
    assert stack.http.get("/api/v1/suites/judged/graders/judge/label-queue", params={"labeler": "alice", "n": 200}).json().__len__() == 20
    bad = stack.http.post("/api/v1/labels", json={"trial_id": queue_items[0]["trial_id"], "grader_id": "nope", "label": 1, "labeler": "alice"})
    assert bad.status_code == 400


def test_client_run_and_wait(stack, background_worker):
    from regressionlens.spec import ExecutionSpec, load_model, load_prompt, load_suite

    client = client_for(stack)
    result = client.run_and_wait(
        suite=load_suite(EXAMPLE / "suite.yaml"), prompt=load_prompt(EXAMPLE / "prompts" / "v1.yaml"),
        model=load_model(EXAMPLE / "models" / "simulated.yaml"), execution=ExecutionSpec(repeats=2), timeout_s=60,
    )
    assert result["verdict"]["verdict"] in {"pass", "fail"} and result["summary"]["n_cases"] == 30
    client.promote(result["run_id"], "main")
    assert client.get("/api/v1/baselines")[0]["run_id"] == result["run_id"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_cli_against_real_server(stack, background_worker):
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(stack.app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        runner = CliRunner()
        args = ["run", "--url", f"http://127.0.0.1:{port}", "--token", ADMIN, "-s", str(EXAMPLE / "suite.yaml"),
                "-m", str(EXAMPLE / "models" / "simulated.yaml"), "-r", "5"]
        first = runner.invoke(cli_app, [*args, "-p", str(EXAMPLE / "prompts" / "v1.yaml"), "--seed", "1", "--promote", "main"])
        assert first.exit_code == 0, first.output
        second = runner.invoke(cli_app, [*args, "-p", str(EXAMPLE / "prompts" / "v2-shorter.yaml"), "--seed", "2", "--baseline", "main"])
        assert second.exit_code == 1, second.output
        assert "FAIL" in second.output
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    with stack.db.session() as s:
        assert s.scalar(select(func.count()).select_from(Run)) == 2
