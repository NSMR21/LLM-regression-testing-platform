from typer.testing import CliRunner

from regressionlens.cli.main import app, exit_code
from regressionlens.engine import plan_trials
from regressionlens.local import compare_records, run_local
from regressionlens.records import RunRecord
from regressionlens.spec import ExecutionSpec, SuiteSpec

from .conftest import EXAMPLE, make_suite, sim_model


def test_plan_trials_is_deterministic_and_shuffled(suite):
    ex = ExecutionSpec(repeats=3)
    a, b = plan_trials(suite, ex, 11), plan_trials(suite, ex, 11)
    assert [(p.case_key, p.repeat_idx, p.seed) for p in a] == [(p.case_key, p.repeat_idx, p.seed) for p in b]
    assert len(a) == 90 and len({(p.case_key, p.repeat_idx) for p in a}) == 90
    assert [p.case_key for p in a[:5]] != [c.key for c in suite.cases[:5]]
    assert plan_trials(suite, ExecutionSpec(repeats=3, seed_policy="none"), 1)[0].seed is None


def test_run_local_records_everything(suite, prompt, execution):
    record = run_local(suite, prompt, sim_model(accuracy=0.8), execution, run_seed=5)
    assert len(record.trials) == 120
    trial = record.trials[0]
    assert trial.messages[0] == {"role": "system", "content": "Be brief."}
    assert trial.latency_ms and trial.seed is not None and "exact" in trial.scores
    assert record.summary.metrics["pass_rate"] == record.summary.metrics["exact"]
    assert 0.5 < record.summary.metrics["pass_rate"] < 1.0
    assert record.snapshot["model"]["resolved"]["digest"].startswith("sim:")
    assert record.snapshot["graders"]["versions"] == {"exact": "exact_match@1.0"}
    assert record.summary.mde is not None
    RunRecord.model_validate_json(record.model_dump_json())


def test_same_seed_reproduces_outputs_and_fingerprint(suite, prompt, execution):
    a = run_local(suite, prompt, sim_model(), execution, run_seed=9)
    b = run_local(suite, prompt, sim_model(), execution, run_seed=9)
    c = run_local(suite, prompt, sim_model(), execution, run_seed=10)
    def key(r):
        return [(t.case_key, t.repeat_idx, t.output) for t in sorted(r.trials, key=lambda t: (t.case_key, t.repeat_idx))]
    assert key(a) == key(b)
    assert key(a) != key(c)
    assert a.fingerprint == b.fingerprint == c.fingerprint
    assert run_local(suite, prompt, sim_model(shift=-0.1), execution, run_seed=9).fingerprint != a.fingerprint


def test_transient_errors_are_retried_and_counted(suite, prompt):
    flaky_server = sim_model(error_rate=0.3)
    record = run_local(suite, prompt, flaky_server, ExecutionSpec(repeats=2, retries=3), run_seed=1)
    assert record.summary.error_rate < 0.05
    assert max(t.attempts for t in record.trials) > 1
    no_retry = run_local(suite, prompt, flaky_server, ExecutionSpec(repeats=2, retries=0), run_seed=1)
    assert no_retry.summary.error_rate > 0.15
    assert compare_records(no_retry, None).verdict == "error"


def test_regression_detected_end_to_end(prompt):
    suite = make_suite(60, non_inferiority_margin=0.05)
    ex = ExecutionSpec(repeats=5)
    base = run_local(suite, prompt, sim_model(accuracy=0.85), ex, run_seed=1)
    same = run_local(suite, prompt, sim_model(accuracy=0.85), ex, run_seed=2)
    worse = run_local(suite, prompt, sim_model(accuracy=0.85, shift=-0.15), ex, run_seed=3)
    assert compare_records(same, base).verdict in {"pass", "inconclusive"}
    report = compare_records(worse, base)
    assert report.verdict == "fail"
    assert "model.simulation.shift" in report.config_diff


def test_judge_grader_runs_in_grading_phase(prompt):
    suite = SuiteSpec.model_validate(
        {
            "name": "judged",
            "graders": [
                {
                    "id": "judge",
                    "type": "llm_judge",
                    "rubric": "Correct answer?",
                    "judge_model": {"provider": "simulated", "model": "j", "params": {"temperature": 0}, "simulation": {"judge_accuracy": 1.0}},
                }
            ],
            "cases": [{"key": f"k{i}", "input": {"q": str(i)}, "expected": f"a{i}"} for i in range(10)],
        }
    )
    record = run_local(suite, prompt, sim_model(accuracy=0.7), ExecutionSpec(repeats=2), run_seed=4)
    assert "judge" in record.snapshot["graders"]["judges"]
    for trial in record.trials:
        assert trial.scores["judge"].passed == (trial.output == f"a{trial.case_key[1:]}")


def test_cli_local_flow(tmp_path):
    runner = CliRunner()
    args = ["run", "--local", "-s", str(EXAMPLE / "suite.yaml"), "-m", str(EXAMPLE / "models" / "simulated.yaml"), "-r", "5"]
    first = runner.invoke(app, [*args, "-p", str(EXAMPLE / "prompts" / "v1.yaml"), "--seed", "1", "--promote", "main", "-o", str(tmp_path / "a.json")])
    assert first.exit_code == 0, first.output
    report = tmp_path / "r.md"
    second = runner.invoke(
        app, [*args, "-p", str(EXAMPLE / "prompts" / "v2-shorter.yaml"), "--seed", "2", "--baseline", "main", "-o", str(tmp_path / "b.json"), "--report-md", str(report)]
    )
    assert second.exit_code == 1, second.output
    assert "FAIL" in report.read_text()
    compared = runner.invoke(app, ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert compared.exit_code == 1
    assert runner.invoke(app, ["validate", "-s", str(EXAMPLE / "suite.yaml")]).exit_code == 0


def test_exit_codes():
    assert exit_code("pass", "warn") == 0
    assert exit_code("fail", "warn") == 1
    assert exit_code("inconclusive", "warn") == 2
    assert exit_code("inconclusive", "fail") == 1
    assert exit_code("inconclusive", "pass") == 0
    assert exit_code("error", "warn") == 3
