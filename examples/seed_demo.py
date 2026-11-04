"""Fill a RegressionLens server with demo runs using the simulated model (no GPU needed).

    rlens server --with-worker          # in one terminal (RLENS_ADMIN_TOKEN=demo-token)
    python examples/seed_demo.py --url http://localhost:8080 --token demo-token

Creates: a baseline, an A/A rerun, a prompt regression, a fix, a reproduction, and a suite with
an LLM judge plus human labels so the calibration view has data.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from regressionlens.cli.client import Client
from regressionlens.spec import (
    ExecutionSpec,
    ModelSpec,
    PromptSpec,
    SuiteSpec,
    load_model,
    load_prompt,
    load_suite,
)

HERE = Path(__file__).resolve().parent / "support-qa"


def judged_suite(n: int = 40) -> SuiteSpec:
    return SuiteSpec.model_validate(
        {
            "name": "support-tone",
            "description": "Free-form replies graded by a (simulated) judge model.",
            "graders": [
                {
                    "id": "judge",
                    "type": "llm_judge",
                    "rubric": "Pass if the reply gives the reference answer politely and without inventing policy.",
                    "judge_model": {"provider": "simulated", "model": "sim-judge", "params": {"temperature": 0}, "simulation": {"judge_accuracy": 0.88}},
                }
            ],
            "cases": [
                {"key": f"reply-{i:02d}", "tags": ["tone"], "input": {"format": "free", "question": f"Customer message #{i}"}, "expected": f"Reply {i}"}
                for i in range(n)
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--token", default="demo-token")
    args = parser.parse_args()
    client = Client(args.url, args.token)
    suite = load_suite(HERE / "suite.yaml")
    sim = load_model(HERE / "models" / "simulated.yaml")
    v1, v2 = load_prompt(HERE / "prompts" / "v1.yaml"), load_prompt(HERE / "prompts" / "v2-shorter.yaml")
    ex = ExecutionSpec(repeats=5)

    def run(prompt: PromptSpec, model: ModelSpec = sim, baseline: str | None = "main", seed: int | None = None, branch: str = "main", s=suite):
        result = client.run_and_wait(suite=s, prompt=prompt, model=model, execution=ex, baseline=baseline, run_seed=seed, ci_meta={"branch": branch})
        print(f"{result['run_id']}: {result['verdict']['verdict']}")
        return result

    first = run(v1, baseline=None, seed=1)
    client.promote(first["run_id"], "main")
    run(v1, seed=2, branch="tidy-readme")  # nothing changed: A/A
    run(v2, seed=3, branch="shorter-prompt")  # the regression
    better = sim.model_copy(update={"name": "simulated-v2", "simulation": {**(sim.simulation or {}), "accuracy": 0.86}})
    fixed = run(v1, model=better, seed=4, branch="model-upgrade")
    client.promote(fixed["run_id"], "main")
    reproduced = client.post(f"/api/v1/runs/{first['run_id']}/rerun")
    client.wait(reproduced["run_id"], timeout_s=600)

    tone = judged_suite()
    judged = run(PromptSpec(name="support-tone-v1", template="{{ question }}"), baseline=None, seed=5, s=tone)
    trials = {t["trial_id"]: t for t in client.get(f"/api/v1/runs/{judged['run_id']}/trials", limit=2000)}
    queue = client.get("/api/v1/suites/support-tone/graders/judge/label-queue", labeler="demo", n=40)
    rng = random.Random(0)
    for item in queue:
        truth = trials[item["trial_id"]]["output"] == item["expected"]
        if rng.random() < 0.05:  # humans disagree with the truth now and then too
            truth = not truth
        client.post("/api/v1/labels", {"trial_id": item["trial_id"], "grader_id": "judge", "label": 1.0 if truth else 0.0, "labeler": "demo"})
    report = client.get("/api/v1/suites/support-tone/graders/judge/calibration")["current"]
    print(f"judge agreement: kappa={report['kappa']:.2f} on {report['n']} labels ({report['trust']})")


if __name__ == "__main__":
    main()
