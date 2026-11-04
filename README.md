# RegressionLens

**Is this prompt, model or settings change safe to ship?** RegressionLens answers that for
LLM-powered features with a verdict you can trust:

- runs your test suite **repeatedly** against a local open model (Ollama, llama.cpp, vLLM, LM Studio)
  and records every output, latency and the exact configuration;
- scores outputs with **exact-match, schema, numeric, regex or model-based graders**;
- compares two runs with **paired statistics** (per-case bootstrap CIs and an exact sign-flip test)
  and says whether a difference is real: **pass**, **fail**, or **inconclusive** (with how many more
  cases or repeats would decide it);
- flags **flaky tests** whose results change between identical runs;
- checks **model-based graders against human labels** (Cohen's κ) and warns when a verdict depends on
  an uncalibrated judge;
- gives **CI a pass/fail exit code** and a Markdown report for the PR;
- keeps **full history**: trends, per-case history, baselines, and one-click reproduction of any run.

Design and rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Measured results for the success criteria: [experiments/results/REPORT.md](experiments/results/REPORT.md).

## Quick start (no GPU needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "backend[server]"
cd examples/support-qa
```

```bash
rlens run --local -s suite.yaml -p prompts/v1.yaml -m models/simulated.yaml -r 5 --promote main
```

```bash
rlens run --local -s suite.yaml -p prompts/v2-shorter.yaml -m models/simulated.yaml -r 5 --baseline main
```

The second command compares a "cleaned-up" prompt with the baseline, prints something like this, and exits with code 1:

```
Verdict: FAIL
  • gate failed: schema = 0.825, required >= 0.85
  • pass_rate dropped: -15.3 pts, 95% CI [-23.3 pts, -7.3 pts], p = 0.0004883
```

`models/simulated.yaml` is a stand-in model with known accuracy, so everything works without a GPU.
To use a real local model, `ollama pull llama3.2:3b` and pass `-m models/llama3.2-3b.yaml`.

## Run the server and dashboard

```bash
RLENS_ADMIN_TOKEN=change-me docker compose up --build
```

Open http://localhost:8080 and sign in with the token. Add `--profile ollama` to also start an
Ollama container, or point model configs at `http://host.docker.internal:11434` to use the host's.

Without Docker, the server also runs as a single process on SQLite:

```bash
RLENS_ADMIN_TOKEN=demo-token rlens server --with-worker
```

To build the dashboard for that mode, run `npm ci && npm run build` in `dashboard/`. To fill a
server with demo data, run `python examples/seed_demo.py --url http://localhost:8080 --token demo-token`.

Then point the CLI at it:

```bash
export RLENS_URL=http://localhost:8080 RLENS_TOKEN=change-me
```

```bash
rlens run -s suite.yaml -p prompts/v1.yaml -m models/llama3.2-3b.yaml -r 5 --baseline main --report-md report.md
```

## Use it in CI

Copy [examples/github-actions/llm-regression.yml](examples/github-actions/llm-regression.yml) into
your repository. On pull requests it compares against the `main` baseline, comments the report on the
PR and gates the merge; on merges to `main` it moves the baseline. The CI runner never talks to your
model: it sends the suite, prompt and model config to RegressionLens, whose worker next to the GPU
does the generation.

Exit codes: `0` pass, `1` fail, `2` inconclusive (or 0/1, set by the suite's `inconclusive` policy),
`3` error.

## Writing a suite

```yaml
name: support-qa
verdict:
  non_inferiority_margin: 0.03     # a drop of up to 3 points still passes
  gates:
    - { metric: schema, min: 0.85 } # absolute threshold on the `schema` grader
cases:
  - key: return-final-sale
    critical: true                   # fails the verdict if it fails in most repeats
    input: { format: yesno, question: "Can I return a jacket marked final sale?" }
    expected: "no"
    graders:
      - { id: answer, type: choice, choices: ["yes", "no"] }
  - key: extract-late-1
    input: { format: json, question: "Order #A-1932 never arrived." }
    expected: { order_id: "A-1932", issue: late }
    graders:
      - { id: schema, type: json_schema, schema_file: schemas/order_issue.json }
      - { id: issue, type: json_field, path: issue }
```

Graders: `exact_match`, `contains`, `regex`, `choice`, `numeric`, `json_schema`, `json_field`,
`llm_judge` (rubric + pinned judge model). Prompts are Jinja2 templates over each case's `input`.
See [examples/support-qa](examples/support-qa) for the full example and
[§7 of the architecture](docs/ARCHITECTURE.md#7-graders) for every option.

## How the verdict works

For each test case, the candidate's mean score over its repeats is compared with the baseline's.
Statistics run on these per-case differences, so correlated repeats can't masquerade as extra
evidence.

- **Fail**: the drop is statistically significant (Holm-adjusted p < α), an absolute gate fails, or a
  critical case fails. On identical configurations this happens at most α/2 = 2.5% of the time.
- **Pass**: the 95% CI rules out a drop bigger than the tolerated margin.
- **Inconclusive**: the CI is too wide. The report says how many more cases or repeats would resolve
  it, and every run shows the smallest drop it can reliably detect.
- **Error**: too many generation or grader errors, or the runs aren't comparable (for example,
  grader versions changed).

Details: [§8 of the architecture](docs/ARCHITECTURE.md#8-statistics-engine).

## Development

```bash
pip install -e "backend[server,dev]"
```

```bash
cd backend && ruff check regressionlens tests && pytest -q
```

```bash
RLENS_TEST_DATABASE_URL=postgresql://rlens:rlens@localhost:5432/regressionlens_test pytest -q tests/test_server.py tests/test_queue.py
```

```bash
python experiments/success_criteria.py --quick
```

```
backend/regressionlens/   engine, stats, graders, providers, API, worker, CLI (one Python package)
backend/tests/            unit + full-stack tests (SQLite by default, Postgres with RLENS_TEST_DATABASE_URL)
dashboard/                React + Vite + TypeScript dashboard (built into the API's static folder)
examples/                 example suite, GitHub Actions workflow, demo seeding script
experiments/              success-criteria harness and its committed report
docs/ARCHITECTURE.md      architecture and design decisions
```
