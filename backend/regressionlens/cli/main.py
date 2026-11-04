"""``rlens``: run suites locally or against a RegressionLens server, and gate CI on the verdict."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from ..records import RunRecord
from ..report import print_console, render_markdown
from ..spec import ExecutionSpec, SpecError, load_model, load_prompt, load_suite

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console(stderr=True, width=None if sys.stderr.isatty() else 120)

EXIT = {"pass": 0, "fail": 1, "inconclusive": 2, "error": 3}
def local_dir() -> Path:
    return Path(os.environ.get("RLENS_HOME", ".rlens"))


def exit_code(verdict: str, inconclusive_policy: str) -> int:
    if verdict == "inconclusive":
        return {"fail": 1, "pass": 0}.get(inconclusive_policy, 2)
    return EXIT.get(verdict, 3)


def _load_specs(suite: Path, prompt: Path, model: Path):
    try:
        return load_suite(suite), load_prompt(prompt), load_model(model)
    except SpecError as exc:
        console.print(f"[red]spec error:[/] {exc}")
        raise typer.Exit(3) from exc


def _read_record(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _local_baseline(name_or_path: str) -> RunRecord | None:
    path = Path(name_or_path)
    if not path.exists():
        path = local_dir() / "baselines" / f"{name_or_path}.json"
    if not path.exists():
        console.print(f"[yellow]baseline {name_or_path!r} not found; running without a baseline[/]")
        return None
    return _read_record(path)


def _github_annotation(verdict: dict) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    level = {"fail": "error", "error": "error", "inconclusive": "warning"}.get(verdict["verdict"])
    if level:
        text = "; ".join(verdict.get("reasons", [])) or verdict["verdict"]
        print(f"::{level} title=RegressionLens {verdict['verdict']}::{text}")


@app.command()
def validate(
    suite: Path = typer.Option(..., "--suite", "-s", exists=True, dir_okay=False),
    prompt: Path | None = typer.Option(None, "--prompt", "-p", exists=True, dir_okay=False),
    model: Path | None = typer.Option(None, "--model", "-m", exists=True, dir_okay=False),
) -> None:
    """Check a suite (and optionally a prompt and model config) without running anything."""
    from ..graders import validate_suite

    try:
        s = load_suite(suite)
        validate_suite(s)
        console.print(f"[green]suite ok[/] {s.name}: {len(s.cases)} cases, graders {', '.join(s.grader_ids())}")
        console.print(f"  content hash {s.content_hash()}")
        if prompt:
            p = load_prompt(prompt)
            for case in s.cases:
                p.render(case.input)
            console.print(f"[green]prompt ok[/] {p.name} {p.content_hash()} (renders for every case)")
        if model:
            m = load_model(model)
            console.print(f"[green]model ok[/] {m.name}: {m.provider}/{m.model}")
    except SpecError as exc:
        console.print(f"[red]invalid:[/] {exc}")
        raise typer.Exit(3) from exc


@app.command()
def run(
    suite: Path = typer.Option(..., "--suite", "-s", exists=True, dir_okay=False, help="Suite YAML"),
    prompt: Path = typer.Option(..., "--prompt", "-p", exists=True, dir_okay=False, help="Prompt version YAML"),
    model: Path = typer.Option(..., "--model", "-m", exists=True, dir_okay=False, help="Model config YAML"),
    repeats: int = typer.Option(3, "--repeats", "-r", min=1, max=200),
    seed: int | None = typer.Option(None, "--seed", help="Run seed (random if omitted)"),
    baseline: str | None = typer.Option(None, "--baseline", "-b", help="Baseline name (or a run JSON file in local mode)"),
    fresh_baseline: bool = typer.Option(False, "--fresh-baseline", help="Re-run the baseline config now (server mode)"),
    local: bool = typer.Option(False, "--local", help="Run in-process instead of on a RegressionLens server"),
    url: str | None = typer.Option(None, "--url", envvar="RLENS_URL", help="RegressionLens server URL"),
    token: str | None = typer.Option(None, "--token", envvar="RLENS_TOKEN", help="API token"),
    project: str = typer.Option("default", "--project", envvar="RLENS_PROJECT"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the run record JSON here (local mode)"),
    report_md: Path | None = typer.Option(None, "--report-md", help="Write a Markdown report (for PR comments)"),
    promote: str | None = typer.Option(None, "--promote", help="After the run, set it as this baseline name"),
    timeout: int = typer.Option(3600, "--timeout", help="Seconds to wait for a server run"),
) -> None:
    """Run a suite against a prompt + model, compare with a baseline, and exit with the verdict.

    Exit codes: 0 pass, 1 fail, 2 inconclusive (or 0/1 per the suite's `inconclusive` policy), 3 error.
    """
    s, p, m = _load_specs(suite, prompt, model)
    execution = ExecutionSpec(repeats=repeats)
    ci_meta = _ci_meta()
    if not local and not url:
        console.print("[dim]no --url / RLENS_URL set: running locally[/]")
        local = True

    if local:
        from ..local import compare_records, run_local

        base = _local_baseline(baseline) if baseline else None
        with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console, transient=True) as bar:
            tasks: dict[str, int] = {}

            def on_progress(phase: str, done: int, total: int) -> None:
                if phase not in tasks:
                    tasks[phase] = bar.add_task(phase, total=total)
                bar.update(tasks[phase], completed=done)

            try:
                record = run_local(s, p, m, execution, run_seed=seed, progress=on_progress)
            except Exception as exc:
                console.print(f"[red]run failed:[/] {exc}")
                raise typer.Exit(3) from exc
        report = compare_records(record, base)
        record.verdict = report.to_dict()
        record.baseline_run_id = base.run_id if base else None
        record.ci_meta = ci_meta
        out = out or local_dir() / "runs" / f"{record.run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        if promote:
            _set_local_baseline(promote, out)
        summary, verdict, link = record.summary.model_dump(mode="json"), record.verdict, None
        console.print(f"[dim]run {record.run_id} saved to {out}[/]")
    else:
        from .client import Client, ClientError

        client = Client(url, token, project=project)
        try:
            result = client.run_and_wait(
                suite=s, prompt=p, model=m, execution=execution, baseline=baseline,
                fresh_baseline=fresh_baseline, ci_meta=ci_meta, timeout_s=timeout,
                on_status=lambda st: console.print(f"[dim]{st}[/]"),
            )
            if promote and result["verdict"]["verdict"] in {"pass", "inconclusive"}:
                client.promote(result["run_id"], promote, suite_name=s.name)
        except ClientError as exc:
            console.print(f"[red]server error:[/] {exc}")
            raise typer.Exit(3) from exc
        summary, verdict, link = result.get("summary"), result["verdict"], result.get("url")
        console.print(f"[dim]run {result['run_id']}: {link}[/]")

    print_console(summary, verdict, console=console)
    if report_md:
        report_md.write_text(render_markdown(summary, verdict, link), encoding="utf-8")
    _github_annotation(verdict)
    raise typer.Exit(exit_code(verdict["verdict"], s.verdict.inconclusive))


@app.command()
def compare(
    baseline: Path = typer.Argument(..., exists=True, dir_okay=False, help="Baseline run JSON"),
    candidate: Path = typer.Argument(..., exists=True, dir_okay=False, help="Candidate run JSON"),
    report_md: Path | None = typer.Option(None, "--report-md"),
) -> None:
    """Compare two local run records and print the verdict."""
    from ..local import compare_records

    cand = _read_record(candidate)
    report = compare_records(cand, _read_record(baseline)).to_dict()
    print_console(cand.summary.model_dump(mode="json") if cand.summary else None, report, console=console)
    if report_md:
        report_md.write_text(render_markdown(cand.summary.model_dump(mode="json") if cand.summary else None, report), encoding="utf-8")
    policy = cand.suite.get("verdict", {}).get("inconclusive", "warn")
    raise typer.Exit(exit_code(report["verdict"], policy))


def _set_local_baseline(name: str, run_file: Path) -> None:
    target = local_dir() / "baselines" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_file, target)
    console.print(f"[green]baseline {name!r} -> {run_file}[/]")


baseline_app = typer.Typer(help="Manage local baselines (server baselines: use the API or --promote).")
app.add_typer(baseline_app, name="baseline")


@baseline_app.command("set")
def baseline_set(name: str, run_file: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    _set_local_baseline(name, run_file)


@baseline_app.command("list")
def baseline_list() -> None:
    folder = local_dir() / "baselines"
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        record = _read_record(path)
        console.print(f"{path.stem}: {record.run_id} ({record.created_at:%Y-%m-%d %H:%M}) {record.fingerprint[:19]}")


@app.command()
def show(run_file: Path = typer.Argument(..., exists=True, dir_okay=False), as_json: bool = typer.Option(False, "--json")) -> None:
    """Print a local run record."""
    record = _read_record(run_file)
    if as_json:
        sys.stdout.write(record.model_dump_json(indent=2) + "\n")
        return
    print_console(record.summary.model_dump(mode="json") if record.summary else None, record.verdict, console=console)


def _ci_meta() -> dict:
    env = os.environ
    meta = {
        "git_sha": env.get("GITHUB_SHA") or env.get("CI_COMMIT_SHA"),
        "branch": env.get("GITHUB_HEAD_REF") or env.get("GITHUB_REF_NAME") or env.get("CI_COMMIT_REF_NAME"),
        "pr": env.get("GITHUB_PR_NUMBER") or env.get("CI_MERGE_REQUEST_IID"),
        "repository": env.get("GITHUB_REPOSITORY"),
        "ci": bool(env.get("CI")),
    }
    return {k: v for k, v in meta.items() if v}


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()


# --- server-side commands (need `pip install regressionlens[server]`) -------------------------


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8080, "--port"),
    with_worker: bool = typer.Option(False, "--with-worker", help="Also run a queue worker in this process"),
) -> None:
    """Start the API server (and optionally a worker) for RLENS_DATABASE_URL."""
    import threading

    import uvicorn

    from ..api.app import create_app
    from ..config import Settings

    settings = Settings()
    application = create_app(settings)
    if with_worker:
        from ..worker import Worker

        def start_worker() -> None:
            from ..db.migrate import upgrade

            if settings.auto_migrate:
                upgrade(settings.database_url)
            Worker(application.state.db, settings).run_forever()

        threading.Thread(target=start_worker, daemon=True, name="rlens-worker").start()
    uvicorn.run(application, host=host, port=port, log_level="info")


@app.command()
def worker() -> None:
    """Run a queue worker (executes, grades and analyzes runs)."""
    from ..worker import main as worker_main

    worker_main()


@app.command()
def migrate(database_url: str | None = typer.Option(None, "--database-url", envvar="RLENS_DATABASE_URL")) -> None:
    """Apply database migrations."""
    from ..config import Settings
    from ..db.migrate import upgrade

    upgrade(database_url or Settings().database_url)
    console.print("[green]database is up to date[/]")


token_app = typer.Typer(help="Manage API tokens (talks to the database directly).")
app.add_typer(token_app, name="token")


@token_app.command("create")
def token_create(
    project: str = typer.Option("default", "--project"),
    name: str = typer.Option("", "--name"),
    scopes: str = typer.Option("read,run", "--scopes", help="Comma-separated: read, run, admin"),
) -> None:
    """Create an API token and print it once."""
    from .. import services
    from ..config import Settings
    from ..db.session import Database

    db = Database(Settings().database_url)
    with db.session() as s:
        token = services.create_token(s, services.ensure_project(s, project), name, scopes)
    sys.stdout.write(token + "\n")
