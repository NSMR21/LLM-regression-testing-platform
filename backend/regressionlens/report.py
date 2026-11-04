"""Human-readable renderings of a run and its verdict (console and Markdown)."""

from __future__ import annotations

from typing import Any

VERDICT_EMOJI = {"pass": "✅", "fail": "❌", "inconclusive": "⚠️", "error": "🛑"}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def _pts(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:+.1f}"


def render_markdown(summary: dict[str, Any] | None, verdict: dict[str, Any] | None, url: str | None = None) -> str:
    lines: list[str] = []
    v = (verdict or {}).get("verdict", "n/a")
    lines.append(f"## {VERDICT_EMOJI.get(v, '')} RegressionLens verdict: **{v.upper()}**")
    lines.append("")
    for reason in (verdict or {}).get("reasons", []):
        lines.append(f"- {reason}")
    for note in (verdict or {}).get("notes", []):
        lines.append(f"- _{note}_")
    metrics = (verdict or {}).get("metrics") or []
    if metrics:
        lines += ["", "| Metric | Baseline | Candidate | Δ (pts) | 95% CI (pts) | p | Status |", "|---|---|---|---|---|---|---|"]
        for m in metrics:
            name = f"**{m['name']}**" if m.get("gated") else m["name"]
            ci = f"[{_pts(m.get('ci_low'))}, {_pts(m.get('ci_high'))}]"
            p = "–" if m.get("p_adjusted") is None else f"{m['p_adjusted']:.3g}"
            lines.append(
                f"| {name} | {_pct(m.get('baseline'))} | {_pct(m.get('candidate'))} | {_pts(m.get('delta'))} | {ci} | {p} | {m['status']} |"
            )
    gates = (verdict or {}).get("gates") or []
    if gates:
        lines += ["", "| Gate | Value | Required | OK |", "|---|---|---|---|"]
        for g in gates:
            bound = f">= {g['min']}" if g.get("min") is not None else f"<= {g['max']}"
            value = "–" if g.get("value") is None else f"{g['value']:.3f}"
            lines.append(f"| {g['metric']} | {value} | {bound} | {'yes' if g['passed'] else 'no'} |")
    broken = [c for c in (verdict or {}).get("case_diff", []) if c["category"] == "broken"][:10]
    if broken:
        lines += ["", "<details><summary>Cases that broke</summary>", "", "| Case | Baseline | Candidate |", "|---|---|---|"]
        for c in broken:
            lines.append(f"| `{c['case_key']}` | {_pct(c['baseline'])} | {_pct(c['candidate'])} |")
        lines += ["", "</details>"]
    if summary:
        lines += [
            "",
            f"{summary['n_cases']} cases × {summary['repeats']} repeats · "
            f"flaky cases: {len(summary.get('flaky_cases', []))} · "
            f"error rate: {_pct(summary.get('error_rate'))} · "
            f"p50 latency: {summary.get('latency_p50_ms') and round(summary['latency_p50_ms'])} ms · "
            f"detectable drop (MDE): {_pct(summary.get('mde'))}",
        ]
    diff = (verdict or {}).get("config_diff") or {}
    changed = [k for k in diff if not k.startswith("model.resolved.details")][:8]
    if changed:
        lines += ["", "Changed since baseline: " + ", ".join(f"`{k}`" for k in changed)]
    if url:
        lines += ["", f"[Open in RegressionLens]({url})"]
    return "\n".join(lines) + "\n"


def print_console(summary: dict[str, Any] | None, verdict: dict[str, Any] | None, console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    if summary:
        table = Table(title="Run summary", show_header=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for name, value in sorted(summary.get("metrics", {}).items()):
            table.add_row(name, _pct(value))
        table.add_row("error rate", _pct(summary.get("error_rate")))
        table.add_row("flaky cases", str(len(summary.get("flaky_cases", []))))
        if summary.get("latency_p50_ms") is not None:
            table.add_row("latency p50 / p95", f"{summary['latency_p50_ms']:.0f} / {summary['latency_p95_ms']:.0f} ms")
        if summary.get("mde") is not None:
            table.add_row("detectable drop (MDE, 80% power)", _pct(summary.get("mde")))
        console.print(table)
    if verdict:
        metrics = verdict.get("metrics") or []
        if metrics:
            table = Table(title="Comparison with baseline")
            for col in ("Metric", "Baseline", "Candidate", "Δ pts", "95% CI pts", "p (adj)", "Status"):
                table.add_column(col, justify="right" if col not in {"Metric", "Status"} else "left")
            for m in metrics:
                table.add_row(
                    ("* " if m.get("gated") else "  ") + m["name"],
                    _pct(m.get("baseline")),
                    _pct(m.get("candidate")),
                    _pts(m.get("delta")),
                    f"[{_pts(m.get('ci_low'))}, {_pts(m.get('ci_high'))}]",
                    "–" if m.get("p_adjusted") is None else f"{m['p_adjusted']:.3g}",
                    m["status"],
                )
            console.print(table)
        colour = {"pass": "green", "fail": "red", "inconclusive": "yellow", "error": "magenta"}.get(verdict["verdict"], "white")
        console.print(f"[bold {colour}]Verdict: {verdict['verdict'].upper()}[/]")
        for reason in verdict.get("reasons", []):
            console.print(f"  • {reason}")
        for note in verdict.get("notes", []):
            console.print(f"  [dim]• {note}[/]")
