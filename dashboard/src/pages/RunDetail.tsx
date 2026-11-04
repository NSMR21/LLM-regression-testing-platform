import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ACTIVE, api, type CaseDiff, type RunDetail, type Trial } from "../api";
import { ForestPlot } from "../components/ForestPlot";
import { CategoryBadge, Dots, ErrorBox, Kpi, Loading, pct, Progress, pts, shortId, StatusBadge, VerdictBadge, when } from "../components/ui";

function groupTrials(trials: Trial[] | undefined): Map<string, Trial[]> {
  const map = new Map<string, Trial[]>();
  for (const t of trials ?? []) {
    const list = map.get(t.case_key) ?? [];
    list.push(t);
    map.set(t.case_key, list);
  }
  for (const list of map.values()) list.sort((a, b) => a.repeat_idx - b.repeat_idx);
  return map;
}

function TrialOutputs({ title, trials }: { title: string; trials: Trial[] | undefined }) {
  if (!trials) return <Loading />;
  return (
    <div>
      <h3>{title}</h3>
      {trials.map((t) => (
        <div key={t.trial_id} style={{ marginBottom: 10 }}>
          <div className="small muted">
            repeat {t.repeat_idx + 1} · {t.error ? <span style={{ color: "var(--error)" }}>error</span> : t.passed ? <span style={{ color: "var(--pass)" }}>pass</span> : <span style={{ color: "var(--fail)" }}>fail</span>}
            {t.latency_ms ? ` · ${Math.round(t.latency_ms)} ms` : ""}
            {Object.entries(t.scores)
              .filter(([, s]) => !s.passed && s.rationale)
              .map(([gid, s]) => ` · ${gid}: ${s.rationale}`)
              .join("")}
          </div>
          <pre className="output">{t.error ?? t.output}</pre>
        </div>
      ))}
    </div>
  );
}

function CaseRow({ run, item, trials }: { run: RunDetail; item: CaseDiff | { case_key: string; candidate: number }; trials: Trial[] }) {
  const [open, setOpen] = useState(false);
  const baselineId = run.verdict_report?.baseline_run_id ?? null;
  const baseTrials = useQuery({
    queryKey: ["trials", baselineId, item.case_key],
    queryFn: () => api.trials(baselineId!, item.case_key),
    enabled: open && !!baselineId,
  });
  const diff = "delta" in item ? item : null;
  const case_ = run.suite_spec.cases.find((c) => c.key === item.case_key);
  return (
    <Fragment>
      <tr className="clickable" onClick={() => setOpen(!open)}>
        <td className="mono">
          {open ? "▾" : "▸"} {item.case_key}
          {case_?.critical ? <span className="badge b-fail" style={{ marginLeft: 6 }}>critical</span> : null}
        </td>
        <td>{diff ? <CategoryBadge category={diff.category} /> : null}</td>
        {baselineId ? <td className="num">{diff ? pct(diff.baseline, 0) : "–"}</td> : null}
        <td className="num">{pct(item.candidate, 0)}</td>
        {baselineId ? <td className="num">{diff ? pts(diff.delta, 0) : "–"}</td> : null}
        <td>
          <Dots outcomes={trials.map((t) => t.passed)} errors={trials.map((t) => !!t.error)} />
        </td>
        <td className="small muted">{case_?.tags.join(", ")}</td>
      </tr>
      {open ? (
        <tr>
          <td colSpan={7} style={{ background: "var(--surface-2)" }}>
            <div className="small" style={{ marginBottom: 8 }}>
              <b>Input</b> <code>{JSON.stringify(case_?.input)}</code> · <b>Expected</b> <code>{JSON.stringify(case_?.expected)}</code> ·{" "}
              <Link to={`/suites/${encodeURIComponent(run.suite)}/cases/${encodeURIComponent(item.case_key)}`}>history</Link>
            </div>
            <div className={baselineId ? "split" : ""}>
              {baselineId ? <TrialOutputs title="Baseline" trials={baseTrials.data} /> : null}
              <TrialOutputs title="This run" trials={trials} />
            </div>
          </td>
        </tr>
      ) : null}
    </Fragment>
  );
}

function Actions({ run }: { run: RunDetail }) {
  const client = useQueryClient();
  const navigate = useNavigate();
  const [name, setName] = useState("main");
  const done = () => client.invalidateQueries();
  const promote = useMutation({ mutationFn: () => api.promote(run.run_id, name), onSuccess: done });
  const rerun = useMutation({ mutationFn: () => api.rerun(run.run_id), onSuccess: (r) => navigate(`/runs/${r.run_id}`) });
  const rescore = useMutation({ mutationFn: () => api.rescore(run.run_id), onSuccess: (r) => navigate(`/runs/${r.run_id}`) });
  const cancel = useMutation({ mutationFn: () => api.cancel(run.run_id), onSuccess: done });
  const error = promote.error ?? rerun.error ?? rescore.error ?? cancel.error;
  return (
    <div>
      <div className="toolbar">
        {run.status === "completed" ? (
          <>
            <input value={name} onChange={(e) => setName(e.target.value)} style={{ width: 90 }} aria-label="Baseline name" />
            <button onClick={() => promote.mutate()} disabled={promote.isPending || !name}>
              {promote.isSuccess ? "Promoted ✓" : "Set as baseline"}
            </button>
            <button onClick={() => rerun.mutate()} disabled={rerun.isPending} title="Execute the stored config again with the same seeds">
              Reproduce
            </button>
            <button onClick={() => rescore.mutate()} disabled={rescore.isPending} title="Re-grade stored outputs with current graders">
              Re-score
            </button>
          </>
        ) : null}
        {ACTIVE.includes(run.status) ? <button onClick={() => cancel.mutate()}>Cancel</button> : null}
      </div>
      {error ? <div style={{ marginTop: 8 }}><ErrorBox error={error} /></div> : null}
    </div>
  );
}

export function RunDetailPage() {
  const { runId = "" } = useParams();
  const [tab, setTab] = useState<"cases" | "config" | "flaky">("cases");
  const [filter, setFilter] = useState<string>("all");
  const run = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.run(runId),
    refetchInterval: (q) => (q.state.data && ACTIVE.includes(q.state.data.status) ? 2000 : false),
  });
  const trials = useQuery({ queryKey: ["trials", runId], queryFn: () => api.trials(runId), enabled: run.data?.status === "completed" });
  const byCase = useMemo(() => groupTrials(trials.data), [trials.data]);

  if (run.isLoading) return <Loading />;
  if (run.error || !run.data) return <ErrorBox error={run.error} />;
  const r = run.data;
  const report = r.verdict_report;
  const summary = r.summary;
  const margin = report?.policy.non_inferiority_margin ?? r.suite_spec.verdict.non_inferiority_margin;
  const rows: (CaseDiff | { case_key: string; candidate: number })[] = report?.case_diff.length
    ? report.case_diff
    : [...byCase.entries()].map(([key, list]) => ({ case_key: key, candidate: list.filter((t) => t.passed).length / list.length }));
  const filtered = rows.filter((row) => filter === "all" || ("category" in row && row.category === filter));
  const categories = report?.case_diff.length ? ["all", "broken", "fixed", "flaky", "changed", "stable_fail", "stable_pass"] : [];
  const changed = Object.entries(report?.config_diff ?? {}).filter(([k]) => !k.startsWith("model.resolved.details"));

  return (
    <>
      <div className="header">
        <div>
          <div className="sub">
            <Link to={`/suites/${encodeURIComponent(r.suite)}`}>{r.suite}</Link> · {r.prompt} · {r.model}
          </div>
          <h1>
            Run <span className="mono">{shortId(r.run_id)}</span>{" "}
            {r.kind !== "run" ? <span className="badge b-neutral">{r.kind.replace("_", " ")}</span> : null}
          </h1>
          <div className="sub">
            {when(r.created_at)} · <StatusBadge status={r.status} />
            {r.baseline_run_id ? (
              <>
                {" "}· compared with <Link to={`/runs/${r.baseline_run_id}`} className="mono">{shortId(r.baseline_run_id)}</Link>
                {r.baseline_name ? ` (${r.baseline_name})` : ""}
              </>
            ) : null}
            {r.source_run_id ? (
              <> · from <Link to={`/runs/${r.source_run_id}`} className="mono">{shortId(r.source_run_id)}</Link></>
            ) : null}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ marginBottom: 8 }}><VerdictBadge verdict={r.verdict} big /></div>
          <Actions run={r} />
        </div>
      </div>

      {ACTIVE.includes(r.status) ? (
        <div className="card">
          <div className="sub" style={{ marginBottom: 6 }}>
            {r.status}: executed {r.progress.executed}/{r.progress.total}, graded {r.progress.graded}/{r.progress.total}
          </div>
          <Progress done={r.progress.executed + r.progress.graded} total={2 * r.progress.total} />
        </div>
      ) : null}
      {r.error ? <div className="card"><ErrorBox error={r.error} /></div> : null}

      {report && (report.reasons.length || report.notes.length) ? (
        <div className="card">
          <h2>Why this verdict</h2>
          <ul className="reasons">{report.reasons.map((x) => <li key={x}>{x}</li>)}</ul>
          <ul className="reasons notes">{report.notes.map((x) => <li key={x}>{x}</li>)}</ul>
        </div>
      ) : null}

      {summary ? (
        <div className="kpis" style={{ marginBottom: 16 }}>
          <Kpi
            label="Pass rate"
            value={pct(summary.metrics.pass_rate)}
            hint={summary.metric_ci.pass_rate ? `95% CI ${pct(summary.metric_ci.pass_rate[0], 0)}–${pct(summary.metric_ci.pass_rate[1], 0)}` : undefined}
          />
          <Kpi label="Detectable drop" value={pct(summary.mde)} hint="80% power, this run's noise" />
          <Kpi label="Flaky cases" value={`${summary.flaky_cases.length}/${summary.n_cases}`} hint={`${summary.repeats} repeats each`} />
          <Kpi label="Error rate" value={pct(summary.error_rate)} />
          <Kpi label="Latency p50 / p95" value={summary.latency_p50_ms ? `${Math.round(summary.latency_p50_ms)} ms` : "–"} hint={summary.latency_p95_ms ? `p95 ${Math.round(summary.latency_p95_ms)} ms` : undefined} />
        </div>
      ) : null}

      {report?.metrics.length ? (
        <div className="card">
          <h2>Change versus baseline</h2>
          <div className="sub" style={{ marginBottom: 8 }}>
            Paired per-case differences with 95% confidence intervals. The shaded band is the tolerated drop ({pct(margin, 0)}).
          </div>
          <ForestPlot metrics={report.metrics} margin={margin} />
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <thead>
                <tr>
                  <th>Metric</th><th className="num">Baseline</th><th className="num">This run</th><th className="num">Δ pts</th>
                  <th className="num">95% CI</th><th className="num">p (adj.)</th><th className="num">Cases</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {report.metrics.map((m) => (
                  <tr key={m.name}>
                    <td>{m.gated ? <b>{m.name}</b> : m.name}{m.gated ? <span className="small muted"> gated</span> : null}</td>
                    <td className="num">{pct(m.baseline)}</td>
                    <td className="num">{pct(m.candidate)}</td>
                    <td className="num">{pts(m.delta)}</td>
                    <td className="num">[{pts(m.ci_low)}, {pts(m.ci_high)}]</td>
                    <td className="num">{m.p_adjusted.toPrecision(2)}</td>
                    <td className="num">{m.n_cases}</td>
                    <td><span className={`badge ${m.status === "fail" || m.status === "worse" ? "b-fail" : m.status === "pass" || m.status === "better" ? "b-pass" : m.status === "inconclusive" ? "b-inconclusive" : "b-neutral"}`}>{m.status.replace("_", " ")}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      {report?.gates.length || (report?.graders && Object.keys(report.graders).length) ? (
        <div className="row">
          {report?.gates.length ? (
            <div className="card">
              <h2>Gates</h2>
              <table>
                <tbody>
                  {report.gates.map((g) => (
                    <tr key={g.metric}>
                      <td>{g.metric}</td>
                      <td className="num">{g.value === null ? "–" : g.value.toFixed(3)}</td>
                      <td className="small muted">{g.min !== null ? `≥ ${g.min}` : `≤ ${g.max}`}</td>
                      <td>{g.passed ? <span className="badge b-pass">ok</span> : <span className="badge b-fail">failed</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          {report?.graders && Object.keys(report.graders).length ? (
            <div className="card">
              <h2>Model-based graders</h2>
              <table>
                <tbody>
                  {Object.entries(report.graders).map(([gid, info]) => (
                    <tr key={gid}>
                      <td>{gid}</td>
                      <td><span className={`badge ${info.trust === "calibrated" ? "b-pass" : info.trust === "weak" ? "b-inconclusive" : "b-neutral"}`}>{info.trust}</span></td>
                      <td className="small muted">{info.kappa !== undefined ? `κ = ${info.kappa.toFixed(2)} on ${info.n} labels` : `${info.n} labels`}</td>
                      <td><Link to={`/label?suite=${encodeURIComponent(r.suite)}&grader=${encodeURIComponent(gid)}`}>label</Link></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="card">
        <div className="tabs">
          <button className={tab === "cases" ? "on" : ""} onClick={() => setTab("cases")}>Cases</button>
          <button className={tab === "flaky" ? "on" : ""} onClick={() => setTab("flaky")}>Flaky ({summary?.flaky_cases.length ?? 0})</button>
          <button className={tab === "config" ? "on" : ""} onClick={() => setTab("config")}>Configuration{changed.length ? ` (${changed.length} changed)` : ""}</button>
        </div>
        {tab === "cases" ? (
          <>
            {categories.length ? (
              <div className="toolbar" style={{ marginBottom: 8 }}>
                {categories.map((c) => (
                  <button key={c} className={filter === c ? "primary" : ""} onClick={() => setFilter(c)}>
                    {c.replace("_", " ")} {c === "all" ? rows.length : rows.filter((x) => "category" in x && x.category === c).length}
                  </button>
                ))}
              </div>
            ) : null}
            {trials.isLoading ? <Loading /> : null}
            {r.status !== "completed" ? <div className="empty">Cases appear when the run completes.</div> : null}
            {r.status === "completed" ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Case</th><th></th>{report?.baseline_run_id ? <th className="num">Baseline</th> : null}<th className="num">This run</th>
                      {report?.baseline_run_id ? <th className="num">Δ</th> : null}<th>Repeats</th><th>Tags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((row) => (
                      <CaseRow key={row.case_key} run={r} item={row} trials={byCase.get(row.case_key) ?? []} />
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </>
        ) : null}
        {tab === "flaky" ? (
          summary?.flaky_cases.length ? (
            <table>
              <thead><tr><th>Case</th><th className="num">Pass rate</th><th className="num">Flakiness</th><th>Repeats</th></tr></thead>
              <tbody>
                {summary.flaky_cases.map((f) => (
                  <tr key={f.case_key}>
                    <td className="mono"><Link to={`/suites/${encodeURIComponent(r.suite)}/cases/${encodeURIComponent(f.case_key)}`}>{f.case_key}</Link></td>
                    <td className="num">{pct(f.pass_rate, 0)}</td>
                    <td className="num">{f.score.toFixed(2)}</td>
                    <td><Dots outcomes={(byCase.get(f.case_key) ?? []).map((t) => t.passed)} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty">No case changed outcome between repeats.</div>
          )
        ) : null}
        {tab === "config" ? (
          <div className="split">
            <div>
              <h3>Frozen configuration</h3>
              <dl className="kv">
                <dt>fingerprint</dt><dd className="mono">{r.fingerprint}</dd>
                <dt>run seed</dt><dd className="mono">{r.run_seed}</dd>
                <dt>model</dt><dd>{String(r.snapshot.model?.provider)} / {String(r.snapshot.model?.model)}</dd>
                <dt>digest</dt><dd className="mono">{String(r.snapshot.model?.resolved?.digest ?? "–")}</dd>
                <dt>params</dt><dd className="mono">{JSON.stringify(r.snapshot.model?.params)}</dd>
                <dt>graders</dt><dd className="mono">{JSON.stringify(r.snapshot.graders?.versions)}</dd>
                <dt>repeats</dt><dd>{String(r.execution.repeats)}</dd>
                <dt>server</dt><dd>{String(r.environment.provider_version ?? "–")}</dd>
              </dl>
              <h3 style={{ marginTop: 16 }}>Prompt</h3>
              {r.prompt_spec.system ? <pre className="output">{r.prompt_spec.system}</pre> : null}
              <pre className="output">{r.prompt_spec.template}</pre>
            </div>
            <div>
              <h3>Changed since baseline</h3>
              {changed.length ? (
                <table>
                  <tbody>
                    {changed.map(([key, v]) => (
                      <tr key={key}>
                        <td className="mono small">{key}</td>
                        <td className="small"><pre className="output" style={{ margin: 0, maxHeight: 140 }}>{String(JSON.stringify(v.baseline) ?? "–").slice(0, 600)}</pre></td>
                        <td className="small"><pre className="output" style={{ margin: 0, maxHeight: 140 }}>{String(JSON.stringify(v.candidate) ?? "–").slice(0, 600)}</pre></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="empty">{report?.baseline_run_id ? "Identical configuration." : "No baseline."}</div>
              )}
              {report && Object.keys(report.excluded_cases).length ? (
                <>
                  <h3 style={{ marginTop: 16 }}>Excluded from comparison</h3>
                  <ul className="reasons">
                    {Object.entries(report.excluded_cases).map(([k, why]) => <li key={k}><code>{k}</code>: {why}</li>)}
                  </ul>
                </>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    </>
  );
}
