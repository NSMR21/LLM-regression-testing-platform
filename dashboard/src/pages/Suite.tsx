import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type TrendPoint } from "../api";
import { Dots, ErrorBox, Loading, pct, VerdictBadge, when } from "../components/ui";

const VERDICT_COLOUR: Record<string, string> = { pass: "var(--pass)", fail: "var(--fail)", inconclusive: "var(--warn)", error: "var(--error)" };

interface ChartRow extends TrendPoint {
  label: string;
  band: [number, number] | null;
  promoted: boolean;
}

function TrendTooltip({ active, payload }: { active?: boolean; payload?: { payload: ChartRow }[] }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="card" style={{ padding: 10, margin: 0, fontSize: 12 }}>
      <div><b>{pct(p.value)}</b> {p.ci ? <span className="muted">CI {pct(p.ci[0], 0)}–{pct(p.ci[1], 0)}</span> : null}</div>
      <div className="muted">{when(p.created_at)} · {p.prompt} · {p.model}</div>
      <div style={{ marginTop: 4 }}><VerdictBadge verdict={p.verdict} /> {p.promoted ? <span className="badge b-active">baseline</span> : null} {p.config_changed ? <span className="badge b-neutral">config changed</span> : null}</div>
    </div>
  );
}

function TrendDot(props: { cx?: number; cy?: number; payload?: ChartRow }) {
  const { cx, cy, payload } = props;
  if (cx === undefined || cy === undefined || !payload) return null;
  const colour = VERDICT_COLOUR[payload.verdict ?? ""] ?? "var(--muted)";
  return (
    <g>
      {payload.promoted ? <circle cx={cx} cy={cy} r={8} fill="none" stroke="var(--accent)" strokeWidth={2} /> : null}
      <circle cx={cx} cy={cy} r={payload.config_changed ? 4.5 : 3.5} fill={colour} stroke="var(--surface)" strokeWidth={1.5} />
    </g>
  );
}

export function SuitePage() {
  const { suite = "" } = useParams();
  const navigate = useNavigate();
  const [metric, setMetric] = useState("pass_rate");
  const suites = useQuery({ queryKey: ["suites"], queryFn: api.suites });
  const info = suites.data?.find((s) => s.name === suite);
  const trends = useQuery({ queryKey: ["trends", suite, metric], queryFn: () => api.trends(suite, metric) });
  const latest = useQuery({ queryKey: ["runs", suite, "latest"], queryFn: () => api.runs({ suite, status: "completed", limit: 1 }) });
  const latestId = latest.data?.[0]?.run_id;
  const latestRun = useQuery({ queryKey: ["run", latestId], queryFn: () => api.run(latestId!), enabled: !!latestId });
  const latestTrials = useQuery({ queryKey: ["trials", latestId], queryFn: () => api.trials(latestId!), enabled: !!latestId });

  const rows: ChartRow[] = useMemo(() => {
    const promoted = new Set(trends.data?.baseline_events.map((e) => e.run_id));
    return (trends.data?.points ?? []).map((p) => ({
      ...p,
      label: new Date(p.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }),
      band: p.ci ? [p.ci[0], p.ci[1]] : null,
      promoted: promoted.has(p.run_id),
    }));
  }, [trends.data]);

  const caseRows = useMemo(() => {
    const byCase = new Map<string, (boolean | null)[]>();
    for (const t of latestTrials.data ?? []) byCase.set(t.case_key, [...(byCase.get(t.case_key) ?? []), t.passed]);
    return (latestRun.data?.suite_spec.cases ?? []).map((c) => {
      const outcomes = byCase.get(c.key) ?? [];
      const rate = outcomes.length ? outcomes.filter(Boolean).length / outcomes.length : null;
      return { ...c, outcomes, rate };
    });
  }, [latestRun.data, latestTrials.data]);

  const metrics = ["pass_rate", ...(info?.graders.map((g) => g.id) ?? [])];

  return (
    <>
      <div className="header">
        <div>
          <div className="sub"><Link to="/suites">Suites</Link></div>
          <h1>{suite}</h1>
          <div className="sub">{info ? `${info.cases ?? "?"} cases · ${info.runs} runs` : null}</div>
        </div>
        <div className="toolbar">
          <label className="sub" htmlFor="metric">Metric</label>
          <select id="metric" value={metric} onChange={(e) => setMetric(e.target.value)}>
            {[...new Set(metrics)].map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      </div>

      <div className="card">
        <h2>{metric} over time</h2>
        <div className="sub" style={{ marginBottom: 8 }}>
          Band: 95% CI across cases. Dot colour: verdict. Ring: promoted to a baseline. Larger dot: configuration changed.
        </div>
        {trends.isLoading ? <Loading /> : trends.error ? <ErrorBox error={trends.error} /> : null}
        {rows.length ? (
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer>
              <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }} onClick={(e) => {
                const row = (e as { activePayload?: { payload: ChartRow }[] } | null)?.activePayload?.[0]?.payload;
                if (row) navigate(`/runs/${row.run_id}`);
              }}>
                <CartesianGrid stroke="var(--border)" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--border)" />
                <YAxis domain={[(min: number) => Math.max(0, Math.floor((min - 0.05) * 10) / 10), 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fill: "var(--muted)", fontSize: 11 }} stroke="var(--border)" width={44} />
                <Tooltip content={<TrendTooltip />} />
                <Area dataKey="band" stroke="none" fill="var(--accent)" fillOpacity={0.12} isAnimationActive={false} />
                <Line dataKey="value" stroke="var(--accent)" strokeWidth={2} dot={<TrendDot />} activeDot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        ) : trends.data ? <div className="empty">No completed runs yet.</div> : null}
      </div>

      <div className="card">
        <h2>Cases {latestId ? <span className="sub">· latest run <Link className="mono" to={`/runs/${latestId}`}>{latestId.slice(4, 12)}</Link></span> : null}</h2>
        {latestTrials.isLoading ? <Loading /> : null}
        <div className="table-wrap">
          <table>
            <thead><tr><th>Case</th><th>Tags</th><th className="num">Pass rate</th><th>Repeats</th></tr></thead>
            <tbody>
              {caseRows.map((c) => (
                <tr key={c.key}>
                  <td className="mono"><Link to={`/suites/${encodeURIComponent(suite)}/cases/${encodeURIComponent(c.key)}`}>{c.key}</Link>{c.critical ? <span className="badge b-fail" style={{ marginLeft: 6 }}>critical</span> : null}</td>
                  <td className="small muted">{c.tags.join(", ")}</td>
                  <td className="num">{pct(c.rate, 0)}</td>
                  <td><Dots outcomes={c.outcomes} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
